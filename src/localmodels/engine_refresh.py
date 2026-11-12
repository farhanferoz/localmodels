"""Refresh, calibration and gates for the container engines (vLLM, SGLang, TensorRT-LLM).

``refresh.py`` is the llama.cpp refresh and calls into this module, loaded only when it runs
(design 4.2, 4.3 and 2's bootstrap and sparkrun move: docs/research/engine-registry-design-2026-11-08.md).

- Recipes: our own recipe per engine for the measurement model, kept as JSON under
  ``$LOCALMODELS_STATE_DIR/engines/<engine>/recipes/`` and realized per candidate into its version record.
- The walk (``EngineWalk``, from ``refresh run``): reap, prune, the sparkrun move check, then per version
  source the newest never-tried tag, as one chain of slurm jobs: pull (CPU), measure (GPU), and for a first
  ``current`` the adjacent version's pull and the calibration job; the llama.cpp same-client job last.
- The jobs (``refresh engine-pull|engine-measure|engine-calibrate|same-client|sparkrun-compare``): classes
  sharing one base for the quiet card, the SIGTERM stop and the result file.
- The gates: pure functions (``judge_engine``, ``compare_scores``, ``calibrated_limits``).

Nothing here starts a container except inside a job, and a job runs only under slurm.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import re
import shutil
import signal
import statistics
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Final, NamedTuple

import httpx
from pydantic import BaseModel, Field

from . import engine_bench, engines, refresh, retune
from .atomic import read_model, write_model, write_text_atomically
from .container_run import (
    SEED_NEED_GIB,
    SPARKRUN_REPO,
    ContainerRun,
    MemoryNeed,
    Reaper,
    RunKind,
    ensure_sparkrun,
    recipe_problems,
    record_result,
    run_sparkrun,
)
from .engines import (
    CONTAINER_ENGINES,
    GateVerdict,
    Publisher,
    TagListing,
    VersionRecord,
    VersionSource,
    docker_runner,
    engines_dir,
    list_tags,
    matching_tags,
    newest_untried,
    prune,
    read_record,
    records,
    run_command,
    sources_for,
    switch,
    tag_key,
    versions_dir,
    write_record,
)
from .enums import ModelSizeClass
from .launcher import LocalServerManager, MemoryGuards, ServerStartError
from .lease import DEFAULT_LEASE_DIR, GpuLease, QuietCard, QuietCardTimeoutError
from .llama_builds import current_commit, current_server, link_target, llama_home
from .llama_tune import KILL_BELOW_GIB, OOM_SCORE_ADJ
from .params import ServingParams
from .process import ServerSpec
from .stack import Engine

if TYPE_CHECKING:
    import argparse
    from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
    from contextlib import AbstractContextManager

    from .engine_bench import EngineRun, TeacherForced
    from .engines import Fetch

log = logging.getLogger(__name__)

EXIT_OK: Final = 0
EXIT_ERROR: Final = 1

# --- constants (decided in the E4 brief) -----------------------------------------------------------

MEASUREMENT_MODEL: Final = "nvidia/Qwen3-30B-A3B-NVFP4"
MEASUREMENT_REVISION: Final = "2538ded2a4edb247b4d2b4a8ba24e44bd4c017c3"
#: The llama.cpp file measured beside each container model (the same model, the Q4_K_M file).
SAME_CLIENT_FILES: Final = {MEASUREMENT_MODEL: refresh.PRODUCTION_MODEL}
SPARKRUN_PINNED: Final = "v0.4.2"  # in use when there is no sparkrun state file
SPARKRUN_TAG: Final = re.compile(r"v(\d+)\.(\d+)\.(\d+)")
HOST_RESERVE_GIB: Final = 10.0  # D22's margin
GUARDS: Final = MemoryGuards(
    kill_below_gib=KILL_BELOW_GIB, host_reserve_gib=HOST_RESERVE_GIB, server_oom_score_adj=OOM_SCORE_ADJ
)
#: Hard stops by ``timeout``, seconds: a measure job (design 4.2, 40 minutes), a calibration, the same-client
#: job, and the sparkrun comparison for each engine that has a ``current``.
MEASURE_LIMIT_S: Final = 2400
CALIBRATE_LIMIT_S: Final = 3600
SAME_CLIENT_LIMIT_S: Final = 600
SPARKRUN_LIMIT_S: Final = 2400
JOB_MEM_MARGIN_GIB: Final = 4  # a GPU job's ``--mem`` is its whole need plus this
#: ``timeout``'s grace after its SIGTERM: a container's stop (sparkrun stop, docker rm, up to 120 s) and the
#: result file come first; refresh's own 30 s would SIGKILL a job mid-stop.
KILL_AFTER_S: Final = 180
PULL_MEM: Final = "4G"
PULL_CPUS: Final = 2
PULL_TIMEOUT_S: Final = 1800.0
MIN_FREE_DISK_GB: Final = 100
TOOL_FETCH_TIMEOUT_S: Final = 600.0  # uvx fetching llama-benchy into its cache
GB: Final = 1000**3
PPL_MARGIN: Final = 0.01  # D18's perplexity rule: the noise arms' largest ratio plus this
SAME_TOP_MARGIN: Final = 0.006  # D18: 0.6 points between the limit and the accepted switch
LLAMA_READY_S: Final = 600.0
STEP_OK: Final = refresh.STEP_OK
NOT_MEASURED: Final = refresh.NOT_MEASURED
SPEED_PREFIXES: Final = ("read_tps_prefill_", "write_tps_")
JUDGE_SPEED: Final = "judge_write_tps"
ADJACENT_REASON: Final = "adjacent version for calibration"


class EngineJob(StrEnum):
    """The job kinds; slurm's ``--name`` is ``<kind>-<engine>`` (no engine for the last two)."""

    PULL = "refresh-pull"
    MEASURE = "refresh-engine"
    CALIBRATE = "refresh-calibrate"
    SAME_CLIENT = "refresh-same-client"
    SPARKRUN = "refresh-sparkrun"


class MeasureMode(StrEnum):
    """Why a version is measured: a first ``current``, a candidate for it, or an approved file."""

    BOOTSTRAP = "bootstrap"
    CANDIDATE = "candidate"
    APPROVED = "approved"


class ResultKind(StrEnum):
    """The ``kind`` of a result entry in a version record: a measure mode, or one of the other starts."""

    BOOTSTRAP = "bootstrap"
    CANDIDATE = "candidate"
    APPROVED = "approved"
    CALIBRATION = "calibration"
    WARM = "warm"
    SPARKRUN = "sparkrun"


class CalibrationArm(StrEnum):
    """The starts whose scores a calibration compares with the reference start."""

    RESTART = "restart"  # same version, autotune cache reused
    EMPTY_CACHE = "empty-cache"
    AUTOTUNE_OFF = "autotune-off"  # SGLang only
    ADJACENT = "adjacent"  # the version before, the size of a real change


NOISE_ARMS: Final = (CalibrationArm.RESTART, CalibrationArm.EMPTY_CACHE, CalibrationArm.AUTOTUNE_OFF)


# --- recipes (design 4.2, 5) -----------------------------------------------------------------------

#: D16's SGLang serve command (run_all.sh row 4) as a recipe template, with the revision pin the Hub folder
#: needs: this machine's Hub cache has no ``refs/`` entry, so an engine started without ``--revision`` could
#: not resolve the model offline (sparkrun warns of it: "unpinned-model-revision").
SGLANG_COMMAND: Final = (
    "sglang serve --model-path {model} --revision {model_revision} --tp-size {tensor_parallel} "
    "--port {port} --mem-fraction-static {gpu_memory_utilization} --context-length {max_model_len} "
    "--max-running-requests {max_num_seqs} --served-model-name {served_model_name} "
    "--quantization {quantization}"
)
AUTOTUNE_OFF_FLAG: Final = "--disable-flashinfer-autotune"  # SGLang v0.5.23, arg_groups/fields/exec_.py:394


def _recipe(runtime: str, defaults: dict[str, Any], **more: Any) -> dict[str, Any]:
    """One base recipe: D16's serve settings (context 16,384, 16 sequences, memory fraction 0.30)."""
    return {
        "recipe_version": "2",
        "model": MEASUREMENT_MODEL,
        "model_revision": MEASUREMENT_REVISION,
        "runtime": runtime,
        "runtime_cache": {"key_by_image": True},
        **more,
        "defaults": {"port": 18102, "tensor_parallel": 1, "max_model_len": 16384, **defaults},
    }


BASE_RECIPES: Final[Mapping[Engine, dict[str, Any]]] = {
    # vllm/vllm-openai declares ENTRYPOINT ["vllm", "serve"], which takes the command sparkrun appends as its
    # own flags, so the server never starts (sparkrun v0.4.2's own diagnosis, job 2264; it advises this key).
    # The GB10 community image has no entrypoint, so clearing it changes nothing there. SGLang's and
    # TensorRT-LLM's images use NVIDIA's pass-through entrypoint script and keep it (registry, 2026-11-09).
    Engine.VLLM: _recipe(
        "vllm",
        {"max_num_seqs": 16, "gpu_memory_utilization": 0.30, "served_model_name": "qwen3"},
        executor_config={"entrypoint": ""},
    ),
    Engine.SGLANG: _recipe(
        "sglang",
        {
            "max_num_seqs": 16,
            "gpu_memory_utilization": 0.30,
            "served_model_name": "qwen3",
            "quantization": "modelopt_fp4",
        },
        command=SGLANG_COMMAND,
    ),
    # TensorRT-LLM has no served-name key: it serves the model's own name (the harness reads /v1/models).
    Engine.TRTLLM: _recipe(
        "trtllm", {"max_batch_size": 16, "free_gpu_memory_fraction": 0.30, "backend": "pytorch"}
    ),
}


def recipes_dir(engine: Engine) -> Path:
    """Where ``engine``'s recipe files are kept."""
    return engines_dir() / engine.value / "recipes"


def recipe_file(engine: Engine, model: str) -> Path:
    """The file of ``model``'s recipe (``<org>--<name>.json``)."""
    return recipes_dir(engine) / f"{model.replace('/', '--')}.json"


def install_recipes() -> list[Path]:
    """Write each base recipe that has no file yet; an existing file is never touched."""
    written = []
    for engine, recipe in BASE_RECIPES.items():
        path = recipe_file(engine, str(recipe["model"]))
        if not path.exists():
            written.append(write_text_atomically(path, json.dumps(recipe, indent=2) + "\n"))
    return written


def recipes_for(engine: Engine) -> dict[str, dict[str, Any]]:
    """Every readable, valid recipe of ``engine`` by model: each is a model the refresh measures."""
    found: dict[str, dict[str, Any]] = {}
    for path in sorted(recipes_dir(engine).glob("*.json")):
        try:
            recipe = json.loads(path.read_text(encoding="utf-8"))
            model = recipe["model"]
        except (OSError, ValueError, KeyError, TypeError) as exc:
            log.warning("recipe %s skipped: %s", path, exc)
            continue
        problems = recipe_problems(recipe) if isinstance(model, str) else ["no model"]
        if problems:
            log.warning("recipe %s skipped: %s", path, "; ".join(problems))
            continue
        found[model] = recipe
    return found


def realized(record: VersionRecord, model: str) -> dict[str, Any] | None:
    """The realized recipe of ``model`` in ``record``; None when there is none or it is unreadable."""
    try:
        recipe = json.loads(record.recipes.get(model, ""))
    except ValueError:
        return None
    return recipe if isinstance(recipe, dict) else None


def autotune_off_recipe(recipe: Mapping[str, Any]) -> dict[str, Any]:
    """SGLang's calibration arm: the serve command written in full, with FlashInfer's autotune off."""
    command = str(recipe.get("command") or SGLANG_COMMAND)
    return {**recipe, "command": f"{command} {AUTOTUNE_OFF_FLAG}"}


# --- the gates (design 4.3, D13) -------------------------------------------------------------------


def speed_figure_names(sizes: engine_bench.Sizes = engine_bench.FULL) -> tuple[str, ...]:
    """The speed figures a container engine is judged on, derived from what the harness produces at ``sizes``.

    Prompt reading at each size and writing at each concurrency, one request's decode, and the judge batch
    (all higher is better).
    """
    one = 1.0
    details: dict[str, Any] = {
        "decode": {"median_decode_tps": one},
        "conc": {
            str(c): {"total_tps_median": one, "ttft_p50_s_median": one, "ttft_max_s_median": one}
            for c in sizes.conc_levels
        },
        "prefill": {
            str(n): {"median_prefill_tps": one, "median_prompt_tokens": one} for n in sizes.prefill_tokens
        },
        "judge": dict.fromkeys(
            (
                "median_wall_s",
                "median_write_tps",
                "median_read_tps_est",
                "median_ttft_p50_s",
                "median_ttft_max_s",
            ),
            one,
        ),
    }
    return tuple(
        name
        for name in engine_bench.figures(details)
        if name.startswith(SPEED_PREFIXES) or name == JUDGE_SPEED
    )


class NotComparableError(ValueError):
    """Two teacher-forced scores that cannot be compared position by position."""


class ScoreComparison(BaseModel):
    """One score against another: the perplexity ratio and the share of positions with the same top token."""

    ppl_ratio: float
    same_top: float
    positions: int


def compare_scores(base: TeacherForced, other: TeacherForced) -> ScoreComparison:
    """``other`` against ``base``; ``NotComparableError`` when the chunks differ in number or length."""
    lengths = [len(chunk) for chunk in base.top_tokens]
    if base.chunks != other.chunks or lengths != [len(chunk) for chunk in other.top_tokens]:
        msg = (
            f"{base.chunks} chunks of {lengths} positions against {other.chunks} chunks of "
            f"{[len(chunk) for chunk in other.top_tokens]}"
        )
        raise NotComparableError(msg)
    positions = sum(lengths)
    if not positions:
        msg = "no scored positions"
        raise NotComparableError(msg)
    if not all(math.isfinite(s.perplexity) and s.perplexity > 0 for s in (base, other)):
        msg = f"perplexity {base.perplexity} against {other.perplexity}: not a finite positive number"
        raise NotComparableError(msg)
    same = sum(
        a == b
        for chunk_a, chunk_b in zip(base.top_tokens, other.top_tokens, strict=True)
        for a, b in zip(chunk_a, chunk_b, strict=True)
    )
    return ScoreComparison(
        ppl_ratio=other.perplexity / base.perplexity, same_top=same / positions, positions=positions
    )


class ArmScore(BaseModel):
    """What one calibration arm scored against the reference start."""

    ppl_ratio: float
    same_top: float


class EngineLimits(BaseModel):
    """``engines/<engine>/limits.json`` (D14): the quality limits of one engine, from its calibration."""

    engine: Engine
    ppl_ratio_max: float
    same_top_min: float
    calibrated_on: str  # the tag
    adjacent: str  # the tag before it
    arms: dict[str, ArmScore]
    run: str = ""
    slurm_job: str = ""
    calibrated_at: str = ""


def limits_path(engine: Engine) -> Path:
    """Where ``engine``'s limits are kept."""
    return engines_dir() / engine.value / "limits.json"


def read_limits(engine: Engine) -> EngineLimits | None:
    """``engine``'s limits; None when it is not calibrated."""
    return read_model(limits_path(engine), EngineLimits)


def calibrated_limits(arms: Mapping[str, ArmScore]) -> tuple[float, float]:
    """(``ppl_ratio_max``, ``same_top_min``) from the arms (design 4.3).

    The perplexity ratio may be the noise arms' largest plus ``PPL_MARGIN``. The same top token must be at
    least the adjacent pair's figure minus ``SAME_TOP_MARGIN`` and never above the noise arms' lowest.
    """
    noise = [score for name, score in arms.items() if name != CalibrationArm.ADJACENT]
    adjacent = arms[CalibrationArm.ADJACENT]
    return (
        max(score.ppl_ratio for score in noise) + PPL_MARGIN,
        min(adjacent.same_top - SAME_TOP_MARGIN, *(score.same_top for score in noise)),
    )


class EngineVerdict(BaseModel):
    """The container gate's verdict on a candidate (or a bootstrap) of one model."""

    gate: GateVerdict
    speed_failures: list[str] = Field(default_factory=list)
    quality_failures: list[str] = Field(default_factory=list)
    slowdown: float = 0.0
    figures: dict[str, dict[str, list[float]]] = Field(default_factory=dict)
    quality: dict[str, float] = Field(default_factory=dict)  # the worst ppl_ratio and same_top over the pairs
    noise: dict[str, float] = Field(default_factory=dict)  # current's round 1 against its round 2
    notes: str = ""

    @property
    def reason(self) -> str:
        """The first two failures, else the notes: what a record's ``reason`` says."""
        if self.gate is GateVerdict.NOT_RUN:
            return self.notes or self.gate.value
        return "; ".join([*self.speed_failures, *self.quality_failures][:2]) or self.notes or self.gate.value


def _quality(worst: Sequence[ScoreComparison]) -> dict[str, float]:
    return {"ppl_ratio": max(c.ppl_ratio for c in worst), "same_top": min(c.same_top for c in worst)}


def judge_engine(
    current_rounds: Sequence[EngineRun], candidate_rounds: Sequence[EngineRun], limits: EngineLimits | None
) -> EngineVerdict:
    """D13 and design 4.3 for a container engine: speed, then teacher-forced quality against the limits.

    Rules in order: a speed failure fails; teacher-forced scoring missing on any round fails (an engine
    without it never switches automatically); no limits means the result is only recorded; a worse
    perplexity ratio or top-token share than the limits fails; otherwise it passes.
    """
    speed = refresh.judge_speed(
        speed_figure_names(), [r.figures for r in current_rounds], [r.figures for r in candidate_rounds]
    )
    quality_failures: list[str] = []
    quality: dict[str, float] = {}
    noise: dict[str, float] = {}
    current_scores = [r.teacher_forced for r in current_rounds if r.teacher_forced is not None]
    candidate_scores = [r.teacher_forced for r in candidate_rounds if r.teacher_forced is not None]
    if (
        not current_rounds
        or not candidate_rounds
        or len(current_scores) < len(current_rounds)
        or len(candidate_scores) < len(candidate_rounds)
    ):
        quality_failures.append("teacher-forced scoring not measured")
    else:
        try:
            quality = _quality(
                [compare_scores(c, k) for c, k in zip(current_scores, candidate_scores, strict=False)]
            )
            if len(current_scores) > 1:
                noise = _quality([compare_scores(current_scores[0], current_scores[1])])
        except NotComparableError as exc:
            quality_failures.append(f"teacher-forced scores not comparable: {exc}")
    if limits is not None and quality and not quality_failures:
        if quality["ppl_ratio"] > limits.ppl_ratio_max:
            quality_failures.append(
                f"perplexity ratio {quality['ppl_ratio']:.4f} over the limit {limits.ppl_ratio_max:.4f}"
            )
        if quality["same_top"] < limits.same_top_min:
            quality_failures.append(
                f"same top token {quality['same_top']:.4f} under the limit {limits.same_top_min:.4f}"
            )
    if speed.failures or quality_failures:
        gate = GateVerdict.FAILED
    elif limits is None:
        gate = GateVerdict.RECORD_ONLY
    else:
        gate = GateVerdict.PASSED
    notes = ", ".join(
        f"{name} {statistics.fmean(f['baseline']):.4g} -> {statistics.fmean(f['candidate']):.4g}"
        for name, f in speed.figures.items()
    )
    return EngineVerdict(
        gate=gate,
        speed_failures=speed.failures,
        quality_failures=quality_failures,
        slowdown=speed.slowdown,
        figures=speed.figures,
        quality=quality,
        noise=noise,
        notes=notes,
    )


def bootstrap_verdict(measured: EngineRun | None, why: str = "") -> EngineVerdict:
    """A first ``current``'s gate: it passes when every harness phase was measured (design 2)."""
    if measured is None:
        return EngineVerdict(gate=GateVerdict.FAILED, speed_failures=[f"not measured: {why or 'no start'}"])
    missing = []
    for phase in engine_bench.Phase:
        outcome = measured.phases.get(phase)
        if outcome is None or outcome.status is not engine_bench.PhaseStatus.MEASURED:
            missing.append(f"not measured: {phase.value}: {outcome.reason if outcome else 'not run'}")
    return EngineVerdict(
        gate=GateVerdict.FAILED if missing else GateVerdict.PASSED,
        speed_failures=missing,
        notes="every phase measured" if not missing else "",
    )


_SEVERITY: Final = {
    GateVerdict.FAILED: 3,
    GateVerdict.RECORD_ONLY: 2,
    GateVerdict.NOT_RUN: 1,
    GateVerdict.PASSED: 0,
}


def worst_verdict(verdicts: Iterable[EngineVerdict]) -> EngineVerdict:
    """The verdict that decides for a version measured on several models: the least favourable one."""
    return max(verdicts, key=lambda v: _SEVERITY[v.gate])


# --- the sparkrun move (design 2) ------------------------------------------------------------------

SPARKRUN_STATE: Final = "sparkrun.json"
#: Values that differ from one sparkrun run to the next, removed before two dry runs are compared.
_NORMALISE: Final = (
    (re.compile(r"sparkrun v\d[\w.+-]*"), "sparkrun <version>"),
    (re.compile(r"sparkrun_[0-9a-f]+_[0-9a-f]+"), "<cluster>"),
    (re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"), "<uuid>"),
    (re.compile(r"\d{4}-\d\d-\d\d[T ]\d\d:\d\d:\d\d(?:\.\d+)?(?:Z|[+-]\d\d:?\d\d)?"), "<time>"),
    (re.compile(r"/\S*localmodels-(?:hub|recipe)-\S+"), "<temp>"),
    (re.compile(r"/tmp/\S+"), "<temp>"),  # noqa: S108 - a pattern, not a path we use
    (re.compile(r"\b[0-9a-f]{8,}\b"), "<id>"),
    (re.compile(r"\b\d+(?:\.\d+)?s\b"), "<seconds>"),
)


DRY_RUN_OK: Final = "exit 0\n"  # the front of ``EngineWalk._dry_run_output`` for a dry run that worked
DOCKER_RUN_MARK: Final = "docker run command:"  # sparkrun's -vvv dry run: executors/docker.py's debug line
SERVE_MARK: Final = "Serve command:"


def launch_lines(text: str) -> str:
    """What a verbose dry run says would start: each ``docker run`` command and the serve command, normalised.

    The other lines (progress, timings, debug notes) differ between sparkrun versions without changing what
    runs. When neither line is there (sparkrun changed its output), the whole text is compared instead, so a
    format change counts as a difference (and gets the GPU comparison), never as a match.
    """
    lines = text.splitlines()
    picked: list[str] = []
    for number, line in enumerate(lines):
        for mark in (DOCKER_RUN_MARK, SERVE_MARK):
            if mark in line:
                rest = line.split(mark, 1)[1].strip()
                follow = lines[number + 1].strip() if number + 1 < len(lines) else ""
                picked.append(f"{mark} {rest or follow}")
    return normalise("\n".join(dict.fromkeys(picked)) if picked else text)


def normalise(text: str) -> str:
    """``text`` with every run-specific value (versions, ids, uuids, times, temporary paths) replaced."""
    for pattern, replacement in _NORMALISE:
        text = pattern.sub(replacement, text)
    return "\n".join(line.rstrip() for line in text.splitlines() if line.strip())


class SparkrunState(BaseModel):
    """``engines/sparkrun.json``: which sparkrun is in use, the one before, and the versions that failed."""

    version: str
    previous: str = ""
    tried: list[str] = Field(default_factory=list)
    reason: str = ""
    moved_at: str = ""


def sparkrun_state_path() -> Path:
    """Where the sparkrun state is kept."""
    return engines_dir() / SPARKRUN_STATE


def read_sparkrun_state() -> SparkrunState | None:
    """The sparkrun state; None when there is no state file."""
    return read_model(sparkrun_state_path(), SparkrunState)


def in_use_version() -> str:
    """The sparkrun version in use: the state file's, else the pinned one."""
    state = read_sparkrun_state()
    return state.version if state is not None else SPARKRUN_PINNED


def version_key(tag: str) -> tuple[int, int, int] | None:
    """The numbers of a ``v<x>.<y>.<z>`` sparkrun tag; None for any other form."""
    match = SPARKRUN_TAG.fullmatch(tag)
    return (int(match[1]), int(match[2]), int(match[3])) if match else None


def list_sparkrun_tags(*, runner: engines.Runner | None = None) -> TagListing:
    """Sparkrun's release tags by ``git ls-remote`` (60 s); never raises."""
    ask = runner or (lambda cmd: run_command(cmd, timeout=refresh.LS_REMOTE_TIMEOUT_S))
    done = ask(["git", "ls-remote", "--tags", SPARKRUN_REPO, "refs/tags/v*"])
    if done.returncode != 0:
        return TagListing(reason=f"git ls-remote exited {done.returncode}: {done.stderr.strip()[-200:]}")
    tags = {line.rpartition("refs/tags/")[2] for line in done.stdout.splitlines()}
    return TagListing(tags=tuple(sorted(t for t in tags if version_key(t) is not None)))


def judged_tags(engine: Engine) -> set[str]:
    """The tags the walk does not offer again: judged by a gate, or whose pull failed (D12's tried rule).

    A version that was pulled but never judged (its measure job did not get the card, or was stopped before
    its verdict) is offered again: it was not tried, only queued.
    """
    return {r.tag for r in records(engine) if r.gate is not GateVerdict.NOT_RUN or not r.image}


def newest_sparkrun(tags: Iterable[str]) -> str | None:
    """The newest release tag by number; None when there is none."""
    keyed = [(key, tag) for tag in tags if (key := version_key(tag)) is not None]
    return max(keyed)[1] if keyed else None


# --- the jobs ---------------------------------------------------------------------------------------


@dataclass
class Harness:
    """What a job starts and measures with: the real container run and harness, replaced by tests."""

    start: Callable[..., AbstractContextManager[Any]] = ContainerRun
    measure: Callable[..., EngineRun] = engine_bench.measure
    sparkrun: Callable[[str], Path] = ensure_sparkrun
    #: llama.cpp's ``llama-server`` on the same client:
    #: (server binary, model file, lease folder, llama-benchy's tokenizer) -> one run.
    llama: Callable[[Path, Path, Path, str], EngineRun] | None = None
    docker: engines.Runner = docker_runner  # ``docker rmi`` of the images a job leaves unkept


class JobResult(BaseModel):
    """What every GPU job writes: when, which slurm job, why nothing was measured, each step's outcome."""

    slurm_job: str = ""
    measured_at: str = ""
    not_measured: str = ""  # why nothing was measured (the card did not go quiet, a refusal)
    error: str = ""  # the process was stopped (slurm's timeout) or failed outside any step
    steps: dict[str, str] = Field(default_factory=dict)  # step -> "ok" or why it did not measure


def _slug(model: str) -> str:
    return model.replace("/", "--")


def engine_folder(run: Path, engine: Engine, tag: str) -> Path:
    """Where a run keeps the files of one engine version's jobs."""
    return run / "engines" / f"{engine.value}-{tag}"


class JobRunner:
    """The part every GPU job shares: the quiet card, SIGTERM as an exit, and the result file in ``finally``.

    A subclass sets ``kind``, ``result_file`` and ``quiet_reason`` and writes ``work()`` (and ``finish()``
    for what runs whatever happened). Every outcome is a recorded result and exits 0, the card not going
    quiet and an unexpected error included (D12's afterok rule: one job must not cancel the rest of the
    chain); only an interrupt (``timeout``'s SIGTERM) exits otherwise.

    ``subject`` is the version under judgment: a start that fails for a reason that is not its own (a
    refusal before launch for want of memory, or another version's start) leaves it unjudged
    (``not_judged``), to be offered again, rather than failed (D12's tried rule is for its own failures).
    """

    kind: ClassVar[EngineJob]
    result_file: ClassVar[str]
    quiet_reason: ClassVar[str]

    def __init__(
        self,
        run: Path,
        folder: Path,
        result: JobResult,
        *,
        harness: Harness | None = None,
        guards: MemoryGuards = GUARDS,
        run_kind: RunKind = RunKind.REFRESH,
    ) -> None:
        self.run_kind = run_kind
        self.run = run
        self.folder = folder
        self.result = result
        self.harness = harness or Harness()
        self.guards = guards
        #: Why no more containers start: nothing starts after a failed start or a guard kill (D22).
        self.stopped: str | None = None
        #: The version under judgment, and why it was not judged when the stop was not its own doing.
        self.subject: str | None = None
        self.not_judged: str | None = None

    def work(self) -> None:
        """The measurements, with the card quiet; fills ``self.result``."""
        raise NotImplementedError

    def finish(self) -> None:
        """What runs after the work, whatever happened (after the result file is written)."""

    def blameless(self, record: VersionRecord, sparkrun: str | None) -> bool:
        """Whether a failed start of ``record`` is not the fault of the version under judgment."""
        del sparkrun  # a sparkrun comparison judges the sparkrun version instead
        return record.tag != self.subject

    def execute(self) -> int:
        """Hold the card quiet and run the work; the result file is always written."""
        self.folder.mkdir(parents=True, exist_ok=True)
        self.result.measured_at = datetime.now(UTC).isoformat(timespec="seconds")
        self.result.slurm_job = _slurm_job()

        def _terminate(_signum: int, _frame: object) -> None:
            raise SystemExit(refresh.SIGTERM_EXIT)

        previous = signal.signal(signal.SIGTERM, _terminate)  # before the card; a run chains its own handler
        try:
            with QuietCard(GpuLease(DEFAULT_LEASE_DIR, refresh.LEASE_PROJECT), reason=self.quiet_reason):
                self.work()
        except QuietCardTimeoutError as exc:
            self.result.not_measured = str(exc)
        except Exception as exc:  # recorded; exit 0 so the rest of the chain still runs
            self.result.error = f"{type(exc).__name__}: {exc}"
            log.exception("%s failed", self.kind)
        except BaseException as exc:
            self.result.error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            signal.signal(signal.SIGTERM, previous)
            write_model(
                self.folder / self.result_file, self.result
            )  # before finish: a kill there loses nothing
            try:
                self.finish()
            except Exception:
                log.exception("%s: finishing failed", self.kind)
        sys.stdout.write(f"{self.kind}: {self.summary()}\n")
        return EXIT_OK

    def summary(self) -> str:
        """One line on the outcome."""
        bad = [f"{step}: {why}" for step, why in self.result.steps.items() if why != STEP_OK]
        return self.result.not_measured or "; ".join(bad[:2]) or "done"

    def serve(
        self,
        label: str,
        record: VersionRecord,
        model: str,
        recipe: dict[str, Any],
        phases: Sequence[engine_bench.Phase],
        kind: ResultKind,
        *,
        sparkrun: str | None = None,
    ) -> EngineRun | None:
        """Start the engine once, measure ``phases``, record the result; None when the start failed.

        A failed start or a guard kill ends every later start of the job (D22). The run is written to
        ``<label>.json`` and added to the version's record; the step says whether every phase measured.
        """
        steps = self.result.steps
        if self.stopped:
            steps[label] = f"{NOT_MEASURED}not started after: {self.stopped}"
            return None
        port = int((recipe.get("defaults") or {}).get("port", 8000))
        run: Any = None
        try:
            binary = self.harness.sparkrun(sparkrun or in_use_version())
            begun = time.monotonic()
            run = self.harness.start(record, model, recipe, binary, self.guards, kind=self.run_kind)
            with run as served:
                startup = time.monotonic() - begun  # sparkrun's --no-follow still waits for readiness
                measured = self.harness.measure(
                    f"http://127.0.0.1:{port}",
                    record.engine.value,
                    stamp=served.stamp(),
                    startup_s=startup,
                    phases=tuple(phases),
                    tokenizer=model,  # the Hub name: llama-benchy cannot load the served name's (slurm 2265)
                )
        except (ServerStartError, OSError) as exc:
            self.stopped = f"{label}: {type(exc).__name__}: {exc}"
            steps[label] = f"{NOT_MEASURED}{type(exc).__name__}: {exc}"
            refused = run is not None and not getattr(run, "launched", True)  # refused before sparkrun ran
            if refused or self.blameless(record, sparkrun):
                self.not_judged = self.not_judged or self.stopped
            return None
        if served.killed:
            self.stopped = f"{label}: the memory guard killed the container ({served.killed})"
            if self.blameless(record, sparkrun):
                self.not_judged = self.not_judged or self.stopped
        unmeasured = [
            f"{phase.value}: {outcome.reason if outcome else 'not run'}"
            for phase in phases
            if (outcome := measured.phases.get(phase)) is None
            or outcome.status is not engine_bench.PhaseStatus.MEASURED
        ]
        write_model(self.folder / f"{label}.json", measured)
        record_result(
            record,
            model,
            served,
            extra={
                "kind": kind.value,
                "job": self.result.slurm_job,
                "result_file": str(self.folder / f"{label}.json"),
                "complete": not unmeasured,
                "figures": measured.figures,
                "startup_s": measured.startup_s,
                "stamp": measured.stamp,
            },
        )
        steps[label] = f"{NOT_MEASURED}{unmeasured[0]}" if unmeasured else STEP_OK
        return measured


def _slurm_job() -> str:
    import os  # noqa: PLC0415 - only the job runner reads the environment

    return os.environ.get("SLURM_JOB_ID", "")


@dataclass(frozen=True)
class StartArm:
    """One side of an interleaved comparison: a name, the version's record, its recipe, a sparkrun version."""

    name: str
    record: VersionRecord
    recipe: dict[str, Any]
    sparkrun: str | None = None


def interleave(
    job: JobRunner, model: str, first: StartArm, second: StartArm, *, prefix: str, kind: ResultKind
) -> tuple[list[EngineRun | None], list[EngineRun | None]]:
    """Warm starts of both arms (unmeasured: each fills its own autotune cache), then the measured rounds.

    Rounds run first, second, first, second (D13's interleaving), every harness phase.
    """
    for arm in (first, second):
        job.serve(
            f"{prefix}warm-{arm.name}",
            arm.record,
            model,
            arm.recipe,
            (engine_bench.Phase.WARM,),
            ResultKind.WARM,
            sparkrun=arm.sparkrun,
        )
    rounds: dict[str, list[EngineRun | None]] = {first.name: [], second.name: []}
    for number in range(1, refresh.ROUNDS + 1):
        for arm in (first, second):
            rounds[arm.name].append(
                job.serve(
                    f"{prefix}r{number}-{arm.name}",
                    arm.record,
                    model,
                    arm.recipe,
                    tuple(engine_bench.Phase),
                    kind,
                    sparkrun=arm.sparkrun,
                )
            )
    return rounds[first.name], rounds[second.name]


def judge_rounds(
    job: JobRunner,
    engine: Engine,
    first: Sequence[EngineRun | None],
    second: Sequence[EngineRun | None],
    *,
    prefix: str,
) -> EngineVerdict:
    """The gate on interleaved rounds; "not measured" (fails) when a start did not give its round."""
    kept_first = [r for r in first if r is not None]
    kept_second = [r for r in second if r is not None]
    if len(kept_first) < len(first) or len(kept_second) < len(second):
        why = [f"{k}: {v}" for k, v in job.result.steps.items() if k.startswith(prefix) and v != STEP_OK]
        return EngineVerdict(gate=GateVerdict.FAILED, speed_failures=why or ["not measured"])
    return judge_engine(kept_first, kept_second, read_limits(engine))


class EngineMeasureResult(JobResult):
    """``measure.json``: what a measure job ran and decided for one engine version."""

    engine: str = ""
    tag: str = ""
    mode: MeasureMode = MeasureMode.CANDIDATE
    verdict: EngineVerdict | None = None
    switched: bool = False
    switch_problem: str = ""


class MeasureJob(JobRunner):
    """`refresh engine-measure`: bootstrap, candidate or approved-file measurement of one version."""

    kind = EngineJob.MEASURE
    result_file = "measure.json"
    quiet_reason = "container engine refresh measurement"

    def __init__(
        self,
        run: Path,
        engine: Engine,
        tag: str,
        mode: MeasureMode,
        *,
        model: str | None = None,
        harness: Harness | None = None,
        guards: MemoryGuards = GUARDS,
    ) -> None:
        self.engine, self.tag, self.mode, self.only_model = engine, tag, mode, model
        folder = engine_folder(run, engine, tag)
        if mode is MeasureMode.APPROVED and model:  # its own files: never over the candidate's measure.json
            folder = folder / "approved" / _slug(model)
        super().__init__(
            run,
            folder,
            EngineMeasureResult(engine=engine.value, tag=tag, mode=mode),
            harness=harness,
            guards=guards,
            run_kind=RunKind.BOOTSTRAP if mode is MeasureMode.BOOTSTRAP else RunKind.REFRESH,
        )
        self.subject = tag

    @property
    def outcome(self) -> EngineMeasureResult:
        """The result, typed."""
        assert isinstance(self.result, EngineMeasureResult)  # noqa: S101 - set in __init__
        return self.result

    def work(self) -> None:
        """Measure the version on every model; judge it; switch on a pass."""
        record = read_record(self.engine, self.tag)
        if record is None:
            self.result.not_measured = f"{engines.record_path(self.engine, self.tag)} is missing"
            return
        if not record.pulled:
            self.result.not_measured = (
                f"{self.tag}: its image is not on disk ({record.reason or 'not pulled'})"
            )
            return
        # The gate and the switch are the measurement model's (design 4.2); each other recipe file (an
        # approved one) is measured in its own APPROVED job, so an approved file never blocks a switch.
        model = self.only_model or (None if self.mode is MeasureMode.APPROVED else MEASUREMENT_MODEL)
        if model is None or model not in recipes_for(self.engine):
            self.result.not_measured = (
                f"{recipes_dir(self.engine)} holds no valid recipe for {model or 'a model'}"
            )
            return
        models = [model]
        current = link_target(versions_dir(self.engine))
        if self.mode is MeasureMode.APPROVED:
            self._approved(record, models, current)
            return
        # Decided now, not when the chain was planned: a version planned behind another engine version's
        # bootstrap in the same chain finds a `current` (or, when that bootstrap failed, none).
        mode = MeasureMode.CANDIDATE if current is not None else MeasureMode.BOOTSTRAP
        if mode is not self.mode:
            log.info("%s %s: planned as %s, measured as %s", self.engine.value, self.tag, self.mode, mode)
        self.outcome.mode = mode
        self.run_kind = RunKind.BOOTSTRAP if mode is MeasureMode.BOOTSTRAP else RunKind.REFRESH
        if mode is MeasureMode.CANDIDATE:
            self._candidate(record, models, current)
        else:
            self._bootstrap(record, models, current)

    def _recipes(self, record: VersionRecord, models: Sequence[str]) -> dict[str, dict[str, Any]]:
        found = {}
        for model in models:
            recipe = realized(record, model)
            if recipe is None:
                self.result.steps[f"{model}"] = f"{NOT_MEASURED}no realized recipe in {record.tag}'s record"
            else:
                found[model] = recipe
        return found

    def _candidate(self, record: VersionRecord, models: Sequence[str], current_tag: str | None) -> None:
        current = read_record(self.engine, current_tag) if current_tag else None
        if current is None or not current.pulled or current.tag == self.tag:
            self.result.not_measured = (
                f"{self.engine.value} has no `current` with its image on disk to compare {self.tag} with"
            )
            return
        verdicts = {}
        ours = self._recipes(record, models)
        for model, recipe in ours.items():
            theirs = realized(current, model)
            if theirs is None:  # current's gap, not the candidate's: it stays unjudged
                self.result.steps[model] = f"{NOT_MEASURED}no realized recipe in {current.tag}'s record"
                self.not_judged = self.not_judged or self.result.steps[model]
                continue
            prefix = f"{_slug(model)}-" if len(models) > 1 else ""
            cur, cand = interleave(
                self,
                model,
                StartArm("current", current, theirs),
                StartArm("candidate", record, recipe),
                prefix=prefix,
                kind=ResultKind.CANDIDATE,
            )
            verdicts[model] = judge_rounds(self, self.engine, cur, cand, prefix=prefix)
        self._conclude(record, verdicts)

    def _bootstrap(self, record: VersionRecord, models: Sequence[str], current_tag: str | None) -> None:
        if current_tag is not None:
            self.result.not_measured = f"{self.engine.value} already has a `current` ({current_tag})"
            return
        verdicts = {}
        for model, recipe in self._recipes(record, models).items():
            prefix = f"{_slug(model)}-" if len(models) > 1 else ""
            self.serve(
                f"{prefix}warm-{self.tag}", record, model, recipe, (engine_bench.Phase.WARM,), ResultKind.WARM
            )
            measured = self.serve(
                f"{prefix}r1-{self.tag}",
                record,
                model,
                recipe,
                tuple(engine_bench.Phase),
                ResultKind.BOOTSTRAP,
            )
            why = self.result.steps.get(f"{prefix}r1-{self.tag}", "")
            verdicts[model] = bootstrap_verdict(measured, why.removeprefix(NOT_MEASURED))
        self._conclude(record, verdicts)

    def _approved(self, record: VersionRecord, models: Sequence[str], current_tag: str | None) -> None:
        if record.tag != current_tag:
            self.result.not_measured = f"{self.tag} is not {self.engine.value}'s `current` ({current_tag})"
            return
        for model, recipe in self._recipes(record, models).items():
            self.serve(f"warm-{self.tag}", record, model, recipe, (engine_bench.Phase.WARM,), ResultKind.WARM)
            self.serve(
                f"r1-{self.tag}", record, model, recipe, tuple(engine_bench.Phase), ResultKind.APPROVED
            )

    def _conclude(self, record: VersionRecord, verdicts: Mapping[str, EngineVerdict]) -> None:
        """Record the verdict on the version and switch to it when it passed and is official."""
        worst = worst_verdict(
            verdicts.values()
            or [
                EngineVerdict(gate=GateVerdict.FAILED, speed_failures=["not measured: no model was measured"])
            ]
        )
        if self.not_judged and worst.gate is GateVerdict.FAILED:  # not its own failure: offered again
            worst = worst.model_copy(
                update={"gate": GateVerdict.NOT_RUN, "notes": f"not judged: {self.not_judged}"}
            )
        self.outcome.verdict = worst
        latest = read_record(self.engine, self.tag) or record
        write_record(latest.model_copy(update={"gate": worst.gate, "reason": worst.reason}))
        if worst.gate is GateVerdict.PASSED and record.publisher is Publisher.OFFICIAL:
            problem = switch(self.engine, self.tag)
            self.outcome.switched, self.outcome.switch_problem = problem is None, problem or ""

    def finish(self) -> None:
        """Remove the images nothing keeps (a failed candidate's goes at once)."""
        report = prune(self.engine, runner=self.harness.docker)
        log.info("%s prune: removed %s, failed %s", self.engine.value, report.removed, report.failed)


class CalibrationResult(JobResult):
    """``calibration.json``: what a calibration measured and the limits it set (none unless every arm)."""

    engine: str = ""
    tag: str = ""
    adjacent: str = ""
    arms: dict[str, ArmScore] = Field(default_factory=dict)
    missing: list[str] = Field(default_factory=list)
    limits: EngineLimits | None = None


def source_of(engine: Engine, tag: str) -> VersionSource | None:
    """The version source whose tag form ``tag`` has."""
    return next((s for s in sources_for(engine) if tag_key(s, tag) is not None), None)


def calibration_refusal(
    engine: Engine, tag: str, adjacent: str, *, adjacent_pending: bool = False
) -> str | None:
    """Why a calibration of ``tag`` against ``adjacent`` may not run; None when it may (design 4.3).

    ``adjacent_pending`` is for the submit-time check, when the adjacent version's pull is still to come.
    """
    current = link_target(versions_dir(engine))
    if current != tag:
        return f"{tag} is not {engine.value}'s `current` ({current or 'none'})"
    needed = [tag] if adjacent_pending else [tag, adjacent]
    for name in needed:
        found = read_record(engine, name)
        if found is None or not found.pulled:
            return f"{name}: its image is not on disk"
    source = source_of(engine, tag)
    key = tag_key(source, tag) if source else None
    for other in records(engine):
        if (
            source is not None
            and key is not None
            and (other_key := tag_key(source, other.tag)) is not None
            and other_key > key
            and other.gate is GateVerdict.NOT_RUN
            and other.pulled
        ):
            return f"{other.tag} is under judgment: a failing candidate must not loosen its own limit"
    return None


class CalibrateJob(JobRunner):
    """`refresh engine-calibrate`: noise arms and the adjacent pair; the limits of design 4.3."""

    kind = EngineJob.CALIBRATE
    result_file = "calibration.json"
    quiet_reason = "container engine calibration"

    def __init__(
        self,
        run: Path,
        engine: Engine,
        tag: str,
        adjacent: str,
        *,
        harness: Harness | None = None,
        guards: MemoryGuards = GUARDS,
    ) -> None:
        self.engine, self.tag, self.adjacent = engine, tag, adjacent
        super().__init__(
            run,
            engine_folder(run, engine, tag),
            CalibrationResult(engine=engine.value, tag=tag, adjacent=adjacent),
            harness=harness,
            guards=guards,
        )

    @property
    def outcome(self) -> CalibrationResult:
        """The result, typed."""
        assert isinstance(self.result, CalibrationResult)  # noqa: S101 - set in __init__
        return self.result

    def work(self) -> None:
        """Start V and the adjacent P in the order of design 4.3; set limits only when every arm measured."""
        refusal = calibration_refusal(self.engine, self.tag, self.adjacent)
        if refusal:
            self.result.not_measured = refusal
            return
        version, other = read_record(self.engine, self.tag), read_record(self.engine, self.adjacent)
        assert version is not None  # noqa: S101 - the refusal checked both records
        assert other is not None  # noqa: S101
        write_record(other.model_copy(update={"reason": other.reason or ADJACENT_REASON}))
        model = MEASUREMENT_MODEL
        recipe, adjacent_recipe = realized(version, model), realized(other, model)
        if recipe is None or adjacent_recipe is None:
            self.result.not_measured = f"{model} has no realized recipe in both records"
            return
        phases = (engine_bench.Phase.WARM, engine_bench.Phase.TEACHER_FORCED)
        calibration = ResultKind.CALIBRATION
        self.serve("warm-current", version, model, recipe, (engine_bench.Phase.WARM,), ResultKind.WARM)
        reference = self.serve("reference", version, model, recipe, phases, calibration)
        runs = {
            CalibrationArm.RESTART: self.serve(
                CalibrationArm.RESTART, version, model, recipe, phases, calibration
            ),
            CalibrationArm.EMPTY_CACHE: self.serve(
                CalibrationArm.EMPTY_CACHE,
                version,
                model,
                {**recipe, "runtime_cache": False},
                phases,
                calibration,
            ),
        }
        if self.engine is Engine.SGLANG:
            runs[CalibrationArm.AUTOTUNE_OFF] = self.serve(
                CalibrationArm.AUTOTUNE_OFF, version, model, autotune_off_recipe(recipe), phases, calibration
            )
        self.serve(
            "warm-adjacent", other, model, adjacent_recipe, (engine_bench.Phase.WARM,), ResultKind.WARM
        )
        runs[CalibrationArm.ADJACENT] = self.serve(
            CalibrationArm.ADJACENT, other, model, adjacent_recipe, phases, calibration
        )
        self._set_limits(reference, runs)

    def _set_limits(
        self, reference: EngineRun | None, runs: Mapping[CalibrationArm, EngineRun | None]
    ) -> None:
        for arm, run in runs.items():
            base = reference.teacher_forced if reference else None
            other = run.teacher_forced if run else None
            if base is None or other is None:
                self.outcome.missing.append(f"{arm.value}: teacher-forced scores not measured")
                continue
            try:
                compared = compare_scores(base, other)
            except NotComparableError as exc:
                self.outcome.missing.append(f"{arm.value}: not comparable ({exc})")
                continue
            self.outcome.arms[arm.value] = ArmScore(ppl_ratio=compared.ppl_ratio, same_top=compared.same_top)
        if self.outcome.missing:
            return  # limits.json stays as it was: a gate is never loosened by half a calibration
        ppl_max, same_min = calibrated_limits(self.outcome.arms)
        limits = EngineLimits(
            engine=self.engine,
            ppl_ratio_max=ppl_max,
            same_top_min=same_min,
            calibrated_on=self.tag,
            adjacent=self.adjacent,
            arms=self.outcome.arms,
            run=self.run.name,
            slurm_job=self.result.slurm_job,
            calibrated_at=datetime.now(UTC).isoformat(timespec="seconds"),
        )
        write_model(limits_path(self.engine), limits)
        self.outcome.limits = limits

    def finish(self) -> None:
        """Remove the images nothing keeps (the adjacent version's, once measured)."""
        prune(self.engine, runner=self.harness.docker)


# --- the llama.cpp same-client job (design 4.1) ---------------------------------------------------

LLAMA_HOST: Final = "127.0.0.1"
#: D16 row C0's layout: ``-ngl 99 -c 32768 -np 16 -kvu -fa on --cache-ram 0``.
SAME_CLIENT_PARAMS: Final = ServingParams(
    n_gpu_layers=-1,
    n_ctx=32768,
    n_parallel=16,
    recommended_quant=None,
    recommended_size_class=ModelSizeClass.LARGE,
    kv_unified=True,
    assumptions=["D16 row C0 layout"],
)
SAME_CLIENT_FLAGS: Final = ("-fa", "on", "--cache-ram", "0")
SAME_CLIENT_PHASES: Final = tuple(p for p in engine_bench.Phase if p is not engine_bench.Phase.TEACHER_FORCED)


def same_client_path(commit: str, gguf: refresh.Model) -> Path:
    """Where llama.cpp's same-client run of the file ``gguf`` is kept for build ``commit``.

    Keyed by the llama.cpp file, not by the container model it is set beside: two projects may pair one
    container file with different llama.cpp files.
    """
    name = f"{_slug(gguf.repository)}@{gguf.revision}--{gguf.filename}.json"
    return engines_dir() / Engine.LLAMA_CPP.value / "same-client" / commit / name


def same_client_record(commit: str, gguf: refresh.Model) -> EngineRun | None:
    """llama.cpp's run of ``gguf`` on the container engines' client at ``commit``; None if none."""
    return read_model(same_client_path(commit, gguf), engine_bench.EngineRun)


class SameClientTarget(NamedTuple):
    """One llama.cpp file for the same-client job, and the Hub name of the container file it is set beside.

    llama-benchy counts its prompt with that container file's tokenizer: it cannot load one from a GGUF path
    (slurm 2233), and the same tokenizer gives llama.cpp the very prompt the container engines were given.
    """

    gguf: refresh.Model
    tokenizer: str


def same_client_targets() -> dict[str, SameClientTarget]:
    """The llama.cpp files measured on the container engines' client, by label.

    ``SAME_CLIENT_FILES`` (labelled by their container model) and each approval's GGUF: a chooser rule needs
    llama.cpp's figures for every file an approval pairs with a container file. Records are keyed by the
    llama.cpp file, so two approvals pairing one container file with different GGUFs never collide.
    """
    targets = {label: SameClientTarget(gguf, label) for label, gguf in SAME_CLIENT_FILES.items()}
    for approval in engines.read_approvals().approvals:
        gguf = refresh.Model(approval.gguf_repository, approval.gguf_revision, approval.gguf_file)
        if all(gguf != t.gguf for t in targets.values()):
            label = f"{gguf.repository}@{gguf.revision}--{gguf.filename}"
            targets[label] = SameClientTarget(gguf, approval.file_repository)
    return targets


def _free_port() -> int:
    import socket  # noqa: PLC0415 - only this one place binds a port

    with socket.socket() as sock:
        sock.bind((LLAMA_HOST, 0))
        return int(sock.getsockname()[1])


async def _measure_llama(binary: Path, model_file: Path, lease_dir: Path, tokenizer: str) -> EngineRun:
    async with httpx.AsyncClient(timeout=engine_bench.REQUEST_TIMEOUT_S) as http:
        port = _free_port()
        manager = LocalServerManager(
            ServerSpec(host=LLAMA_HOST, port=port, model_path=model_file),
            SAME_CLIENT_PARAMS,
            GpuLease(lease_dir, "localmodels-same-client"),  # private: the shared directory is slurm's
            http=http,
            pid_dir=lease_dir / "pid",
            binary=binary,
            ready_timeout_s=LLAMA_READY_S,
            guards=GUARDS,
            extra_flags=SAME_CLIENT_FLAGS,
        )
        begun = time.monotonic()
        await manager.start()
        try:
            return await asyncio.to_thread(  # off the loop, which runs the memory guard
                engine_bench.measure,
                f"http://{LLAMA_HOST}:{port}",
                Engine.LLAMA_CPP.value,
                extra={"cache_prompt": False},
                stamp={
                    "engine": Engine.LLAMA_CPP.value,
                    "model_file": model_file.name,
                    "server": str(binary),
                },
                startup_s=time.monotonic() - begun,
                phases=SAME_CLIENT_PHASES,
                tokenizer=tokenizer,
            )
        finally:
            await manager.stop()


def measure_llama(binary: Path, model_file: Path, lease_dir: Path, tokenizer: str) -> EngineRun:
    """Start ``llama-server`` on D16's layout inside the tune's guards and measure it (one run)."""
    return asyncio.run(_measure_llama(binary, model_file, lease_dir, tokenizer))


class SameClientJob(JobRunner):
    """`refresh same-client`: llama.cpp's `current` on the container engines' client, per container model."""

    kind = EngineJob.SAME_CLIENT
    result_file = "same-client.json"
    quiet_reason = "llama.cpp same-client measurement"

    def __init__(
        self,
        run: Path,
        home: str | None = None,
        *,
        harness: Harness | None = None,
        guards: MemoryGuards = GUARDS,
    ) -> None:
        self.home = home
        super().__init__(run, run / "engines" / "same-client", JobResult(), harness=harness, guards=guards)

    def work(self) -> None:
        """Measure each file of ``same_client_targets`` that has no record for the current build."""
        base = llama_home(self.home)
        commit = current_commit(base)
        if commit is None:
            self.result.not_measured = "llama.cpp has no current build"
            return
        for label, (pin, tokenizer) in same_client_targets().items():
            if same_client_record(commit, pin) is not None:
                self.result.steps[label] = STEP_OK
                continue
            file = refresh.model_file(pin)
            if file is None:
                self.result.steps[label] = f"{NOT_MEASURED}{pin.filename} absent"
                continue
            try:
                measured = (self.harness.llama or measure_llama)(
                    current_server(base), file, self.folder / "lease", tokenizer
                )
            except (ServerStartError, httpx.HTTPError, OSError) as exc:
                self.result.steps[label] = f"{NOT_MEASURED}{type(exc).__name__}: {exc}"
                continue
            write_model(same_client_path(commit, pin), measured)
            write_model(self.folder / f"{_slug(label)}.json", measured)
            self.result.steps[label] = STEP_OK


# --- the sparkrun comparison job (design 2) --------------------------------------------------------


class SparkrunCompareResult(JobResult):
    """``sparkrun.json``: the gate on every engine's `current` under the old and the new sparkrun."""

    old: str = ""
    new: str = ""
    verdicts: dict[str, EngineVerdict] = Field(default_factory=dict)  # "<engine> <model>" -> verdict
    moved: bool = False
    reason: str = ""


def current_records() -> dict[Engine, VersionRecord]:
    """Each container engine's `current` record whose image is on disk."""
    found = {}
    for engine in CONTAINER_ENGINES:
        tag = link_target(versions_dir(engine))
        record = read_record(engine, tag) if tag else None
        if record is not None and record.pulled:
            found[engine] = record
    return found


def move_sparkrun(new: str, old: str, reason: str) -> None:
    """Make ``new`` the sparkrun in use."""
    state = read_sparkrun_state() or SparkrunState(version=old)
    write_model(
        sparkrun_state_path(),
        state.model_copy(
            update={
                "version": new,
                "previous": old,
                "reason": reason,
                "moved_at": datetime.now(UTC).isoformat(timespec="seconds"),
            }
        ),
    )


def keep_sparkrun(old: str, new: str, reason: str, *, tried: bool = True) -> None:
    """Stay on ``old``; ``new`` is tried (not offered again) only when its own gate failed."""
    state = read_sparkrun_state() or SparkrunState(version=old)
    kept = sorted({*state.tried, new}) if tried else state.tried
    write_model(sparkrun_state_path(), state.model_copy(update={"tried": kept, "reason": reason}))


class SparkrunCompareJob(JobRunner):
    """`refresh sparkrun-compare`: each `current` under the old and new sparkrun, through the same gate."""

    kind = EngineJob.SPARKRUN
    result_file = "sparkrun.json"
    quiet_reason = "sparkrun comparison"

    def __init__(
        self, run: Path, old: str, new: str, *, harness: Harness | None = None, guards: MemoryGuards = GUARDS
    ) -> None:
        self.old, self.new = old, new
        super().__init__(
            run,
            run / "engines" / "sparkrun",
            SparkrunCompareResult(old=old, new=new),
            harness=harness,
            guards=guards,
        )

    def work(self) -> None:
        """Warm and measure every `current` recipe under both; move only when every engine passed."""
        outcome = self.result
        assert isinstance(outcome, SparkrunCompareResult)  # noqa: S101 - set in __init__
        for engine, current in current_records().items():
            for model in current.recipes:
                recipe = realized(current, model)
                if recipe is None:
                    continue
                prefix = f"{engine.value}-{_slug(model)}-"
                old, new = interleave(
                    self,
                    model,
                    StartArm(self.old, current, recipe, self.old),
                    StartArm(self.new, current, recipe, self.new),
                    prefix=prefix,
                    kind=ResultKind.SPARKRUN,
                )
                outcome.verdicts[f"{engine.value} {model}"] = judge_rounds(
                    self, engine, old, new, prefix=prefix
                )
        verdicts = outcome.verdicts
        failed = [k for k, v in verdicts.items() if v.gate is GateVerdict.FAILED]
        unjudged = [k for k, v in verdicts.items() if v.gate is not GateVerdict.PASSED and k not in failed]
        if self.not_judged:  # the stop was not the new sparkrun's doing: compared again next run
            outcome.reason = f"not judged: {self.not_judged}"
            keep_sparkrun(self.old, self.new, outcome.reason, tried=False)
        elif failed:
            outcome.reason = f"{self.new} did not pass the gate on: {', '.join(failed)}"
            keep_sparkrun(self.old, self.new, outcome.reason)
        elif unjudged or not verdicts:  # e.g. an engine whose quality limits are not calibrated yet
            outcome.reason = (
                f"{self.new} not judged on: {', '.join(unjudged) or 'no engine'}; stays on {self.old}"
            )
            keep_sparkrun(self.old, self.new, outcome.reason, tried=False)
        else:
            outcome.moved, outcome.reason = True, f"{self.new} passed the gate on every engine"
            move_sparkrun(self.new, self.old, outcome.reason)

    def blameless(self, record: VersionRecord, sparkrun: str | None) -> bool:
        """A start under the old sparkrun failing is not the new version's fault."""
        del record
        return sparkrun != self.new


# --- the pull job (design 4.2) ----------------------------------------------------------------------


def realize_recipes(
    engine: Engine,
    image: str,
    models: Collection[str] | None,
    folder: Path,
    *,
    sparkrun: Callable[[str], Path] = ensure_sparkrun,
) -> tuple[dict[str, str], list[str]]:
    """Realize ``engine``'s recipes for ``image``: (model -> recipe text, one failure per model).

    ``models`` None realizes every recipe of the engine, else only those named. ``folder`` holds the
    recipe sources sparkrun reads. A pull realizes all of them; an approval realizes its own file for
    `current` at once (CPU, seconds), since a measure job finds no recipe otherwise.
    """
    wanted = {m: r for m, r in recipes_for(engine).items() if models is None or m in models}
    texts: dict[str, str] = {}
    failures: list[str] = []
    try:
        binary = sparkrun(in_use_version())
    except ServerStartError as exc:
        return texts, [f"{model}: realize failed: {exc}" for model in wanted]
    for model, recipe in wanted.items():
        source = folder / f"realize-{_slug(model)}.json"
        write_text_atomically(source, json.dumps({**recipe, "container": image}, indent=2))
        done = run_sparkrun(
            binary,
            [
                "export",
                "recipe",
                str(source),
                "--realize",
                "--offline",
                "--hosts",
                "localhost",
                "--tp",
                "1",
                "--json",
            ],
            timeout=engines.DOCKER_TIMEOUT_SECONDS,
        )
        try:
            json.loads(done.stdout)
        except ValueError:
            failures.append(
                f"{model}: realize failed: {done.stderr.strip()[-200:] or f'exit {done.returncode}'}"
            )
            continue
        texts[model] = done.stdout.strip()
    return texts, failures


class PullJob:
    """`refresh engine-pull`: pull one image on the CPU, realize the recipes, write the version record.

    A failed pull still writes the record (``pulled`` false and the reason): the tag is tried (D12). Exit 0
    whatever came of the pull; exit 1 only when the record cannot be written.
    """

    def __init__(  # noqa: PLR0913 - the job's inputs and the seams tests replace
        self,
        run: Path,
        engine: Engine,
        tag: str,
        publisher: Publisher,
        *,
        docker: engines.Runner = docker_runner,
        pull: engines.Runner | None = None,
        sparkrun: Callable[[str], Path] = ensure_sparkrun,
        free_bytes: Callable[[str], int] | None = None,
        tools: engines.Runner | None = None,
    ) -> None:
        self.engine, self.tag, self.publisher = engine, tag, publisher
        self.tools = tools or (lambda cmd: run_command(cmd, timeout=TOOL_FETCH_TIMEOUT_S))
        self.folder = engine_folder(run, engine, tag)
        self.docker = docker
        self.pull = pull or (lambda cmd: run_command(cmd, timeout=PULL_TIMEOUT_S))
        self.sparkrun = sparkrun
        self.free_bytes = free_bytes or (lambda path: shutil.disk_usage(path).free)

    def disk_problem(self) -> str | None:
        """Why the pull is refused for want of disk: the Docker root has under 100 GB free."""
        done = self.docker(["docker", "info", "--format", "{{.DockerRootDir}}"])
        root = done.stdout.strip()
        if done.returncode != 0 or not root:
            return (
                f"cannot read Docker's root folder: {done.stderr.strip()[-200:] or f'exit {done.returncode}'}"
            )
        try:
            free = self.free_bytes(root)
        except OSError as exc:
            return f"cannot read the free space of {root}: {exc}"
        if free < MIN_FREE_DISK_GB * GB:
            return f"refused: {root} has {free / GB:.0f} GB free, a pull needs {MIN_FREE_DISK_GB} GB"
        return None

    def pulled_image(self, record: VersionRecord) -> tuple[str, str | None]:
        """Pull the tag (or the digest the record names) and read its digest: (image, problem)."""
        source = source_of(self.engine, self.tag)
        if source is None:
            return "", f"{self.tag} is not a tag of any {self.engine.value} source"
        reference = record.image or f"{source.reference}:{self.tag}"  # a rollback pulls what was measured
        done = self.pull(["docker", "pull", reference])
        if done.returncode != 0:
            return "", f"docker pull {reference}: {done.stderr.strip()[-300:] or f'exit {done.returncode}'}"
        found = self.docker(["docker", "image", "inspect", reference, "--format", "{{index .RepoDigests 0}}"])
        digest = found.stdout.strip()
        try:
            VersionRecord.model_validate({**record.model_dump(), "image": digest})
        except ValueError:
            return (
                "",
                f"docker gave no usable digest for {reference}: {digest!r} {found.stderr.strip()[-100:]}",
            )
        return digest, None

    def execute(self) -> int:
        """Pull, realize and record; see the class."""
        record = read_record(self.engine, self.tag) or VersionRecord(
            engine=self.engine, tag=self.tag, publisher=self.publisher
        )
        try:
            self.folder.mkdir(parents=True, exist_ok=True)
            install_recipes()
            problem = self.disk_problem()
        except OSError as exc:  # recorded like a failed pull: the chain goes on
            problem = f"{type(exc).__name__}: {exc}"
        image = ""
        if problem is None:
            image, problem = self.pulled_image(record)
        if problem is not None:  # ``pulled`` stays as it was: a failed pull removes nothing from disk
            updated = record.model_copy(update={"reason": problem})
        else:
            texts, failures = realize_recipes(self.engine, image, None, self.folder, sparkrun=self.sparkrun)
            verdict_kept = record.gate is not GateVerdict.NOT_RUN
            updated = record.model_copy(
                update={
                    "image": image,
                    "sparkrun_version": in_use_version(),
                    "recipes": {**record.recipes, **texts},
                    "pulled": True,
                    "reason": "; ".join(failures) if failures else (record.reason if verdict_kept else ""),
                }
            )
        try:
            write_record(updated)
        except OSError as exc:
            sys.stderr.write(f"cannot write the record of {self.engine.value} {self.tag}: {exc}\n")
            return EXIT_ERROR
        sys.stdout.write(f"pull {self.engine.value} {self.tag}: {problem or 'ok'}\n")
        self.fetch_tools()
        self.ensure_prompts()
        return EXIT_OK

    @staticmethod
    def ensure_prompts() -> None:
        """Install the pinned measuring prompts now when they are missing or changed, on the CPU.

        Without them every phase but llama-benchy's fails after the GPU start (slurm 2265, where `engines
        prompts install` had never been run). A refused install is only logged: the phases say what to do.
        """
        try:
            engine_bench.load_prompts()
        except engine_bench.PromptFileError as missing:
            try:
                for path in engine_bench.install_prompts():
                    log.info("installed %s (%s)", path, missing)
            except engine_bench.PromptFileError as exc:
                log.warning("measuring prompts not installed: %s", exc)

    def fetch_tools(self) -> None:
        """Put the pinned llama-benchy in uv's cache now, on the CPU.

        The measure job's benchy phase then neither downloads it on the GPU's time nor fails a bootstrap for
        want of the network. A failure is only logged: the phase fetches it itself.
        """
        done = self.tools(["uvx", engine_bench.BENCHY_PACKAGE, "--help"])
        if done.returncode != 0:
            log.warning(
                "fetching %s failed (exit %d): %s",
                engine_bench.BENCHY_PACKAGE,
                done.returncode,
                done.stderr.strip()[-200:],
            )


# --- the job commands (argv) -----------------------------------------------------------------------


def job_name(kind: EngineJob, engine: Engine | None) -> str:
    """The job's ``--name``: ``<kind>-<engine>`` (no engine part for the same-client and sparkrun jobs)."""
    return f"{kind.value}-{engine.value}" if engine is not None else kind.value


def job_need_gib(engine: Engine, tags: Sequence[str], models: Sequence[str]) -> float:
    """The whole memory need of a job that starts ``tags`` on ``models``: the largest recorded, else the seed.

    GPU memory is charged to no cgroup on this machine, so ``--mem`` is set by hand from this.
    """
    needs = []
    for tag in tags:
        record = read_record(engine, tag)
        for model in models or [MEASUREMENT_MODEL]:
            recorded = MemoryNeed.recorded(record, model) if record is not None else None
            needs.append(recorded if recorded is not None else SEED_NEED_GIB[engine])
    return max(needs, default=SEED_NEED_GIB[engine])


def _after_flags(after: Sequence[str]) -> list[str]:
    return [part for job in dict.fromkeys(after) for part in ("--after", job)]  # repeated: afterok each


def _gpu_flags(need_gib: float, after: Sequence[str]) -> list[str]:
    return [
        "--class",
        "restart",
        "--gpu",
        "--mem",
        f"{math.ceil(need_gib + JOB_MEM_MARGIN_GIB)}G",
        *_after_flags(after),
    ]


def _timed(limit_s: int, command: Sequence[str]) -> list[str]:
    """``command`` under ``timeout``: the hard stop of the whole job."""
    return ["timeout", f"--kill-after={KILL_AFTER_S}", str(limit_s), *command]


def _job_argv(
    script: Path,
    run: Path,
    kind: EngineJob,
    engine: Engine | None,
    tag: str,
    what: str,
    flags: Sequence[str],
    command: Sequence[str],
) -> list[str]:
    stem = "-".join(
        part for part in (kind.value.removeprefix("refresh-"), engine.value if engine else "", tag) if part
    )
    return refresh._submit(script, run, job_name(kind, engine), what, flags, command, log_stem=stem)  # noqa: SLF001


def pull_argv(
    script: Path,
    run: Path,
    engine: Engine,
    tag: str,
    publisher: Publisher,
    home: str | None,
    after: Sequence[str] = (),
) -> list[str]:
    """The CPU pull job: small, resumable (docker pull continues), and not placed on another machine."""
    flags = ["--class", "resume", "--resume-ok", "--no-anywhere", "--cpus", str(PULL_CPUS), "--mem", PULL_MEM]
    command = [
        *refresh._refresh_command(home, "engine-pull"),  # noqa: SLF001
        *("--run", str(run), "--engine", engine.value, "--tag", tag, "--publisher", publisher.value),
    ]
    what = f"{engine.value} {tag} image pull (refresh {run.name})"
    return _job_argv(script, run, EngineJob.PULL, engine, tag, what, [*flags, *_after_flags(after)], command)


def measure_job_argv(  # noqa: PLR0913 - one job's argv: every argument is a distinct input
    script: Path,
    run: Path,
    engine: Engine,
    tag: str,
    mode: MeasureMode,
    *,
    home: str | None = None,
    after: Sequence[str] = (),
    model: str | None = None,
    also_starts: Sequence[str] = (),
) -> list[str]:
    """The GPU measure job of one version (``--class restart``, 40-minute stop, ``--mem`` the whole need).

    ``also_starts`` names a version that may be `current` by the time the job runs (one bootstrapped
    earlier in the same chain): it is started too, so its need counts.
    """
    tags = [t for t in (link_target(versions_dir(engine)), *also_starts, tag) if t]
    need = job_need_gib(engine, tags, [model or MEASUREMENT_MODEL])
    command = [
        *refresh._refresh_command(home, "engine-measure"),  # noqa: SLF001
        *("--run", str(run), "--engine", engine.value, "--tag", tag, "--mode", mode.value),
        *(["--model", model] if model else []),
    ]
    what = f"{engine.value} {tag} {mode.value} measurement (refresh {run.name})"
    return _job_argv(
        script,
        run,
        EngineJob.MEASURE,
        engine,
        f"{tag}-approved-{_slug(model)}" if mode is MeasureMode.APPROVED and model else tag,  # its own log
        what,
        _gpu_flags(need, after),
        _timed(MEASURE_LIMIT_S, command),
    )


def calibrate_argv(
    script: Path,
    run: Path,
    engine: Engine,
    tag: str,
    adjacent: str,
    *,
    home: str | None,
    after: Sequence[str] = (),
) -> list[str]:
    """The GPU calibration job: ``tag`` (the `current`) against ``adjacent`` (one hour stop)."""
    need = job_need_gib(engine, [tag, adjacent], [MEASUREMENT_MODEL])
    command = [
        *refresh._refresh_command(home, "engine-calibrate"),  # noqa: SLF001
        *("--run", str(run), "--engine", engine.value, "--tag", tag, "--adjacent", adjacent),
    ]
    what = f"{engine.value} {tag} calibration against {adjacent} (refresh {run.name})"
    return _job_argv(
        script,
        run,
        EngineJob.CALIBRATE,
        engine,
        tag,
        what,
        _gpu_flags(need, after),
        _timed(CALIBRATE_LIMIT_S, command),
    )


def same_client_argv(script: Path, run: Path, *, home: str | None, after: Sequence[str] = ()) -> list[str]:
    """The GPU job that measures llama.cpp's `current` on the container engines' client (10-minute stop)."""
    command = [*refresh._refresh_command(home, "same-client"), "--run", str(run)]  # noqa: SLF001
    flags = ["--class", "restart", "--gpu", "--mem", refresh.MEASURE_MEM, *_after_flags(after)]
    what = f"llama.cpp same-client measurement (refresh {run.name})"
    return _job_argv(
        script, run, EngineJob.SAME_CLIENT, None, "", what, flags, _timed(SAME_CLIENT_LIMIT_S, command)
    )


def sparkrun_argv(
    script: Path, run: Path, old: str, new: str, *, home: str | None, after: Sequence[str] = ()
) -> list[str]:
    """The GPU job that compares sparkrun ``old`` and ``new`` on every `current` (40 minutes per engine)."""
    current = current_records()
    need = max((job_need_gib(e, [r.tag], list(r.recipes)) for e, r in current.items()), default=0.0)
    command = [
        *refresh._refresh_command(home, "sparkrun-compare"),  # noqa: SLF001
        *("--run", str(run), "--old", old, "--new", new),
    ]
    what = f"sparkrun {old} against {new} on {len(current)} engine(s) (refresh {run.name})"
    return _job_argv(
        script,
        run,
        EngineJob.SPARKRUN,
        None,
        new,
        what,
        _gpu_flags(need, after),
        _timed(SPARKRUN_LIMIT_S * max(len(current), 1), command),
    )


# --- the walk (refresh run) ------------------------------------------------------------------------


def reap_leftovers() -> list[str]:
    """Remove the containers of ended jobs (``Reaper``); the report's lines."""
    report = Reaper().run()
    return [f"reaper removed {line}" for line in report.removed] + ([report.reason] if report.reason else [])


@dataclass
class JobSpec:
    """A job of the chain, planned: how to build its argv once its parents' job ids are known."""

    kind: EngineJob
    engine: Engine | None
    tag: str
    build: Callable[[Sequence[str]], list[str]]
    also_after: tuple[JobSpec, ...] = ()  # parents besides the job before it in the chain
    job: str | None = field(default=None, init=False)

    @property
    def name(self) -> str:
        """The job's ``--name``."""
        return job_name(self.kind, self.engine)


class EngineWalk:
    """The container engines' part of ``refresh run``: decide what is due, then submit it as one chain.

    ``decide()`` reaps, prunes, installs the recipes, checks the sparkrun move and lists each source's tags;
    ``submit()`` queues the chain, each job ``--after`` the one before (so one image is pulled at a time and
    GPU jobs never wait on each other's memory), and the llama.cpp same-client job last.
    """

    def __init__(  # noqa: PLR0913 - the run's pieces and the seams tests replace
        self,
        args: argparse.Namespace,
        run: Path,
        record: refresh.RunRecord,
        submitter: refresh.JobSubmitter,
        script: Path,
        *,
        fetch: Fetch | None = None,
        sparkrun_tags: Callable[[], TagListing] | None = None,
        sparkrun: Callable[[str], Path] = ensure_sparkrun,
        dry_runner: Callable[[Path, VersionRecord, str, dict[str, Any]], str] | None = None,
        docker: engines.Runner = docker_runner,
    ) -> None:
        self.args, self.run, self.record, self.submitter, self.script = args, run, record, submitter, script
        self.home: str | None = getattr(args, "home", None)
        self.dry = bool(args.dry_run)
        self.fetch = fetch
        self.docker = docker
        self.sparkrun_tags = sparkrun_tags
        self.sparkrun = sparkrun
        self.dry_runner = dry_runner or self._dry_run_output
        self.specs: list[JobSpec] = []
        self.same_client = False
        self._bootstrapping: dict[Engine, str] = {}  # an engine's official bootstrap planned in this chain
        self._current_pulls: set[Engine] = set()

    # --- deciding ---

    def decide(self) -> None:
        """Fill ``record.engines`` and the chain's specs; reap and prune (a dry run only reports)."""
        reports = self.record.engine_reports
        if not self.dry:
            reports += reap_leftovers()
            install_recipes()
        for engine in CONTAINER_ENGINES:
            pruned = prune(engine, dry_run=self.dry, runner=self.docker)
            if pruned.removed or pruned.failed:
                reports.append(f"{engine.value} prune: removed {pruned.removed}, failed {pruned.failed}")
        self._sparkrun_move()
        for engine in CONTAINER_ENGINES:
            for source in sources_for(engine):
                self._plan(engine, source)
        commit = current_commit(llama_home(self.home))
        self.same_client = commit is not None and any(
            same_client_record(commit, target.gguf) is None for target in same_client_targets().values()
        )

    @staticmethod
    def _dry_run_output(binary: Path, record: VersionRecord, model: str, recipe: dict[str, Any]) -> str:
        done = ContainerRun(record, model, recipe, binary, GUARDS, kind=RunKind.REFRESH).dry_run(verbose=True)
        return f"exit {done.returncode}\n{launch_lines(done.stdout + chr(10) + done.stderr)}"

    def _sparkrun_move(self) -> None:
        """Design 2: a newer sparkrun is adopted only after the dry runs of every realized recipe agree."""
        reports = self.record.engine_reports
        listing = (self.sparkrun_tags or list_sparkrun_tags)()
        if listing.reason:
            reports.append(f"sparkrun: tag list unreadable: {listing.reason}")
            return
        newest, in_use, state = newest_sparkrun(listing.tags), in_use_version(), read_sparkrun_state()
        if newest is None or (version_key(newest) or ()) <= (version_key(in_use) or ()):
            return
        if state is not None and newest in state.tried:
            return
        if self.dry:
            reports.append(f"sparkrun {newest} is newer than {in_use}: not checked in a dry run")
            return
        try:
            different = self._differences(self.sparkrun(in_use), self.sparkrun(newest))
        except (ServerStartError, OSError) as exc:
            reports.append(f"sparkrun {newest}: not checked: {exc}")
            return
        if different is None:
            reports.append(f"sparkrun {newest}: a dry run failed: not checked")
            return
        if different:
            reports.append(
                f"sparkrun {newest} differs from {in_use} on {', '.join(different)}: comparison job"
            )
            self.specs.append(
                JobSpec(
                    EngineJob.SPARKRUN,
                    None,
                    newest,
                    lambda after: sparkrun_argv(
                        self.script, self.run, in_use, newest, home=self.home, after=after
                    ),
                )
            )
        else:
            move_sparkrun(newest, in_use, "the dry runs of every realized recipe are identical")
            reports.append(f"sparkrun moved {in_use} -> {newest}: the dry runs are identical")

    def _differences(self, old: Path, new: Path) -> list[str] | None:
        """The realized recipes whose dry run differs between two sparkrun binaries; None if one failed.

        Two identical failures are no evidence that the launch is the same.
        """
        different = []
        for engine, current in current_records().items():
            for model in current.recipes:
                recipe = realized(current, model)
                if recipe is None:
                    continue
                before = self.dry_runner(old, current, model, recipe)
                after = self.dry_runner(new, current, model, recipe)
                if not (before.startswith(DRY_RUN_OK) and after.startswith(DRY_RUN_OK)):
                    log.warning("dry run of %s %s failed:\n%s\n%s", engine.value, model, before, after)
                    return None
                if before != after:
                    different.append(f"{engine.value} {model}")
        return different

    def _plan(self, engine: Engine, source: VersionSource) -> None:
        plan = refresh.EnginePlan(
            engine=engine.value, repository=source.repository, publisher=source.publisher.value
        )
        self.record.engines.append(plan)
        listing = list_tags(source, fetch=self.fetch or engines.urllib_fetch)
        if listing.reason:
            plan.reason = f"tag list unreadable: {listing.reason}"
            return
        tag = newest_untried(source, listing.tags, judged_tags(engine))
        if tag is None:
            plan.reason = "no new tag"
            return
        plan.tag = tag
        official = source.publisher is Publisher.OFFICIAL
        current = link_target(versions_dir(engine))
        expected = current or self._bootstrapping.get(engine)  # what is `current` when the job runs
        mode = MeasureMode.CANDIDATE if expected else MeasureMode.BOOTSTRAP
        plan.mode = mode.value
        self._pull_current(engine, current)
        self._pull(engine, tag, source.publisher)
        also = [expected] if expected and expected != current else []
        measure = self._measure(engine, tag, mode, also_starts=also)
        if official and mode is MeasureMode.BOOTSTRAP:
            self._bootstrapping[engine] = tag
        if official:  # each approved file on what is then `current` (refused there unless this tag became it)
            for model in recipes_for(engine):
                if model != MEASUREMENT_MODEL:
                    self._measure(engine, tag, MeasureMode.APPROVED, model)
        if not (official and mode is MeasureMode.BOOTSTRAP):
            return  # a community build is measured and recorded, never calibrated
        key = tag_key(source, tag)
        older = [
            t
            for t in matching_tags(source, listing.tags)
            if key is not None and (tag_key(source, t) or ()) < key
        ]
        if not older:
            plan.reason = "no adjacent tag: not calibrated"
            return
        adjacent = older[-1]
        have = read_record(engine, adjacent)
        parents: tuple[JobSpec, ...] = (measure,)
        if have is None or not have.pulled:
            parents = (self._pull(engine, adjacent, source.publisher), measure)
        self._add(
            EngineJob.CALIBRATE,
            engine,
            tag,
            lambda after: calibrate_argv(
                self.script, self.run, engine, tag, adjacent, home=self.home, after=after
            ),
            also_after=parents,
        )

    def _pull_current(self, engine: Engine, current: str | None) -> None:
        """Pull `current` again when its image is gone (a rollback): a candidate is compared with it."""
        record = read_record(engine, current) if current else None
        if record is None or record.pulled or not record.image or engine in self._current_pulls:
            return
        self._current_pulls.add(engine)
        self._pull(engine, record.tag, record.publisher)

    def _pull(self, engine: Engine, tag: str, publisher: Publisher) -> JobSpec:
        return self._add(
            EngineJob.PULL,
            engine,
            tag,
            lambda after: pull_argv(self.script, self.run, engine, tag, publisher, self.home, after),
        )

    def _measure(
        self,
        engine: Engine,
        tag: str,
        mode: MeasureMode,
        model: str | None = None,
        *,
        also_starts: Sequence[str] = (),
    ) -> JobSpec:
        return self._add(
            EngineJob.MEASURE,
            engine,
            tag,
            lambda after: measure_job_argv(
                self.script,
                self.run,
                engine,
                tag,
                mode,
                home=self.home,
                after=after,
                model=model,
                also_starts=also_starts,
            ),
        )

    def _add(
        self,
        kind: EngineJob,
        engine: Engine,
        tag: str,
        build: Callable[[Sequence[str]], list[str]],
        *,
        also_after: tuple[JobSpec, ...] = (),
    ) -> JobSpec:
        spec = JobSpec(kind, engine, tag, build, also_after)
        self.specs.append(spec)
        return spec

    # --- submitting ---

    def submit(self, llama_measure_job: str | None = None) -> None:
        """Queue the chain, then the same-client job (after llama.cpp's measure job, else the last link)."""
        last: str | None = None
        for spec in self.specs:
            after = [job for job in (last, *(p.job for p in spec.also_after)) if job]
            spec.job = self.submitter.submit(spec.name, spec.build(after))
            if spec.job is None:
                self.record.engine_reports.append(f"chain stopped: {spec.name} {spec.tag} was not submitted")
                return
            last = spec.job
        if self.same_client or llama_measure_job:
            after = [llama_measure_job] if llama_measure_job else ([last] if last else [])
            self.submitter.submit(
                job_name(EngineJob.SAME_CLIENT, None),
                same_client_argv(self.script, self.run, home=self.home, after=after),
            )


# --- the digest and status lines (design 4.2) ---------------------------------------------------


def _line(  # noqa: PLR0911 - one line per outcome
    plan: refresh.EnginePlan, result: EngineMeasureResult | None, record: VersionRecord | None
) -> str:
    who = f"{plan.engine} {plan.tag} ({plan.publisher})"
    if result is None:
        if record is not None and not record.pulled and record.reason:
            return f"{who}: pull failed: {record.reason}"
        return f"{who}: measure job pending"
    verdict = result.verdict
    if verdict is None:
        return f"{who}: not measured: {result.not_measured or result.error or 'no verdict'}"
    community = plan.publisher == Publisher.COMMUNITY.value
    if verdict.gate is GateVerdict.NOT_RUN:
        return f"{who}: {verdict.reason} (offered again next run)"
    if result.mode is MeasureMode.BOOTSTRAP:
        if verdict.gate is GateVerdict.PASSED:
            return f"{who}: bootstrap passed: " + (
                "recorded for the owner (community image: never switched)" if community else "now current"
            )
        return f"{who}: bootstrap failed: {verdict.reason}"
    if result.switched:
        return f"{who}: switched"
    if verdict.gate is GateVerdict.RECORD_ONLY:
        return f"{who}: recorded for the owner (limits not calibrated)"
    if verdict.gate is GateVerdict.PASSED:
        why = (
            "recorded for the owner (community image: never switched)" if community else result.switch_problem
        )
        return f"{who}: " + (why or "passed")
    return f"{who}: kept back: {verdict.reason}"


def engine_lines(view: refresh.RunView) -> list[str]:
    """One line per engine the run touched, for the digest and `refresh status`."""
    if view.run is None:
        return []
    lines = [r for r in view.run.engine_reports if r.startswith("sparkrun")]
    for plan in view.run.engines:
        if not plan.tag:
            lines.append(f"{plan.engine}: {plan.reason}")
            continue
        engine = Engine(plan.engine)
        result = read_model(
            engine_folder(view.folder, engine, plan.tag) / "measure.json", EngineMeasureResult
        )
        lines.append(_line(plan, result, read_record(engine, plan.tag)))
    return list(dict.fromkeys(lines))


# --- commands --------------------------------------------------------------------------------------


def _home(args: argparse.Namespace) -> str | None:
    return getattr(args, "home", None)


def cmd_pull(args: argparse.Namespace) -> int:
    """`refresh engine-pull`: the CPU pull job."""
    job = PullJob(Path(args.run), Engine(args.engine), args.tag, Publisher(args.publisher))
    return job.execute()


def cmd_measure(args: argparse.Namespace) -> int:
    """`refresh engine-measure`: the GPU measure job."""
    job = MeasureJob(
        Path(args.run), Engine(args.engine), args.tag, MeasureMode(args.mode), model=args.model or None
    )
    return job.execute()


def cmd_calibrate(args: argparse.Namespace) -> int:
    """`refresh engine-calibrate`: the GPU calibration job."""
    return CalibrateJob(Path(args.run), Engine(args.engine), args.tag, args.adjacent).execute()


def cmd_same_client(args: argparse.Namespace) -> int:
    """`refresh same-client`: llama.cpp on the container engines' client."""
    return SameClientJob(Path(args.run), _home(args)).execute()


def cmd_sparkrun_compare(args: argparse.Namespace) -> int:
    """`refresh sparkrun-compare`: the sparkrun comparison job."""
    return SparkrunCompareJob(Path(args.run), args.old, args.new).execute()


def _submitter() -> refresh.JobSubmitter:
    return refresh.JobSubmitter(
        refresh.RunRecord(time=datetime.now(UTC).isoformat(timespec="seconds")), dry_run=False
    )


def owner_folder(engine: Engine, kind: str) -> Path:
    """A new run folder for a job the owner submits by hand (`engines pull|calibrate|approve`)."""
    folder = engines_dir() / engine.value / kind / datetime.now(UTC).strftime(refresh.RUN_ID_FORMAT)
    folder.mkdir(parents=True, exist_ok=True)  # the jobs' --workdir and --log: there before they queue
    return folder


def cmd_engines_pull(args: argparse.Namespace) -> int:
    """`engines pull <engine> <tag>`: submit one pull job."""
    engine, tag = Engine(args.engine), args.version
    source = source_of(engine, tag)
    script = retune._script()  # noqa: SLF001
    if source is None:
        sys.stderr.write(f"{tag} is not a tag of any {engine.value} source: see `localmodels engines list`\n")
        return EXIT_ERROR
    if script is None:
        sys.stderr.write(f"no submit script (${retune.SUBMIT_ENV} is empty)\n")
        return EXIT_ERROR
    folder = owner_folder(engine, "pulls")
    job = _submitter().submit(
        job_name(EngineJob.PULL, engine), pull_argv(script, folder, engine, tag, source.publisher, None)
    )
    return EXIT_OK if job else EXIT_ERROR


def approved_recipe(approval: engines.Approval) -> dict[str, Any]:
    """The recipe an approval installs for its container file: the engine's base recipe on that file."""
    return {
        **BASE_RECIPES[approval.engine],
        "model": approval.file_repository,
        "model_revision": approval.file_revision,
    }


def recipe_conflict(approval: engines.Approval) -> str | None:
    """Why ``approval`` cannot run on its file's recipe; None when it can.

    A recipe is kept per file repository and names one revision, which is what the engine measures and
    serves, so an approval of another revision is refused rather than measured and stamped on the wrong
    file. When no recipe file exists yet the refresh's own base recipe for that repository decides.
    """
    path = recipe_file(approval.engine, approval.file_repository)
    try:
        installed: Any = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        base = BASE_RECIPES[approval.engine]
        installed = base if base["model"] == approval.file_repository else None
    except (OSError, ValueError) as exc:
        return f"the recipe {path} cannot be read: {exc!r}"
    if installed is None:
        return None
    revision = installed.get("model_revision") if isinstance(installed, dict) else None
    if revision == approval.file_revision:
        return None
    return (
        f"{approval.engine.value} measures and serves {approval.file_repository} at revision {revision} "
        f"(its recipe {path}, one per repository), not {approval.file_revision}: approve that revision"
    )


def approve_measurement(  # noqa: PLR0911 - one line per way the measurement has to wait
    approval: engines.Approval,
    *,
    dry_run: bool = False,
    sparkrun: Callable[[str], Path] = ensure_sparkrun,
    docker: engines.Runner = docker_runner,
) -> list[str]:
    """What `engines approve` does after the approval is stored: the recipe, its realization, the job.

    Installs a recipe for the approved file when absent; when the engine has a `current` whose image is on
    disk, realizes it for that image now (CPU, seconds) and submits an APPROVED measure job of `current`;
    otherwise says the measurement waits for the next refresh. Returns the lines to print; a dry run
    writes and submits nothing and prints the job's command.
    """
    engine, model = approval.engine, approval.file_repository
    path = recipe_file(engine, model)
    lines = []
    if not path.exists():
        if not dry_run:
            write_text_atomically(path, json.dumps(approved_recipe(approval), indent=2) + "\n")
        lines.append(f"{'would install' if dry_run else 'installed'} the recipe {path}")
    waits = "the measurement waits: the next refresh realizes every recipe and measures it"
    tag = link_target(versions_dir(engine))
    record = read_record(engine, tag) if tag else None
    if record is None or not record.pulled or not record.image:
        return [*lines, f"{engine.value} has no `current` with its image on disk; {waits}"]
    if docker(["docker", "image", "inspect", record.image]).returncode != 0:
        return [*lines, f"{engine.value} {record.tag}: its image is not on disk; {waits}"]
    script = retune._script()  # noqa: SLF001
    if script is None:
        return [*lines, f"no submit script (${retune.SUBMIT_ENV} is empty); {waits}"]
    folder = engines_dir() / engine.value / "approvals" if dry_run else owner_folder(engine, "approvals")
    argv = measure_job_argv(script, folder, engine, record.tag, MeasureMode.APPROVED, model=model)
    if dry_run:
        return [*lines, f"would realize the recipe for {record.tag}, then submit: {' '.join(argv)}"]
    texts, failures = realize_recipes(engine, record.image, [model], folder, sparkrun=sparkrun)
    if failures or model not in texts:
        return [*lines, *(failures or [f"{model}: no recipe to realize"]), waits]
    latest = read_record(engine, record.tag) or record
    write_record(latest.model_copy(update={"recipes": {**latest.recipes, **texts}}))
    lines.append(f"realized the recipe for {record.tag}")
    job, failure = retune._run_submit(argv)  # noqa: SLF001
    if failure:
        return [*lines, f"measure job not submitted: {failure}", waits]
    return [*lines, f"submitted measure job {job} ({engine.value} {record.tag} on {model})"]


def cmd_engines_calibrate(args: argparse.Namespace) -> int:
    """`engines calibrate <engine>`: pull the adjacent version if needed, then calibrate the `current`."""
    engine = Engine(args.engine)
    tag = link_target(versions_dir(engine))
    script = retune._script()  # noqa: SLF001
    source = source_of(engine, tag) if tag else None
    if tag is None or source is None or script is None:
        sys.stderr.write(f"{engine.value} has no `current`, or there is no submit script\n")
        return EXIT_ERROR
    listing = list_tags(source, fetch=engines.urllib_fetch)
    key = tag_key(source, tag)
    older = [
        t for t in matching_tags(source, listing.tags) if key is not None and (tag_key(source, t) or ()) < key
    ]
    if listing.reason or not older:
        sys.stderr.write(f"no adjacent tag to calibrate against: {listing.reason or 'none older'}\n")
        return EXIT_ERROR
    adjacent = older[-1]
    have = read_record(engine, adjacent)
    pending = have is None or not have.pulled
    refusal = calibration_refusal(engine, tag, adjacent, adjacent_pending=pending)
    if refusal:
        sys.stderr.write(f"not calibrating: {refusal}\n")
        return EXIT_ERROR
    submitter = _submitter()
    pull = None
    if pending:
        pull = submitter.submit(
            job_name(EngineJob.PULL, engine),
            pull_argv(script, owner_folder(engine, "pulls"), engine, adjacent, source.publisher, None),
        )
        if pull is None:
            return EXIT_ERROR
    job = submitter.submit(
        job_name(EngineJob.CALIBRATE, engine),
        calibrate_argv(
            script,
            owner_folder(engine, "calibrations"),
            engine,
            tag,
            adjacent,
            home=None,
            after=[pull] if pull else [],
        ),
    )
    return EXIT_OK if job else EXIT_ERROR
