"""The llama.cpp tune: measure ``-ub``, speculative decoding and a quantised file, adopt what holds.

Run it as ``python -m localmodels.llama_tune`` (the ready recipe command for llama.cpp, design 5.4);
``--help`` lists its arguments. Everything is measured on this machine and nothing is assumed:

* ``-ub``, the prompt batch (design 6.1 stage A). ``llama-bench`` times 512, 1024 and 2048 in one run; a
  larger one is kept for a ``llama-perplexity`` teacher-forced check against 512 (the same text read by both,
  their probabilities compared at every position), with a restart of 512 as the control. Only when the
  project's prompts are at least one batch long, because for shorter ones ``-ub`` changes nothing.
* Speculative decoding (stage B): a server is started with each declared candidate at the serving role's
  concurrency, temperature 0 and a fixed output length. It is kept only if the outputs repeat exactly across a
  restart, the draft acceptance is above zero, the outputs stay near the reference's and the memory fits.
* A quantised file (stage C): only with a task-quality check supplied by the project.

Memory is the server's own peak from ``nvidia-smi`` per process. The tune keeps to a soft time limit by
estimating each further start from the reference's and skipping what would not fit, most valuable first
(design 6.7), and ends by adopting its record (``settings.adopt``). A candidate other than the reference or
the incumbent wins only when it is faster by more than the margin of design 5.5.

The card is held quiet for the whole run (``lease.QuietCard``) and the servers lease on a private directory
under the run folder: slurm's lease keeper already holds the shared one for the job. Imported on demand:
``import localmodels`` does not load this module. No GPU work happens at import time.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import json
import logging
import math
import os
import re
import shlex
import shutil
import signal
import socket
import statistics
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import IntEnum, StrEnum
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, NamedTuple

import httpx
from pydantic import BaseModel, Field

from localmodels.bench import BenchError, run_bench
from localmodels.enums import ServingRole, WorkloadKind
from localmodels.hardware import (
    available_memory,
    gpu_process_memory_mib,
    inspect_hardware,
    network_filesystem,
    process_anon_mib,
)
from localmodels.launcher import LocalServerManager, MemoryGuards
from localmodels.lease import DEFAULT_LEASE_DIR, GpuLease, QuietCard, QuietCardTimeoutError
from localmodels.llama_builds import current_server, llama_home
from localmodels.params import ServingParams, derive_serving_params
from localmodels.pinned import snapshot_dir
from localmodels.process import LocalModelsError, ServerSpec, resolve_llama_server_bin
from localmodels.pull import _default_hub_cache
from localmodels.retune import TUNE_LIMIT_S
from localmodels.serving.protocol import ResultMessage
from localmodels.serving.tune import Agreement, hardware_key, text_agreement
from localmodels.settings import (
    MIN_PASSES,
    SPEED_MARGIN,
    AdoptError,
    ModelIdentity,
    SettingsAnswer,
    SettingsRecord,
    adopt,
    read_store,
    record_path,
)
from localmodels.settings import _pick as newest_record
from localmodels.stack import (
    Engine,
    Settings,
    SettingsSource,
    SettingsStatus,
    SpecType,
    fingerprint,
    llama_cpp_build,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence

    from localmodels.pinned import PinnedFile

log = logging.getLogger(__name__)

GIB: Final = 1024**3
MIB_PER_GIB: Final = 1024
PROJECT: Final = "localmodels-tune"  # the name on the private lease and the quiet request

# -ub (design 6.1, 6.4)
UB_REFERENCE: Final = 512  # llama.cpp's own default
UB_CANDIDATES: Final = (1024, 2048)
UB_FIRST: Final = 2048  # the one tried first when time is short (design 6.7)
# the chunk the limits were calibrated at (HL2c; at 4,096 the same -ub moved three times as much)
KL_CHUNK: Final = 2048
KL_CHUNKS: Final = 28  # HL2c's text: llama-perplexity is told to read no more, so disk and time stay bounded
BYTES_PER_TOKEN: Final = 3.5  # E: HL2c's text, 199,173 bytes, was 28 chunks of 2,048 tokens
# E: the base file holds every position of every chunk over the whole vocabulary as 16-bit values: about
# 14.2 GB for 28 chunks and a 248k vocabulary (HL2d: "about 13 GB"). The vocabulary is not read from the
# model, so this covers vocabularies up to about 299k; deleted after use.
KL_BASE_GIB: Final = 16.0
# The teacher-forced text: llama.cpp's docs/*.md at ccc3646 (MIT licence), sorted by name and concatenated
# (HL2c). Checked by SHA-256 at use.
PINNED_TEXT_SHA256: Final = "29fc9436e8c142cadea8dc00e48400fe59fbac7361e40ccf3ae1d833d0064d2c"
PINNED_TEXT: Final = Path(__file__).parent / "data" / "kl-text.txt"

# speculation (design 6.4)
SPEC_MIN_SIMILARITY: Final = 0.65  # a tripwire for a broken draft setup, not a quality measure

# servers (design 6.2)
HOST: Final = "127.0.0.1"
SERVER_READY_TIMEOUT_S: Final = 1800.0  # a 94 GB model read from the local disk (service_c's allowance)
KILL_BELOW_GIB: Final = 10.0  # service_c's floor: above earlyoom's SIGTERM at 8 GiB on this host
OOM_SCORE_ADJ: Final = 1000  # the server, never the tune, is the kernel's first pick
REQUEST_TIMEOUT_S: Final = 900.0
WARMUP_TOKENS: Final = 8
SAMPLE_INTERVAL_S: Final = 0.5
CHECK_URL_ENV: Final = "LLAMA_TUNE_SERVER_URL"
CHECK_FILE_ENV: Final = "LLAMA_TUNE_MODEL_FILE"
CHECK_TIMEOUT_FACTOR: Final = 2.0
CHECK_TIMEOUT_MIN_S: Final = 60.0

# time (design 5.4, 6.7)
# both limits follow the job's ceiling, ``retune.TUNE_LIMIT_S`` (D19): set the time there, nowhere else
STOP_MARGIN_S: Final = 60.0
HARD_LIMIT_S: Final = TUNE_LIMIT_S - STOP_MARGIN_S  # no step runs past it: to stop and write before SIGTERM
SOFT_LIMIT_S: Final = HARD_LIMIT_S - STOP_MARGIN_S  # no step starts that would end past it
BENCH_REPETITIONS: Final = 3
OUT_OF_BUDGET: Final = "not measured: out of budget"
OUT_OF_TIME: Final = "not measured: out of time"  # a step the hard limit cut short
SIGTERM_EXIT: Final = 128 + signal.SIGTERM

# exit codes of the command
EXIT_ADOPTED: Final = 0
EXIT_FAILED: Final = 1
EXIT_NOT_ADOPTED: Final = 2

DEFAULT_MAX_PROMPTS: Final = 24
DEFAULT_MAX_TOKENS: Final = 128
DEFAULT_HEADROOM_GIB: Final = 4.0  # as service_b's tune uses (commands/tune.py:42)


class LlamaTuneError(LocalModelsError):
    """The tune cannot give a result: the reference could not be measured."""


class Step(StrEnum):
    """What a measurement in the record is."""

    BENCH = "bench"
    KL = "teacher-forced"
    SERVER = "server"
    NOT_MEASURED = "not-measured"
    LAYOUT = "layout"


class Priority(IntEnum):
    """Which candidate is started first when time is short (design 6.7); lower comes first."""

    OWN_SPECULATION = 1  # the model's own multi-token-prediction head: the largest known gain
    UBATCH_2048 = 2
    REST = 3


# ---------------------------------------------------------------------------------------------------------
# What the tools print


class KlStats(BaseModel):
    """What ``llama-perplexity --kl-divergence`` reports against the saved base."""

    ppl_ratio: float  # Mean PPL(Q)/PPL(base)
    mean_kl: float
    same_top: float  # the share of positions with the same top token, 0 to 1


_PPL_RATIO = re.compile(r"Mean PPL\(Q\)/PPL\(base\)\s*:\s*(-?[\d.]+)")
_MEAN_KL = re.compile(r"^Mean\s+KLD:\s*(-?[\d.]+)", re.MULTILINE)
_SAME_TOP = re.compile(r"Same top p:\s*([\d.]+)")
_ACCEPTANCE = re.compile(r"draft acceptance = [\d.]+ \(\s*(\d+) accepted /\s*(\d+) generated\)")


def parse_kl(output: str) -> KlStats:
    """The three figures the limits use, from ``llama-perplexity``'s own output.

    Raises:
        ValueError: one of them is missing (the run failed or the build prints another format).
    """
    found = [pattern.search(output) for pattern in (_PPL_RATIO, _MEAN_KL, _SAME_TOP)]
    if not all(found):
        msg = "llama-perplexity printed no KL divergence statistics"
        raise ValueError(msg)
    ratio, kl, top = (float(match.group(1)) for match in found if match)
    return KlStats(ppl_ratio=ratio, mean_kl=kl, same_top=top / 100)


def parse_acceptance(server_log: str) -> float | None:
    """The share of drafted tokens the target kept over the log's requests; None if the server logged none."""
    pairs = [(int(m.group(1)), int(m.group(2))) for m in _ACCEPTANCE.finditer(server_log)]
    generated = sum(drafted for _, drafted in pairs)
    return sum(kept for kept, _ in pairs) / generated if generated else None


class BenchRow(BaseModel):
    """``llama-bench``'s prompt-reading speed at one ``-ub``."""

    step: Step = Step.BENCH
    n_ubatch: int
    tokens_per_second: float
    samples: list[float] = Field(default_factory=list)

    @property
    def spread(self) -> float:
        """How far the repetitions differ, as a fraction of the slowest; 0 with fewer than two."""
        if len(self.samples) < MIN_PASSES or min(self.samples) <= 0:
            return 0.0
        return (max(self.samples) - min(self.samples)) / min(self.samples)


def bench_rows(raw_rows: Sequence[dict[str, Any]], n_prompt: int) -> dict[int, BenchRow]:
    """The prompt-reading rows of a ``llama-bench`` run, one per ``-ub`` (``_parse_rows`` keeps the last)."""
    rows: dict[int, BenchRow] = {}
    for row in raw_rows:
        if row.get("n_prompt") == n_prompt and not row.get("n_gen") and "avg_ts" in row and "n_ubatch" in row:
            samples = [float(sample) for sample in row.get("samples_ts", [])]
            rows[int(row["n_ubatch"])] = BenchRow(
                n_ubatch=int(row["n_ubatch"]), tokens_per_second=float(row["avg_ts"]), samples=samples
            )
    return rows


def parse_check(output: str, returncode: int) -> Agreement:
    """A task-quality check's verdict: its last line is ``{"score": ..., "passed": ...}`` and it exits 0."""
    lines = [line for line in output.splitlines() if line.strip()]
    try:
        verdict = json.loads(lines[-1])
        score, passed = float(verdict["score"]), verdict["passed"] is True  # a string "false" is not a pass
        if not math.isfinite(score):
            raise ValueError(score)  # noqa: TRY301 - a NaN or infinite score is no score
    except (IndexError, ValueError, KeyError, TypeError):
        return Agreement(score=0.0, passed=False, detail="task-quality check printed no score and verdict")
    ok = passed and returncode == 0
    return Agreement(
        score=score, passed=ok, detail=f"task-quality check score {score:.3f}, exit {returncode}"
    )


# ---------------------------------------------------------------------------------------------------------
# The limits (design 6.4)


class UbLimits(NamedTuple):
    """What a larger ``-ub`` must stay within, against a restart of the reference."""

    ppl_margin: float  # the perplexity ratio may exceed the control's by this much
    max_kl: float
    min_top: float


#: One set for writers and scorers alike (D21), calibrated on the pinned text (HL2c: 0.016-0.018, 97.7 %); the
#: refresh gate judges a new build by the same limits (D18), which every project, scorers included, accepts
UB_LIMITS: Final = UbLimits(ppl_margin=0.01, max_kl=0.03, min_top=0.97)


def judge_ub(candidate: KlStats, control: KlStats, limits: UbLimits) -> Agreement:
    """Whether the candidate's teacher-forced figures are within ``limits`` of the restart control's."""
    ratio_ok = candidate.ppl_ratio <= control.ppl_ratio + limits.ppl_margin
    kl_ok = candidate.mean_kl <= limits.max_kl
    top_ok = candidate.same_top >= limits.min_top
    detail = (
        f"ppl ratio {candidate.ppl_ratio:.4f} (control {control.ppl_ratio:.4f} +{limits.ppl_margin:g}), "
        f"mean KL divergence {candidate.mean_kl:.4f} (at most {limits.max_kl:g}), "
        f"same top token {candidate.same_top:.2%} (at least {limits.min_top:.0%})"
    )
    return Agreement(score=candidate.same_top, passed=ratio_ok and kl_ok and top_ok, detail=detail)


def combine(parts: Sequence[Agreement]) -> Agreement:
    """Several checks as one: all must pass; the score is the lowest (each is a 0 to 1 share)."""
    return Agreement(
        score=min(part.score for part in parts),
        passed=all(part.passed for part in parts),
        detail="; ".join(part.detail for part in parts),
    )


def judge_speculation(
    reference: Sequence[str],
    outputs: Sequence[str],
    *,
    acceptance: float | None,
    reference_repeats: bool | None,
    repeats: bool | None,
) -> Agreement:
    """Design 6.4's checks for a speculative arm (memory is judged apart).

    (a) the reference and the arm each repeat themselves exactly across a restart (None: not measured, which
    fails), (b) the draft acceptance is above zero, (c) the mean word similarity to the reference's outputs
    (``text_agreement``) is at least ``SPEC_MIN_SIMILARITY``. The score is that similarity.
    """
    wrap = [
        [ResultMessage(request_id=str(i), raw_text=text) for i, text in enumerate(run)]
        for run in (reference, outputs)
    ]
    similarity = text_agreement(SPEC_MIN_SIMILARITY)(wrap[0], wrap[1])
    repeat_ok = reference_repeats is True and repeats is True
    repeat_note = {True: "repeats", False: "does not repeat", None: "repeat not measured"}
    accepted_ok = acceptance is not None and acceptance > 0
    shown = "none logged" if acceptance is None else f"{acceptance:.3f}"
    detail = (
        f"reference {repeat_note[reference_repeats]} across a restart, arm {repeat_note[repeats]}; "
        f"draft acceptance {shown}; {similarity.detail}"
    )
    return Agreement(
        score=similarity.score, passed=repeat_ok and accepted_ok and similarity.passed, detail=detail
    )


# ---------------------------------------------------------------------------------------------------------
# Time (design 5.4, 6.7)


class Planned(NamedTuple):
    """One candidate step waiting for its turn."""

    name: str
    priority: Priority
    cost: Callable[[], float]  # the estimate in seconds, read when its turn comes
    run: Callable[[], Awaitable[None]]


def in_priority_order(steps: Sequence[Planned]) -> list[Planned]:
    """The steps most valuable first; equal priorities keep their order (the stages' own)."""
    return sorted(steps, key=lambda step: step.priority)


def fits_budget(elapsed_s: float, cost_s: float, soft_limit_s: float) -> bool:
    """Whether a step estimated at ``cost_s`` can finish inside the soft limit."""
    return elapsed_s + cost_s <= soft_limit_s


# ---------------------------------------------------------------------------------------------------------
# What a tune measures


type SettingsKey = tuple[int, str, str, tuple[str, ...]]


def settings_key(settings: Settings) -> SettingsKey:
    """The settings with the defaults filled in, so ``Settings()`` and ``-ub 512, spec none`` are one."""
    return (
        settings.n_ubatch or UB_REFERENCE,
        str(settings.spec or SpecType.NONE),
        settings.spec_draft_file or "",
        tuple(f.path for f in settings.model_file or ()),
    )


REFERENCE_KEY: Final = settings_key(Settings())


def arm_name(settings: Settings) -> str:
    """A short name for an arm: what differs from the reference, "reference" if nothing does."""
    ubatch, spec, draft, files = settings_key(settings)
    parts = [f"ub{ubatch}"] if ubatch != UB_REFERENCE else []
    if spec != SpecType.NONE:
        parts.append(f"{spec}:{Path(draft).name}" if draft else spec)
    if files:
        parts.append(f"file:{files[0]}")
    return "+".join(parts) or "reference"


def fit_verdict(peak_gib: float | None, free_gib: float | None, headroom_gib: float) -> bool | None:
    """Whether the measured peak plus the headroom fits what was available before the start.

    None when either figure is missing: memory was not measured, which is not the same as "fits".
    """
    if peak_gib is None or free_gib is None:
        return None
    return peak_gib + headroom_gib <= free_gib


class ArmResult(BaseModel):
    """What one server start showed (stages B and C); ``error`` is set when it could not run."""

    step: Step = Step.SERVER
    name: str
    settings: Settings
    start_s: float | None = None
    run_s: float | None = None
    stop_s: float | None = None
    request_s: float | None = None  # the median server time per request: reading the prompt plus writing
    prompt_per_second: float | None = None  # the median of the server's own ``timings``
    predicted_per_second: float | None = None
    acceptance: float | None = None
    peak_gib: float | None = None
    free_gib: float | None = None  # available memory just before the start
    fits: bool | None = None  # None: memory not measured
    check: Agreement | None = None  # the project's task-quality check
    repeats: bool | None = None  # identical outputs across a restart; None: not measured
    agreement: Agreement | None = None
    eligible: bool = False
    error: str | None = None
    outputs: list[str] = Field(default_factory=list, exclude=True)  # large; the record keeps the figures


class KlResult(BaseModel):
    """One ``llama-perplexity`` teacher-forced run against the base."""

    step: Step = Step.KL
    n_ubatch: int
    control: bool = False
    seconds: float | None = None
    stats: KlStats | None = None
    agreement: Agreement | None = None
    error: str | None = None


class Layout(BaseModel):
    """The serving layout every candidate was measured at; the role fixes it, so it is never a candidate."""

    step: Step = Step.LAYOUT
    n_ctx: int
    n_parallel: int
    kv_unified: bool


class NotMeasured(BaseModel):
    """Something the tune did not measure, and why."""

    step: Step = Step.NOT_MEASURED
    name: str
    reason: str


def choose(arms: Sequence[ArmResult], *, incumbent: SettingsKey | None, spread: float) -> ArmResult:
    """The fastest eligible arm, after the speed margin of design 5.5.

    The incumbent is the arm with the settings the store holds, if it ran and is eligible, else the reference.
    An arm other than the reference or the incumbent wins only when its median request time beats the
    incumbent's by more than the larger of ``SPEED_MARGIN`` and the reference's ``spread``; otherwise the
    incumbent stays, so timing noise cannot change the settings.
    """
    usable = [arm for arm in arms if arm.eligible and arm.request_s]
    reference = next(arm for arm in arms if settings_key(arm.settings) == REFERENCE_KEY)
    held = next((arm for arm in usable if settings_key(arm.settings) == incumbent), reference)
    best = min(usable, key=lambda arm: arm.request_s or float("inf"), default=reference)
    if best is not reference and best is not held and held.request_s and best.request_s:
        faster = held.request_s / best.request_s - 1
        if faster <= max(SPEED_MARGIN, spread):
            best = held
    return best


@dataclass(frozen=True)
class FileSet:
    """A model's files as a tune loads them: the identity the store keys on, and the file the server gets."""

    identity: ModelIdentity
    path: Path

    @property
    def pinned(self) -> tuple[PinnedFile, ...]:
        """The files with their SHA-256 (a stage C winner records them as ``model_file``)."""
        return self.identity.files


@dataclass(frozen=True)
class SpecCandidate:
    """One declared speculation setting: a type, and a draft file for ``draft-simple``."""

    spec: SpecType
    draft_file: str | None = None


@dataclass(frozen=True)
class TuneSpec:
    """Everything a tune is asked: the model, the work, the declared candidates and the limits."""

    reference: FileSet
    alternatives: tuple[FileSet, ...]
    kind: WorkloadKind
    params: ServingParams  # the role's layout: the slot count is never tuned
    hardware: str  # ``hardware_key``
    server_bin: Path
    prompts: Sequence[str]
    prompt_tokens: int
    run_dir: Path
    text: Path | None = None
    speculation: tuple[SpecCandidate, ...] = ()
    quality_command: tuple[str, ...] = ()
    quality_name: str = ""
    check_s: float = 60.0  # what the project's check is expected to take, for the time estimate
    headroom_gib: float = DEFAULT_HEADROOM_GIB
    soft_limit_s: float = SOFT_LIMIT_S
    hard_limit_s: float = HARD_LIMIT_S
    max_tokens: int = DEFAULT_MAX_TOKENS
    project: str | None = None
    guards: MemoryGuards = field(default_factory=MemoryGuards)


def base_file_gib(text_bytes: int) -> float:
    """The disk the teacher-forced base file needs for a text of ``text_bytes``.

    Whole chunks, at most ``KL_CHUNKS`` (llama-perplexity is told to read no more). Scaled from
    ``KL_BASE_GIB``, which holds for the pinned text's 28 chunks and a vocabulary up to about 299k.
    """
    chunks = min(KL_CHUNKS, max(1, math.ceil(text_bytes / BYTES_PER_TOKEN / KL_CHUNK)))
    return KL_BASE_GIB * chunks / KL_CHUNKS


def load_prompts(path: Path, limit: int) -> list[str]:
    """The project's prompts: a JSON array of strings, or JSON lines each a string or ``{"prompt": ...}``."""
    text = path.read_text(encoding="utf-8")
    try:
        items = json.loads(text)
        if not isinstance(items, list):
            items = [items]
    except json.JSONDecodeError:
        items = [json.loads(line) for line in text.splitlines() if line.strip()]
    prompts = [item["prompt"] if isinstance(item, dict) else item for item in items]
    if not prompts or not all(isinstance(prompt, str) and prompt for prompt in prompts):
        msg = f"{path}: needs at least one prompt, each a non-empty string"
        raise ValueError(msg)
    return prompts[:limit]


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind((HOST, 0))
        return int(sock.getsockname()[1])


class LlamaTune:
    """One llama.cpp tune. ``tune()`` measures and returns the record; ``adopt`` is the caller's step.

    The methods that run a program (``_bench``, ``_perplexity``, ``_screen``, ``_serve``) and read the clock
    and memory are the ones a test replaces.
    """

    def __init__(
        self,
        spec: TuneSpec,
        *,
        clock: Callable[[], float] = time.monotonic,
        started: float | None = None,
    ) -> None:
        self.spec = spec
        self._clock = clock
        self._started = clock() if started is None else started  # the command's own start, before any wait
        self.measurements: list[BaseModel] = []
        self.arms: list[ArmResult] = []  # the first run of each arm; a restart is only kept in the record
        self.bench: dict[int, BenchRow] = {}
        self.kl: dict[int, KlResult] = {}
        self.ub_star: int | None = None  # the larger -ub that passed, once one has
        self.start_s = 0.0  # the reference's start (t0), measured once
        self.run_s = 0.0  # the reference's run over the prompts
        self.stop_s = 0.0  # the reference's stop
        self.check_s = spec.check_s
        self.spread = 0.0  # how far the reference's own timings differ
        self.ppl_s: float | None = None  # a teacher-forced run's measured time
        self.text: Path | None = None
        self.control: KlStats | None = None
        self.lease = GpuLease(spec.run_dir / "lease", PROJECT)
        self._servers = 0
        self._unified = False  # whether the GPU shares the host's RAM, read with the available memory
        self.unverified: set[str] = set()  # neither passed nor failed: not restarted in time, or cut short
        self.writes = spec.kind is not WorkloadKind.K3

    # -- the clock and the plan ---------------------------------------------------------------------------

    def elapsed(self) -> float:
        """Seconds since the tune began."""
        return self._clock() - self._started

    def left(self) -> float:
        """Seconds before the hard limit; no step is allowed to run past it."""
        return max(self.spec.hard_limit_s - self.elapsed(), 0.0)

    def _skip(self, name: str, reason: str) -> None:
        log.info("%s: %s", name, reason)
        self.measurements.append(NotMeasured(name=name, reason=reason))

    @property
    def reference(self) -> ArmResult:
        """The reference arm (measured first)."""
        return self.arms[0]

    def _arm_cost(self, arm: ArmResult | None = None, *, checked: bool = False) -> float:
        """What one more server start is estimated at: its start, run and stop (design 6.7).

        From ``arm``'s own figures when it ran before (a restart), else from the reference's.
        """
        if arm is not None and arm.start_s is not None and arm.run_s is not None:
            start, run, stop = arm.start_s, arm.run_s, arm.stop_s or 0.0
        else:
            start, run, stop = self.start_s, self.run_s, self.stop_s
        return start + run + stop + (self.check_s if checked else 0.0)

    def _ppl_cost(self) -> float:
        """What a teacher-forced run is estimated at.

        Measured once one has run; before that, the reference's start plus reading the text at its speed.
        """
        if self.ppl_s is not None:
            return self.ppl_s
        speed = self.reference.prompt_per_second
        tokens = min(self.text.stat().st_size / BYTES_PER_TOKEN, KL_CHUNKS * KL_CHUNK) if self.text else 0.0
        return self.start_s + (tokens / speed if speed else 0.0)

    def _ub_cost(self) -> float:
        base = 0.0 if self.control is not None else 2 * self._ppl_cost()  # the base and the restart control
        bench = 0.0
        if not self.bench:
            speed = self.reference.prompt_per_second or 1.0
            bench = self.start_s + len(UB_CANDIDATES) * BENCH_REPETITIONS * self.spec.prompt_tokens / speed
        return bench + base + self._ppl_cost()

    def _ub_arm_cost(self) -> float:
        """Nothing when no larger -ub has passed or an arm already carries it, else one more start."""
        if self.ub_star is None or any(arm.settings.n_ubatch == self.ub_star for arm in self.arms):
            return 0.0
        return self._arm_cost()

    def _plan(self) -> list[Planned]:
        """The steps in the order stages A, B, C list them; ``in_priority_order`` decides who goes first."""
        spec = self.spec
        steps: list[Planned] = []
        declared = [c for c in spec.speculation if c.spec is not SpecType.NONE]
        if declared and not self.writes:
            self._skip("speculation", "not measured: a scorer writes nothing a draft could speed up")
        elif declared:
            checked = bool(spec.quality_command)
            steps += [
                Planned(
                    arm_name(Settings(spec=c.spec, spec_draft_file=c.draft_file)),
                    Priority.OWN_SPECULATION if c.spec is SpecType.DRAFT_MTP else Priority.REST,
                    partial(self._arm_cost, checked=checked),
                    partial(self._speculation_arm, c),
                )
                for c in declared
            ]
        if self._stage_a_ready():
            # a passed -ub is carried by an arm of its own, because the memory figure comes from a server
            arm = Planned("-ub arm", Priority.UBATCH_2048, self._ub_arm_cost, self._ub_arm)
            steps.append(
                Planned(
                    f"-ub {UB_FIRST}", Priority.UBATCH_2048, self._ub_cost, partial(self._ub_step, UB_FIRST)
                )
            )
            steps.append(arm)
            steps += [
                Planned(f"-ub {ub}", Priority.REST, self._ub_cost, partial(self._ub_step, ub))
                for ub in UB_CANDIDATES
                if ub != UB_FIRST
            ]
            # and again at the end: a smaller -ub that passed after the first arm step needs its arm too
            steps.append(arm._replace(priority=Priority.REST))
        if spec.alternatives and not spec.quality_command:
            self._skip("quantised file", "not measured: no task-quality check")
        elif spec.alternatives:
            steps += [
                Planned(
                    f"file {alt.pinned[0].path}",
                    Priority.REST,
                    partial(self._arm_cost, checked=True),
                    partial(self._file_arm, alt),
                )
                for alt in spec.alternatives
            ]
        return in_priority_order(steps)

    def _stage_a_ready(self) -> bool:
        """Whether -ub can be measured at all; otherwise the record says why not."""
        if self.spec.prompt_tokens < UB_REFERENCE:
            self._skip("-ub", "`-ub` not measured: prompts shorter than one batch")
            return False
        self.text = self._find_text()
        if self.text is None:
            self._skip("-ub", "`-ub` not measured: no text for the teacher-forced check")
            return False
        need_gib = base_file_gib(self.text.stat().st_size)
        free_gib = shutil.disk_usage(self.spec.run_dir).free / GIB
        if free_gib < need_gib:
            self._skip(
                "-ub",
                f"`-ub` not measured: the base file needs about {need_gib:.0f} GiB, {free_gib:.0f} free",
            )
            return False
        return True

    def _find_text(self) -> Path | None:
        """The recipe's text, else the pinned one if its SHA-256 is the recorded one."""
        if self.spec.text is not None:
            return self.spec.text if self.spec.text.is_file() else None
        pinned = PINNED_TEXT.expanduser()
        try:
            digest = hashlib.sha256(pinned.read_bytes()).hexdigest()
        except OSError:
            return None
        return pinned if digest == PINNED_TEXT_SHA256 else None

    # -- the tune ---------------------------------------------------------------------------------------

    async def tune(self) -> SettingsRecord:
        """Measure the reference, then each candidate while time allows, and return the record to adopt.

        Raises:
            LlamaTuneError: the reference could not be measured.
        """
        self.spec.run_dir.mkdir(parents=True, exist_ok=True)
        base = self.spec.run_dir / "base.kld"
        base.unlink(missing_ok=True)  # one a killed run left would count as used space
        try:
            await self._measure_reference()
            for step in self._plan():
                cost = step.cost()  # 0: nothing to do now, and not worth a line in the record
                if cost and not fits_budget(self.elapsed(), cost, self.spec.soft_limit_s):
                    self._skip(step.name, OUT_OF_BUDGET)
                    continue
                await step.run()
            chosen = await self._decide()
        except (asyncio.CancelledError, LlamaTuneError) as exc:
            self.write_partial("stopped by a signal" if isinstance(exc, asyncio.CancelledError) else str(exc))
            raise
        finally:
            base.unlink(missing_ok=True)
        return self._record(chosen)

    @property
    def candidates_measured(self) -> bool:
        """Whether anything but the reference was measured (a server that ran, or a teacher-forced run)."""
        return any(arm.request_s for arm in self.arms[1:]) or any(r.stats for r in self.kl.values())

    def _layout(self) -> Layout:
        params = self.spec.params
        return Layout(n_ctx=params.n_ctx, n_parallel=params.n_parallel, kv_unified=params.kv_unified)

    def write_partial(self, reason: str) -> None:
        """Keep what was measured in ``record.json`` when the tune ends without a record (not adopted)."""
        everything = [self._layout(), *self.measurements, *self.arms]
        partial = {
            "partial": True,
            "reason": reason,
            "llama_measurements": [m.model_dump(mode="json") for m in everything],
        }
        (self.spec.run_dir / "record.json").write_text(json.dumps(partial, indent=2), encoding="utf-8")

    def _wants_candidates(self) -> bool:
        """Whether any candidate may be measured, so that the reference is worth timing twice (design 5.5)."""
        spec = self.spec
        speculating = self.writes and any(c.spec is not SpecType.NONE for c in spec.speculation)
        return (
            speculating
            or bool(spec.alternatives and spec.quality_command)
            or spec.prompt_tokens >= UB_REFERENCE
        )

    async def _measure_reference(self) -> None:
        """Start the reference first: it gives t0, the run time, the memory and the outputs to judge by.

        With any candidate to measure it is started a second time: for design 6.4 (a) and for the timing
        spread that the speed margin of design 5.5 needs.
        """
        reference = await self._serve(Settings(), self.spec.reference, checked=False)
        self.arms.append(reference)
        if reference.error:
            msg = f"the reference could not be measured: {reference.error}"
            raise LlamaTuneError(msg)
        self.start_s, self.run_s = reference.start_s or 0.0, reference.run_s or 0.0
        reference.eligible = True
        self.stop_s = reference.stop_s or 0.0
        if not self._wants_candidates():
            return
        if not fits_budget(self.elapsed(), self._arm_cost(), self.spec.soft_limit_s):
            self._skip("reference restart", OUT_OF_BUDGET)
            return
        again = await self._serve(Settings(), self.spec.reference, checked=False)
        self.measurements.append(again.model_copy(update={"name": "reference#restart"}))
        if again.error:
            self._skip("reference restart", f"not measured: {again.error}")
            return
        reference.repeats = again.repeats = again.outputs == reference.outputs
        if again.request_s and reference.request_s:
            self.spread = abs(again.request_s - reference.request_s) / min(
                again.request_s, reference.request_s
            )

    async def _speculation_arm(self, candidate: SpecCandidate) -> None:
        if self.reference.repeats is not True:
            why = (
                "does not repeat itself across a restart"
                if self.reference.repeats is False
                else "was not restarted"
            )
            name = arm_name(Settings(spec=candidate.spec, spec_draft_file=candidate.draft_file))
            self._skip(name, f"not measured: the reference {why}")
            return
        settings = Settings(n_ubatch=self.ub_star, spec=candidate.spec, spec_draft_file=candidate.draft_file)
        if self._draft_fits(arm_name(settings), candidate.draft_file):
            await self._run_arm(settings, self.spec.reference, checked=bool(self.spec.quality_command))

    def _draft_fits(self, name: str, draft_file: str | None) -> bool:
        """Whether the reference's measured peak plus the draft file leaves the margin available; else skip.

        From the reference's own peak, the figure for this model at this layout, so a draft that cannot fit
        costs no server start. Nothing to estimate (no draft file, or no figure) lets the arm run.
        """
        peak, free = self.reference.peak_gib, self._free_gib()
        if peak is None or free is None or draft_file is None:
            return True
        try:
            need = peak + Path(draft_file).stat().st_size / GIB
        except OSError:
            return True  # the server's own start reports a missing draft
        if need + self.margin_gib <= free:
            return True
        self._skip(
            name,
            f"not tried: about {need:.1f} GiB with its draft (the reference's peak and the file) plus "
            f"{self.margin_gib:g} kept free, {free:.1f} GiB available",
        )
        return False

    async def _file_arm(self, alternative: FileSet) -> None:
        """A quantised file at the reference's settings; other stages' settings are not combined with it."""
        settings = Settings(model_file=alternative.pinned)
        await self._run_arm(settings, alternative, checked=True)

    async def _ub_arm(self) -> None:
        """The arm that carries a passed ``-ub``: the best speculation so far (or none) at that ``-ub``.

        A larger ``-ub`` is adopted only from a server that ran with it, because the memory figure comes from
        there (design 6.3).
        """
        if self.ub_star is None or any(arm.settings.n_ubatch == self.ub_star for arm in self.arms):
            return
        best = self._pick(assume_repeat=True)
        spec = best.settings if best.settings.spec not in (None, SpecType.NONE) else Settings()
        settings = Settings(n_ubatch=self.ub_star, spec=spec.spec, spec_draft_file=spec.spec_draft_file)
        await self._run_arm(
            settings, self.spec.reference, checked=bool(spec.spec and self.spec.quality_command)
        )

    async def _run_arm(self, settings: Settings, files: FileSet, *, checked: bool) -> ArmResult:
        arm = await self._serve(settings, files, checked=checked)
        self.arms.append(arm)
        self._judge(arm, assume_repeat=False)
        return arm

    # -- stage A ----------------------------------------------------------------------------------------

    async def _ub_step(self, ubatch: int) -> None:
        name = f"-ub {ubatch}"
        if not self.bench:
            await self._run_bench()
        row, base = self.bench.get(ubatch), self.bench.get(UB_REFERENCE)
        if row is None or base is None:
            self._skip(name, "not measured: llama-bench gave no prompt-reading row for it")
            return
        margin = max(SPEED_MARGIN, base.spread)
        gain = row.tokens_per_second / base.tokens_per_second - 1
        star = self.bench.get(self.ub_star or 0)
        if gain <= margin:
            self._skip(
                name, f"not tried: prompt reading is {gain:+.1%} against 512, within the {margin:.1%} margin"
            )
            return
        if star is not None and row.tokens_per_second <= star.tokens_per_second * (1 + margin):
            self._skip(name, f"not tried: not faster than -ub {self.ub_star}, which already passed")
            return
        if not await self._fits_screen(ubatch):
            return
        if self.control is None and not await self._kl_base():
            return
        result = await self._kl(ubatch)
        if result.agreement is not None and result.agreement.passed:
            self.ub_star = ubatch

    async def _run_bench(self) -> None:
        """One ``llama-bench`` run for every -ub (it reloads the model only when its parameters change)."""
        bench_bin = self.spec.server_bin.with_name("llama-bench")
        ubatches = ",".join(str(u) for u in (UB_REFERENCE, *UB_CANDIDATES))
        try:
            result = await asyncio.to_thread(
                run_bench, bench_bin, self.spec.reference.path, n_prompt=self.spec.prompt_tokens, n_gen=0,
                repetitions=BENCH_REPETITIONS, timeout_s=max(self.left(), 1.0), extra_flags=("-ub", ubatches),
            )  # fmt: skip
        except BenchError as exc:
            self._skip(
                "-ub", OUT_OF_TIME if self.left() <= 0 else f"not measured: llama-bench failed ({exc})"
            )
            return
        self.bench = bench_rows(result.raw_rows, self.spec.prompt_tokens)
        self.measurements.extend(self.bench.values())

    async def _fits_screen(self, ubatch: int) -> bool:
        """Whether a server at this ``-ub`` leaves the margin free: it keeps what cannot fit from starting.

        The reference's measured peak, the figure for this model as it runs, plus what ``llama-fit-params``
        projects this ``-ub`` to add over the default; the projection alone without a measured reference.
        """
        name = f"-ub {ubatch}"
        free = self._free_gib()
        if free is None:
            self._skip(name, "not measured: memory not measured (available memory cannot be read)")
            return False
        peak = self.reference.peak_gib if self.arms else None
        try:
            need = await self._screen(ubatch)
            if peak is not None:
                need = peak + max(0.0, need - await self._screen(UB_REFERENCE))
        except LocalModelsError as exc:
            self._skip(name, f"not measured: llama-fit-params gave no projection ({exc})")
            return False
        if need + self.margin_gib > free:
            source = (
                "the reference's peak and llama.cpp's projected difference"
                if peak is not None
                else "projected"
            )
            self._skip(
                name,
                f"not tried: about {need:.1f} GiB at -ub {ubatch} ({source}) plus {self.margin_gib:g} "
                f"kept free, {free:.1f} GiB available",
            )
            return False
        return True

    @staticmethod
    def _failed(what: str, error: str | None) -> str:
        return error if error == OUT_OF_TIME else f"not measured: {what} failed ({error})"

    async def _kl_base(self) -> bool:
        """Save the reference's probabilities and run the restart control. False if either failed."""
        base = await self._perplexity(UB_REFERENCE, base=True)
        if base.error:
            self._skip("-ub", self._failed("the base run", base.error))
            return False
        control = await self._perplexity(UB_REFERENCE, base=False)
        if control.stats is None:
            self._skip("-ub", self._failed("the restart control", control.error))
            return False
        self.control = control.stats
        return True

    async def _kl(self, ubatch: int) -> KlResult:
        result = await self._perplexity(ubatch, base=False)
        if result.stats is not None and self.control is not None:
            result.agreement = judge_ub(result.stats, self.control, UB_LIMITS)
        self.kl[ubatch] = result
        return result

    # -- the programs -----------------------------------------------------------------------------------

    async def _run_tool(self, cmd: list[str], timeout_s: float) -> str:
        """Run a llama.cpp program to completion; its output, stderr included."""
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT
        )
        try:
            out, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
        finally:
            if proc.returncode is None:
                proc.kill()
                await proc.wait()
        if proc.returncode != 0:
            msg = f"{Path(cmd[0]).name} exited {proc.returncode}: {out.decode(errors='replace')[-300:]}"
            raise LlamaTuneError(msg)
        return out.decode(errors="replace")

    async def _perplexity(self, ubatch: int, *, base: bool) -> KlResult:
        """One ``llama-perplexity`` run at the calibrated chunk length: saving the base, or against it."""
        assert self.text is not None  # noqa: S101 - narrowing for mypy; stage A needs the text
        kld = self.spec.run_dir / "base.kld"
        mode = (
            ["-f", str(self.text), "--chunks", str(KL_CHUNKS), "--kl-divergence-base", str(kld)]
            if base
            else ["--kl-divergence-base", str(kld), "--kl-divergence"]
        )
        cmd = [
            str(self.spec.server_bin.with_name("llama-perplexity")),
            *("-m", str(self.spec.reference.path), "-c", str(KL_CHUNK), "-b", str(KL_CHUNK)),
            *("-ngl", "99", "-fa", "on", "-ub", str(ubatch)),
            *mode,
        ]
        result = KlResult(
            n_ubatch=ubatch, control=not base and ubatch == UB_REFERENCE and self.control is None
        )
        started = self._clock()
        try:
            output = await self._run_tool(cmd, self.left())
            result.seconds = self._clock() - started
            self.ppl_s = result.seconds
            if not base:
                result.stats = parse_kl(output)
        except TimeoutError:
            result.error = OUT_OF_TIME
        except (LlamaTuneError, OSError, ValueError) as exc:
            result.error = f"{type(exc).__name__}: {exc}"
        self.measurements.append(result)
        return result

    async def _screen(self, ubatch: int) -> float:
        """llama.cpp's projection, in GiB, of the reference file at this ``-ub`` (the manager's screen)."""
        async with httpx.AsyncClient() as http:
            manager = self._manager(Settings(n_ubatch=ubatch), self.spec.reference, http)
            return await manager._projected_gib()  # noqa: SLF001 - the manager's pre-load screen, used as it is

    def _free_gib(self) -> float | None:
        reading = available_memory("cuda")
        if reading is not None:
            self._unified = reading.unified
        return None if reading is None else reading.bytes / GIB

    @property
    def margin_gib(self) -> float:
        """What a server must leave available: the headroom, never less than the floor its guard kills at."""
        return max(self.spec.headroom_gib, KILL_BELOW_GIB)

    # -- a server arm ----------------------------------------------------------------------------------

    def _manager(self, settings: Settings, files: FileSet, http: httpx.AsyncClient) -> LocalServerManager:
        """A manager for one server, on the private lease directory (the shared one is slurm's)."""
        self._servers += 1
        answer = None
        if settings != Settings():
            answer = SettingsAnswer(
                model=files.identity, engine=Engine.LLAMA_CPP, kind=self.spec.kind, settings=settings,
                status=SettingsStatus.UNTUNED, source=SettingsSource.DEFAULT,
            )  # fmt: skip
        guards = self.spec.guards.model_copy(
            update={
                "host_reserve_gib": self.margin_gib,
                "kill_below_gib": KILL_BELOW_GIB,
                "server_oom_score_adj": OOM_SCORE_ADJ,
            }
        )
        return LocalServerManager(
            ServerSpec(host=HOST, port=_free_port(), model_path=files.path),
            self.spec.params,
            self.lease,
            http=http,
            pid_dir=self.spec.run_dir / f"server-{self._servers}",
            binary=self.spec.server_bin,
            ready_timeout_s=SERVER_READY_TIMEOUT_S,
            guards=guards,
            settings=answer,
        )

    async def _serve(self, settings: Settings, files: FileSet, *, checked: bool) -> ArmResult:
        """Start a server with ``settings``, send the prompts at the role's concurrency, and measure."""
        arm = ArmResult(name=arm_name(settings), settings=settings, free_gib=self._free_gib())
        peaks: list[int] = []
        sampler: asyncio.Task[None] | None = None
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_S) as http:
            manager = self._manager(settings, files, http)
            try:
                async with asyncio.timeout(self.left()):  # no step runs past the hard limit
                    started = self._clock()
                    handle = await manager.start()
                    arm.start_s = self._clock() - started
                    sampler = asyncio.create_task(self._sample(handle.pid, peaks))
                    started = self._clock()
                    await self._send_prompts(http, manager, arm)
                    arm.run_s = self._clock() - started
                    if checked and self.spec.quality_command:
                        arm.check = await self._check(manager.spec.port, files)
                        if arm.check is None:
                            arm.error = OUT_OF_TIME  # the check did not finish: not measured, not failed
            except TimeoutError:
                arm.error = OUT_OF_TIME
            except (LocalModelsError, httpx.HTTPError, OSError, KeyError, ValueError) as exc:
                arm.error = f"{type(exc).__name__}: {exc}"
            finally:
                if sampler is not None:
                    sampler.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await sampler
                stopping = self._clock()
                await manager.stop()
                arm.stop_s = self._clock() - stopping
        if peaks:
            arm.peak_gib = max(peaks) / MIB_PER_GIB
        arm.fits = fit_verdict(arm.peak_gib, arm.free_gib, self.margin_gib)
        return arm

    async def _sample(self, pid: int, peaks: list[int]) -> None:
        """Note the server process's memory every half second until cancelled.

        Its GPU memory, and on a GPU that shares the host's RAM its host memory too, which comes out of the
        same available figure.
        """
        while True:
            held = await asyncio.to_thread(gpu_process_memory_mib)
            if held and pid in held:
                host = (process_anon_mib(pid) or 0) if self._unified else 0
                peaks.append(held[pid] + host)
            await asyncio.sleep(SAMPLE_INTERVAL_S)

    async def _complete(
        self, http: httpx.AsyncClient, port: int, prompt: str, n_predict: int
    ) -> dict[str, Any]:
        reply = await http.post(
            f"http://{HOST}:{port}/completion",
            json={"prompt": prompt, "n_predict": n_predict, "temperature": 0, "cache_prompt": False,
                  "ignore_eos": True, "stream": False},
        )  # fmt: skip
        reply.raise_for_status()
        body: dict[str, Any] = reply.json()
        return body

    async def _send_prompts(
        self, http: httpx.AsyncClient, manager: LocalServerManager, arm: ArmResult
    ) -> None:
        """Warm up, then send every prompt at the role's concurrency; fill in the arm's figures."""
        port = manager.spec.port
        prompts = list(self.spec.prompts)
        await self._complete(http, port, prompts[0], WARMUP_TOKENS)
        log_path = manager.pid_dir / f"{port}.log"
        mark = log_path.stat().st_size if log_path.exists() else 0
        gate = asyncio.Semaphore(self.spec.params.n_parallel)
        n_predict = self.spec.max_tokens if self.writes else 1

        async def one(prompt: str) -> dict[str, Any]:
            async with gate:
                return await self._complete(http, port, prompt, n_predict)

        replies = await asyncio.gather(*(one(prompt) for prompt in prompts))
        timings = [reply["timings"] for reply in replies]
        arm.outputs = [str(reply["content"]) for reply in replies]
        arm.prompt_per_second = statistics.median(t["prompt_per_second"] for t in timings)
        arm.predicted_per_second = statistics.median(t["predicted_per_second"] for t in timings)
        arm.request_s = statistics.median((t["prompt_ms"] + t["predicted_ms"]) / 1000 for t in timings)
        if log_path.exists():
            arm.acceptance = parse_acceptance(log_path.read_bytes()[mark:].decode(errors="replace"))

    async def _check(self, port: int, files: FileSet) -> Agreement | None:
        """The project's task-quality check, run against the live server; its time counts inside the limit.

        None when it did not finish in time: ``check_s`` then becomes the time waited, so the shrinking rule
        skips what is left.
        """
        env = {**os.environ, CHECK_URL_ENV: f"http://{HOST}:{port}", CHECK_FILE_ENV: str(files.path)}
        started = self._clock()
        proc = await asyncio.create_subprocess_exec(
            *self.spec.quality_command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=env,
        )
        timeout = min(max(self.spec.check_s * CHECK_TIMEOUT_FACTOR, CHECK_TIMEOUT_MIN_S), self.left())
        try:
            out, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except TimeoutError:
            self.check_s = self._clock() - started
            return None
        finally:
            if proc.returncode is None:
                proc.kill()
                await proc.wait()
        self.check_s = self._clock() - started
        return parse_check(out.decode(errors="replace"), proc.returncode or 0)

    # -- judging and choosing --------------------------------------------------------------------------

    def _judge(self, arm: ArmResult, *, assume_repeat: bool) -> None:
        """Set the arm's agreement (every check that applies to it) and whether it can be adopted.

        ``assume_repeat`` treats an arm not yet restarted as repeating, to find the one worth restarting.
        """
        reference = self.reference
        settings = arm.settings
        ubatch, spec, _, files = settings_key(settings)
        parts: list[Agreement] = []
        if arm.error:
            parts.append(Agreement(score=0.0, passed=False, detail=f"did not run: {arm.error}"))
        if ubatch != UB_REFERENCE:
            kl = self.kl.get(ubatch)
            parts.append(
                kl.agreement
                if kl and kl.agreement
                else Agreement(score=0.0, passed=False, detail="no teacher-forced check")
            )
        if spec != SpecType.NONE and not files:
            repeats = True if assume_repeat and arm.repeats is None else arm.repeats
            ref_repeats = reference.repeats
            parts.append(
                judge_speculation(
                    reference.outputs,
                    arm.outputs,
                    acceptance=arm.acceptance,
                    reference_repeats=ref_repeats,
                    repeats=repeats,
                )
                if arm.outputs
                else Agreement(score=0.0, passed=False, detail="no outputs")
            )
        if arm.check is not None:
            parts.append(arm.check)
        elif files:
            parts.append(Agreement(score=0.0, passed=False, detail="no task-quality check ran"))
        if arm is not reference:
            parts.append(self._memory_verdict(arm))
        arm.agreement = combine(parts) if parts else Agreement(score=1.0, passed=True, detail="reference")
        arm.eligible = arm is reference or arm.agreement.passed

    def _memory_verdict(self, arm: ArmResult) -> Agreement:
        if arm.fits is None:
            return Agreement(score=0.0, passed=False, detail="memory not measured")
        figures = (
            f"peak {arm.peak_gib:.1f} GiB plus {self.margin_gib:g} kept free, "
            f"{arm.free_gib:.1f} GiB available"
        )
        return Agreement(
            score=1.0, passed=arm.fits, detail=f"memory {'fits' if arm.fits else 'does not fit'}: {figures}"
        )

    def _pick(self, *, assume_repeat: bool, skip: frozenset[str] = frozenset()) -> ArmResult:
        for arm in self.arms:
            self._judge(arm, assume_repeat=assume_repeat)
            if arm.name in skip:
                arm.eligible = False
        return choose(self.arms, incumbent=self._incumbent(), spread=self.spread)

    def _incumbent(self) -> SettingsKey | None:
        spec = self.spec
        records = read_store(record_path(spec.reference.identity, spec.kind)).records
        stored = newest_record(records, spec.hardware, None, Engine.LLAMA_CPP)
        return settings_key(stored.chosen) if stored else None

    async def _decide(self) -> ArmResult:
        """Choose, restarting the leading speculative arm first: it is adopted only if its outputs repeat."""
        unverified = self.unverified
        # an arm whose check the hard limit cut short was neither passed nor failed
        unverified.update(arm.name for arm in self.arms[1:] if arm.error == OUT_OF_TIME)
        while True:
            best = self._pick(assume_repeat=True, skip=frozenset(unverified))
            speculates = settings_key(best.settings)[1] != SpecType.NONE
            if best is self.reference or not speculates or best.repeats is not None:
                break
            if not fits_budget(self.elapsed(), self._arm_cost(best), self.spec.soft_limit_s):
                self._skip(f"{best.name} restart", OUT_OF_BUDGET)
                unverified.add(best.name)
                continue
            again = await self._serve(best.settings, self.spec.reference, checked=False)
            self.measurements.append(again.model_copy(update={"name": f"{best.name}#restart"}))
            if again.error == OUT_OF_TIME:  # cut by the hard limit: its repeat is unknown, not failed
                unverified.add(best.name)
                continue
            best.repeats = again.error is None and again.outputs == best.outputs
        return self._pick(assume_repeat=False, skip=frozenset(unverified))

    # -- the record ------------------------------------------------------------------------------------

    def _record(self, chosen: ArmResult) -> SettingsRecord:
        """The record of this tune; the defaults that were measured are written out in its settings."""
        spec = self.spec
        settings = chosen.settings
        measured_ub = self.control is not None
        measured_spec = self.writes and any(arm.settings.spec for arm in self.arms)
        stored = settings.model_copy(
            update={
                "n_ubatch": settings.n_ubatch or (UB_REFERENCE if measured_ub else None),
                "spec": settings.spec or (SpecType.NONE if measured_spec else None),
            }
        )
        agreement = chosen.agreement or Agreement(score=1.0, passed=True, detail="reference")
        checks = [name for name, present in (
            ("teacher-forced KL of -ub", settings_key(settings)[0] != UB_REFERENCE),
            ("speculation tripwires", settings_key(settings)[1] != SpecType.NONE),
            (spec.quality_name or "task-quality check", chosen.check is not None),
        ) if present]  # fmt: skip
        none = "reference kept; no candidate " + ("passed" if self.candidates_measured else "measured")
        quality = "llama.cpp tune: " + (", ".join(checks) if checks else none)
        self.measurements.extend(self.arms)
        layout = self._layout()
        return SettingsRecord(
            model=spec.reference.identity,
            kind=spec.kind,
            hardware=spec.hardware,
            measured_at=datetime.now(UTC),
            chosen=stored,
            candidate=chosen.name,
            quality=quality,
            stack=fingerprint(),
            build=llama_cpp_build(spec.server_bin),
            recipe_project=spec.project,
            engine=Engine.LLAMA_CPP,
            agreement=agreement,
            reference="reference",
            tune_job=os.environ.get("SLURM_JOB_ID"),
            llama_measurements=[m.model_dump(mode="json") for m in (layout, *self.measurements)],
        )


# ---------------------------------------------------------------------------------------------------------
# The command


async def run_guarded[T](work: Awaitable[T]) -> T:
    """Await ``work``; SIGTERM and SIGINT cancel it through the loop's own handler.

    ``timeout`` sends SIGTERM when the hard limit passes. Cancelling the task unwinds ``_serve``'s finally,
    which stops the server; a handler that raised inside the running loop could leave one loaded. The task is
    cancelled once: a second signal (``timeout`` signals its child and then its whole group) must not cut the
    cleanup short.
    """
    loop = asyncio.get_running_loop()
    task = asyncio.current_task()
    assert task is not None  # noqa: S101 - narrowing for mypy; asyncio.run runs this as a task
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, lambda: task.cancelling() or task.cancel())
    try:
        return await work
    finally:
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.remove_signal_handler(sig)


async def tune_and_adopt(spec: TuneSpec, started: float | None = None) -> int:
    """Hold the card quiet, tune, write ``record.json`` in the run folder and adopt it; the exit code.

    ``started`` is the command's own start (``time.monotonic()``), so the quiet card's wait counts against
    the limits. A tune that measured nothing but the reference adopts nothing (exit 2): D3's revisit
    condition when the time budget was the reason.
    """
    card = QuietCard(
        GpuLease(DEFAULT_LEASE_DIR, PROJECT), reason=f"llama.cpp tune of {spec.reference.identity.repository}"
    )
    try:
        await asyncio.to_thread(card.__enter__)
    except QuietCardTimeoutError as exc:
        log.error("%s", exc)  # noqa: TRY400 - the message is the whole story
        return EXIT_FAILED
    tune = LlamaTune(spec, started=started)
    try:
        record = await tune.tune()
    except LlamaTuneError as exc:
        log.error("%s", exc)  # noqa: TRY400 - the message is the whole story
        return EXIT_FAILED
    finally:
        card.__exit__(None, None, None)
    (spec.run_dir / "record.json").write_text(record.model_dump_json(indent=2), encoding="utf-8")
    if not tune.candidates_measured:
        why = (
            OUT_OF_BUDGET
            if any(isinstance(m, NotMeasured) and m.reason == OUT_OF_BUDGET for m in tune.measurements)
            else ""
        )
        log.warning(
            "measured, not adopted: no candidate was measured%s; llama.cpp tunes cost more GPU time than the "
            "limit allows (design D3's revisit condition)",
            f" ({why})" if why else "",
        )
        return EXIT_NOT_ADOPTED
    if tune.unverified and record.candidate == tune.reference.name:
        log.warning(
            "measured, not adopted: %s could not be checked within the time limit, so the reference "
            "is not a tuned answer (design D3's revisit condition)",
            ", ".join(sorted(tune.unverified)),
        )
        return EXIT_NOT_ADOPTED
    try:
        adopt(record)
    except AdoptError as exc:
        log.warning("measured, not adopted: %s", exc)
        return EXIT_NOT_ADOPTED
    return EXIT_ADOPTED


def _snapshot_file(repository: str, revision: str, name: str) -> Path:
    """Where the Hub cache keeps ``name`` of ``repository`` at ``revision``; it must be there already."""
    path = snapshot_dir(repository, revision, _default_hub_cache()) / name
    if not path.is_file():
        msg = f"{path} is not in the Hub cache; pull it first (a tune never downloads)"
        raise SystemExit(msg)
    return path


def _file_set(repository: str, revision: str, names: list[str]) -> FileSet:
    paths = [_snapshot_file(repository, revision, name) for name in names]
    try:
        identity = ModelIdentity.from_hub_files(repository, revision, names)
    except ValueError as exc:  # a file the Hub holds outside LFS has no SHA-256 to key a record on
        raise SystemExit(str(exc)) from exc
    return FileSet(identity, paths[0])


def run_dir_problem(run_dir: Path) -> str | None:
    """Why ``run_dir`` is no place for a tune's files, in plain words; None when it is fine.

    The teacher-forced base file is some 14 GB: under ~/dev Syncthing would copy it to every machine, and a
    write that size to an NFS share can stall the host.
    """
    target = run_dir.expanduser().resolve()
    if target.is_relative_to((Path.home() / "dev").resolve()):
        return f"{target} is under ~/dev, which Syncthing copies to every machine; use ~/cache/<project>/"
    if (kind := network_filesystem(target)) is not None:
        return f"{target} is on a network filesystem ({kind}); write locally and copy at the end"
    return None


def build_parser() -> argparse.ArgumentParser:
    """The command's arguments (design 5.4: the recipe command for llama.cpp)."""
    doc = __doc__ or ""
    parser = argparse.ArgumentParser(
        prog="python -m localmodels.llama_tune", description=doc.split("\n\n")[0]
    )
    add = parser.add_argument
    add("--repository", required=True, help="the model's Hub repository")
    add("--revision", required=True, help="its 40-character commit")
    add(
        "--file",
        action="append",
        required=True,
        dest="files",
        help="a file of the reference set (the first is loaded); repeat",
    )
    add(
        "--alternative",
        action="append",
        default=[],
        help="a quantised file set of the same repository and revision, for stage C: FILE[,FILE...]; repeat",
    )
    add(
        "--kind",
        required=True,
        choices=[k.value for k in (WorkloadKind.K1, WorkloadKind.K2, WorkloadKind.K3)],
    )
    add(
        "--role",
        required=True,
        choices=[r.value for r in ServingRole],
        help="the serving layout; its slot count is never tuned",
    )
    add(
        "--parallel",
        type=int,
        help="fewer slots than the role's, when the caller lays its server out so (1: one request at a "
        "time); recorded in the layout, never tuned",
    )
    add(
        "--prompts",
        required=True,
        type=Path,
        help="the project's real inputs: a JSON array or JSON lines of strings",
    )
    add("--prompt-tokens", required=True, type=int, help="their typical length in tokens")
    add("--text", type=Path, help="a text for the teacher-forced check (default: the pinned one)")
    add(
        "--speculation",
        action="append",
        default=[],
        choices=[s.value for s in SpecType],
        help="a candidate to measure; repeat",
    )
    add(
        "--draft-file",
        help="the file draft-mtp or draft-simple loads with -md, by name in the Hub cache snapshot of "
        "--repository at --revision (or of --draft-repository at --draft-revision)",
    )
    add("--draft-repository", help="where --draft-file is, when not the reference's repository")
    add("--draft-revision", help="and its 40-character commit")
    add(
        "--quality-command",
        help="the project's task-quality check, run against a live server (see the README)",
    )
    add("--quality-name", default="", help="its name, for the record")
    add(
        "--check-seconds",
        type=float,
        default=60.0,
        help="how long the check really takes: it is the estimate for the time limit",
    )
    add("--headroom-gib", type=float, default=DEFAULT_HEADROOM_GIB)
    add(
        "--run-dir",
        required=True,
        type=Path,
        help="a local folder outside ~/dev for logs and the base file (about 16 GiB while it runs)",
    )
    add(
        "--soft-limit-s",
        type=float,
        default=SOFT_LIMIT_S,
        help="no candidate is started that would end after this",
    )
    add("--max-prompts", type=int, default=DEFAULT_MAX_PROMPTS)
    add("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    add("--project", help="the project asking, for the record")
    return parser


def _draft_path(args: argparse.Namespace) -> str | None:
    """The path in the Hub cache of ``--draft-file`` (for draft-mtp and draft-simple), or None without one."""
    takers = [s for s in args.speculation if s in (SpecType.DRAFT_MTP, SpecType.DRAFT_SIMPLE)]
    if args.draft_file is None:
        if SpecType.DRAFT_SIMPLE in args.speculation:
            msg = "--speculation draft-simple needs --draft-file"
            raise SystemExit(msg)
        return None
    if not takers:
        msg = "--draft-file is for --speculation draft-mtp or draft-simple"
        raise SystemExit(msg)
    if bool(args.draft_repository) != bool(args.draft_revision):
        msg = "--draft-repository and --draft-revision go together"
        raise SystemExit(msg)
    repository = args.draft_repository or args.repository
    revision = args.draft_revision or args.revision
    return str(_snapshot_file(repository, revision, args.draft_file).absolute())


def spec_from_args(args: argparse.Namespace) -> TuneSpec:
    """Resolve the arguments: the files in the Hub cache, the binaries, the layout of the role."""
    if problem := run_dir_problem(args.run_dir):
        raise SystemExit(problem)
    draft = _draft_path(args)
    candidates = tuple(
        SpecCandidate(SpecType(s), draft if s in (SpecType.DRAFT_MTP, SpecType.DRAFT_SIMPLE) else None)
        for s in args.speculation
    )
    report = inspect_hardware()
    reference = _file_set(args.repository, args.revision, args.files)
    server = resolve_llama_server_bin(explicit=current_server(llama_home()))
    params = derive_serving_params(report.device, report.total_memory_gb, role=ServingRole(args.role))
    if args.parallel is not None:
        if not 1 <= args.parallel <= params.n_parallel:
            msg = (
                f"--parallel {args.parallel}: between 1 and the {args.role} role's {params.n_parallel} slots"
            )
            raise SystemExit(msg)
        params = params.model_copy(update={"n_parallel": args.parallel})
    return TuneSpec(
        reference=reference,
        alternatives=tuple(_file_set(args.repository, args.revision, a.split(",")) for a in args.alternative),
        kind=WorkloadKind(args.kind),
        params=params,
        hardware=hardware_key(report),
        server_bin=server,
        prompts=load_prompts(args.prompts, args.max_prompts),
        prompt_tokens=args.prompt_tokens,
        run_dir=args.run_dir,
        text=args.text,
        speculation=candidates,
        quality_command=tuple(shlex.split(args.quality_command or "")),
        quality_name=args.quality_name,
        check_s=args.check_seconds,
        headroom_gib=args.headroom_gib,
        soft_limit_s=args.soft_limit_s,
        max_tokens=args.max_tokens,
        project=args.project,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the tune. Exit 0 adopted, 1 no reference, 2 measured but not adopted, 143 stopped by SIGTERM."""
    started = time.monotonic()  # the limits count from here, the quiet card's wait included
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    spec = spec_from_args(build_parser().parse_args(argv))
    try:
        return asyncio.run(run_guarded(tune_and_adopt(spec, started)))
    except asyncio.CancelledError:
        return SIGTERM_EXIT


if __name__ == "__main__":
    sys.exit(main())
