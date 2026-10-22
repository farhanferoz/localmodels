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
from localmodels.hardware import available_memory, gpu_process_memory_mib, inspect_hardware
from localmodels.launcher import LocalServerManager, MemoryGuards
from localmodels.lease import DEFAULT_LEASE_DIR, GpuLease, QuietCard, QuietCardTimeoutError
from localmodels.llama_builds import current_server, llama_home
from localmodels.params import ServingParams, derive_serving_params
from localmodels.pinned import snapshot_dir
from localmodels.process import LocalModelsError, ServerSpec, resolve_llama_server_bin
from localmodels.pull import _default_hub_cache
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
KL_CHUNK: Final = (
    2048  # the chunk the limits were calibrated at (HL2c; at 4,096 the same -ub moved three times as much)
)
BYTES_PER_TOKEN: Final = 3.5  # E: HL2c's text, 199,173 bytes, was 28 chunks of 2,048 tokens
KL_BASE_GIB: Final = (
    16.0  # E: the base file of HL2d (4B model, 248k vocabulary) was about 13 GB; deleted after use
)
PINNED_TEXT_SHA256: Final = "29fc9436e8c142cadea8dc00e48400fe59fbac7361e40ccf3ae1d833d0064d2c"
PINNED_TEXT: Final = Path("~/cache/localmodels/spikes-2026-10-22/hl2c/text.txt")

# speculation (design 6.4)
SPEC_MIN_SIMILARITY: Final = 0.65  # a tripwire for a broken draft setup, not a quality measure

# servers (design 6.2)
HOST: Final = "127.0.0.1"
SERVER_READY_TIMEOUT_S: Final = 1800.0  # a 94 GB model read from the local disk (service_c's allowance)
SERVER_STOP_GRACE_S: Final = 120.0
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
SOFT_LIMIT_S: Final = 600.0
BENCH_REPETITIONS: Final = 3
OUT_OF_BUDGET: Final = "not measured: out of budget"
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
        score, passed = float(verdict["score"]), bool(verdict["passed"])
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


GENERATION_LIMITS: Final = UbLimits(
    ppl_margin=0.01, max_kl=0.03, min_top=0.97
)  # K1, K2 (HL2c: 0.016-0.018, 97.7 %)
SCORING_LIMITS: Final = UbLimits(ppl_margin=0.01, max_kl=0.02, min_top=0.98)  # K3's own (rules.md K3)


def ub_limits(kind: WorkloadKind) -> UbLimits:
    """The limits for ``kind``: a scorer's numbers are compared across runs, so it keeps K3's own."""
    return SCORING_LIMITS if kind is WorkloadKind.K3 else GENERATION_LIMITS


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
    max_tokens: int = DEFAULT_MAX_TOKENS
    project: str | None = None
    guards: MemoryGuards = field(default_factory=MemoryGuards)


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

    def __init__(self, spec: TuneSpec, *, clock: Callable[[], float] = time.monotonic) -> None:
        self.spec = spec
        self._clock = clock
        self._started = clock()
        self.measurements: list[BaseModel] = []
        self.arms: list[ArmResult] = []  # the first run of each arm; a restart is only kept in the record
        self.bench: dict[int, BenchRow] = {}
        self.kl: dict[int, KlResult] = {}
        self.ub_star: int | None = None  # the larger -ub that passed, once one has
        self.start_s = 0.0  # the reference's start (t0), measured once
        self.run_s = 0.0  # the reference's run over the prompts
        self.check_s = spec.check_s
        self.spread = 0.0  # how far the reference's own timings differ
        self.ppl_s: float | None = None  # a teacher-forced run's measured time
        self.text: Path | None = None
        self.control: KlStats | None = None
        self.lease = GpuLease(spec.run_dir / "lease", PROJECT)
        self._servers = 0
        self.writes = spec.kind is not WorkloadKind.K3

    # -- the clock and the plan ---------------------------------------------------------------------------

    def elapsed(self) -> float:
        """Seconds since the tune began."""
        return self._clock() - self._started

    def _skip(self, name: str, reason: str) -> None:
        log.info("%s: %s", name, reason)
        self.measurements.append(NotMeasured(name=name, reason=reason))

    @property
    def reference(self) -> ArmResult:
        """The reference arm (measured first)."""
        return self.arms[0]

    def _arm_cost(self, *, checked: bool = False) -> float:
        """What one more server start is estimated at: the reference's start and run (design 6.7)."""
        return self.start_s + self.run_s + (self.check_s if checked else 0.0)

    def _ppl_cost(self) -> float:
        """What a teacher-forced run is estimated at.

        Measured once one has run; before that, the reference's start plus reading the text at its speed.
        """
        if self.ppl_s is not None:
            return self.ppl_s
        speed = self.reference.prompt_per_second
        tokens = self.text.stat().st_size / BYTES_PER_TOKEN if self.text else 0.0
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
        free_gib = shutil.disk_usage(self.spec.run_dir).free / GIB
        if free_gib < KL_BASE_GIB:
            self._skip(
                "-ub",
                f"`-ub` not measured: the base file needs about {KL_BASE_GIB:g} GiB, {free_gib:.0f} free",
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
        try:
            await self._measure_reference()
            for step in self._plan():
                cost = step.cost()  # 0: nothing to do now, and not worth a line in the record
                if cost and not fits_budget(self.elapsed(), cost, self.spec.soft_limit_s):
                    self._skip(step.name, OUT_OF_BUDGET)
                    continue
                await step.run()
            chosen = await self._decide()
        finally:
            (self.spec.run_dir / "base.kld").unlink(missing_ok=True)
        return self._record(chosen)

    async def _measure_reference(self) -> None:
        """Start the reference first: it gives t0, the run time, the memory and the outputs to judge by.

        With speculation declared it is started a second time, for design 6.4 (a) and its own timing spread.
        """
        reference = await self._serve(Settings(), self.spec.reference, checked=False)
        self.arms.append(reference)
        if reference.error:
            msg = f"the reference could not be measured: {reference.error}"
            raise LlamaTuneError(msg)
        self.start_s, self.run_s = reference.start_s or 0.0, reference.run_s or 0.0
        reference.eligible = True
        if not (self.writes and any(c.spec is not SpecType.NONE for c in self.spec.speculation)):
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
            self._skip(arm_name(Settings(spec=candidate.spec)), f"not measured: the reference {why}")
            return
        settings = Settings(n_ubatch=self.ub_star, spec=candidate.spec, spec_draft_file=candidate.draft_file)
        await self._run_arm(settings, self.spec.reference, checked=bool(self.spec.quality_command))

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
        remaining = max(self.spec.soft_limit_s - self.elapsed(), 60.0)
        try:
            result = await asyncio.to_thread(
                run_bench, bench_bin, self.spec.reference.path, n_prompt=self.spec.prompt_tokens, n_gen=0,
                repetitions=BENCH_REPETITIONS, timeout_s=remaining, extra_flags=("-ub", ubatches),
            )  # fmt: skip
        except BenchError as exc:
            self._skip("-ub", f"not measured: llama-bench failed ({exc})")
            return
        self.bench = bench_rows(result.raw_rows, self.spec.prompt_tokens)
        self.measurements.extend(self.bench.values())

    async def _fits_screen(self, ubatch: int) -> bool:
        """``llama-fit-params`` at this ``-ub``: it keeps what cannot fit from starting, never picks."""
        name = f"-ub {ubatch}"
        free = self._free_gib()
        if free is None:
            self._skip(name, "not measured: memory not measured (available memory cannot be read)")
            return False
        try:
            projected = await self._screen(ubatch)
        except LocalModelsError as exc:
            self._skip(name, f"not measured: llama-fit-params gave no projection ({exc})")
            return False
        if projected + self.spec.headroom_gib > free:
            self._skip(
                name,
                f"not tried: llama.cpp projects {projected:.1f} GiB plus {self.spec.headroom_gib:g} "
                f"headroom, {free:.1f} free",
            )
            return False
        return True

    async def _kl_base(self) -> bool:
        """Save the reference's probabilities and run the restart control. False if either failed."""
        base = await self._perplexity(UB_REFERENCE, base=True)
        if base.error:
            self._skip("-ub", f"not measured: the base run failed ({base.error})")
            return False
        control = await self._perplexity(UB_REFERENCE, base=False)
        if control.stats is None:
            self._skip("-ub", f"not measured: the restart control failed ({control.error})")
            return False
        self.control = control.stats
        return True

    async def _kl(self, ubatch: int) -> KlResult:
        result = await self._perplexity(ubatch, base=False)
        if result.stats is not None and self.control is not None:
            result.agreement = judge_ub(result.stats, self.control, ub_limits(self.spec.kind))
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
            ["-f", str(self.text), "--kl-divergence-base", str(kld)]
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
            output = await self._run_tool(cmd, max(self.spec.soft_limit_s, 60.0))
            result.seconds = self._clock() - started
            self.ppl_s = result.seconds
            if not base:
                result.stats = parse_kl(output)
        except (LlamaTuneError, TimeoutError, OSError, ValueError) as exc:
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
        return None if reading is None else reading.bytes / GIB

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
                "host_reserve_gib": self.spec.headroom_gib,
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
            stop_grace_s=SERVER_STOP_GRACE_S,
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
                started = self._clock()
                handle = await manager.start()
                arm.start_s = self._clock() - started
                sampler = asyncio.create_task(self._sample(handle.pid, peaks))
                started = self._clock()
                await self._send_prompts(http, manager, arm)
                arm.run_s = self._clock() - started
                if checked and self.spec.quality_command:
                    arm.check = await self._check(manager.spec.port, files)
            except (LocalModelsError, httpx.HTTPError, OSError, KeyError, ValueError, TimeoutError) as exc:
                arm.error = f"{type(exc).__name__}: {exc}"
            finally:
                if sampler is not None:
                    sampler.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await sampler
                await manager.stop()
        if peaks:
            arm.peak_gib = max(peaks) / MIB_PER_GIB
        arm.fits = fit_verdict(arm.peak_gib, arm.free_gib, self.spec.headroom_gib)
        return arm

    async def _sample(self, pid: int, peaks: list[int]) -> None:
        """Note the server process's GPU memory every half second until cancelled."""
        while True:
            held = await asyncio.to_thread(gpu_process_memory_mib)
            if held and pid in held:
                peaks.append(held[pid])
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

    async def _check(self, port: int, files: FileSet) -> Agreement:
        """The project's task-quality check, run against the live server; its time counts inside the limit."""
        env = {**os.environ, CHECK_URL_ENV: f"http://{HOST}:{port}", CHECK_FILE_ENV: str(files.path)}
        started = self._clock()
        proc = await asyncio.create_subprocess_exec(
            *self.spec.quality_command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=env,
        )
        timeout = max(self.spec.check_s * CHECK_TIMEOUT_FACTOR, CHECK_TIMEOUT_MIN_S)
        try:
            out, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except TimeoutError:
            return Agreement(
                score=0.0, passed=False, detail=f"task-quality check not finished in {timeout:g} s"
            )
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
            f"peak {arm.peak_gib:.1f} GiB plus {self.spec.headroom_gib:g} headroom, "
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
        unverified: set[str] = set()
        while True:
            best = self._pick(assume_repeat=True, skip=frozenset(unverified))
            speculates = settings_key(best.settings)[1] != SpecType.NONE
            if best is self.reference or not speculates or best.repeats is not None:
                break
            if not fits_budget(self.elapsed(), self._arm_cost(), self.spec.soft_limit_s):
                self._skip(f"{best.name} restart", OUT_OF_BUDGET)
                unverified.add(best.name)
                continue
            again = await self._serve(best.settings, self.spec.reference, checked=False)
            self.measurements.append(again.model_copy(update={"name": f"{best.name}#restart"}))
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
        quality = "llama.cpp tune: " + (
            ", ".join(checks) if checks else "reference kept; no candidate passed"
        )
        self.measurements.extend(self.arms)
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
            llama_measurements=[m.model_dump(mode="json") for m in self.measurements],
        )


# ---------------------------------------------------------------------------------------------------------
# The command


async def run_guarded[T](work: Awaitable[T]) -> T:
    """Await ``work``; SIGTERM and SIGINT cancel it through the loop's own handler.

    ``timeout`` sends SIGTERM when the hard limit passes. Cancelling the task unwinds ``_serve``'s finally,
    which stops the server; a handler that raised inside the running loop could leave one loaded.
    """
    loop = asyncio.get_running_loop()
    task = asyncio.current_task()
    assert task is not None  # noqa: S101 - narrowing for mypy; asyncio.run runs this as a task
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, task.cancel)
    try:
        return await work
    finally:
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.remove_signal_handler(sig)


async def tune_and_adopt(spec: TuneSpec) -> int:
    """Hold the card quiet, tune, write ``record.json`` in the run folder and adopt it; the exit code."""
    card = QuietCard(
        GpuLease(DEFAULT_LEASE_DIR, PROJECT), reason=f"llama.cpp tune of {spec.reference.identity.repository}"
    )
    try:
        await asyncio.to_thread(card.__enter__)
    except QuietCardTimeoutError as exc:
        log.error("%s", exc)  # noqa: TRY400 - the message is the whole story
        return EXIT_FAILED
    try:
        record = await LlamaTune(spec).tune()
    except LlamaTuneError as exc:
        log.error("%s", exc)  # noqa: TRY400 - the message is the whole story
        return EXIT_FAILED
    finally:
        card.__exit__(None, None, None)
    (spec.run_dir / "record.json").write_text(record.model_dump_json(indent=2), encoding="utf-8")
    try:
        adopt(record)
    except AdoptError as exc:
        log.warning("measured, not adopted: %s", exc)
        return EXIT_NOT_ADOPTED
    return EXIT_ADOPTED


def _file_set(repository: str, revision: str, names: list[str]) -> FileSet:
    identity = ModelIdentity.from_hub_files(repository, revision, names)
    return FileSet(identity, snapshot_dir(repository, revision, _default_hub_cache()) / names[0])


def _alternative(text: str) -> FileSet:
    """``REPOSITORY@REVISION:FILE[,FILE...]`` as a file set."""
    repository, _, rest = text.partition("@")
    revision, _, names = rest.partition(":")
    if not (repository and revision and names):
        msg = f"{text!r}: expected REPOSITORY@REVISION:FILE[,FILE...]"
        raise argparse.ArgumentTypeError(msg)
    return _file_set(repository, revision, names.split(","))


def build_parser() -> argparse.ArgumentParser:
    """The command's arguments (design 5.4: the recipe command for llama.cpp)."""
    parser = argparse.ArgumentParser(
        prog="python -m localmodels.llama_tune", description=__doc__.split("\n\n")[0]
    )
    parser.add_argument("--repository", required=True, help="the model's Hub repository")
    parser.add_argument("--revision", required=True, help="its 40-character commit")
    parser.add_argument(
        "--file",
        action="append",
        required=True,
        dest="files",
        help="a file of the reference set (the first is loaded); repeat",
    )
    parser.add_argument(
        "--alternative",
        action="append",
        default=[],
        type=_alternative,
        help="a quantised file set for stage C, REPOSITORY@REVISION:FILE[,FILE...]",
    )
    parser.add_argument(
        "--kind",
        required=True,
        choices=[k.value for k in (WorkloadKind.K1, WorkloadKind.K2, WorkloadKind.K3)],
    )
    parser.add_argument(
        "--role",
        required=True,
        choices=[r.value for r in ServingRole],
        help="the serving layout; its slot count is never tuned",
    )
    parser.add_argument(
        "--prompts",
        required=True,
        type=Path,
        help="the project's real inputs: a JSON array or JSON lines of strings",
    )
    parser.add_argument("--prompt-tokens", required=True, type=int, help="their typical length in tokens")
    parser.add_argument(
        "--text", type=Path, help="a text for the teacher-forced check (default: the pinned one)"
    )
    parser.add_argument(
        "--speculation",
        action="append",
        default=[],
        choices=[s.value for s in SpecType],
        help="a candidate to measure; repeat",
    )
    parser.add_argument("--draft-file", type=Path, help="the draft file for draft-simple")
    parser.add_argument(
        "--quality-command",
        help="the project's task-quality check, run against a live server (see the module docstring)",
    )
    parser.add_argument("--quality-name", default="", help="its name, for the record")
    parser.add_argument(
        "--check-seconds", type=float, default=60.0, help="how long the check is expected to take"
    )
    parser.add_argument("--headroom-gib", type=float, default=DEFAULT_HEADROOM_GIB)
    parser.add_argument(
        "--run-dir",
        required=True,
        type=Path,
        help="a folder for logs and the teacher-forced base file (about 16 GiB while it runs)",
    )
    parser.add_argument("--soft-limit-s", type=float, default=SOFT_LIMIT_S)
    parser.add_argument("--max-prompts", type=int, default=DEFAULT_MAX_PROMPTS)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--project", help="the project asking, for the record")
    return parser


def spec_from_args(args: argparse.Namespace) -> TuneSpec:
    """Resolve the arguments: the files in the Hub cache, the binaries, the layout of the role."""
    candidates = tuple(
        SpecCandidate(SpecType(s), str(args.draft_file) if s == SpecType.DRAFT_SIMPLE else None)
        for s in args.speculation
    )
    if any(c.spec is SpecType.DRAFT_SIMPLE and c.draft_file is None for c in candidates):
        msg = "--speculation draft-simple needs --draft-file"
        raise SystemExit(msg)
    report = inspect_hardware()
    reference = _file_set(args.repository, args.revision, args.files)
    if not reference.path.is_file():
        msg = f"{reference.path} is not in the Hub cache; pull it first (a tune never downloads)"
        raise SystemExit(msg)
    server = resolve_llama_server_bin(explicit=current_server(llama_home()))
    return TuneSpec(
        reference=reference,
        alternatives=tuple(args.alternative),
        kind=WorkloadKind(args.kind),
        params=derive_serving_params(report.device, report.total_memory_gb, role=ServingRole(args.role)),
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
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    spec = spec_from_args(build_parser().parse_args(argv))
    try:
        return asyncio.run(run_guarded(tune_and_adopt(spec)))
    except asyncio.CancelledError:
        return SIGTERM_EXIT


if __name__ == "__main__":
    sys.exit(main())
