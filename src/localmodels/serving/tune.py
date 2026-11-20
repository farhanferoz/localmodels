"""Choose a model's numeric precision on this machine by measuring it. Torch-free.

``run_tune`` starts a fresh model server per candidate (a precision plus load options), times the same inputs
on each and compares every candidate's outputs with the first, the reference, through a quality check. It
picks the fastest candidate that agrees with the reference and fits in memory. The result is a ``TuneRecord``
that ``save_record`` keeps per machine (``hardware_key``) and ``load_choice`` reads back, so a service runs
the measured settings and says so (``untuned_message``) when it has none. Running a tune is GPU work: submit
it through the scheduler like any other GPU job.
"""

from __future__ import annotations

import contextlib
import difflib
import json
import logging
import math
import statistics
import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Final

from pydantic import BaseModel, ConfigDict, Field

from localmodels.atomic import write_text_atomically
from localmodels.enums import Precision  # noqa: TC001 - pydantic needs it at runtime
from localmodels.hardware import HardwareReport, available_memory, inspect_hardware
from localmodels.serving.client import ModelServerProcess
from localmodels.serving.errors import ServingCancelled, ServingError
from localmodels.serving.protocol import InferMessage, LoadMessage, ResultMessage
from localmodels.stack import fingerprint, stale_components
from localmodels.version import package_version

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from pathlib import Path

    from localmodels.lease import QuietCard

GIB: Final = 1024**3
DEADLINE_ERROR: Final = "not measured: the tune's deadline passed"

# shared by every tune command (llama_tune, the in-process tune); the numbers are facts of this host's guards
KILL_BELOW_GIB: Final = 10.0  # service_c's floor: above earlyoom's SIGTERM at 8 GiB on this host
DEFAULT_HEADROOM_GIB: Final = 4.0  # as service_b's tune uses (commands/tune.py:42)
STOP_MARGIN_S: Final = (
    60.0  # a tune stops this long before the job's ceiling, and starts no step this long before that stop
)

# exit codes of a tune command
EXIT_ADOPTED: Final = 0
EXIT_FAILED: Final = 1
EXIT_NOT_ADOPTED: Final = 2

UNKNOWN: Final = "-"  # a cell of the printed table that was not measured

log = logging.getLogger(__name__)


def fits_budget(elapsed_s: float, cost_s: float, soft_limit_s: float) -> bool:
    """Whether a step estimated at ``cost_s`` can finish inside the soft limit."""
    return elapsed_s + cost_s <= soft_limit_s


def hard_limit_s(limit_s: float) -> float:
    """The time after which nothing runs: the job's ceiling ``limit_s`` less one stop margin."""
    return limit_s - STOP_MARGIN_S


def soft_limit_s(limit_s: float) -> float:
    """The time after which no step is started: the job's ceiling ``limit_s`` less two stop margins."""
    return hard_limit_s(limit_s) - STOP_MARGIN_S


def free_margin_gib(headroom_gib: float) -> float:
    """What a server must leave available: the headroom, never less than the floor its guard kills at."""
    return max(headroom_gib, KILL_BELOW_GIB)


class TuneError(Exception):
    """A tune that cannot give a choice: the reference failed, or no candidate is eligible."""


class ReferenceFailedError(TuneError):
    """The reference could not be measured, so nothing could be compared: the tune failed."""


class NoEligibleCandidateError(TuneError):
    """Everything was measured, and no candidate agrees with the reference and fits: nothing to adopt."""


class TuneCandidate(BaseModel):
    """One setting to measure: a precision and load options merged over the base load's."""

    model_config = ConfigDict(frozen=True)

    name: str
    dtype: Precision
    options: dict[str, Any] = Field(default_factory=dict)


class Agreement(BaseModel):
    """How a candidate's outputs compare with the reference's."""

    score: float
    passed: bool
    detail: str


type QualityCheck = Callable[[Sequence[ResultMessage], Sequence[ResultMessage]], Agreement]
"""Reference outputs and candidate outputs (one per input, same order) -> how well they agree."""


class TuneMeasurement(BaseModel):
    """What running one candidate showed; ``error`` is set when it could not run."""

    candidate: TuneCandidate
    load_s: float | None = None
    infer_s: list[float] = Field(default_factory=list)
    infer_s_median: float | None = None
    pass_s: list[float] = Field(default_factory=list)  # the total seconds of each timed pass over the inputs
    peak_gib: float | None = None
    agreement: Agreement | None = None
    fits: bool = False
    error: str | None = None


class TuneBudget(BaseModel):
    """The time a tune may use, in seconds from ``started`` (a ``time.monotonic()`` instant).

    No candidate starts that would end past ``soft_limit_s``; one running at ``hard_limit_s`` is stopped.
    """

    model_config = ConfigDict(frozen=True)

    started: float
    soft_limit_s: float
    hard_limit_s: float

    @classmethod
    def for_job(cls, started: float, limit_s: float) -> TuneBudget:
        """The budget of a job whose ceiling (its ``timeout``) is ``limit_s`` seconds after ``started``."""
        return cls(started=started, soft_limit_s=soft_limit_s(limit_s), hard_limit_s=hard_limit_s(limit_s))


class TuneRecord(BaseModel):
    """The outcome of a tune on one machine."""

    hardware: str
    localmodels_version: str
    measured_at: datetime
    reference: str
    quality: str
    chosen: TuneCandidate
    measurements: list[TuneMeasurement]
    #: The host stack (``stack.fingerprint()``) it was measured on; None for a record made before stamps.
    stack: dict[str, str] | None = None
    #: The library versions the reference's model server reported (empty from a server that reports none).
    libraries: dict[str, str] = Field(default_factory=dict)


def _ranks(values: Sequence[float]) -> list[float]:
    """Ranks from 1, tied values sharing the average of their positions."""
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start
        while end + 1 < len(order) and values[order[end + 1]] == values[order[start]]:
            end += 1
        for index in order[start : end + 1]:
            ranks[index] = (start + end) / 2 + 1
        start = end + 1
    return ranks


def _pearson(a: Sequence[float], b: Sequence[float]) -> float:
    """Pearson correlation; 1 for two constant series that are equal, 0 for any other with no spread."""
    mean_a, mean_b = statistics.fmean(a), statistics.fmean(b)
    cov = sum((x - mean_a) * (y - mean_b) for x, y in zip(a, b, strict=True))
    spread = math.sqrt(sum((x - mean_a) ** 2 for x in a) * sum((y - mean_b) ** 2 for y in b))
    if spread == 0:
        return 1.0 if list(a) == list(b) else 0.0
    return cov / spread


def spearman(a: Sequence[float], b: Sequence[float]) -> float:
    """Spearman rank correlation of two equally long series (average ranks for ties)."""
    return _pearson(_ranks(a), _ranks(b))


def text_agreement(min_ratio: float = 0.9) -> QualityCheck:
    """Mean word-level similarity of ``raw_text`` (``difflib`` ratio per input); passes at ``min_ratio``."""

    def check(reference: Sequence[ResultMessage], candidate: Sequence[ResultMessage]) -> Agreement:
        ratios = [
            difflib.SequenceMatcher(None, ref.raw_text.split(), cand.raw_text.split()).ratio()
            for ref, cand in zip(reference, candidate, strict=True)
        ]
        mean = statistics.fmean(ratios)
        return Agreement(score=mean, passed=mean >= min_ratio, detail=f"mean word similarity {mean:.3f}")

    return check


def score_agreement(key: str, min_spearman: float = 0.99, max_abs_diff: float = 0.05) -> QualityCheck:
    """Rank agreement and largest difference of the name -> float scores in ``native_outputs[key]``.

    Every (input, name) pair present in both outputs is pooled; passes when the Spearman correlation is at
    least ``min_spearman`` and no pair differs by more than ``max_abs_diff``. The score is the correlation.
    """

    def check(reference: Sequence[ResultMessage], candidate: Sequence[ResultMessage]) -> Agreement:
        ref_scores: list[float] = []
        cand_scores: list[float] = []
        for ref, cand in zip(reference, candidate, strict=True):
            ref_map: Mapping[str, float] = ref.native_outputs.get(key, {})
            cand_map: Mapping[str, float] = cand.native_outputs.get(key, {})
            for name in ref_map.keys() & cand_map.keys():
                ref_scores.append(float(ref_map[name]))
                cand_scores.append(float(cand_map[name]))
        if not ref_scores:
            return Agreement(score=0.0, passed=False, detail=f"no scores named in {key!r} to compare")
        rho = spearman(ref_scores, cand_scores)
        worst = max(abs(x - y) for x, y in zip(ref_scores, cand_scores, strict=True))
        return Agreement(
            score=rho,
            passed=rho >= min_spearman and worst <= max_abs_diff,
            detail=f"Spearman {rho:.4f} over {len(ref_scores)} scores, largest difference {worst:.4f}",
        )

    return check


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return 0.0 if norm == 0 else sum(x * y for x, y in zip(a, b, strict=True)) / norm


def embedding_cosine(key: str, min_cosine: float = 0.999) -> QualityCheck:
    """The smallest cosine similarity over inputs of the float list in ``native_outputs[key]``."""

    def check(reference: Sequence[ResultMessage], candidate: Sequence[ResultMessage]) -> Agreement:
        cosines = [
            _cosine(ref.native_outputs[key], cand.native_outputs[key])
            for ref, cand in zip(reference, candidate, strict=True)
        ]
        worst = min(cosines)
        return Agreement(score=worst, passed=worst >= min_cosine, detail=f"smallest cosine {worst:.5f}")

    return check


def measurement_line(measurement: TuneMeasurement, chosen: str) -> str:
    """One printed line of a candidate: load seconds, median inference seconds, peak GiB, agreement."""
    name = measurement.candidate.name
    if measurement.error is not None:
        return f"{name:<10} failed: {measurement.error}"
    agreement = measurement.agreement
    cells = [
        f"load {measurement.load_s:.1f} s" if measurement.load_s is not None else UNKNOWN,
        f"median {measurement.infer_s_median:.2f} s" if measurement.infer_s_median is not None else UNKNOWN,
        f"peak {measurement.peak_gib:.1f} GiB" if measurement.peak_gib is not None else UNKNOWN,
        f"agreement {agreement.score:.4f} ({'pass' if agreement.passed else 'fail'})"
        if agreement
        else UNKNOWN,
        "fits" if measurement.fits else "does not fit",
        "<- chosen" if name == chosen else "",
    ]
    return f"{name:<10} " + "  ".join(cells).rstrip()


def hardware_key(report: HardwareReport) -> str:
    """Device type, device name, memory in whole GB and architecture, joined with ':'."""
    return ":".join(
        [report.device.value, report.device_name, f"{round(report.total_memory_gb)}GB", report.arch]
    )


def _measure(
    candidate: TuneCandidate,
    *,
    argv: Sequence[str],
    env: Mapping[str, str],
    load: LoadMessage,
    inputs: Sequence[InferMessage],
    repeats: int,
    cancel_check: Callable[[], bool],
) -> tuple[TuneMeasurement, list[ResultMessage], dict[str, str]]:
    """Load and time one candidate on a fresh server.

    Its last pass's outputs and the library versions its server reported come back with the numbers.
    """
    server = ModelServerProcess(argv, env, candidate.name)
    request = load.model_copy(
        update={"dtype": candidate.dtype, "options": {**load.options, **candidate.options}}
    )
    times: list[float] = []
    passes: list[float] = []
    outputs: list[ResultMessage] = []
    try:
        server.start()
        loaded = server.load(request, cancel_check)
        peak = server.infer(
            inputs[0], cancel_check, lambda _p: None
        ).peak_allocated_gib  # warm-up, unmeasured
        for _ in range(repeats):
            outputs = []
            first = len(times)
            for message in inputs:
                started = time.perf_counter()
                result = server.infer(message, cancel_check, lambda _p: None)
                times.append(time.perf_counter() - started)
                peak = max(peak, result.peak_allocated_gib)
                outputs.append(result)
            passes.append(sum(times[first:]))
    finally:
        server.stop()
    measurement = TuneMeasurement(
        candidate=candidate,
        load_s=loaded.seconds,
        infer_s=times,
        infer_s_median=statistics.median(times),
        pass_s=passes,
        peak_gib=peak,
    )
    return measurement, outputs, loaded.libraries


class _Timekeeper:
    """A tune's time against its budget: whether the next candidate may start, and the hard stop."""

    def __init__(self, budget: TuneBudget | None, cancel_check: Callable[[], bool]) -> None:
        self.budget = budget
        self.cancel_check = cancel_check
        self.longest_s = 0.0  # the longest whole measurement so far: what the next candidate is taken to cost
        self.out_of_time = False  # the budget's hard limit stopped a candidate

    def _elapsed(self) -> float:
        return time.monotonic() - self.budget.started if self.budget is not None else 0.0

    def may_start(self) -> bool:
        """Whether a candidate costing ``longest_s`` ends inside the soft limit (always, with no budget)."""
        if self.budget is None:
            return True
        return not self.out_of_time and fits_budget(self._elapsed(), self.longest_s, self.budget.soft_limit_s)

    def measured(self, wall_s: float) -> None:
        """Count one whole measurement's wall time."""
        self.longest_s = max(self.longest_s, wall_s)

    def stop_check(self) -> bool:
        """The caller's cancel check, also true from the hard limit on (remembered in ``out_of_time``)."""
        if self.budget is not None and self._elapsed() >= self.budget.hard_limit_s:
            self.out_of_time = True
            return True
        return self.cancel_check()

    def stopped_reason(self) -> str:
        """Why the reference had to be given up when the hard limit stopped it."""
        limit = self.budget.hard_limit_s if self.budget is not None else math.inf
        return f"was still running at the tune's hard limit ({limit:g} s); raise the job's limit"


def _not_measured(
    position: int, candidate: TuneCandidate, error: str, reference_reason: str
) -> TuneMeasurement:
    """A candidate not measured, recorded with ``error``; for the reference, ``TuneError`` instead."""
    if position == 0:
        msg = f"Reference candidate {candidate.name} {reference_reason}, so nothing can be compared"
        raise ReferenceFailedError(msg)
    return TuneMeasurement(candidate=candidate, error=error)


def run_tune(  # noqa: PLR0913 - one keyword-only entry point; the arguments are the whole contract
    *,
    argv: Sequence[str],
    env: Mapping[str, str],
    load: LoadMessage,
    inputs: Sequence[InferMessage],
    candidates: Sequence[TuneCandidate],
    quality: QualityCheck,
    quality_name: str,
    headroom_gib: float,
    repeats: int = 1,
    cancel_check: Callable[[], bool] = lambda: False,
    hardware: HardwareReport | None = None,
    quiet_card: QuietCard | None = None,
    budget: TuneBudget | None = None,
) -> TuneRecord:
    """Measure every candidate and choose the fastest that agrees with ``candidates[0]`` and fits in memory.

    ``quiet_card``, when given, is held around the measurements so that stay-up services have unloaded
    (``lease.QuietCard``); without it nothing changes. ``budget``, when given, is checked before each
    candidate after the reference (which always starts): a candidate that, costing as long as the longest
    whole measurement so far (load, warm-up and timed passes), would end past ``budget.soft_limit_s`` is not
    started but recorded with ``DEADLINE_ERROR``. A candidate still running at ``budget.hard_limit_s`` is
    stopped and recorded the same way, as is every one after it; the choice is made among those measured, so
    the record is written before the job's own ceiling.

    Raises:
        ValueError: two candidates have one name (nothing has been measured).
        ReferenceFailedError: the reference failed or reached the budget's hard limit (a ``TuneError``).
        NoEligibleCandidateError: no candidate is eligible (a ``TuneError``).
        ServingCancelled: ``cancel_check()`` turned true.
        QuietCardTimeoutError: ``quiet_card`` found a service still loaded after its timeout.
    """
    if len({c.name for c in candidates}) != len(candidates):
        msg = "Candidate names must be unique: " + ", ".join(c.name for c in candidates)
        raise ValueError(msg)
    report = hardware or inspect_hardware()
    measurements: list[TuneMeasurement] = []
    reference_outputs: list[ResultMessage] = []
    clock = _Timekeeper(budget, cancel_check)
    libraries: dict[str, str] = {}
    with quiet_card if quiet_card is not None else contextlib.nullcontext():
        for position, candidate in enumerate(candidates):
            if position and not clock.may_start():
                measurements.append(TuneMeasurement(candidate=candidate, error=DEADLINE_ERROR))
                continue
            free = available_memory(report.device.value)
            began = time.monotonic()
            try:
                measurement, outputs, reported = _measure(
                    candidate,
                    argv=argv,
                    env=env,
                    load=load,
                    inputs=inputs,
                    repeats=repeats,
                    cancel_check=clock.stop_check,
                )
            except ServingCancelled:
                if not clock.out_of_time:
                    raise
                measurements.append(
                    _not_measured(position, candidate, DEADLINE_ERROR, clock.stopped_reason())
                )
                continue
            except ServingError as exc:
                measurements.append(_not_measured(position, candidate, exc.message, f"failed: {exc.message}"))
                continue
            assert measurement.peak_gib is not None  # noqa: S101 - narrowing for mypy
            measurement.fits = free is None or measurement.peak_gib + headroom_gib <= free.bytes / GIB
            clock.measured(time.monotonic() - began)
            if position == 0:
                reference_outputs, libraries = outputs, reported
                measurement.agreement = Agreement(score=1.0, passed=True, detail="reference")
            else:
                measurement.agreement = quality(reference_outputs, outputs)
            if free is None:
                measurement.agreement.detail += " (free memory unknown, so it is taken to fit)"
            measurements.append(measurement)
    eligible = [
        m
        for m in measurements
        if m.error is None and m.fits and m.agreement is not None and m.agreement.passed
    ]
    if not eligible:
        msg = "No candidate agrees with the reference and fits in memory; see the measurements."
        raise NoEligibleCandidateError(msg)
    best = min(eligible, key=lambda m: m.infer_s_median if m.infer_s_median is not None else math.inf)
    return TuneRecord(
        hardware=hardware_key(report),
        localmodels_version=package_version(),
        measured_at=datetime.now(UTC),
        reference=candidates[0].name,
        quality=quality_name,
        chosen=best.candidate,
        measurements=measurements,
        stack=fingerprint(),
        libraries=libraries,
    )


def _read_records(path: Path) -> list[TuneRecord]:
    if not path.is_file():
        return []
    return [
        TuneRecord.model_validate(item) for item in json.loads(path.read_text(encoding="utf-8"))["records"]
    ]


def save_record(path: Path, record: TuneRecord) -> None:
    """Write ``record`` into the file at ``path``, replacing the record for the same hardware."""
    kept = [r for r in _read_records(path) if r.hardware != record.hardware]
    payload = {"records": [r.model_dump(mode="json") for r in [*kept, record]]}
    write_text_atomically(path, json.dumps(payload, indent=2) + "\n")


def load_choice(path: Path, report: HardwareReport) -> TuneCandidate | None:
    """The candidate chosen for this machine, or None when the file or this machine's record is missing."""
    key = hardware_key(report)
    record = next((r for r in _read_records(path) if r.hardware == key), None)
    if record is None:
        return None
    if record.stack is not None and (stale := stale_components(record.stack, fingerprint())):
        log.warning(
            "The tune record for %s was measured on a different stack (%s); re-run the tune step.",
            key,
            "; ".join(stale),
        )
    return record.chosen


def untuned_message(model: str, key: str) -> str:
    """The plain-words notice that ``model`` has no tune record for the hardware ``key``."""
    return (
        f"{model} is not tuned for this hardware ({key}); it runs its default settings. "
        "Run the tune step on this machine to measure them."
    )
