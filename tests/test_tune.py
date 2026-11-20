"""The tune step: measuring precision candidates on the echo server and keeping the choice per machine."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

import pytest

from localmodels.enums import DeviceTier, DeviceType, Precision
from localmodels.hardware import HardwareReport
from localmodels.lease import GpuLease, QuietCard
from localmodels.serving import (
    InferMessage,
    LoadMessage,
    NoEligibleCandidateError,
    ReferenceFailedError,
    ResultMessage,
    ServingCancelled,
    TuneBudget,
    TuneCandidate,
    TuneMeasurement,
    TuneRecord,
    embedding_cosine,
    hardware_key,
    load_choice,
    run_tune,
    save_record,
    score_agreement,
    text_agreement,
)
from localmodels.serving import tune as tune_module
from localmodels.serving.tune import DEADLINE_ERROR, free_margin_gib, measurement_line
from tests.test_serving import ECHO_ARGV, ECHO_ENV

if TYPE_CHECKING:
    from pathlib import Path

CPU = HardwareReport(
    platform="linux", arch="x86_64", device=DeviceType.CPU, device_name="CPU", total_memory_gb=8.0,
    tier=DeviceTier.SLOW_CPU,
)  # fmt: skip
OTHER = CPU.model_copy(update={"device_name": "Another CPU"})
LOAD = LoadMessage(model_dir="/m", revision="r", dtype=Precision.FLOAT32)
INPUTS = [InferMessage(request_id="a", input_dir="same words every time")]
SLOW = TuneCandidate(
    name="fp32",
    dtype=Precision.FLOAT32,
    options={"sleep_s": 0.2, "text": "one two three", "libraries": {"torch": "9.9"}},
)
FAST = TuneCandidate(name="bf16", dtype=Precision.BFLOAT16, options={"sleep_s": 0.0, "text": "one two three"})
FAST_BUT_WRONG = TuneCandidate(name="fp16", dtype=Precision.FLOAT16, options={"text": "nothing alike here"})
BROKEN = TuneCandidate(name="bad", dtype=Precision.FLOAT16, options={"fail_dtype": "float16"})


def tune(
    *candidates: TuneCandidate,
    repeats: int = 1,
    budget: TuneBudget | None = None,
    inputs: list[InferMessage] = INPUTS,
) -> TuneRecord:
    return run_tune(
        argv=ECHO_ARGV, env=ECHO_ENV, load=LOAD, inputs=inputs, candidates=candidates,
        quality=text_agreement(0.9), quality_name="text 0.9", headroom_gib=0.0, hardware=CPU,
        repeats=repeats, budget=budget,
    )  # fmt: skip


def result(**native: object) -> ResultMessage:
    return ResultMessage(request_id="r", raw_text="", native_outputs=native)


def test_the_fastest_candidate_that_passes_quality_is_chosen() -> None:
    record = tune(SLOW, FAST_BUT_WRONG, FAST)
    assert record.chosen.name == "bf16"
    by_name = {m.candidate.name: m for m in record.measurements}
    assert by_name["fp16"].agreement is not None
    assert not by_name["fp16"].agreement.passed
    assert by_name["fp32"].agreement is not None
    assert by_name["fp32"].agreement.detail.startswith("reference")


def test_the_reference_reports_its_libraries_and_each_timed_pass_is_totalled() -> None:
    two_inputs = [*INPUTS, InferMessage(request_id="b", input_dir="the second input")]
    record = tune(SLOW, FAST, repeats=2, inputs=two_inputs)
    assert record.libraries == {"torch": "9.9"}
    reference = record.measurements[0]
    times = reference.infer_s
    assert reference.pass_s == pytest.approx([times[0] + times[1], times[2] + times[3]])


_MEASURE = tune_module._measure


class _Clock:
    """The tune's clock: still unless a measurement moves it (the timed passes keep the real one)."""

    def __init__(self) -> None:
        self.now = 0.0
        self.perf_counter = time.perf_counter

    def monotonic(self) -> float:
        return self.now


def _costed(monkeypatch: pytest.MonkeyPatch, costs: dict[str, float], stopped: str = "") -> None:
    """Each candidate's measurement takes ``costs[name]`` seconds of the tune's clock.

    ``stopped`` is still running when its time has passed: the client then asks the stop check, as it does
    once a second, and is cancelled if the check says so.
    """
    clock = _Clock()
    monkeypatch.setattr(tune_module, "time", clock)
    measure = _MEASURE  # the real one, also when an earlier call already replaced it

    def costed(candidate: TuneCandidate, **kw: Any) -> Any:
        clock.now += costs[candidate.name]
        if candidate.name == stopped and kw["cancel_check"]():
            raise ServingCancelled
        return measure(candidate, **kw)

    monkeypatch.setattr(tune_module, "_measure", costed)


def test_a_budget_stops_a_candidate_that_would_end_past_the_soft_limit_but_never_the_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # the reference costs 100 s, so the next candidate, asked about at 100 s, would end at 200 s
    _costed(monkeypatch, {"fp32": 100, "bf16": 100})
    roomy = TuneBudget(started=0, soft_limit_s=200, hard_limit_s=1e9)
    assert tune(SLOW, FAST, budget=roomy).chosen.name == "bf16"
    _costed(monkeypatch, {"fp32": 300, "bf16": 1})
    record = tune(SLOW, FAST, budget=roomy)
    assert record.chosen.name == "fp32"  # the reference runs whatever the budget, and is chosen alone
    skipped = record.measurements[1]
    assert (skipped.candidate.name, skipped.error) == ("bf16", DEADLINE_ERROR)
    assert record.measurements[0].infer_s_median is not None
    # the cost is the longest measurement so far, not the reference's: fp16's 100 s stops bf16 at 110 s
    _costed(monkeypatch, {"fp32": 10, "fp16": 100, "bf16": 1})
    record = tune(
        SLOW, FAST_BUT_WRONG, FAST, budget=TuneBudget(started=0, soft_limit_s=150, hard_limit_s=1e9)
    )
    assert [m.error for m in record.measurements] == [None, None, DEADLINE_ERROR]


def test_the_hard_limit_stops_a_running_candidate_and_the_tune_still_chooses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    budget = TuneBudget.for_job(started=0, limit_s=200)  # hard limit 140 s, soft limit 80 s
    assert (budget.hard_limit_s, budget.soft_limit_s) == (140, 80)
    _costed(monkeypatch, {"fp32": 10, "bf16": 200, "fp16": 1}, stopped="bf16")
    record = tune(SLOW, FAST, FAST_BUT_WRONG, budget=budget)
    assert record.chosen.name == "fp32"
    assert [m.error for m in record.measurements] == [None, DEADLINE_ERROR, DEADLINE_ERROR]  # none after it
    _costed(monkeypatch, {"fp32": 200}, stopped="fp32")
    with pytest.raises(ReferenceFailedError, match="still running at the tune's hard limit"):
        tune(SLOW, FAST, budget=budget)


def test_a_tune_where_nothing_fits_has_nothing_to_adopt_which_is_not_a_failed_reference() -> None:
    too_big = SLOW.model_copy(update={"options": {**SLOW.options, "peak_gib": 1e6}})
    with pytest.raises(NoEligibleCandidateError, match="No candidate agrees"):
        tune(too_big)


def test_the_free_memory_margin_is_never_below_the_kill_floor() -> None:
    assert free_margin_gib(4.0) == 10.0
    assert free_margin_gib(12.0) == 12.0


def test_a_measurement_prints_as_one_line_with_dashes_for_what_was_not_measured() -> None:
    failed = TuneMeasurement(candidate=FAST, error="boom")
    assert measurement_line(failed, "bf16") == "bf16       failed: boom"
    partial = TuneMeasurement(candidate=FAST, load_s=1.25, fits=True)
    assert measurement_line(partial, "bf16") == "bf16       load 1.2 s  -  -  -  fits  <- chosen"


def test_a_candidate_that_cannot_run_is_recorded_and_skipped() -> None:
    record = tune(SLOW, BROKEN, FAST)
    failed = next(m for m in record.measurements if m.candidate.name == "bad")
    assert failed.error == "float16 is not supported"
    assert record.chosen.name == "bf16"


def test_a_reference_that_cannot_run_or_two_candidates_of_one_name_end_the_tune() -> None:
    reference = TuneCandidate(name="fp32", dtype=Precision.FLOAT32, options={"fail_dtype": "float32"})
    with pytest.raises(ReferenceFailedError, match="Reference candidate fp32 failed"):
        tune(reference, FAST)
    with pytest.raises(ValueError, match="unique"):
        tune(SLOW, FAST, FAST.model_copy(update={"dtype": Precision.FLOAT16}))


def test_score_agreement_uses_rank_correlation_with_ties_and_the_largest_difference() -> None:
    ref = [result(s={"a": 0.1, "b": 0.1, "c": 0.9})]
    check = score_agreement("s", min_spearman=0.99, max_abs_diff=0.05)
    same_order = check(ref, [result(s={"a": 0.12, "b": 0.12, "c": 0.8})])
    assert same_order.score == pytest.approx(1.0)
    assert not same_order.passed  # c moved by 0.1
    swapped = check(ref, [result(s={"a": 0.1, "b": 0.5, "c": 0.9})])
    assert swapped.score == pytest.approx(0.8660, abs=1e-4)  # ranks (1.5, 1.5, 3) against (1, 2, 3)
    assert not swapped.passed


def test_embedding_cosine_is_the_smallest_over_inputs() -> None:
    ref = [result(e=[1.0, 0.0]), result(e=[0.0, 1.0])]
    cand = [result(e=[2.0, 0.0]), result(e=[1.0, 1.0])]
    agreement = embedding_cosine("e", min_cosine=0.9)(ref, cand)
    assert agreement.score == pytest.approx(0.7071, abs=1e-4)
    assert not agreement.passed


def test_saving_a_record_replaces_its_own_hardware_and_keeps_the_others(tmp_path: Path) -> None:
    path = tmp_path / "tuning" / "model.json"
    here = tune(SLOW, FAST)
    save_record(path, here.model_copy(update={"hardware": hardware_key(OTHER), "chosen": SLOW}))
    save_record(path, here.model_copy(update={"chosen": SLOW}))
    save_record(path, here)
    assert load_choice(path, CPU) == FAST
    assert load_choice(path, OTHER) == SLOW
    assert load_choice(tmp_path / "missing.json", CPU) is None


def test_a_record_made_on_another_stack_warns_once_and_still_gives_its_choice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    stack = {"driver": "580.1", "cuda": "13.0", "kernel": "6.17", "dgx_os": "7.6.0", "localmodels": "0.5.0"}
    monkeypatch.setattr("localmodels.serving.tune.fingerprint", lambda: stack)
    path = tmp_path / "model.json"
    record = tune(SLOW, FAST)
    assert record.stack == stack
    save_record(path, record)

    def warnings_for(current: dict[str, str]) -> list[str]:
        monkeypatch.setattr("localmodels.serving.tune.fingerprint", lambda: current)
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="localmodels.serving.tune"):
            assert load_choice(path, CPU) == FAST
        return [r.getMessage() for r in caplog.records]

    assert warnings_for(stack) == []
    assert warnings_for({**stack, "localmodels": "0.6.0"}) == []  # this package's version is not a stale part
    (message,) = warnings_for({**stack, "driver": "590.2", "kernel": "6.18"})
    assert "driver: 580.1 -> 590.2" in message
    assert "kernel: 6.17 -> 6.18" in message
    assert "cuda" not in message
    save_record(path, record.model_copy(update={"stack": None}))
    assert warnings_for({**stack, "driver": "590.2"}) == []


def test_the_quiet_card_is_withdrawn_when_the_tune_fails(tmp_path: Path) -> None:
    card = QuietCard(GpuLease(tmp_path, project="tuner"), reason="tune", timeout_s=1, poll_s=0.02)
    with pytest.raises(ReferenceFailedError, match="Reference candidate bad failed"):
        run_tune(
            argv=ECHO_ARGV, env=ECHO_ENV, load=LOAD, inputs=INPUTS, candidates=[BROKEN, FAST],
            quality=text_agreement(0.9), quality_name="text 0.9", headroom_gib=0.0, hardware=CPU,
            quiet_card=card,
        )  # fmt: skip
    assert list((tmp_path / "quiet").glob("*.json")) == []
