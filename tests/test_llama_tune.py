"""The llama.cpp tune: its limits on saved measurements, its stages with every program faked, its guards."""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import signal
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Self

import httpx
import pytest

from localmodels import launcher, llama_tune
from localmodels.enums import DeviceType, ServingRole, WorkloadKind
from localmodels.hardware import AvailableMemory
from localmodels.lease import GpuLease
from localmodels.llama_tune import (
    UB_LIMITS,
    ArmResult,
    FileSet,
    KlResult,
    LlamaTune,
    NotMeasured,
    Planned,
    Priority,
    SpecCandidate,
    TuneSpec,
    bench_rows,
    fit_verdict,
    fits_budget,
    in_priority_order,
    judge_speculation,
    judge_ub,
    parse_acceptance,
    parse_check,
    parse_kl,
    run_guarded,
    tune_and_adopt,
)
from localmodels.params import derive_serving_params
from localmodels.pinned import PinnedFile, snapshot_dir
from localmodels.process import _is_pid_alive
from localmodels.serving.tune import Agreement
from localmodels.settings import ModelIdentity, SettingsRecord, adopt, read_store, record_path
from localmodels.stack import Settings, SpecType

FIXTURES = Path(__file__).parent / "fixtures" / "llama_tune"
K1, K3 = WorkloadKind.K1, WorkloadKind.K3
GIB = 1024**3
MODEL = ModelIdentity(repository="Qwen/Qwen3-30B-A3B-GGUF", revision="c" * 40)
ALTERNATIVE_FILE = PinnedFile(path="model-iq4.gguf", sha256="b" * 64)
TEXTS = [f"the answer to question {i} is forty two and then some more words follow here" for i in range(6)]
PLAIN = json.loads((FIXTURES / "hl3_plain1_texts.json").read_text())
MTP = json.loads((FIXTURES / "hl3_mtp1_texts.json").read_text())


def stats(name: str) -> Any:
    return parse_kl((FIXTURES / name).read_text())


DRIFTED = stats("hl2c_kld_ub2048.txt").model_copy(update={"same_top": 0.96})  # outside UB_LIMITS


def test_ub_limits_applied_to_the_saved_teacher_forced_outputs() -> None:
    control = stats("hl2c_kld_ub512.txt")  # the restart: the base file's 16-bit storage gives 1.0013
    assert control.ppl_ratio == pytest.approx(1.001281)
    for ub in (1024, 2048):
        saved = stats(f"hl2c_kld_ub{ub}.txt")
        assert 0.016 < saved.mean_kl < 0.019
        assert 0.976 < saved.same_top < 0.978
        assert judge_ub(saved, control, UB_LIMITS).passed  # within 0.03 and 97 %, writers and scorers (D21)
        edges = (({"same_top": 0.969}, False), ({"mean_kl": 0.031}, False), ({"mean_kl": 0.029}, True))
        for change, passes in edges:  # each limit's edge
            assert judge_ub(saved.model_copy(update=change), control, UB_LIMITS).passed is passes
    longer = judge_ub(stats("hl2d_kld_ub2048.txt"), stats("hl2d_kld_ub512.txt"), UB_LIMITS)
    assert not longer.passed
    assert "0.0604" in longer.detail  # at 4,096-token chunks the mean KL divergence is three times as large
    worse = stats("hl2c_kld_ub2048.txt").model_copy(update={"ppl_ratio": control.ppl_ratio + 0.02})
    assert not judge_ub(worse, control, UB_LIMITS).passed  # a perplexity loss beyond the control's
    base_storage = control.model_copy(update={"ppl_ratio": 1.02})  # the base file's 16-bit storage offsets it
    near = stats("hl2c_kld_ub2048.txt").model_copy(update={"ppl_ratio": 1.025})
    assert judge_ub(near, base_storage, UB_LIMITS).passed  # judged against the control, not against 1.0
    assert not judge_ub(near, control, UB_LIMITS).passed


def test_speculation_tripwires_on_the_hl3_arms() -> None:
    acceptance = parse_acceptance((FIXTURES / "hl3_mtp1_acceptance.log").read_text())
    assert acceptance == pytest.approx(0.573, abs=0.005)  # accepted over generated, six requests
    ok = judge_speculation(PLAIN, MTP, acceptance=acceptance, reference_repeats=True, repeats=True)
    assert ok.passed
    assert ok.score == pytest.approx(0.785, abs=0.001)  # HL3's mean word similarity
    for broken in (
        {"acceptance": acceptance, "reference_repeats": True, "repeats": False},  # not exact across a restart
        {
            "acceptance": acceptance,
            "reference_repeats": None,
            "repeats": True,
        },  # the reference never restarted
        {"acceptance": 0.0, "reference_repeats": True, "repeats": True},  # the draft was never used
    ):
        assert not judge_speculation(PLAIN, MTP, **broken).passed
    unrelated = judge_speculation(PLAIN, TEXTS, acceptance=0.5, reference_repeats=True, repeats=True)
    assert not unrelated.passed  # words with nothing in common: a broken draft setup
    assert unrelated.score < 0.65


def test_the_task_check_is_read_strictly() -> None:
    assert parse_check('log line\n{"score": 0.9, "passed": true}\n', 0).passed
    assert not parse_check('{"score": 0.9, "passed": true}', 1).passed  # a failing exit code
    assert not parse_check('{"score": 0.4, "passed": "false"}', 0).passed  # a string is not a verdict
    assert not parse_check(
        '{"score": "NaN", "passed": true}', 0
    ).passed  # nor is a score that is not a number
    assert not parse_check("no verdict here", 0).passed


def test_bench_rows_are_one_per_ub_and_the_generation_row_is_left_out() -> None:
    rows = bench_rows(json.loads((FIXTURES / "llama_bench_ub_sweep.json").read_text()), 512)
    assert sorted(rows) == [512, 1024, 2048]
    assert rows[2048].tokens_per_second > rows[1024].tokens_per_second > rows[512].tokens_per_second
    assert 0.05 < rows[512].spread < 0.1  # the three repetitions' own spread: the margin's second term


def test_shrinking_rule_orders_by_value_and_skips_what_would_pass_the_limit() -> None:
    async def noop() -> None: ...

    steps = [
        Planned(name, priority, lambda: 1.0, noop)
        for name, priority in (
            ("ngram-mod", Priority.REST),
            ("-ub 2048", Priority.UBATCH_2048),
            ("-ub 1024", Priority.REST),
            ("draft-mtp", Priority.OWN_SPECULATION),
        )
    ]
    assert [s.name for s in in_priority_order(steps)] == ["draft-mtp", "-ub 2048", "ngram-mod", "-ub 1024"]
    assert fits_budget(500, 100, 600)
    assert not fits_budget(500, 100.1, 600)


def test_a_restart_is_estimated_from_its_own_start_run_and_stop(tmp_path: Path) -> None:
    tune = LlamaTune(make_spec(tmp_path))
    tune.start_s, tune.run_s, tune.stop_s = 10.0, 10.0, 5.0
    quick = ArmResult(name="draft-mtp", settings=Settings(), start_s=3.0, run_s=4.0, stop_s=1.0)
    assert tune._arm_cost() == 25.0
    assert tune._arm_cost(quick) == 8.0
    assert tune._arm_cost(checked=True) == 25.0 + tune.check_s


# -- the stages, with every program faked ------------------------------------------------------------------


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def arm_script() -> dict[str, dict[str, Any]]:
    """What the faked servers report, by arm name: the reference writes TEXTS in 10 s a request."""
    return {
        "reference": {"request_s": 10.0, "outputs": TEXTS, "peak_gib": 19.0, "prompt_per_second": 1000.0},
        "draft-mtp": {"request_s": 5.0, "outputs": TEXTS, "peak_gib": 20.0, "acceptance": 0.57},
        "ub2048": {"request_s": 8.0, "outputs": TEXTS, "peak_gib": 19.5},
        "file:model-iq4.gguf": {
            "request_s": 7.0,
            "outputs": ["x"] * 6,
            "peak_gib": 12.0,
            "check": Agreement(score=0.95, passed=True, detail="task check"),
        },
    }


class FakeTune(LlamaTune):
    """The tune with its programs and clock faked; the choosing, judging and recording are the real ones."""

    def __init__(
        self,
        spec: TuneSpec,
        script: dict[str, dict[str, Any]],
        *,
        flaky: str = "",
        cut: str = "",
        started: float | None = None,
    ) -> None:
        self.clock = Clock()
        super().__init__(spec, clock=self.clock, started=started)
        self.script, self.flaky, self.cut, self.calls, self.tools = script, flaky, cut, [], []

    def _find_text(self) -> Path | None:
        return self.spec.text

    def _free_gib(self) -> float | None:
        return 100.0

    async def _serve(self, settings: Settings, files: FileSet, *, checked: bool) -> ArmResult:
        name = llama_tune.arm_name(settings)
        self.calls.append(name)
        fields = dict(self.script[name])
        if name == self.flaky and self.calls.count(name) > 1:
            fields["outputs"] = [*TEXTS[:-1], "a restart gave different words"]
        if name == self.cut and self.calls.count(name) > 1:
            fields["error"] = llama_tune.OUT_OF_TIME  # the hard limit cut the restart short
        arm = ArmResult(name=name, settings=settings, start_s=10.0, run_s=10.0, free_gib=100.0, **fields)
        arm.fits = fit_verdict(arm.peak_gib, arm.free_gib, self.margin_gib)
        self.clock.now += 20.0
        return arm

    async def _run_bench(self) -> None:
        self.tools.append("llama-bench")
        self.bench = bench_rows(json.loads((FIXTURES / "llama_bench_ub_sweep.json").read_text()), 512)

    async def _screen(self, ubatch: int) -> float:
        self.tools.append("llama-fit-params")
        return 10.0

    async def _perplexity(self, ubatch: int, *, base: bool) -> KlResult:
        self.tools.append(f"llama-perplexity {ubatch}{' base' if base else ''}")
        result = KlResult(
            n_ubatch=ubatch, seconds=30.0, control=not base and ubatch == 512 and self.control is None
        )
        if base:
            (self.spec.run_dir / "base.kld").write_text("a base file")
        else:  # a case may script the figures of a -ub in "kl"; otherwise the saved HL2c outputs
            result.stats = self.script.get("kl", {}).get(ubatch) or stats(f"hl2c_kld_ub{ubatch}.txt")
        self.ppl_s = 30.0
        self.measurements.append(result)
        return result


def make_spec(tmp_path: Path, **changes: Any) -> TuneSpec:
    text = tmp_path / "text.txt"
    text.write_text("a text " * 100)
    values: dict[str, Any] = {
        "reference": FileSet(MODEL, tmp_path / "model.gguf"),
        "alternatives": (),
        "kind": K1,
        "params": derive_serving_params(DeviceType.CUDA, 122.0, role=ServingRole.INTERACTIVE),
        "hardware": "cuda:NVIDIA GB10:122GB:aarch64",
        "server_bin": tmp_path / "bin" / "llama-server",
        "prompts": TEXTS,
        "prompt_tokens": 512,
        "run_dir": tmp_path / "run",
        "text": text,
    }
    return TuneSpec(**{**values, **changes})


@pytest.fixture(autouse=True)
def _store_and_disk(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOCALMODELS_STATE_DIR", str(tmp_path / "state"))
    free = SimpleNamespace(free=10**12)
    monkeypatch.setattr(llama_tune.shutil, "disk_usage", lambda _path: free)


MTP_ONLY = (SpecCandidate(SpecType.DRAFT_MTP),)


async def measured(
    tmp_path: Path, script: dict[str, dict[str, Any]], **changes: Any
) -> tuple[FakeTune, SettingsRecord]:
    tune = FakeTune(make_spec(tmp_path, **changes), script)
    return tune, await tune.tune()


async def test_stage_a_is_not_measured_for_short_prompts(tmp_path: Path) -> None:
    tune, record = await measured(tmp_path, arm_script(), prompt_tokens=300)
    assert tune.tools == []  # no llama-bench, no llama-perplexity, no llama-fit-params
    reasons = [m.reason for m in tune.measurements if isinstance(m, NotMeasured)]
    assert reasons == ["`-ub` not measured: prompts shorter than one batch"]
    assert record.chosen.n_ubatch is None  # never measured, so the record has no opinion


async def test_stage_a_needs_the_disk_for_its_base_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    left_over = []

    def disk_usage(_path: Path) -> SimpleNamespace:
        left_over.append((tmp_path / "run" / "base.kld").exists())
        return SimpleNamespace(free=GIB // 4)

    monkeypatch.setattr(llama_tune.shutil, "disk_usage", disk_usage)
    (tmp_path / "run").mkdir()
    (tmp_path / "run" / "base.kld").write_text("what a killed run left")
    tune, _ = await measured(tmp_path, arm_script())
    assert left_over == [False]  # removed before the free space is counted
    assert llama_tune.base_file_gib(700) < 1  # a short text needs few chunks
    assert llama_tune.base_file_gib(10**9) == llama_tune.KL_BASE_GIB  # a long one is cut at 28
    assert tune.tools == []
    reasons = [m.reason for m in tune.measurements if isinstance(m, NotMeasured)]
    assert reasons == ["`-ub` not measured: the base file needs about 1 GiB, 0 free"]


def test_the_pinned_text_ships_with_the_package_and_matches_its_hash(tmp_path: Path) -> None:
    assert LlamaTune(make_spec(tmp_path, text=None))._find_text() == llama_tune.PINNED_TEXT
    assert llama_tune.PINNED_TEXT.stat().st_size == 199_173


def test_a_text_other_than_the_pinned_one_is_not_used(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    other = tmp_path / "other.txt"
    other.write_text("not the pinned text")
    monkeypatch.setattr(llama_tune, "PINNED_TEXT", other)  # its SHA-256 is not the recorded one
    assert LlamaTune(make_spec(tmp_path, text=None))._find_text() is None


@pytest.mark.parametrize(
    ("changes", "script", "candidate", "ubatch", "spec"),
    [
        # speculation twice as fast, outputs close, repeats, acceptance above zero: adopted
        ({"speculation": MTP_ONLY, "prompt_tokens": 300}, {}, "draft-mtp", None, SpecType.DRAFT_MTP),
        # 2 % faster is inside the margin of 5 %: the reference stays
        (
            {"speculation": MTP_ONLY, "prompt_tokens": 300},
            {"draft-mtp": {**arm_script()["draft-mtp"], "request_s": 9.8}},
            "reference",
            None,
            None,
        ),
        # the project's task check fails for the speculative arm: it is not adopted, however fast
        (
            {"speculation": MTP_ONLY, "prompt_tokens": 300, "quality_command": ("check",)},
            {
                "draft-mtp": {
                    **arm_script()["draft-mtp"],
                    "check": Agreement(score=0.2, passed=False, detail="the task check failed"),
                }
            },
            "reference",
            None,
            None,
        ),
        # a larger -ub that passes, from a server that ran with it
        ({}, {}, "ub2048", 2048, None),
        # a scorer is judged by the same limits (D21): its larger -ub passes and is adopted under its own kind
        ({"kind": K3}, {}, "ub2048", 2048, None),
        # ... and refused when its figures fall outside them: the reference's -ub stays
        ({"kind": K3}, {"kl": dict.fromkeys((1024, 2048), DRIFTED)}, "reference", 512, None),
        # no memory figure from nvidia-smi: a larger -ub is not adopted
        ({}, {"ub2048": {"request_s": 8.0, "outputs": TEXTS}}, "reference", 512, None),
    ],
)
async def test_chosen_by_the_limits_the_margin_and_a_measured_memory_figure(
    tmp_path: Path,
    changes: dict[str, Any],
    script: dict[str, Any],
    candidate: str,
    ubatch: int | None,
    spec: SpecType | None,
) -> None:
    _, record = await measured(tmp_path, {**arm_script(), **script}, **changes)
    assert not (tmp_path / "run" / "base.kld").exists()  # the 14 GB base file is deleted once used
    assert record.candidate == candidate
    assert record.chosen.n_ubatch == ubatch
    assert record.chosen.spec == spec or (spec is None and record.chosen.spec in (None, SpecType.NONE))
    assert record.agreement is not None
    assert record.agreement.passed
    stored = adopt(record)  # the record is accepted by the store and carries its measurements
    assert (
        read_store(record_path(MODEL, K1 if "kind" not in changes else K3)).records[0].chosen == stored.chosen
    )
    assert {m["step"] for m in stored.llama_measurements} >= {"server", "layout"}


async def test_a_speculative_arm_that_does_not_repeat_across_a_restart_is_not_adopted(tmp_path: Path) -> None:
    spec = make_spec(tmp_path, speculation=MTP_ONLY, prompt_tokens=300)
    tune = FakeTune(spec, arm_script(), flaky="draft-mtp")
    record = await tune.tune()
    assert record.candidate == "reference"
    restart = next(m for m in tune.measurements if isinstance(m, ArmResult) and m.name == "draft-mtp#restart")
    assert restart.outputs != tune.arms[1].outputs
    assert tune.calls == [
        "reference",
        "reference",
        "draft-mtp",
        "draft-mtp",
    ]  # restarted once each, nothing more
    # a restart the hard limit cut short leaves the arm unverified, not failed
    tune = FakeTune(spec, arm_script(), cut="draft-mtp")
    record = await tune.tune()
    assert (record.candidate, tune.unverified) == ("reference", {"draft-mtp"})


async def test_a_draft_that_cannot_fit_beside_the_reference_is_not_started(tmp_path: Path) -> None:
    draft = tmp_path / "mtp.gguf"
    with draft.open("wb") as fh:
        fh.truncate(3 * GIB)  # sparse
    candidate = SpecCandidate(SpecType.DRAFT_MTP, str(draft))
    spec = make_spec(tmp_path, speculation=(candidate,), prompt_tokens=300)
    script = arm_script()
    script["draft-mtp:mtp.gguf"] = script.pop("draft-mtp")
    # 88 GiB measured, 3 GiB of draft and 10 GiB kept free (the kill floor) pass the 100 GiB available
    script["reference"] = {**script["reference"], "peak_gib": 88.0}
    tune = FakeTune(spec, script)
    await tune.tune()
    assert tune.calls == ["reference", "reference"]
    skipped = {m.name: m.reason for m in tune.measurements if isinstance(m, NotMeasured)}
    assert skipped["draft-mtp:mtp.gguf"].startswith("not tried: about 91.0 GiB with its draft")

    # a larger -ub: the reference's peak plus what llama.cpp projects it to add over the default
    async def screen(ubatch: int) -> float:
        return {512: 10.0, 2048: 13.0}[ubatch]

    tune._screen = screen  # type: ignore[method-assign]
    assert not await tune._fits_screen(2048)
    assert tune.measurements[-1].reason.startswith("not tried: about 91.0 GiB at -ub 2048")
    # 86 GiB leave room: the draft is started
    script["reference"] = {**script["reference"], "peak_gib": 86.0}
    tune = FakeTune(spec, script)
    await tune.tune()
    assert "draft-mtp:mtp.gguf" in tune.calls


async def test_a_short_budget_starts_the_models_own_speculation_first_and_skips_the_rest(
    tmp_path: Path,
) -> None:
    spec = make_spec(tmp_path, speculation=(SpecCandidate(SpecType.NGRAM_MOD), *MTP_ONLY), soft_limit_s=70.0)
    script = {
        **arm_script(),
        "ngram-mod": {"request_s": 6.0, "outputs": TEXTS, "peak_gib": 19.0, "acceptance": 0.3},
    }
    tune = FakeTune(spec, script)
    record = await tune.tune()
    assert tune.calls == [
        "reference",
        "reference",
        "draft-mtp",
    ]  # 20 s each: the model's own head, then no more
    skipped = {m.name: m.reason for m in tune.measurements if isinstance(m, NotMeasured)}
    assert skipped["ngram-mod"] == skipped["-ub 2048"] == llama_tune.OUT_OF_BUDGET
    assert skipped["draft-mtp restart"] == llama_tune.OUT_OF_BUDGET
    assert record.candidate == "reference"  # its repeat could not be checked, so it is not adopted unchecked


async def test_a_quantised_file_needs_the_projects_check(tmp_path: Path) -> None:
    alternative = FileSet(
        ModelIdentity(repository=MODEL.repository, revision=MODEL.revision, files=(ALTERNATIVE_FILE,)),
        tmp_path / "a.gguf",
    )
    tune, record = await measured(tmp_path, arm_script(), alternatives=(alternative,), prompt_tokens=300)
    assert [m.reason for m in tune.measurements if isinstance(m, NotMeasured)][
        -1
    ] == "not measured: no task-quality check"
    assert record.candidate == "reference"
    tune, record = await measured(
        tmp_path, arm_script(), alternatives=(alternative,), prompt_tokens=300, quality_command=("check",)
    )
    assert record.candidate == "file:model-iq4.gguf"
    assert record.chosen.model_file == (ALTERNATIVE_FILE,)
    assert "task-quality check" in record.quality


async def test_servers_lease_on_a_private_directory_not_the_one_slurm_holds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shared = tmp_path / "gpu-lease"
    shared.mkdir()
    (shared / "slurm.json").write_text(json.dumps({"pid": os.getpid(), "project": "slurm"}))
    assert [o.project for o in GpuLease(shared, "tune").live_others()] == [
        "slurm"
    ]  # the control: this refuses
    monkeypatch.setattr(llama_tune, "DEFAULT_LEASE_DIR", shared)
    tune = LlamaTune(make_spec(tmp_path))
    async with httpx.AsyncClient() as http:
        manager = tune._manager(Settings(n_ubatch=2048), tune.spec.reference, http)
    assert manager.lease.directory == tmp_path / "run" / "lease"
    assert manager.lease.live_others() == []
    # the server must leave the kill floor free, not only the 4 GiB headroom: it is killed below the floor
    assert manager.guards.host_reserve_gib == manager.guards.kill_below_gib == llama_tune.KILL_BELOW_GIB


async def test_sigterm_cancels_once_through_the_loops_own_handler_so_the_servers_are_stopped() -> None:
    reached = []
    seen = []
    previous = signal.signal(signal.SIGTERM, lambda *_: seen.append("a plain python handler ran"))

    async def work() -> None:
        try:
            await asyncio.sleep(5)
        finally:
            await asyncio.sleep(0.3)  # stopping a server takes a moment, as manager.stop() polls
            reached.append("server stopped")

    loop = asyncio.get_running_loop()
    loop.call_later(0.05, os.kill, os.getpid(), signal.SIGTERM)
    loop.call_later(
        0.15, os.kill, os.getpid(), signal.SIGTERM
    )  # timeout signals its child and then its group
    try:
        with pytest.raises(asyncio.CancelledError):
            await asyncio.create_task(run_guarded(work()))  # its own task: the cancel is the tune's
    finally:
        signal.signal(signal.SIGTERM, previous)
    assert reached == ["server stopped"]
    assert seen == []


FAKE_SERVER = """#!{python}
import json, sys, os, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

args = sys.argv[1:]
port = int(args[args.index("--port") + 1])
open(os.environ["FAKE_SERVER_NOTES"], "w").write(json.dumps({{"argv": args, "pid": os.getpid()}}))
drafting = "--spec-type" in args


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        pass

    def reply(self, body):
        data = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self.reply({{"status": "ok"}})

    def do_POST(self):
        asked = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        time.sleep(float(os.environ.get("FAKE_SERVER_DELAY", "0")))
        if drafting:  # the warm-up (8 tokens) is accepted once in a hundred, the real requests six in ten
            kept = 1 if asked["n_predict"] == 8 else 60
            line = "slot print_timing: id  0 | draft acceptance = 0.5 ({{:d}} accepted / 100 generated)"
            print(line.format(kept), flush=True)
        timings = {{"prompt_per_second": 1000.0, "predicted_per_second": 50.0,
                    "prompt_ms": 500.0, "predicted_ms": 2000.0}}
        self.reply({{"content": "words " + asked["prompt"][:6], "timings": timings}})


ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
"""
FAKE_FIT_PARAMS = """#!{python}
print("llama_params_fit: projected to use 4341 MiB of device memory")
"""


FAKE_PERPLEXITY = """#!{python}
import json, os, sys, time
open(os.environ["FAKE_PERPLEXITY_NOTES"], "w").write(json.dumps(sys.argv[1:]))
time.sleep(5)
"""
TASK_CHECK = (
    "import json, os; open(os.environ['FAKE_CHECK_NOTES'], 'w').write("
    "os.environ['LLAMA_TUNE_SERVER_URL'] + ' ' + os.environ['LLAMA_TUNE_MODEL_FILE']); "
    "print(json.dumps({'score': 0.9, 'passed': True}))"
)


@pytest.fixture
def fake_bin(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A folder of fake llama.cpp programs; the memory readings are plentiful. Returns the server's path."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name, code in (
        ("llama-server", FAKE_SERVER),
        ("llama-fit-params", FAKE_FIT_PARAMS),
        ("llama-perplexity", FAKE_PERPLEXITY),
    ):
        (bin_dir / name).write_text(code.format(python=sys.executable))
        (bin_dir / name).chmod(0o755)
    monkeypatch.setenv("FAKE_SERVER_NOTES", str(tmp_path / "notes.json"))
    monkeypatch.setenv("FAKE_PERPLEXITY_NOTES", str(tmp_path / "perplexity.json"))
    plenty = AvailableMemory(bytes=100 * GIB, device=DeviceType.CUDA, unified=False, source="test")
    monkeypatch.setattr(llama_tune, "available_memory", lambda _device: plenty)
    monkeypatch.setattr(launcher, "available_memory", lambda _device: plenty)
    return bin_dir / "llama-server"


async def test_a_real_server_start_carries_the_flags_and_gives_speeds_acceptance_and_peak_memory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_bin: Path
) -> None:
    samples = iter([4000, 4468])  # the server's memory rises to 4,468 MiB and falls to 3,000 MiB

    class AnyPid(dict[int, int]):  # the server's pid is only known once it runs
        def __contains__(self, _key: object) -> bool:
            return True

        def __getitem__(self, _key: int) -> int:
            return next(samples, 3000)

        def __len__(self) -> int:
            return 1

    monkeypatch.setattr(llama_tune, "gpu_process_memory_mib", AnyPid)
    # a GPU that shares the host's RAM: the server's host memory counts in its peak (2 GiB here)
    # 14 GiB available: the 6.4 GiB peak leaves the 4 GiB headroom but not the 10 GiB kill floor
    unified = AvailableMemory(bytes=14 * GIB, device=DeviceType.CUDA, unified=True, source="test")
    monkeypatch.setattr(llama_tune, "available_memory", lambda _device: unified)
    monkeypatch.setattr(llama_tune, "process_anon_mib", lambda _pid: 2048)
    monkeypatch.setattr(llama_tune, "SAMPLE_INTERVAL_S", 0.05)
    monkeypatch.setenv("FAKE_SERVER_DELAY", "0.1")  # three waves of requests: the samples see the peak
    monkeypatch.setenv("FAKE_CHECK_NOTES", str(tmp_path / "check.txt"))
    command = (sys.executable, "-c", TASK_CHECK)
    tune = LlamaTune(make_spec(tmp_path, server_bin=fake_bin, quality_command=command))
    head = tmp_path / "mtp.gguf"
    head.touch()  # the launcher counts an MTP head as its size
    settings = Settings(n_ubatch=2048, spec=SpecType.DRAFT_MTP, spec_draft_file=str(head))
    arm = await tune._serve(settings, tune.spec.reference, checked=True)
    assert arm.error is None
    assert arm.request_s == 2.5  # the server's own prompt_ms plus predicted_ms
    assert arm.prompt_per_second == 1000.0
    assert arm.predicted_per_second == 50.0
    assert arm.acceptance == pytest.approx(0.6)  # the warm-up's line is left out
    assert arm.peak_gib == pytest.approx((4468 + 2048) / 1024)  # the largest sample, not the last
    # a GPU with its own memory: the host side is not added
    tune._unified, peaks = False, []
    sampler = asyncio.create_task(tune._sample(4242, peaks))
    await asyncio.sleep(0.12)
    sampler.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await sampler
    assert peaks
    assert max(peaks) <= 4468
    assert arm.fits is False
    assert arm.outputs == [f"words {text[:6]}" for text in TEXTS]
    assert arm.check is not None
    assert arm.check.passed  # the project's check ran against the live server
    assert (tmp_path / "check.txt").read_text().startswith("http://127.0.0.1:")
    started = json.loads((tmp_path / "notes.json").read_text())
    assert started["argv"][started["argv"].index("-ub") + 1] == "2048"
    assert started["argv"][started["argv"].index("--spec-type") + 1] == "draft-mtp"
    assert started["argv"][started["argv"].index("--spec-draft-model") + 1] == str(head)  # the head file
    assert not _is_pid_alive(started["pid"])  # stopped when its arm ended
    assert arm.stop_s is not None


async def test_no_step_runs_past_the_hard_limit(tmp_path: Path, fake_bin: Path) -> None:
    clock = Clock()
    assert (
        LlamaTune(make_spec(tmp_path), clock=clock, started=-100.0).elapsed() == 100.0
    )  # the command's own start
    tune = LlamaTune(make_spec(tmp_path, server_bin=fake_bin, hard_limit_s=0.4))
    arm = await tune._serve(Settings(), tune.spec.reference, checked=False)
    assert arm.error == llama_tune.OUT_OF_TIME  # cut short while loading, and the server is stopped
    notes = tmp_path / "notes.json"
    assert not notes.exists() or not _is_pid_alive(json.loads(notes.read_text())["pid"])
    tune = LlamaTune(make_spec(tmp_path, server_bin=fake_bin, hard_limit_s=0.3))
    tune.text = tmp_path / "text.txt"
    result = await tune._perplexity(512, base=True)  # a program that would run for 5 s
    assert result.error == llama_tune.OUT_OF_TIME
    argv = json.loads((tmp_path / "perplexity.json").read_text())
    assert argv[argv.index("--chunks") + 1] == "28"  # the calibrated text and no more: bounded disk and time
    assert argv[argv.index("-c") + 1] == argv[argv.index("-b") + 1] == "2048"


async def test_a_task_check_that_does_not_finish_is_not_measured_and_sets_its_own_estimate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(llama_tune, "CHECK_TIMEOUT_MIN_S", 0.1)
    command = (sys.executable, "-c", "import time; time.sleep(5)")
    tune = LlamaTune(make_spec(tmp_path, quality_command=command, check_s=0.01))
    assert await tune._check(1234, tune.spec.reference) is None
    assert 0.1 <= tune.check_s < 2  # the time waited: the shrinking rule skips the checks still to come
    late = LlamaTune(make_spec(tmp_path, quality_command=command, hard_limit_s=0.3))  # 60 s would be allowed
    assert await asyncio.wait_for(late._check(1234, late.spec.reference), 3) is None  # but 0.3 s are left


# -- the command ---------------------------------------------------------------------------------------------


@pytest.fixture
def commanded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """The command with its tune and quiet card faked; the events (card, tune) come back in order."""
    events: list[str] = []

    class Card:
        def __init__(self, _lease: object, *, reason: str) -> None:
            self.reason = reason

        def __enter__(self) -> Self:
            events.append("enter")
            return self

        def __exit__(self, *_exc: object) -> None:
            events.append("exit")

    monkeypatch.setattr(llama_tune, "QuietCard", Card)
    return events


def command_with(
    monkeypatch: pytest.MonkeyPatch, events: list[str], script: dict[str, dict[str, Any]]
) -> None:
    class Recording(FakeTune):
        async def tune(self) -> SettingsRecord:
            events.append("tune")
            return await super().tune()

    monkeypatch.setattr(
        llama_tune, "LlamaTune", lambda spec, started=None: Recording(spec, script, started=started)
    )


@pytest.mark.parametrize(
    ("script", "changes", "exit_code", "adopted"),
    [
        ({}, {"speculation": MTP_ONLY, "prompt_tokens": 300}, 0, True),
        # the reference cannot be measured: nothing is adopted, and the measurements so far are kept
        ({"reference": {"error": "the server would not start"}}, {}, 1, False),
        # no time for anything but the reference: not adopted, so the lookup does not read it as tuned
        ({}, {"speculation": MTP_ONLY, "soft_limit_s": 25.0}, 2, False),
        # the speculative arm led but its restart did not fit: the reference is not adopted as tuned
        ({}, {"speculation": MTP_ONLY, "soft_limit_s": 70.0, "prompt_tokens": 300}, 2, False),
        # the arm ran but the hard limit cut its check short: neither passed nor failed, so the same
        (
            {"draft-mtp": {**arm_script()["draft-mtp"], "error": llama_tune.OUT_OF_TIME}},
            {"speculation": MTP_ONLY, "prompt_tokens": 300},
            2,
            False,
        ),
    ],
)
async def test_the_command_holds_the_card_writes_the_record_and_adopts_only_what_was_measured(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    commanded: list[str],
    script: dict[str, Any],
    changes: dict[str, Any],
    exit_code: int,
    adopted: bool,
) -> None:
    command_with(monkeypatch, commanded, {**arm_script(), **script})
    code = await tune_and_adopt(make_spec(tmp_path, **changes))
    assert code == exit_code
    assert commanded == ["enter", "tune", "exit"]  # the card is held for the whole tune and let go after it
    written = json.loads((tmp_path / "run" / "record.json").read_text())
    assert bool(written.get("partial")) is (exit_code == 1)
    assert bool(read_store(record_path(MODEL, K1)).records) is adopted


async def test_a_cancelled_tune_keeps_what_it_measured_and_deletes_its_base_file(tmp_path: Path) -> None:
    class Cancelled(FakeTune):
        async def _serve(self, settings: Settings, files: FileSet, *, checked: bool) -> ArmResult:
            if llama_tune.arm_name(settings) == "ub2048":
                raise asyncio.CancelledError  # SIGTERM arrives while the -ub arm is loading
            return await super()._serve(settings, files, checked=checked)

    tune = Cancelled(make_spec(tmp_path), arm_script())
    with pytest.raises(asyncio.CancelledError):
        await tune.tune()
    assert not (tmp_path / "run" / "base.kld").exists()
    kept = json.loads((tmp_path / "run" / "record.json").read_text())
    assert kept["partial"] is True
    assert {m["step"] for m in kept["llama_measurements"]} >= {"layout", "teacher-forced", "server"}


async def test_the_reference_is_timed_twice_whenever_a_candidate_is_planned(tmp_path: Path) -> None:
    alone, _ = await measured(tmp_path, arm_script(), prompt_tokens=300)
    assert alone.calls == ["reference"]  # nothing to compare it with
    with_ub, _ = await measured(tmp_path, arm_script())
    assert with_ub.calls[:2] == ["reference", "reference"]  # the margin's spread is measured
    short, _ = await measured(tmp_path, arm_script(), soft_limit_s=25.0)  # 20 s a start: no time for a second
    assert short.calls == ["reference"]
    assert next(m.name for m in short.measurements if isinstance(m, NotMeasured)) == "reference restart"


def hub_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, names: list[str]) -> Path:
    """A Hub cache with a snapshot of MODEL holding ``names`` (each a link to a blob named by its SHA-256)."""
    hub = tmp_path / "hub"
    monkeypatch.setenv("HF_HUB_CACHE", str(hub))
    root = snapshot_dir(MODEL.repository, MODEL.revision, hub)
    for number, name in enumerate(names):
        blob = hub / "blobs" / f"{number:064x}"
        blob.parent.mkdir(parents=True, exist_ok=True)
        blob.write_bytes(b"gguf")
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).symlink_to(blob)
    return root


def test_the_arguments_resolve_the_models_head_file_and_alternatives_from_the_hub_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = hub_cache(tmp_path, monkeypatch, ["UD-IQ4_XS/m-00001.gguf", "Q4/m.gguf", "MTP/mtp.gguf"])
    server = tmp_path / "llama-server"
    server.write_text("#!/bin/sh\n")
    server.chmod(0o755)
    monkeypatch.setenv("LLAMA_SERVER_BIN", str(server))
    prompts = tmp_path / "prompts.json"
    prompts.write_text(json.dumps(TEXTS))
    base = [
        *("--repository", MODEL.repository, "--revision", MODEL.revision, "--file", "UD-IQ4_XS/m-00001.gguf"),
        *("--kind", "K1", "--role", "interactive", "--prompts", str(prompts), "--prompt-tokens", "600"),
        *("--run-dir", str(tmp_path / "run")),
    ]
    parse = llama_tune.build_parser().parse_args
    candidates = ["--speculation", "draft-mtp", "--speculation", "ngram-mod", "--draft-file", "MTP/mtp.gguf"]
    spec = llama_tune.spec_from_args(parse([*base, *candidates, "--alternative", "Q4/m.gguf"]))
    head = str(root / "MTP" / "mtp.gguf")
    assert spec.speculation == (
        SpecCandidate(SpecType.DRAFT_MTP, head),
        SpecCandidate(SpecType.NGRAM_MOD, None),
    )
    assert spec.alternatives[0].path == root / "Q4" / "m.gguf"  # the reference's own repository and revision
    assert spec.alternatives[0].identity.revision == MODEL.revision
    assert spec.params.n_parallel == 4  # the interactive role's slots
    one_at_a_time = llama_tune.spec_from_args(parse([*base, "--parallel", "1"]))
    assert one_at_a_time.params.n_parallel == 1
    for bad in (
        ["--parallel", "0"],
        ["--parallel", "5"],  # more slots than the role's
        ["--speculation", "draft-simple"],  # a draft model needs its file
        ["--speculation", "ngram-mod", "--draft-file", "MTP/mtp.gguf"],  # nothing takes the file
        ["--speculation", "draft-mtp", "--draft-file", "MTP/missing.gguf"],  # a tune never downloads
        ["--alternative", "Q4/missing.gguf"],
    ):
        with pytest.raises(SystemExit):
            llama_tune.spec_from_args(parse([*base, *bad]))
    in_dev = [*base[:-1], str(Path.home() / "dev" / "x")]
    with pytest.raises(SystemExit, match="Syncthing"):
        llama_tune.spec_from_args(parse(in_dev))
    monkeypatch.setattr(llama_tune, "network_filesystem", lambda _path: "nfs4")
    with pytest.raises(SystemExit, match="network filesystem"):
        llama_tune.spec_from_args(parse(base))
