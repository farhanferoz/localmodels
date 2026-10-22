"""The llama.cpp tune: its limits on saved measurements, its stages with every program faked, its guards."""

from __future__ import annotations

import asyncio
import json
import os
import signal
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from localmodels import launcher, llama_tune
from localmodels.enums import DeviceType, ServingRole, WorkloadKind
from localmodels.lease import GpuLease
from localmodels.llama_tune import (
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
    parse_kl,
    run_guarded,
    ub_limits,
)
from localmodels.params import derive_serving_params
from localmodels.pinned import PinnedFile
from localmodels.process import _is_pid_alive
from localmodels.serving.tune import Agreement
from localmodels.settings import ModelIdentity, SettingsRecord, adopt, read_store, record_path
from localmodels.stack import Settings, SpecType

FIXTURES = Path(__file__).parent / "fixtures" / "llama_tune"
K1, K3 = WorkloadKind.K1, WorkloadKind.K3
MODEL = ModelIdentity(repository="Qwen/Qwen3-30B-A3B-GGUF", revision="c" * 40)
ALTERNATIVE_FILE = PinnedFile(path="model-iq4.gguf", sha256="b" * 64)
TEXTS = [f"the answer to question {i} is forty two and then some more words follow here" for i in range(6)]
PLAIN = json.loads((FIXTURES / "hl3_plain1_texts.json").read_text())
MTP = json.loads((FIXTURES / "hl3_mtp1_texts.json").read_text())


def stats(name: str) -> Any:
    return parse_kl((FIXTURES / name).read_text())


def test_ub_limits_applied_to_the_saved_teacher_forced_outputs() -> None:
    control = stats("hl2c_kld_ub512.txt")  # the restart: the base file's 16-bit storage gives 1.0013
    assert control.ppl_ratio == pytest.approx(1.001281)
    for ub in (1024, 2048):
        saved = stats(f"hl2c_kld_ub{ub}.txt")
        assert 0.016 < saved.mean_kl < 0.019
        assert 0.976 < saved.same_top < 0.978
        assert judge_ub(saved, control, ub_limits(K1)).passed  # a generator: 0.03 and 97 %
        assert not judge_ub(saved, control, ub_limits(K3)).passed  # a scorer: 98 % is missed by 0.3 points
    longer = judge_ub(stats("hl2d_kld_ub2048.txt"), stats("hl2d_kld_ub512.txt"), ub_limits(K1))
    assert not longer.passed
    assert "0.0604" in longer.detail  # at 4,096-token chunks the mean KL divergence is three times as large
    worse = stats("hl2c_kld_ub2048.txt").model_copy(update={"ppl_ratio": control.ppl_ratio + 0.02})
    assert not judge_ub(worse, control, ub_limits(K1)).passed  # a perplexity loss beyond the control's


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

    def __init__(self, spec: TuneSpec, script: dict[str, dict[str, Any]], *, flaky: str = "") -> None:
        self.clock = Clock()
        super().__init__(spec, clock=self.clock)
        self.script, self.flaky, self.calls, self.tools = script, flaky, [], []

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
        arm = ArmResult(name=name, settings=settings, start_s=10.0, run_s=10.0, free_gib=100.0, **fields)
        arm.fits = fit_verdict(arm.peak_gib, arm.free_gib, self.spec.headroom_gib)
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
        if not base:
            result.stats = stats(f"hl2c_kld_ub{ubatch}.txt")
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
        # a larger -ub that passes, from a server that ran with it
        ({}, {}, "ub2048", 2048, None),
        # a scorer keeps K3's own limits: 97.7 % is under 98 %, so -ub stays
        ({"kind": K3}, {}, "reference", 512, None),
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
        ModelIdentity(repository=MODEL.repository, revision="d" * 40, files=(ALTERNATIVE_FILE,)),
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


async def test_sigterm_cancels_through_the_loops_own_handler_so_the_servers_are_stopped() -> None:
    reached = []
    seen = []
    previous = signal.signal(signal.SIGTERM, lambda *_: seen.append("a plain python handler ran"))

    async def work() -> None:
        try:
            await asyncio.sleep(5)
        finally:
            reached.append("server stopped")

    asyncio.get_running_loop().call_later(0.05, os.kill, os.getpid(), signal.SIGTERM)
    try:
        with pytest.raises(asyncio.CancelledError):
            await asyncio.create_task(
                run_guarded(work())
            )  # its own task: the cancel is the tune's, not the test's
    finally:
        signal.signal(signal.SIGTERM, previous)
    assert reached == ["server stopped"]
    assert seen == []


FAKE_SERVER = """#!{python}
import json, sys, os
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


async def test_a_real_server_start_carries_the_flags_and_gives_speeds_acceptance_and_peak_memory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name, code in (("llama-server", FAKE_SERVER), ("llama-fit-params", FAKE_FIT_PARAMS)):
        (bin_dir / name).write_text(code.format(python=sys.executable))
        (bin_dir / name).chmod(0o755)
    notes = tmp_path / "notes.json"
    monkeypatch.setenv("FAKE_SERVER_NOTES", str(notes))
    plenty = SimpleNamespace(bytes=100 * 1024**3)
    monkeypatch.setattr(llama_tune, "available_memory", lambda _device: plenty)
    monkeypatch.setattr(launcher, "available_memory", lambda _device: plenty)

    class AnyPid(dict[int, int]):  # the server's pid is only known once it runs: it holds 4,468 MiB
        def __contains__(self, _key: object) -> bool:
            return True

        def __getitem__(self, _key: int) -> int:
            return 4468

        def __len__(self) -> int:
            return 1

    monkeypatch.setattr(llama_tune, "gpu_process_memory_mib", AnyPid)
    monkeypatch.setattr(llama_tune, "SAMPLE_INTERVAL_S", 0.05)
    tune = LlamaTune(make_spec(tmp_path, server_bin=bin_dir / "llama-server"))
    settings = Settings(n_ubatch=2048, spec=SpecType.DRAFT_MTP)
    arm = await tune._serve(settings, tune.spec.reference, checked=False)
    assert arm.error is None
    assert arm.request_s == 2.5  # the server's own prompt_ms plus predicted_ms
    assert arm.prompt_per_second == 1000.0
    assert arm.predicted_per_second == 50.0
    assert arm.acceptance == pytest.approx(0.6)  # the warm-up's line is left out
    assert arm.peak_gib == pytest.approx(4468 / 1024)
    assert arm.fits is True
    assert arm.outputs == [f"words {text[:6]}" for text in TEXTS]
    started = json.loads(notes.read_text())
    assert started["argv"][started["argv"].index("-ub") + 1] == "2048"
    assert started["argv"][started["argv"].index("--spec-type") + 1] == "draft-mtp"
    assert not _is_pid_alive(started["pid"])  # stopped when its arm ended
