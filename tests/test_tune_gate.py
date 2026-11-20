"""`localmodels settings require`: a long GPU job is refused until its model has a measured record here."""

from __future__ import annotations

import json
import sys
import types
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import pytest

from localmodels import cli, settings, tune_gate
from localmodels.enums import DeviceTier, DeviceType, Precision, WorkloadKind
from localmodels.hardware import HardwareReport
from localmodels.pinned import PinnedFile
from localmodels.serving import LoaderDeclaration
from localmodels.serving.server import library_versions
from localmodels.serving.tune import Agreement, hardware_key
from localmodels.settings import ModelIdentity, SettingsRecord, adopt
from localmodels.stack import Engine, Settings, SettingsStatus, TuneState, fingerprint
from tests.test_retune import GPU, Slurm, slurm  # noqa: F401 - the fake slurm fixture, re-used

if TYPE_CHECKING:
    from pathlib import Path

MESMER = ModelIdentity(repository="example/mesmer-4b", revision="c" * 40)  # tracecraft's case, renamed
CPU = HardwareReport(
    platform="linux", arch="aarch64", device=DeviceType.CPU, device_name="CPU", total_memory_gb=122.0,
    tier=DeviceTier.SLOW_CPU,
)  # fmt: skip
DECLARATION = LoaderDeclaration(
    dtypes=(Precision.BFLOAT16,),
    options={"int8_weights": (False, True)},
    reference=Settings(dtype=Precision.BFLOAT16, options={"int8_weights": False}),
)


@pytest.fixture(autouse=True)
def host(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A GB10 that sees its GPU, an empty store, no driver file to find, and a project's loader module."""
    monkeypatch.setenv(settings.STATE_DIR_ENV, str(tmp_path / "state"))
    monkeypatch.setattr(settings, "inspect_hardware", lambda: GPU)
    monkeypatch.setattr(tune_gate, "inspect_hardware", lambda: GPU)
    monkeypatch.setattr(tune_gate, "DRIVER_FILES", (tmp_path / "no-driver",))
    monkeypatch.setattr(settings, "_announced", set())
    module = types.ModuleType("fake_gate_project")
    module.model = lambda _entry: types.SimpleNamespace(declare=lambda: DECLARATION)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "fake_gate_project", module)


def _require(*extra: str, model: ModelIdentity = MESMER, hours: str = "24") -> int:
    flags = ["--model", model.repository, "--revision", model.revision, "--kind", "K6", "--engine", "pytorch"]
    return cli.main(["settings", "require", *flags, "--hours", hours, *extra])


def _log() -> list[dict[str, Any]]:
    path = settings.settings_dir() / tune_gate.GATE_LOG
    return [json.loads(line) for line in path.read_text().splitlines()]


def _store(model: ModelIdentity, **change: Any) -> None:
    record = SettingsRecord(
        model=model, kind=WorkloadKind.K6, hardware=hardware_key(GPU), measured_at=datetime.now(UTC),
        chosen=Settings(dtype=Precision.BFLOAT16, options={"int8_weights": False}), candidate="bfloat16",
        quality="score_agreement(scores)", agreement=Agreement(score=1.0, passed=True, detail="reference"),
        stack=fingerprint(), engine=Engine.PYTORCH, libraries=library_versions(),
    )  # fmt: skip
    adopt(record.model_copy(update=change))


def test_a_long_job_on_an_unmeasured_model_is_refused_with_the_tune_to_run_and_logged(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert _require() == 1  # tracecraft's 24 GPU-hours on a setting never measured here
    err = capsys.readouterr().err
    assert "REFUSED example/mesmer-4b (K6, pytorch, 24 h): no measured record" in err
    assert "localmodels settings tune --project <project> --loader <module:function> --entry <entry> " in err
    assert f"--model example/mesmer-4b --revision {'c' * 40} --kind K6 --texts <texts.jsonl>" in err
    assert "(automatic tune: no-recipe)" in err
    (line,) = _log()
    assert (line["allowed"], line["hours"], line["max_untuned_hours"], line["status"]) == (
        False,
        24,
        1.0,
        "rule-other-model",  # K6's rule, measured on other models: not a record of this one
    )
    assert line["stamp"]["status"] == "rule-other-model"
    assert _require(hours="0.5") == 0  # a short job needs no record
    assert "ALLOWED" in capsys.readouterr().err
    assert _log()[-1]["allowed"] is True


def test_a_measured_record_allows_the_job_and_a_stale_one_still_does_with_its_staleness_logged() -> None:
    _store(MESMER)
    assert _require() == 0
    allowed = _log()[-1]
    assert (allowed["allowed"], allowed["status"], allowed["stale"]) == (True, "tuned", [])
    assert allowed["record_id"] is not None
    assert allowed["stamp"]["record_id"] == allowed["record_id"]
    host_stack = fingerprint()
    old = {**host_stack, "driver": "1.0.0"}  # a stamped part of the stack changed since the tune
    _store(MESMER, stack=old)
    assert _require() == 0  # D2: a stale record stays in use until its confirm tune lands
    stale = _log()[-1]
    assert stale["status"] == "stale"
    assert stale["stale"]


def test_a_rule_or_another_identity_is_not_a_measured_record(capsys: pytest.CaptureFixture[str]) -> None:
    ruled = ModelIdentity(
        repository="mlinslab/neurovfm-dx-ct", revision="d" * 40
    )  # K6's rule names this model
    assert _require(model=ruled) == 1
    assert "only rule K6.precision applies, measured on this model" in capsys.readouterr().err
    assert _log()[-1]["status"] == SettingsStatus.RULE.value
    _store(MESMER)
    other = MESMER.model_copy(update={"revision": "e" * 40})
    assert _require(model=other) == 1
    assert f"records exist under another identity ({'c' * 40})" in capsys.readouterr().err


def test_a_call_that_can_never_find_a_record_is_refused_naming_what_to_fix(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with_file = MESMER.model_copy(update={"files": (PinnedFile(path="m.gguf", sha256="0" * 64),)})
    refused = tune_gate.require(with_file, WorkloadKind.K6, engine=Engine.PYTORCH, hours=24)
    assert "filed by its snapshot" in refused.reason
    llama = tune_gate.require(MESMER, WorkloadKind.K1, engine=Engine.LLAMA_CPP, hours=0.1)  # even a short job
    assert (llama.allowed, "give --file for each part" in llama.reason) == (False, True)
    vllm = tune_gate.require(MESMER, WorkloadKind.K2, engine=Engine.VLLM, hours=24)
    assert "vllm has no tune here" in vllm.reason
    with pytest.raises(SystemExit):
        cli.main(["settings", "require", "--model", "m", "--revision", "c" * 40, "--kind", "K2",
                  "--engine", "vllm", "--hours", "24"])  # fmt: skip
    assert len(_log()) == 3  # every decision is logged, the parser's refusal is not a decision
    capsys.readouterr()


def test_a_hidden_gpu_is_named_instead_of_looking_up_a_cpu_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    driver = tmp_path / "version"
    driver.write_text("NVRM version")
    monkeypatch.setattr(tune_gate, "inspect_hardware", lambda: CPU)  # what a CPU-only slurm job sees
    monkeypatch.setattr(tune_gate, "DRIVER_FILES", (driver,))
    assert _require() == 1
    err = capsys.readouterr().err
    assert f"this process sees no GPU though the host has one ({driver})" in err
    assert "the submitting shell, not a CPU-only job" in err
    monkeypatch.setattr(tune_gate, "DRIVER_FILES", (tmp_path / "absent",))  # a host without a GPU
    assert _require() == 1
    assert "sees no GPU" not in capsys.readouterr().err


def test_with_the_loader_given_the_check_queues_the_tune_and_names_its_job(
    slurm: Slurm,  # noqa: F811 - the fixture imported above
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(tune_gate, "inspect_hardware", lambda: GPU)
    texts = tmp_path / "texts.jsonl"
    loader = [
        "--project",
        "proj",
        "--loader",
        "fake_gate_project:model",
        "--entry",
        "mesmer-4b",
        "--texts",
        str(texts),
    ]
    assert _require(*loader) == 1
    assert "wait for the tune, slurm job 101, to end; then submit again" in capsys.readouterr().err
    (submitted,) = slurm.submissions()
    command = submitted[submitted.index("--") + 1 :]
    assert (
        command[command.index("settings") + 1] == "tune"
    )  # the project's in-process tune, as the recipe has it
    line = _log()[-1]
    assert (line["tune"], line["tune_detail"]) == (TuneState.QUEUED.value, "101")
    assert _require(*loader[:4]) == 1  # some of the loader's flags but not all
    assert "give --project, --loader, --entry and --texts together" in capsys.readouterr().err
