"""Unit tests for `localmodels setup`'s CLI wiring (argument plumbing, catalogue loading).

`run_setup`'s own behaviour (gating, model choice, backend refusal, bench-under-lease) is
covered by `tests/test_setup.py`; these tests only check that `main(["setup", ...])` parses
its flags and calls `run_setup`/`record_setup` correctly -- `run_setup` itself is faked here.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

import localmodels.cli as cli_mod
from localmodels.catalogue import ModelCategory, ModelSpec
from localmodels.cli import EXIT_ERROR, EXIT_OK, _load_catalogue, main
from localmodels.enums import ModelSizeClass
from localmodels.setup import Role, SetupReport, SetupStepName, StepResult, StepStatus


def _fake_report(*, ok: bool = True) -> SetupReport:
    status = StepStatus.DONE if ok else StepStatus.FAILED
    return SetupReport(
        dry_run=False,
        chosen_model="qwen3-30b-a3b",
        results=[StepResult(step=SetupStepName.HARDWARE, name="preflight", status=status, detail="x")],
    )


def test_load_catalogue_parses_a_json_file(tmp_path: Path) -> None:
    catalogue_path = tmp_path / "catalogue.json"
    catalogue_path.write_text(
        json.dumps(
            [
                {
                    "alias": "m1",
                    "category": "generator",
                    "repo_id": "org/m1",
                    "filename": "m1.gguf",
                    "revision": "abc123",
                    "description": "d",
                    "licence": "mit",
                    "osi_approved": True,
                    "size_class": "small",
                }
            ]
        )
    )
    specs = _load_catalogue(str(catalogue_path))
    assert specs == [
        ModelSpec(
            alias="m1",
            category=ModelCategory.GENERATOR,
            repo_id="org/m1",
            filename="m1.gguf",
            revision="abc123",
            description="d",
            licence="mit",
            osi_approved=True,
            size_class=ModelSizeClass.SMALL,
        )
    ]


def test_cmd_setup_forwards_args_to_run_setup(monkeypatch, tmp_path: Path) -> None:
    captured = {}

    def fake_run_setup(**kwargs):
        captured.update(kwargs)
        return _fake_report()

    monkeypatch.setattr(cli_mod, "run_setup", fake_run_setup)
    monkeypatch.setattr(cli_mod, "record_setup", lambda *_a, **_k: None)

    code = main(
        [
            "setup",
            "--role",
            "chat",
            "--models-dir",
            str(tmp_path / "models"),
            "--project",
            "test-project",
            "--lease-dir",
            str(tmp_path / "lease"),
            "--state-path",
            str(tmp_path / "state.json"),
            "--llama-server-bin",
            "/opt/llama-server",
            "--yes",
        ]
    )
    assert code == EXIT_OK
    assert captured["role"] is Role.CHAT
    assert captured["models_dir"] == tmp_path / "models"
    assert captured["lease"].project == "test-project"
    assert captured["lease"].directory == tmp_path / "lease"
    assert captured["binary"] == Path("/opt/llama-server")
    assert captured["yes"] is True
    assert captured["dry_run"] is False


def test_cmd_setup_dry_run_never_calls_record_setup(monkeypatch) -> None:
    record_calls = []
    monkeypatch.setattr(cli_mod, "run_setup", lambda **_k: _fake_report())
    monkeypatch.setattr(cli_mod, "record_setup", lambda *a, **k: record_calls.append((a, k)))

    code = main(["setup", "--dry-run"])
    assert code == EXIT_OK
    assert record_calls == []


def test_cmd_setup_returns_error_when_report_not_ok(monkeypatch) -> None:
    monkeypatch.setattr(cli_mod, "run_setup", lambda **_k: _fake_report(ok=False))
    monkeypatch.setattr(cli_mod, "record_setup", lambda *_a, **_k: None)

    code = main(["setup", "--yes"])
    assert code == EXIT_ERROR


def test_cmd_setup_loads_a_custom_catalogue(monkeypatch, tmp_path: Path) -> None:
    catalogue_path = tmp_path / "catalogue.json"
    catalogue_path.write_text(
        json.dumps(
            [
                {
                    "alias": "custom",
                    "category": "generator",
                    "repo_id": "org/custom",
                    "filename": "c.gguf",
                    "description": "d",
                    "licence": "mit",
                    "osi_approved": True,
                    "size_class": "small",
                }
            ]
        )
    )
    captured = {}
    monkeypatch.setattr(cli_mod, "run_setup", lambda **kwargs: captured.update(kwargs) or _fake_report())
    monkeypatch.setattr(cli_mod, "record_setup", lambda *_a, **_k: None)

    main(["setup", "--dry-run", "--catalogue", str(catalogue_path)])
    assert captured["catalogue"][0].alias == "custom"


@pytest.mark.parametrize("blocked_step", [SetupStepName.BACKEND, SetupStepName.HARDWARE])
def test_setup_exits_nonzero_when_backend_refused(monkeypatch, blocked_step: SetupStepName) -> None:
    """A refused CPU-only build (or infeasible hardware) is a failed setup, not a summary to skim."""
    report = SetupReport(
        dry_run=False,
        results=[StepResult(step=blocked_step, name="x", status=StepStatus.BLOCKED, detail="rebuild it")],
    )
    out = io.StringIO()
    monkeypatch.setattr(cli_mod, "_STDOUT", out)
    monkeypatch.setattr(cli_mod, "run_setup", lambda **_k: report)
    monkeypatch.setattr(cli_mod, "record_setup", lambda *_a, **_k: None)

    assert main(["setup", "--yes"]) == EXIT_ERROR
    assert "start serving" not in out.getvalue()
