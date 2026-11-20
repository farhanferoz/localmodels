"""`localmodels settings tune` end to end, its candidates served by the echo server (no PyTorch, no GPU)."""

from __future__ import annotations

import hashlib
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest

from localmodels import cli, inprocess_tune
from localmodels.enums import Precision, WorkloadKind
from localmodels.serving import LoaderDeclaration
from localmodels.serving.tune import EXIT_ADOPTED, EXIT_FAILED, EXIT_NOT_ADOPTED
from localmodels.settings import ModelIdentity, read_store, record_path
from localmodels.stack import Settings
from tests.test_serving import ECHO_ARGV, ECHO_ENV

if TYPE_CHECKING:
    from collections.abc import Callable

REPO = "org/scorer"
REVISION = "a" * 40
BF16 = Precision.BFLOAT16
# the reference is unshifted and sleeps 0.05 s a pass; a 0.1 shift disagrees past K6's 0.05, 0.001 agrees
DECLARATION = LoaderDeclaration(
    dtypes=(BF16,),
    options={"score_shift": (0.0, 0.1, 0.001), "sleep_s": (0.05, 0.0)},
    reference=Settings(dtype=BF16, options={"score_shift": 0.0, "sleep_s": 0.05}),
)


class _Loader:
    def __init__(self, entry: str) -> None:
        self.entry = entry

    def declare(self) -> LoaderDeclaration:
        if self.entry == "broken":  # the reference's dtype is refused by the server: the reference fails
            return DECLARATION.model_copy(
                update={
                    "options": {**DECLARATION.options, "fail_dtype": ("bfloat16",)},
                    "reference": Settings(
                        dtype=BF16, options={**DECLARATION.reference.options, "fail_dtype": "bfloat16"}
                    ),
                }
            )
        return DECLARATION

    def load(
        self, model_dir: Path, settings: Settings, device: str
    ) -> Any:  # never called: the echo server serves
        raise AssertionError


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Callable[..., int]:
    """A snapshot in a Hub cache, a store, a lease folder, the loader module, the echo server as its server.

    Returns a runner of `localmodels settings tune` with the fixture's paths and any extra flags.
    """
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))
    (tmp_path / "hub" / "models--org--scorer" / "snapshots" / REVISION).mkdir(parents=True)
    monkeypatch.setenv("LOCALMODELS_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("PYTHONPATH", ECHO_ENV["PYTHONPATH"])  # the echo child imports tests.echo_server
    module = types.ModuleType("fake_tune_project")
    module.model = _Loader  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "fake_tune_project", module)
    started: list[str] = []

    def argv(spec: str, entry: str) -> list[str]:
        started.append(entry)
        return ECHO_ARGV

    monkeypatch.setattr(inprocess_tune, "loader_argv", argv)
    monkeypatch.setattr("localmodels.serving.tune.available_memory", lambda _device: None)  # taken to fit
    texts = tmp_path / "texts.jsonl"
    texts.write_text("".join(json.dumps({"id": f"t{i}", "text": "x" * i}) + "\n" for i in range(5)))

    def run(*extra: str, entry: str = "ok") -> int:
        flags = {
            "--project": "proj",
            "--loader": "fake_tune_project:model",
            "--entry": entry,
            "--model": REPO,
            "--revision": REVISION,
            "--kind": "K6",
            "--texts": str(texts),
            "--min-texts": "3",
            "--lease-dir": str(tmp_path / "lease"),
        }
        given = dict(zip(extra[::2], extra[1::2], strict=True))
        args = [item for flag, value in {**flags, **given}.items() for item in (flag, value)]
        return cli.main(["settings", "tune", *args])

    run.started = started  # type: ignore[attr-defined]
    run.texts = texts  # type: ignore[attr-defined]
    return run


def _stored() -> Any:
    return read_store(record_path(ModelIdentity(repository=REPO, revision=REVISION), WorkloadKind.K6)).records


def test_the_fastest_agreeing_candidate_is_adopted_with_the_texts_it_was_measured_on(
    project: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    assert project() == EXIT_ADOPTED
    (record,) = _stored()
    assert record.chosen.options["sleep_s"] == 0.0  # faster than the reference by far more than the margin
    assert record.chosen.options["score_shift"] != 0.1  # the shifted scores disagree, so never chosen
    shifted = [m for m in record.measurements if m.candidate.options["score_shift"] == 0.1]
    assert shifted
    assert all(m.agreement is not None and not m.agreement.passed for m in shifted)
    assert record.inputs is not None
    assert (record.inputs.count, record.inputs.sha256) == (
        5,
        hashlib.sha256(project.texts.read_bytes()).hexdigest(),
    )
    assert record.recipe_project == "proj"
    assert record.quality == inprocess_tune.QUALITY_NAME
    assert len(capsys.readouterr().out.splitlines()) == 1 + 6  # a heading and one line per candidate


@pytest.mark.parametrize(
    ("extra", "refusal"),
    [
        (("--min-texts", "6"), "holds 5 texts; a tune needs between 6 and 500"),
        (("--max-texts", "4"), "holds 5 texts; a tune needs between 3 and 4"),
        (("--revision", "short"), "40-character commit"),
        (("--revision", "b" * 40), "is not in the Hub cache"),
        (("--max-candidates", "5"), "6 candidates, above the cap of 5"),
        (("--loader", "fake_tune_project"), "MODULE:ATTRIBUTE"),
        (("--limit-s", "100"), "leaves no time"),
    ],
    ids=[
        "too-few",
        "too-many",
        "bad-revision",
        "missing-snapshot",
        "too-many-candidates",
        "bad-loader",
        "limit",
    ],
)
def test_input_that_cannot_be_tuned_is_refused_before_any_server_starts(
    project: Any, capsys: pytest.CaptureFixture[str], extra: tuple[str, str], refusal: str
) -> None:
    assert project(*extra) == EXIT_FAILED
    assert refusal in capsys.readouterr().err
    assert project.started == []
    assert _stored() == []


def test_a_texts_file_with_a_repeated_id_is_refused_before_any_server_starts(
    project: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    with project.texts.open("a") as handle:
        handle.write(json.dumps({"id": "t1", "text": "again"}) + "\n")
    assert project() == EXIT_FAILED
    assert "repeats the id 't1'" in capsys.readouterr().err
    assert project.started == []


def test_a_failed_reference_is_a_failure_and_nothing_eligible_is_measured_not_adopted(
    project: Any, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    assert project(entry="broken") == EXIT_FAILED
    assert "Reference candidate" in capsys.readouterr().err
    # 8 GiB free: the 4 GiB headroom alone would fit, but no candidate may leave less than the 10 GiB floor
    full = SimpleNamespace(bytes=8 * 1024**3)
    monkeypatch.setattr("localmodels.serving.tune.available_memory", lambda _device: full)
    assert project() == EXIT_NOT_ADOPTED
    assert "measured, nothing adopted" in capsys.readouterr().err
    assert _stored() == []


def test_the_recipe_is_the_command_the_cli_parses(tmp_path: Path) -> None:
    model = ModelIdentity(repository=REPO, revision=REVISION)
    recipe = inprocess_tune.in_process_recipe(
        project="proj",
        loader="pkg.loaders:model",
        entry="scorer-4b",
        model=model,
        kind=WorkloadKind.K6,
        texts=tmp_path / "t.jsonl",
        workdir=tmp_path,
        reference="bfloat16,score_shift=0.0",
    )
    assert recipe.command[0] == str(Path(sys.executable).with_name("localmodels"))
    args = cli.build_parser().parse_args(recipe.command[1:])
    assert args.fn is inprocess_tune.cmd_settings_tune
    parsed = (args.project, args.loader, args.entry, args.model, args.revision, args.kind, args.texts)
    assert parsed == (
        "proj",
        "pkg.loaders:model",
        "scorer-4b",
        REPO,
        REVISION,
        "K6",
        str(tmp_path / "t.jsonl"),
    )
    assert (recipe.workdir, recipe.reference, recipe.quality) == (
        tmp_path,
        "bfloat16,score_shift=0.0",
        inprocess_tune.QUALITY_NAME,
    )
