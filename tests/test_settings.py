"""The settings lookup: the fallback order, a store that never fails a load, and what stays out of imports."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import pytest

from localmodels import settings
from localmodels.cli import main
from localmodels.enums import DeviceTier, DeviceType, Precision, WorkloadKind
from localmodels.hardware import HardwareReport
from localmodels.rules import rules_text, stamps
from localmodels.settings import (
    ModelIdentity,
    Settings,
    SettingsRecord,
    SettingsSource,
    SettingsStatus,
    StoreState,
    TunePlan,
    TuneRecipe,
    TuneState,
    lookup,
    record_path,
)
from localmodels.stack import Engine

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

GB10 = HardwareReport(
    platform="linux", arch="aarch64", device=DeviceType.CUDA, device_name="NVIDIA GB10",
    total_memory_gb=122.0, tier=DeviceTier.FAST_ACCELERATED,
)  # fmt: skip
GB10_KEY = "cuda:NVIDIA GB10:122GB:aarch64"
HOST = {"driver": "580.1", "cuda": "13.0", "kernel": "6.17", "dgx_os": "7.6.0", "localmodels": "0.6.0"}
NEWER_DRIVER = {**HOST, "driver": "590.2"}
QWEN = ModelIdentity(repository="Qwen/Qwen3-30B-A3B-GGUF", revision="c" * 40)  # K1's rule was measured on it
FLASH = ModelIdentity(repository="Qwen/Qwen3.8-Flash-Next-GGUF", revision="d" * 40)  # K1's rule was not
DETECTOR = ModelIdentity(
    repository="service_c/machine-text-detector", revision="e" * 40
)  # K6's rule was not
MEDGEMMA = ModelIdentity(repository="google/medgemma-1.5-4b-it", revision="a" * 40)
FP32 = Settings(dtype=Precision.FLOAT32)

# service_b's config/tuning/medgemma-1-5-4b.json as localmodels 0.4.0 wrote it: a record with no stack key.
LEGACY_MEDGEMMA = {
    "records": [
        {
            "hardware": GB10_KEY,
            "localmodels_version": "0.4.0",
            "measured_at": "2026-10-18T12:09:14.586923Z",
            "reference": "float32",
            "quality": "text_agreement(min_ratio=0.9)",
            "chosen": {"name": "bfloat16", "dtype": "bfloat16", "options": {}},
            "measurements": [
                {
                    "candidate": {"name": "bfloat16", "dtype": "bfloat16", "options": {}},
                    "load_s": 46.68,
                    "infer_s": [21.11],
                    "infer_s_median": 21.11,
                    "peak_gib": 14.13,
                    "agreement": {"score": 0.907, "passed": True, "detail": "mean word similarity 0.907"},
                    "fits": True,
                    "error": None,
                }
            ],
        }
    ]
}


@pytest.fixture(autouse=True)
def host(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    """A GB10 host whose stack is ``HOST`` (change the returned dict to change it), and an empty store."""
    stack = dict(HOST)
    monkeypatch.setenv(settings.STATE_DIR_ENV, str(tmp_path / "state"))
    monkeypatch.setattr(settings, "inspect_hardware", lambda: GB10)
    monkeypatch.setattr(settings, "fingerprint", lambda: stack)
    monkeypatch.setattr(settings, "_current_build", lambda _home=None: "ccc3646")
    monkeypatch.setattr(settings, "_announced", set())
    return stack


def put(model: ModelIdentity, kind: WorkloadKind, **fields: Any) -> Path:
    """Write one stored record of the model (bf16 on the current stack unless ``fields`` say otherwise)."""
    record = SettingsRecord(
        model=model, kind=kind, hardware=GB10_KEY, measured_at=datetime(2026, 10, 20, tzinfo=UTC),
        chosen=Settings(dtype=Precision.BFLOAT16), candidate="bfloat16", stack=HOST,
    )  # fmt: skip
    path = record_path(model, kind)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"schema": 1, "records": [{**record.model_dump(mode="json"), **fields}]}), encoding="utf-8"
    )
    return path


def expect(
    status: SettingsStatus, source: SettingsSource, tune: TuneState, plan: TunePlan | None, **more: Any
) -> dict[str, Any]:
    """What a row of design 3.4 must answer; ``more`` names the settings and extras it also checks."""
    return {"status": status, "source": source, "tune": tune, "plan": plan, **more}


T, S, RL, RO, U = (
    SettingsStatus.TUNED,
    SettingsStatus.STALE,
    SettingsStatus.RULE,
    SettingsStatus.RULE_OTHER_MODEL,
    SettingsStatus.UNTUNED,
)
REC, RUL, DEF = SettingsSource.RECORD, SettingsSource.RULE, SettingsSource.DEFAULT
NOT_NEEDED, NO_RECIPE, UNAVAILABLE = TuneState.NOT_NEEDED, TuneState.NO_RECIPE, TuneState.NOT_AVAILABLE
FULL, CONFIRM = TunePlan.FULL, TunePlan.CONFIRM
BF16, F32 = Precision.BFLOAT16, Precision.FLOAT32
RECIPE = TuneRecipe(project="proj", command=("python", "tune.py"))

# Rows of design 3.4: model, kind, stored record (changed fields; None: no record), lookup arguments, answer.
CASES = [
    pytest.param(
        QWEN,
        WorkloadKind.K6,
        {"stack": {**HOST, "localmodels": "0.5.0"}},
        {},
        expect(T, REC, NOT_NEEDED, None, dtype=BF16),
        id="record from another localmodels version is still tuned",
    ),
    pytest.param(
        QWEN,
        WorkloadKind.K6,
        {"stack": NEWER_DRIVER},
        {},
        expect(S, REC, NO_RECIPE, CONFIRM, dtype=BF16, stale=["driver: 590.2 -> 580.1"]),
        id="record from another driver stays in use but is stale",
    ),
    pytest.param(
        QWEN,
        WorkloadKind.K1,
        {"build": "aaa1111", "engine": "llama.cpp"},
        {"engine": Engine.LLAMA_CPP, "build": "bbb2222"},
        expect(
            S,
            REC,
            NO_RECIPE,
            CONFIRM,
            dtype=BF16,
            stale=["llama_cpp: aaa1111 -> bbb2222"],
            notes=["revision only"],
        ),
        id="record from another llama.cpp build is stale",
    ),
    pytest.param(
        QWEN,
        WorkloadKind.K1,
        {"build": "aaa1111"},
        {"engine": Engine.LLAMA_CPP},
        expect(
            S,
            REC,
            NO_RECIPE,
            CONFIRM,
            dtype=BF16,
            stale=["llama_cpp: aaa1111 -> ccc3646"],
            notes=["no llama.cpp build named"],
        ),
        id="a llama.cpp lookup naming no build compares with the current one",
    ),
    pytest.param(
        QWEN,
        WorkloadKind.K6,
        {"libraries": {"torch": "2.10.0"}},
        {"libraries": {"torch": "2.11.0"}},
        expect(S, REC, NO_RECIPE, CONFIRM, dtype=BF16, stale=["torch: 2.10.0 -> 2.11.0"]),
        id="a library the record names at another version is stale",
    ),
    pytest.param(
        QWEN,
        WorkloadKind.K6,
        {},
        {"libraries": {"torch": "2.10.0"}},
        expect(T, REC, NOT_NEEDED, None, dtype=BF16, notes=["torch not recorded"]),
        id="a library the record does not name is reported, not stale",
    ),
    pytest.param(
        QWEN,
        WorkloadKind.K6,
        {"engine": "llama.cpp"},
        {"default": FP32},
        expect(RO, DEF, NO_RECIPE, FULL, dtype=F32),
        id="record made through another engine is not used",
    ),
    pytest.param(
        QWEN,
        WorkloadKind.K1,
        None,
        {"recipe": RECIPE},
        expect(RL, RUL, UNAVAILABLE, FULL, n_ubatch=512, rule_applied=True),
        id="rule names this model, recipe given: first-use tune wanted, never tuned",
    ),
    pytest.param(
        QWEN,
        WorkloadKind.K1,
        None,
        {},
        expect(RL, RUL, NO_RECIPE, FULL, n_ubatch=512, rule_applied=True),
        id="rule names this model, no recipe",
    ),
    pytest.param(
        FLASH,
        WorkloadKind.K1,
        None,
        {"default": Settings(n_ubatch=256)},
        expect(RO, DEF, NO_RECIPE, FULL, n_ubatch=256, rule_applied=False),
        id="rule names only other models: the caller's default stays",
    ),
    pytest.param(
        DETECTOR,
        WorkloadKind.K6,
        None,
        {"default": FP32},
        expect(RO, DEF, NO_RECIPE, FULL, dtype=F32, rule_applied=False, not_applied=BF16),
        id="K6 detector keeps its fp32 default; bf16 is listed as not applied",
    ),
    pytest.param(
        QWEN,
        WorkloadKind.K4,
        None,
        {"default": FP32, "recipe": RECIPE},
        expect(U, DEF, UNAVAILABLE, FULL, dtype=F32),
        id="kind with no rule values is untuned",
    ),
    pytest.param(
        QWEN,
        WorkloadKind.K6,
        {"hardware": "cpu:CPU:8GB:x86_64"},
        {"default": FP32},
        expect(RO, DEF, NO_RECIPE, FULL, dtype=F32),
        id="record made on another machine is not used",
    ),
    pytest.param(
        QWEN,
        WorkloadKind.K1,
        {},
        {"engine": Engine.LLAMA_CPP_PYTHON, "default": FP32},
        expect(U, DEF, UNAVAILABLE, None, dtype=F32),
        id="llama-cpp-python never takes a llama-server record",
    ),
]


@pytest.mark.parametrize(("model", "kind", "record", "call", "expect"), CASES)
def test_the_fallback_order_of_design_3_4(
    model: ModelIdentity,
    kind: WorkloadKind,
    record: dict[str, Any] | None,
    call: dict[str, Any],
    expect: dict[str, Any],
) -> None:
    if record is not None:
        put(model, kind, **record)
    answer = lookup(model, kind, **call)
    assert answer.status is expect["status"]
    assert answer.source is expect["source"]
    assert answer.tune is expect["tune"]
    assert answer.tune_plan == expect["plan"]
    assert answer.settings.dtype == expect.get("dtype")
    assert answer.settings.n_ubatch == expect.get("n_ubatch")
    assert answer.stale == expect.get("stale", [])
    assert answer.store is StoreState.OK
    if "rule_applied" in expect:
        assert answer.rule is not None
        assert answer.rule.applied is expect["rule_applied"]
    if "not_applied" in expect:
        assert answer.rule is not None
        assert answer.rule.values.dtype == expect["not_applied"]
    assert all(any(text in note for note in answer.notes) for text in expect.get("notes", []))
    assert (answer.record is None) == (answer.source is not SettingsSource.RECORD)


@pytest.mark.parametrize(
    ("damage", "state"),
    [
        pytest.param(
            lambda p: p.write_text('{"schema": 1, "records": [{"model"'),
            StoreState.UNREADABLE,
            id="truncated",
        ),
        pytest.param(
            lambda p: p.write_text('{"schema": 2, "records": []}'), StoreState.NEWER_SCHEMA, id="newer schema"
        ),
        pytest.param(lambda p: p.write_text("[1, 2]"), StoreState.UNREADABLE, id="not an object"),
        pytest.param(lambda p: p.write_text('{"records": []}'), StoreState.UNREADABLE, id="no schema key"),
        pytest.param(lambda p: p.write_text("[" * 200000), StoreState.UNREADABLE, id="nested too deeply"),
        pytest.param(
            lambda p: p.parent.chmod(0),
            StoreState.UNREADABLE,
            id="folder that cannot be searched",
            marks=pytest.mark.skipif(os.geteuid() == 0, reason="root searches any folder"),
        ),
        pytest.param(lambda p: p.write_bytes(b"\xff\xfe\x00"), StoreState.UNREADABLE, id="not text"),
        pytest.param(
            lambda p: p.write_text(p.read_text().replace("bfloat16", "float8")),
            StoreState.UNREADABLE,
            id="an enum value this version lacks",
        ),
    ],
)
def test_a_bad_store_file_is_absent_and_announced_never_an_error(
    damage: Callable[[Path], object], state: StoreState, caplog: pytest.LogCaptureFixture
) -> None:
    path = put(QWEN, WorkloadKind.K4)
    damage(path)
    try:
        answer = lookup(QWEN, WorkloadKind.K4, default=FP32)
    finally:
        path.parent.chmod(0o700)  # the folder case; harmless otherwise
    assert (answer.status, answer.settings, answer.store) == (SettingsStatus.UNTUNED, FP32, state)
    with caplog.at_level(logging.WARNING):
        answer.announce(logging.getLogger("test"))
    assert str(path) in caplog.text


def test_a_record_with_a_field_this_version_lacks_is_still_used() -> None:
    put(QWEN, WorkloadKind.K4, added_by_a_newer_version={"x": 1})
    answer = lookup(QWEN, WorkloadKind.K4)
    assert (answer.status, answer.store) == (SettingsStatus.TUNED, StoreState.OK)


def test_service_bs_legacy_file_answers_stale_until_a_stamped_record_exists(tmp_path: Path) -> None:
    legacy = tmp_path / "medgemma-1-5-4b.json"
    legacy.write_text(json.dumps(LEGACY_MEDGEMMA), encoding="utf-8")
    answer = lookup(MEDGEMMA, WorkloadKind.K5, legacy_path=legacy, recipe=object())
    assert answer.status is SettingsStatus.STALE
    assert answer.stale == ["recorded without a stack stamp"]
    assert answer.settings.dtype is Precision.BFLOAT16
    assert answer.record is not None
    assert answer.record.path == str(legacy)
    assert (answer.tune_plan, answer.tune) == (TunePlan.CONFIRM, TuneState.NOT_AVAILABLE)
    put(MEDGEMMA, WorkloadKind.K5, candidate="float16", chosen={"dtype": "float16"})
    assert lookup(MEDGEMMA, WorkloadKind.K5, legacy_path=legacy).settings.dtype is Precision.FLOAT16
    legacy.write_text("{", encoding="utf-8")  # a bad legacy file is absent, whichever store it is read after
    assert lookup(MEDGEMMA, WorkloadKind.K6, legacy_path=legacy).store is StoreState.UNREADABLE


def test_explicit_settings_win_and_are_listed_without_changing_the_status() -> None:
    put(
        QWEN,
        WorkloadKind.K6,
        chosen={"dtype": "bfloat16", "options": {"token_budget": 512, "batch_size": 32}},
    )
    answer = lookup(QWEN, WorkloadKind.K6, explicit={"dtype": "float32", "options": {"batch_size": 8}})
    assert answer.settings == Settings(
        dtype=Precision.FLOAT32, options={"token_budget": 512, "batch_size": 8}
    )
    assert (answer.status, answer.overridden) == (SettingsStatus.TUNED, ["dtype", "options.batch_size"])
    with pytest.raises(ValueError, match="dtpye"):
        lookup(QWEN, WorkloadKind.K6, explicit={"dtpye": "float32"})


def test_rules_md_is_read_at_run_time_and_the_value_table_matches_it(monkeypatch: pytest.MonkeyPatch) -> None:
    text = rules_text()
    for kind, rule in settings.RULE_VALUES.items():
        assert kind.value in {name for name, _ in stamps(text)}
        assert f"Rule {rule.id}" in text
    k1 = text.replace("model=Qwen/Qwen3-30B-A3B-GGUF,unsloth/Qwen3.5-4B-GGUF", "", 1)
    monkeypatch.setattr(settings, "rules_text", lambda: k1)  # a stamp that names no model
    nameless = lookup(QWEN, WorkloadKind.K1)
    assert nameless.status is SettingsStatus.RULE_OTHER_MODEL
    assert nameless.rule is not None
    assert nameless.rule.models == []
    monkeypatch.setattr(settings, "rules_text", lambda: k1.replace("date=2026-10-18", "model=any", 1))
    assert lookup(QWEN, WorkloadKind.K1).status is SettingsStatus.RULE


def test_a_model_is_its_file_set_and_a_hub_cache_names_the_hashes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    blobs, snap = tmp_path / "models--o--m" / "blobs", tmp_path / "models--o--m" / "snapshots" / ("b" * 40)
    blobs.mkdir(parents=True)
    snap.mkdir(parents=True)
    for name, sha in (("m-Q4.gguf", "1" * 64), ("m-Q8.gguf", "2" * 64)):
        (blobs / sha).write_bytes(b"x")
        (snap / name).symlink_to(blobs / sha)
    q4 = ModelIdentity.from_hub_files("o/m", "b" * 40, ["m-Q4.gguf"], cache_dir=tmp_path)
    q8 = ModelIdentity.from_hub_files("o/m", "b" * 40, ["m-Q8.gguf"], cache_dir=tmp_path)
    assert q4.files[0].sha256 == "1" * 64
    assert q4.key != q8.key  # one repository and revision, two quantisations
    assert record_path(q4, WorkloadKind.K1) != record_path(q8, WorkloadKind.K1)
    for bad in ({}, {"revision": "main"}):  # a repository alone, or a revision that is not a commit
        with pytest.raises(ValueError, match="revision"):
            ModelIdentity(repository="o/m", **bad)
    put(q4, WorkloadKind.K1, chosen={"model_file": [f.model_dump() for f in q8.files]})
    with_files = lookup(q4, WorkloadKind.K1, engine=Engine.LLAMA_CPP, build="ccc3646")
    assert with_files.settings.model_file == q8.files  # the chosen file set, not a bare name
    assert not any("revision only" in note for note in with_files.notes)
    assert main(["settings", "show", "--model", "o/m", "--revision", "b" * 40, "--kind", "K1"]) == 0
    assert "another identity of this repository: m-Q4.gguf" in capsys.readouterr().out
    assert main(["settings", "show", "--model", "o/m", "--kind", "K1"]) == 1  # no --revision, no --file
    (snap / "plain.gguf").write_bytes(b"not a blob link")
    with pytest.raises(ValueError, match="sha256"):
        ModelIdentity.from_hub_files("o/m", "b" * 40, ["plain.gguf"], cache_dir=tmp_path)
    unpinned = lookup(ModelIdentity.from_path("o/m", snap / "plain.gguf"), WorkloadKind.K1)
    assert "path, size and modification time" in unpinned.notes[0]


def test_settings_show_prints_an_answer_and_check_lists_stale_records(
    host: dict[str, str], capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    put(QWEN, WorkloadKind.K6, stack=NEWER_DRIVER)
    assert (
        main(["settings", "show", "--model", QWEN.repository, "--revision", QWEN.revision, "--kind", "K6"])
        == 0
    )
    assert "status: stale (from record)\nsettings: dtype=bfloat16" in capsys.readouterr().out
    assert main(["settings", "check", "--home", str(tmp_path)]) == 1
    out = capsys.readouterr().out
    assert "Qwen/Qwen3-30B-A3B-GGUF K6" in out
    assert "driver: 590.2 -> 580.1" in out
    assert "1 of 1 records stale." in out
    host["driver"] = "590.2"
    assert main(["settings", "check", "--home", str(tmp_path)]) == 0


def test_importing_localmodels_or_its_cli_does_not_import_the_lookup() -> None:
    """In a fresh interpreter: slurm's lease keeper imports this package and must not pay for the lookup."""
    code = (
        "import sys, localmodels, localmodels.cli, localmodels.lease\n"
        "assert not [m for m in sys.modules if m.startswith('localmodels.settings')], sorted(sys.modules)\n"
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)  # noqa: S603
    assert done.returncode == 0, done.stderr
