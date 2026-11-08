"""The settings lookup: the fallback order, a store that never fails a load, and what stays out of imports."""

from __future__ import annotations

import fcntl
import json
import logging
import os
import subprocess
import sys
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from localmodels import retune, settings
from localmodels.atomic import write_model
from localmodels.cli import main
from localmodels.engine_bench import EngineRun
from localmodels.engine_choice import ChoiceRule, JobShape, choose
from localmodels.engine_refresh import same_client_path
from localmodels.engines import (
    Approval,
    GateVerdict,
    Publisher,
    VersionRecord,
    pins_for,
    register_approval,
    switch,
    write_record,
)
from localmodels.enums import DeviceTier, DeviceType, Precision, WorkloadKind
from localmodels.hardware import HardwareReport
from localmodels.refresh import Model
from localmodels.rules import rules_text, stamps
from localmodels.serving.tune import Agreement, TuneCandidate, TuneMeasurement, TuneRecord
from localmodels.settings import (
    AdoptError,
    ModelIdentity,
    Settings,
    SettingsRecord,
    SettingsSource,
    SettingsStatus,
    StoreState,
    TunePlan,
    TuneRecipe,
    TuneState,
    adopt,
    lookup,
    read_store,
    record_from_tune,
    record_path,
)
from localmodels.stack import Engine, LoadOptions, RetunePolicy

if TYPE_CHECKING:
    from collections.abc import Callable

GB10 = HardwareReport(
    platform="linux", arch="aarch64", device=DeviceType.CUDA, device_name="NVIDIA GB10",
    total_memory_gb=122.0, tier=DeviceTier.FAST_ACCELERATED,
)  # fmt: skip
GB10_KEY = "cuda:NVIDIA GB10:122GB:aarch64"
HOST = {"driver": "580.1", "cuda": "13.0", "kernel": "6.17", "dgx_os": "7.6.0", "localmodels": "0.6.0"}
NEWER_DRIVER = {**HOST, "driver": "590.2"}
QWEN = ModelIdentity(repository="Qwen/Qwen3-30B-A3B-GGUF", revision="c" * 40)  # K1's rule was measured on it
FLASH = ModelIdentity(repository="Qwen/Qwen3.8-Flash-Next-GGUF", revision="d" * 40)  # K1's rule was not
DETECTOR = ModelIdentity(repository="service_c/machine-text-detector", revision="e" * 40)  # K6's rule was not
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
    """A GB10 host with stack ``HOST`` (change the returned dict to change it), an empty store, no slurm."""
    stack = dict(HOST)
    monkeypatch.setenv(
        retune.SUBMIT_ENV, str(tmp_path / "no-submit.py")
    )  # never the machine's real submit.py
    monkeypatch.setenv(settings.RETUNE_ENV, settings.RETUNE_ON)  # pytest turns it off unless it is on
    monkeypatch.delenv(settings.TUNING_ENV, raising=False)
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
PINNED = SettingsStatus.PINNED
REC, RUL, DEF = SettingsSource.RECORD, SettingsSource.RULE, SettingsSource.DEFAULT
NOT_NEEDED, NO_RECIPE, NO_SCHEDULER = TuneState.NOT_NEEDED, TuneState.NO_RECIPE, TuneState.NO_SCHEDULER
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
        expect(RL, RUL, NO_SCHEDULER, FULL, n_ubatch=512, rule_applied=True),
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
        expect(U, DEF, NO_SCHEDULER, FULL, dtype=F32),
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
        expect(U, DEF, TuneState.NOT_AVAILABLE, None, dtype=F32),
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
    answer = lookup(MEDGEMMA, WorkloadKind.K5, legacy_path=legacy, recipe=RECIPE)
    assert answer.status is SettingsStatus.STALE
    assert answer.stale == ["recorded without a stack stamp"]
    assert answer.settings.dtype is Precision.BFLOAT16
    assert answer.record is not None
    assert answer.record.path == str(legacy)
    assert (answer.tune_plan, answer.tune) == (TunePlan.CONFIRM, TuneState.NO_SCHEDULER)
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


def test_a_pin_keeps_the_first_answer_for_each_model_kind_and_engine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    pin = Path("study/pin.json")  # relative, as the design writes it: the stamp must name the file in full
    put(QWEN, WorkloadKind.K6, recipe_project="proj")
    first = lookup(QWEN, WorkloadKind.K6, pin=pin)
    assert (first.status, first.settings.dtype, first.pinned) == (T, BF16, str(tmp_path / pin))
    stamp = first.server_stamp("/models/qwen", "2.10.0")
    assert (stamp.engine, stamp.load) == (Engine.PYTORCH, LoadOptions(dtype=BF16))
    assert first.record is not None
    assert stamp.settings is not None
    assert (stamp.settings.status, stamp.settings.pinned, stamp.settings.pinned_status) == (
        T, str(tmp_path / pin), None,
    )  # fmt: skip
    record_id = f"{first.record.path}@2026-10-20T00:00:00+00:00"
    assert (stamp.settings.record_id, stamp.settings.recipe_project) == (record_id, "proj")
    # the store moves on: a newer record and a changed driver; the pin does not
    put(QWEN, WorkloadKind.K6, chosen={"dtype": "float16"}, candidate="float16", stack=NEWER_DRIVER)
    again = lookup(QWEN, WorkloadKind.K6, pin=pin, recipe=RECIPE)
    assert (again.status, again.source, again.settings.dtype) == (PINNED, SettingsSource.PIN, BF16)
    assert (again.tune, again.tune_plan, again.stale) == (NOT_NEEDED, None, [])
    assert (again.pinned_vs_store or "").startswith("stale (dtype=float16)")
    assert any("stay in use" in note for note in again.notes)
    pinned = again.stamp()  # the stamp still says how the pinned settings were decided
    assert (pinned.status, pinned.pinned_status, pinned.record_id, pinned.recipe_project) == (
        PINNED, T, record_id, "proj",
    )  # fmt: skip
    assert (pinned.stale, pinned.rule, pinned.tune) == ([], None, NOT_NEEDED)
    # a pin made from a stale answer, with a caller's own dtype on top that is not pinned
    put(QWEN, WorkloadKind.K5, stack=NEWER_DRIVER)
    stale = lookup(QWEN, WorkloadKind.K5, pin=pin, explicit={"dtype": "float32"}).stamp()
    assert (stale.status, stale.stale, stale.overridden, stale.applied.dtype) == (
        S, ["driver: 590.2 -> 580.1"], ["dtype"], F32,
    )  # fmt: skip
    later = lookup(QWEN, WorkloadKind.K5, pin=pin).stamp()
    assert (later.status, later.pinned_status, later.stale, later.overridden, later.applied.dtype) == (
        PINNED, S, ["driver: 590.2 -> 580.1"], [], BF16,
    )  # fmt: skip
    # another model, another kind and another engine each get their own entry; explicit wins and is not pinned
    fp16 = Settings(dtype=Precision.FLOAT16)
    explicit = {"dtype": "float32"}
    other = lookup(FLASH, WorkloadKind.K6, pin=pin, default=fp16, explicit=explicit)
    in_process = lookup(QWEN, WorkloadKind.K6, pin=pin, engine=Engine.LLAMA_CPP_PYTHON)
    assert in_process.status is not PINNED
    entries = {
        (e["model"]["repository"], e["kind"], e["engine"]): e for e in json.loads(pin.read_text())["entries"]
    }
    assert set(entries) == {
        (FLASH.repository, "K6", "pytorch"), (QWEN.repository, "K5", "pytorch"),
        (QWEN.repository, "K6", "pytorch"), (QWEN.repository, "K6", "llama-cpp-python"),
    }  # fmt: skip
    assert entries[(FLASH.repository, "K6", "pytorch")]["settings"]["dtype"] == "float16"
    assert entries[(QWEN.repository, "K6", "pytorch")]["settings"]["dtype"] == "bfloat16"
    assert (other.settings.dtype, other.overridden) == (Precision.FLOAT32, ["dtype"])
    over = lookup(FLASH, WorkloadKind.K6, pin=pin, default=fp16, explicit=explicit)
    assert (over.status, over.settings.dtype, over.overridden) == (PINNED, Precision.FLOAT32, ["dtype"])
    untuned = over.stamp()  # pinned from a rule measured on other models: still untuned on its face
    assert (untuned.pinned_status, untuned.rule) == (RO, "K6.precision")
    assert lookup(FLASH, WorkloadKind.K6, pin=pin).settings == fp16
    record_path(QWEN, WorkloadKind.K6).write_text(
        "{"
    )  # a damaged store is still reported beside a pinned answer
    damaged = lookup(QWEN, WorkloadKind.K6, pin=pin)
    assert (damaged.status, damaged.store) == (PINNED, StoreState.UNREADABLE)
    assert any(str(record_path(QWEN, WorkloadKind.K6)) in note for note in damaged.notes)


@pytest.mark.parametrize(
    ("written", "pinned"),
    [
        pytest.param(None, True, id="a valid entry"),
        pytest.param("{", False, id="a damaged file"),
    ],
)
def test_a_lookup_waits_for_the_pin_lock_and_reads_the_file_again_before_writing(
    tmp_path: Path, written: str | None, pinned: bool
) -> None:
    pin, theirs = tmp_path / "pin.json", tmp_path / "theirs.json"
    put(QWEN, WorkloadKind.K6, chosen={"dtype": "float16"})
    lookup(QWEN, WorkloadKind.K6, pin=theirs)
    put(QWEN, WorkloadKind.K6)
    answers: list[settings.SettingsAnswer] = []
    worker = threading.Thread(target=lambda: answers.append(lookup(QWEN, WorkloadKind.K6, pin=pin)))
    with pin.with_name("pin.json.lock").open("a") as held:  # a flock belongs to the open file
        fcntl.flock(held, fcntl.LOCK_EX)
        worker.start()
        worker.join(0.3)
        assert worker.is_alive()
        assert not pin.exists()
        # another process was first
        pin.write_text(theirs.read_text(encoding="utf-8") if written is None else written, encoding="utf-8")
    worker.join(5)
    answer = answers[0]
    assert (answer.status is PINNED) == pinned
    assert answer.settings.dtype is (Precision.FLOAT16 if pinned else BF16)
    assert pin.read_text(encoding="utf-8") == (theirs.read_text(encoding="utf-8") if pinned else written)


@pytest.mark.parametrize(
    "damage",
    [
        pytest.param(lambda p: p.write_text("{"), id="not JSON"),
        pytest.param(lambda p: p.write_text('{"schema": 2, "entries": []}'), id="newer schema"),
        pytest.param(
            lambda p: p.write_text('{"schema": 1, "entries": [{"kind": "K6"}]}'), id="invalid entry"
        ),
        pytest.param(lambda p: p.write_text(""), id="empty file"),
        pytest.param(
            lambda p: p.parent.chmod(0o500),
            id="folder that cannot be written",
            marks=pytest.mark.skipif(os.geteuid() == 0, reason="root writes anywhere"),
        ),
    ],
)
def test_a_pin_file_that_cannot_be_used_is_left_as_it_is_and_the_result_says_so(
    damage: Callable[[Path], object], tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    folder = tmp_path / "study"
    folder.mkdir()
    pin = folder / "pin.json"
    put(QWEN, WorkloadKind.K6)
    damage(pin)
    before = pin.read_bytes() if pin.exists() else None
    try:
        answer = lookup(QWEN, WorkloadKind.K6, pin=pin)
    finally:
        folder.chmod(0o700)
    assert (answer.status, answer.pinned, answer.pinned_vs_store) == (T, None, None)
    assert str(pin) in (answer.pin_problem or "")
    assert answer.stamp().pin_problem == answer.pin_problem
    with caplog.at_level(logging.WARNING):  # the store answer is tuned, but the pin that was asked for failed
        answer.announce(logging.getLogger("test"))
    assert [r.levelno for r in caplog.records if str(pin) in r.getMessage()] == [logging.WARNING]
    assert (pin.read_bytes() if pin.exists() else None) == before


def test_a_pin_that_cannot_take_an_entry_unpins_only_that_model(tmp_path: Path) -> None:
    pin = tmp_path / "pin.json"
    put(QWEN, WorkloadKind.K6)
    lookup(QWEN, WorkloadKind.K6, pin=pin)
    payload = json.loads(pin.read_text())
    payload["entries"].append({"model": "an entry a later version could write"})
    pin.write_text(json.dumps(payload), encoding="utf-8")
    before = pin.read_bytes()
    put(QWEN, WorkloadKind.K6, chosen={"dtype": "float16"})
    kept = lookup(QWEN, WorkloadKind.K6, pin=pin)  # its own entry is valid
    assert (kept.status, kept.settings.dtype) == (PINNED, BF16)
    # a file with an entry this version cannot read takes no more entries
    refused = lookup(FLASH, WorkloadKind.K6, pin=pin)
    assert (refused.pinned, "entry 1" in (refused.pin_problem or "")) == (None, True)
    assert pin.read_bytes() == before
    # settings that are not JSON: the answer comes back unpinned, not as an error
    odd = Settings(options={"matmul": object()})
    unpinned = lookup(DETECTOR, WorkloadKind.K6, pin=tmp_path / "odd.json", default=odd)
    assert (unpinned.pinned, "cannot pin" in (unpinned.pin_problem or "")) == (None, True)
    assert not (tmp_path / "odd.json").exists()


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


K5 = WorkloadKind.K5
PASSED = Agreement(score=0.907, passed=True, detail="mean word similarity 0.907")


def adoptable(
    hardware: str = GB10_KEY, dtype: Precision = Precision.BFLOAT16, **fields: Any
) -> SettingsRecord:
    """A record that may be adopted (it names its quality check and its passed agreement)."""
    return SettingsRecord.model_validate(
        {
            "model": MEDGEMMA, "kind": K5, "hardware": hardware, "chosen": Settings(dtype=dtype),
            "measured_at": datetime(2026, 10, 22, tzinfo=UTC), "candidate": dtype.value,
            "quality": "text 0.9", "agreement": PASSED, "engine": Engine.PYTORCH, **fields,
        }
    )  # fmt: skip


def history_files(model: ModelIdentity, kind: WorkloadKind) -> list[Path]:
    folder = settings.settings_dir() / "history" / record_path(model, kind).parent.name / kind.value
    return sorted(folder.glob("*.json"))


def test_adoption_needs_a_quality_check_and_a_passed_agreement_and_never_writes_a_newer_file() -> None:
    failed = PASSED.model_copy(update={"passed": False})
    for refused in (adoptable(quality=""), adoptable(agreement=None), adoptable(agreement=failed)):
        with pytest.raises(AdoptError, match=r"quality check|agreement"):
            adopt(refused)
    unfit = [measured("bfloat16", Precision.BFLOAT16, 0.5, fits=False)]
    with pytest.raises(AdoptError, match="measurements show 'bfloat16'"):
        adopt(adoptable(measurements=unfit))  # its own measurement says it does not fit
    path = record_path(MEDGEMMA, K5)
    assert not path.exists()
    path.parent.mkdir(parents=True)
    for newer in (
        '{"schema": 2, "records": []}',
        '{"schema": 2, "entries": []}',
        '{"schema": 3, "records": {}}',
    ):
        path.write_text(newer)  # a newer layout is never taken for a damaged file
        with pytest.raises(AdoptError, match="newer localmodels"):
            adopt(adoptable())
        assert path.read_text() == newer
        assert read_store(path).state is StoreState.NEWER_SCHEMA


def test_an_adopt_replaces_only_its_own_entry_and_keeps_five_replaced_records(
    caplog: pytest.LogCaptureFixture,
) -> None:
    path = record_path(MEDGEMMA, K5)
    path.parent.mkdir(parents=True)
    future = {"model": "a shape this version cannot read", "added_by": "a newer localmodels"}
    other = adoptable(hardware="cuda:Other:24GB:x86_64").model_dump(mode="json")
    path.write_text(json.dumps({"schema": 1, "records": [future, other]}))
    with caplog.at_level(logging.INFO, logger="localmodels.settings"):
        first = adopt(adoptable())
        assert (first.previous, first.adopted_at is not None) == (None, True)
        assert history_files(MEDGEMMA, K5) == []
        for dtype in (
            Precision.FLOAT32,
            Precision.BFLOAT16,
            Precision.FLOAT32,
            Precision.BFLOAT16,
            Precision.FLOAT32,
        ):
            adopt(adoptable(dtype=dtype))
        again = adopt(adoptable(dtype=Precision.FLOAT32))
    assert again.previous == "bfloat16"  # unchanged: it keeps the change before it, not its own candidate
    assert lookup(MEDGEMMA, K5).stamp().changed_from == "bfloat16"  # every later stamp shows the last change
    records = json.loads(path.read_text())["records"]
    assert [records[0], records[1]] == [future, other]  # entries it does not own stay exactly as they were
    assert [r["candidate"] for r in records[2:]] == ["float32"]  # one entry for this slot, replaced
    assert len(history_files(MEDGEMMA, K5)) == 5  # six were replaced; the oldest was deleted
    changes = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert changes[0].startswith("settings for google/medgemma-1.5-4b-it (K5) changed: bfloat16 -> float32, ")
    assert "agreement 0.907 (text 0.9)" in changes[0]
    assert len(changes) == 5  # the last choice was unchanged: it only says it was adopted
    for damaged in (b'{"version": 2, "records": []}', b"\xff\xfe not text"):
        path.write_bytes(damaged)  # no schema number: unreadable, so copied byte for byte and replaced
        adopt(adoptable())
        assert damaged in [f.read_bytes() for f in history_files(MEDGEMMA, K5)]
        assert [r.candidate for r in read_store(path).records] == ["bfloat16"]
    k6 = WorkloadKind.K6  # a new build writes beside the old build's entry, and still reports the change
    adopt(adoptable(kind=k6, build="aaa1111"))
    with caplog.at_level(logging.WARNING, logger="localmodels.settings"):
        caplog.clear()
        moved = adopt(adoptable(kind=k6, build="bbb2222", dtype=Precision.FLOAT32))
    assert moved.previous == "bfloat16"
    assert [r.build for r in read_store(record_path(MEDGEMMA, k6)).records] == ["aaa1111", "bbb2222"]
    assert "changed: bfloat16 -> float32" in caplog.text
    assert len(history_files(MEDGEMMA, k6)) == 1


def test_an_adopt_in_a_tune_job_tells_the_pending_file_how_it_went(monkeypatch: pytest.MonkeyPatch) -> None:
    pending = retune.pending_dir() / "0123abcd.json"
    pending.parent.mkdir(parents=True)
    pending.write_text(
        json.dumps({"state": "queued", "at": 1.0, "job": "7", "key": MEDGEMMA.key, "kind": "K5"})
    )
    adopt(adoptable())  # not in a tune job: the file is none of its business
    assert json.loads(pending.read_text())["state"] == "queued"
    monkeypatch.setenv(settings.TUNE_CHANGE_ENV, "0123abcd")
    adopt(adoptable(model=QWEN))  # another model's record, adopted by the same job
    with pytest.raises(AdoptError, match="quality check"):
        adopt(adoptable(model=QWEN, quality=""))
    assert json.loads(pending.read_text())["state"] == "queued"
    with pytest.raises(AdoptError, match="quality check"):
        adopt(adoptable(quality=""))
    held = json.loads(pending.read_text())
    assert (held["state"], held["job"], "quality check" in held["reason"]) == (
        "measured_not_adopted",
        "7",
        True,
    )
    adopt(adoptable())  # the file stays, so the same change is never queued again
    held = json.loads(pending.read_text())
    assert (held["state"], "bfloat16 measured 2026-10-22" in held["record"]) == ("adopted", True)


def test_two_adopts_racing_lose_nothing() -> None:
    barrier = threading.Barrier(8)

    def racer(index: int) -> None:
        barrier.wait()
        adopt(adoptable(hardware=f"cuda:Card {index}:24GB:x86_64"))

    threads = [threading.Thread(target=racer, args=(i,)) for i in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(read_store(record_path(MEDGEMMA, K5)).records) == 8


def measured(name: str, dtype: Precision, median: float, **fields: Any) -> TuneMeasurement:
    return TuneMeasurement.model_validate(
        {
            "candidate": TuneCandidate(name=name, dtype=dtype), "infer_s": [median],
            "infer_s_median": median, "peak_gib": 1.0, "agreement": PASSED, "fits": True, **fields,
        }
    )  # fmt: skip


def tune_of(medians: dict[str, float], passes: list[float] | None = None, failed: str = "") -> TuneRecord:
    """The candidates named in ``medians`` (the first is the reference) timed so; ``failed`` did not agree."""
    ms = [
        measured(
            name, Precision(name), median, pass_s=passes or [] if index == 0 else [],
            **({"agreement": PASSED.model_copy(update={"passed": False})} if name == failed else {}),
        )
        for index, (name, median) in enumerate(medians.items())
    ]  # fmt: skip
    best = min((m for m in ms if m.agreement and m.agreement.passed), key=lambda m: m.infer_s_median or 0)
    return TuneRecord(
        hardware=GB10_KEY, localmodels_version="0.6.0", measured_at=datetime(2026, 10, 22, tzinfo=UTC),
        reference=next(iter(medians)), quality="text 0.9", chosen=best.candidate, measurements=ms,
        stack=HOST, libraries={"torch": "2.9"},
    )  # fmt: skip


F32, B16, F16 = "float32", "bfloat16", "float16"


def test_a_faster_candidate_displaces_the_incumbent_only_beyond_the_margin_and_the_timing_spread(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    put(MEDGEMMA, K5, candidate="bfloat16", engine="pytorch")  # the incumbent
    ask = {"model": MEDGEMMA, "kind": K5, "libraries": {"torch": "2.8", "transformers": "5.0"}}
    cases = [  # medians by candidate, reference passes, candidate not agreeing, chosen, spread noted
        ({F32: 1.0, B16: 0.5, F16: 0.485}, [1.0], "", B16, True),  # 3 % faster than the incumbent
        ({F32: 1.0, B16: 0.5, F16: 0.45}, [1.0], "", F16, True),  # 11 % faster
        ({F32: 1.0, B16: 0.5, F16: 0.45}, [1.0, 1.2], "", B16, False),  # the reference's passes differ 20 %
        ({F32: 1.0, B16: 0.5, F16: 0.485}, [1.0], B16, F16, True),  # the incumbent did not agree: float32 is
        (
            {F32: 1.0, F16: 0.97},
            [1.0],
            "",
            F32,
            True,
        ),  # the incumbent was not measured: float32 is, 3 % ahead
        ({F32: 0.49, B16: 0.5, F16: 0.6}, [1.0], "", F32, False),  # the reference wins without the margin
    ]
    for medians, passes, failed, expected, noted in cases:
        with caplog.at_level(logging.INFO, logger="localmodels.settings"):
            caplog.clear()
            chosen = record_from_tune(tune_of(medians, passes, failed), **ask).candidate
        assert (medians, chosen) == (medians, expected)
        assert ("spread is not measured" in caplog.text) is noted
    record = record_from_tune(
        tune_of({F32: 1.0, B16: 0.5, F16: 0.45}), **ask, recipe_project="service_b", tune_job="12"
    )
    assert (record.chosen, record.engine, record.agreement) == (
        Settings(dtype=Precision.FLOAT16),
        Engine.PYTORCH,
        PASSED,
    )
    assert (record.reference, len(record.measurements), record.recipe_project, record.tune_job) == (
        "float32", 3, "service_b", "12",
    )  # fmt: skip
    assert record.libraries == {"torch": "2.9", "transformers": "5.0"}  # the tune's own report wins
    unfit = tune_of({F32: 1.0, F16: 0.4})
    unfit.measurements[1].fits = False
    with pytest.raises(AdoptError, match="not eligible"):
        record_from_tune(unfit, **ask)  # a choice that does not fit is never recorded
    record_path(MEDGEMMA, K5).unlink()  # no stored record: the caller's own file holds the incumbent
    legacy = tmp_path / "legacy.json"
    legacy.write_text(json.dumps(LEGACY_MEDGEMMA))
    close = tune_of({F32: 1.0, B16: 0.5, F16: 0.485}, [1.0])
    assert record_from_tune(close, **ask, legacy_path=legacy).candidate == B16
    assert record_from_tune(close, **ask).candidate == F16


def test_importing_localmodels_or_its_cli_does_not_import_the_lookup() -> None:
    """In a fresh interpreter: slurm's lease keeper imports this package and must not pay for the lookup."""
    code = (
        "import sys, localmodels, localmodels.cli, localmodels.lease\n"
        "lazy = ('localmodels.settings', 'localmodels.llama_tune')\n"
        "assert not [m for m in sys.modules if m.startswith(lazy)], sorted(sys.modules)\n"
        "import localmodels.settings\n"
        "assert 'localmodels.retune' not in sys.modules, 'the lookup imports retune only to queue a tune'\n"
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)  # noqa: S603
    assert done.returncode == 0, done.stderr


# --- the engine chooser (design 6) -----------------------------------------------------------------

FILE_REPO, FILE_REVISION = "nvidia/Qwen3-30B-A3B-NVFP4", "b" * 40
IMAGE = "vllm/vllm-openai@sha256:" + "ab" * 32
BULK_JOB = JobShape(project="proj", requests=1000, prompt_tokens=1000, output_tokens=500, concurrency=16)
LLAMA_FIGURES = {
    "write_tps_16_at_once": 100.0,
    "read_tps_prefill_512": 900.0,
    "read_tps_prefill_8192": 1000.0,
}


def stage(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
    engine: Engine = Engine.VLLM,
    *,
    write_tps: float = 300.0,
    read_tps: float = 3000.0,
    startup_s: float = 200.0,
    **flaws: bool,
) -> None:
    """A passing world for ``QWEN`` K2 work: an approval, a complete record on `current`, llama.cpp's figures.

    llama.cpp writes 100 tokens/s and reads 1,000 at 16 at once; the container writes ``write_tps`` and
    reads 3,000, so ``BULK_JOB`` takes 6,000 s on llama.cpp and 2,000 s (at 300) on the container, which
    starts 150 s later. ``flaws`` breaks one piece: no_approval, no_current, incomplete, no_same_client,
    no_memory (no recorded need).
    """
    llama_home = root / "llama"
    commit = "abc1234"
    monkeypatch.setenv("LOCALMODELS_LLAMA_HOME", str(llama_home))
    (llama_home / "builds" / commit).mkdir(parents=True, exist_ok=True)
    if not (llama_home / "builds" / "current").is_symlink():
        (llama_home / "builds" / "current").symlink_to(commit)
    gguf = Model(QWEN.repository, QWEN.revision, "q4.gguf")
    if not flaws.get("no_same_client"):
        run = EngineRun(
            engine="llama.cpp", model="m", url="u", started="t", startup_s=50.0, figures=LLAMA_FIGURES
        )  # fmt: skip
        write_model(same_client_path(commit, gguf), run)
    if not flaws.get("no_approval"):
        evidence = root / "check.json"
        evidence.write_text("{}", encoding="utf-8")
        approval = Approval(
            project="proj", model_key=QWEN.key, gguf_repository=gguf.repository, gguf_revision=gguf.revision,
            gguf_file=gguf.filename, engine=engine, file_repository=FILE_REPO, file_revision=FILE_REVISION,
            evidence=str(evidence), evidence_sha256="0" * 64, approved_at="2026-11-08T00:00:00+00:00",
        )  # fmt: skip
        assert register_approval(approval) is None
    if flaws.get("no_current"):
        return
    figures = {
        "write_tps_16_at_once": write_tps,
        "read_tps_prefill_512": read_tps,
        "read_tps_prefill_8192": read_tps,
    }
    result = {
        "model": FILE_REPO, "kind": "approved", "complete": not flaws.get("incomplete"), "figures": figures,
        "startup_s": startup_s, "memory": {} if flaws.get("no_memory") else {"fall_gib": 30.0},
    }  # fmt: skip
    write_record(
        VersionRecord(
            engine=engine,
            tag="v0.31.0",
            image=IMAGE,
            publisher=Publisher.OFFICIAL,
            gate=GateVerdict.PASSED,
            pulled=True,
            results=[result],
        )
    )
    assert switch(engine, "v0.31.0") is None


def test_a_container_engine_is_chosen_only_when_every_rule_holds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(settings.STATE_DIR_ENV, str(tmp_path / "ok" / "state"))
    stage(tmp_path / "ok", monkeypatch)
    chosen = choose(QWEN, WorkloadKind.K2, BULK_JOB, available_gib=100.0)
    assert (chosen.engine, chosen.rule, chosen.version, chosen.image) == (
        Engine.VLLM, ChoiceRule.CHOSEN, "v0.31.0", IMAGE,
    )  # fmt: skip
    assert chosen.file == f"{FILE_REPO}@{FILE_REVISION}"
    assert chosen.numbers == pytest.approx(
        {
            "t_llama_s": 6000.0,
            "t_container_s": 2000.0,
            "saving_s": 4000.0,
            "extra_start_s": 150.0,
            "need_gib": 30.0,
            "available_gib": 100.0,
        }
    )
    # each rule broken once from that passing world
    small = BULK_JOB.model_copy(update={"requests": 1})  # saves 4 s, under twice the 150 s extra start
    eight = BULK_JOB.model_copy(update={"concurrency": 8})
    broken = {
        "no approval": ({"no_approval": True}, BULK_JOB, 100.0, ChoiceRule.NO_APPROVAL),
        "no current": ({"no_current": True}, BULK_JOB, 100.0, ChoiceRule.NO_CURRENT),
        "incomplete record": ({"incomplete": True}, BULK_JOB, 100.0, ChoiceRule.NO_COMPLETE_RECORD),
        "no same-client record": ({"no_same_client": True}, BULK_JOB, 100.0, ChoiceRule.NO_SAME_CLIENT),
        "8 at once": ({}, eight, 100.0, ChoiceRule.LOW_CONCURRENCY),
        "saving too small": ({}, small, 100.0, ChoiceRule.SAVING_TOO_SMALL),
        "need over available": ({}, BULK_JOB, 39.0, ChoiceRule.MEMORY),
        "no recorded need": ({"no_memory": True}, BULK_JOB, 100.0, ChoiceRule.MEMORY),
        "no saving at all": (
            {"write_tps": 100.0, "read_tps": 1000.0, "startup_s": 40.0},
            BULK_JOB,
            100.0,
            ChoiceRule.SAVING_TOO_SMALL,
        ),
    }
    for index, (name, (flaws, job, available, rule)) in enumerate(broken.items()):
        root = tmp_path / f"case{index}"
        monkeypatch.setenv(settings.STATE_DIR_ENV, str(root / "state"))
        stage(root, monkeypatch, **flaws)  # type: ignore[arg-type]
        answer = choose(QWEN, WorkloadKind.K2, job, available_gib=available)
        assert (answer.engine, answer.rule) == (Engine.LLAMA_CPP, rule), name
        assert answer.version is None
        if rule is not ChoiceRule.NO_APPROVAL:
            assert answer.considered[0].startswith("vllm "), name
    # K1 and K3 are llama.cpp, K4 is PyTorch, K2 without a job is llama.cpp, whatever the approvals say
    monkeypatch.setenv(settings.STATE_DIR_ENV, str(tmp_path / "ok" / "state"))
    monkeypatch.setenv("LOCALMODELS_LLAMA_HOME", str(tmp_path / "ok" / "llama"))
    routes = [
        (WorkloadKind.K1, BULK_JOB),
        (WorkloadKind.K3, BULK_JOB),
        (WorkloadKind.K4, BULK_JOB),
        (WorkloadKind.K2, None),
    ]
    answers = [choose(QWEN, kind, job, available_gib=100.0) for kind, job in routes]
    assert [(a.engine, a.rule) for a in answers] == [
        (Engine.LLAMA_CPP, ChoiceRule.INTERACTIVE_OR_SCORING),
        (Engine.LLAMA_CPP, ChoiceRule.INTERACTIVE_OR_SCORING),
        (Engine.PYTORCH, ChoiceRule.PYTORCH_KIND),
        (Engine.LLAMA_CPP, ChoiceRule.NO_JOB),
    ]


def test_the_approval_with_the_largest_saving_is_chosen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(settings.STATE_DIR_ENV, str(tmp_path / "state"))
    stage(tmp_path, monkeypatch, Engine.VLLM, write_tps=300.0)
    stage(tmp_path, monkeypatch, Engine.SGLANG, write_tps=600.0)
    best = choose(QWEN, WorkloadKind.K2, BULK_JOB, available_gib=100.0)
    assert (best.engine, best.rule) == (Engine.SGLANG, ChoiceRule.CHOSEN)
    assert len(best.considered) == 2


def test_lookup_chooses_the_engine_only_when_asked_and_a_pin_keeps_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(settings.STATE_DIR_ENV, str(tmp_path / "state"))
    stage(tmp_path, monkeypatch)
    monkeypatch.setattr("localmodels.container_run.available_gib", lambda: 100.0)
    off = RetunePolicy.OFF
    put(
        QWEN,
        WorkloadKind.K2,
        engine="llama.cpp",
        build="ccc3646",
        chosen={"n_ubatch": 1024},
        candidate="ub1024",
    )
    # an existing call is what it always was: no chooser, no choice, nothing extra in the stamp
    for engine in (Engine.PYTORCH, Engine.LLAMA_CPP):
        plain = lookup(QWEN, WorkloadKind.K2, engine=engine, retune=off)
        assert (plain.engine, plain.choice, plain.stamp().engine_choice) == (engine, None, None)
        assert "engine_choice" not in plain.stamp().model_dump(mode="json")
    default = lookup(QWEN, WorkloadKind.K2, retune=off, job=BULK_JOB)  # a job alone does not ask for a choice
    assert (default.engine, default.choice) == (Engine.PYTORCH, None)

    pin = tmp_path / "study" / "pin.json"
    asked = lookup(QWEN, WorkloadKind.K2, engine=None, job=BULK_JOB, retune=off, pin=pin)
    llama = lookup(QWEN, WorkloadKind.K2, engine=Engine.LLAMA_CPP, retune=off)
    assert llama.settings.n_ubatch == 1024  # the tuned llama.cpp record: what a fallback would run
    assert asked.choice is not None
    assert (asked.engine, asked.choice.rule, asked.settings) == (
        Engine.VLLM,
        ChoiceRule.CHOSEN,
        llama.settings,
    )
    assert asked.stamp().engine_choice == {
        "engine": "vllm", "rule": "chosen", "version": "v0.31.0", "image": IMAGE,
        "file": f"{FILE_REPO}@{FILE_REVISION}",
    }  # fmt: skip
    assert any("engine vllm (chosen)" in note for note in asked.notes)
    entry = json.loads(pin.read_text(encoding="utf-8"))["entries"][0]
    assert (entry["engine"], entry["engine_version"], entry["engine_file"]) == (
        "vllm", "v0.31.0", f"{FILE_REPO}@{FILE_REVISION}",
    )  # fmt: skip
    assert [(p.version, p.file) for p in pins_for(Engine.VLLM)] == [("v0.31.0", pin.resolve())]

    # the world changes (the approval goes), and the chooser alone would now say llama.cpp; the pin does not
    (tmp_path / "state" / "engines" / "approvals.json").unlink()
    free = lookup(QWEN, WorkloadKind.K2, engine=None, job=BULK_JOB, retune=off)
    assert free.choice is not None
    assert (free.engine, free.choice.rule) == (Engine.LLAMA_CPP, ChoiceRule.NO_APPROVAL)
    again = lookup(QWEN, WorkloadKind.K2, engine=None, job=BULK_JOB, retune=off, pin=pin)
    assert again.choice is not None
    assert (again.engine, again.choice.rule, again.choice.version, again.status) == (
        Engine.VLLM, ChoiceRule.PINNED, "v0.31.0", PINNED,
    )  # fmt: skip
    assert (again.choice.image, again.choice.file) == (IMAGE, f"{FILE_REPO}@{FILE_REVISION}")
    # an explicit engine beats the pin's: no chooser, no choice
    assert lookup(QWEN, WorkloadKind.K2, engine=Engine.LLAMA_CPP, retune=off, pin=pin).choice is None
