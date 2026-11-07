"""The settings lookup: what a model should be loaded with on this machine, and how far that can be trusted.

``lookup`` answers from, in order, a tune record in the store (``~/.local/state/localmodels/settings``, or
``$LOCALMODELS_STATE_DIR``/settings), a record the caller's own earlier tool left (``legacy_path``), the
written rule of the model's kind, and the caller's default. Only a record is ever TUNED. A rule measured on
this very model answers RULE; a rule measured on other models only answers RULE_OTHER_MODEL and changes
nothing, because every setting here can change what a model computes and nothing has checked it on this one.
The store is shared by projects pinning different localmodels versions, so a bad or newer file is treated as
absent and announced: ``lookup`` never raises on the store. Torch-free. Imported on demand: ``import
localmodels`` does not load this module (slurm's lease keeper runs this package).
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import logging
import os
import re
import sys
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, NamedTuple, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from localmodels.atomic import write_text_atomically
from localmodels.enums import DeviceType, Precision, WorkloadKind
from localmodels.hardware import inspect_hardware
from localmodels.pinned import _REVISION_RE, PinnedFile, snapshot_dir
from localmodels.pull import _default_hub_cache, _repo_blob_name
from localmodels.rules import EXIT_STALE, StampKey, rules_text, stamps
from localmodels.serving.tune import (
    Agreement,
    TuneMeasurement,
    TuneRecord,
    _read_records,
    hardware_key,
    untuned_message,
)
from localmodels.stack import (
    UNKNOWN,
    Engine,
    LoadOptions,
    RetunePolicy,
    ServerStamp,
    Settings,
    SettingsSource,
    SettingsStamp,
    SettingsStatus,
    StoreState,
    TunePlan,
    TuneState,
    fingerprint,
    llama_cpp_build,
    stale_components,
)
from localmodels.stack import SpecType as SpecType  # noqa: PLC0414 - re-exported: the types live in stack.py

if TYPE_CHECKING:
    import argparse
    from collections.abc import Iterator, Mapping

    from localmodels.hardware import HardwareReport

STATE_DIR_ENV: Final = "LOCALMODELS_STATE_DIR"
DEFAULT_STATE_DIR: Final = Path("~/.local/state/localmodels")
SCHEMA_VERSION: Final = 1
NO_STACK_STAMP: Final = "recorded without a stack stamp"
_KEY_CHARS: Final = 16
_SLUG_UNSAFE = re.compile(r"[^A-Za-z0-9._-]")
_BAD_FILE: Final = (OSError, ValueError, KeyError, TypeError, RecursionError)  # what a damaged file raises
_ANY_MODEL: Final = "any"  # a rules.md stamp's model= for a training method
HISTORY_KEPT: Final = 5  # replaced records kept per model and kind
MIN_PASSES: Final = 2  # timed passes needed to measure a spread
SPEED_MARGIN: Final = 0.05  # a candidate must beat the incumbent by more than this (and the timing spread)
TUNING_ENV: Final = "LOCALMODELS_TUNING"  # set to TUNING_ON in a tune job: its loads queue no tune
TUNING_ON: Final = "1"
RETUNE_ENV: Final = (
    "LOCALMODELS_RETUNE"  # any value but ``on`` turns the re-tune off; ``on`` allows it under pytest
)
RETUNE_ON: Final = "on"
PYTEST_ENV: Final = "PYTEST_CURRENT_TEST"  # set by pytest while a test runs
TUNE_CHANGE_ENV: Final = "LOCALMODELS_TUNE_CHANGE"  # the change a tune job was queued for (``retune.py``)
#: Enabled only after the first real llama.cpp tune has run through slurm (HL5).
LLAMA_AUTO_RETUNE: Final = False
#: The settings fields a tune sets, by engine; a caller whose ``explicit`` holds all has nothing to tune.
_TUNED_FIELDS: Final = {
    Engine.PYTORCH: frozenset({"dtype", "options"}),
    Engine.LLAMA_CPP: frozenset({"n_ubatch", "spec", "spec_draft_file", "model_file"}),
}

log = logging.getLogger(__name__)


class ModelIdentity(BaseModel):
    """What a model is: its repository plus the files loaded, so two quantisations never share a record.

    A GGUF model is its file set (names and SHA-256 per file, from the Hub cache's blob names, never hashed
    on demand). A PyTorch snapshot loaded whole is its repository and a 40-character revision. ``unpinned``
    marks the fallback for a bare path with no content identity. Nothing else is an identity: a repository
    alone would give every revision and quantisation of it one shared record.
    """

    model_config = ConfigDict(frozen=True)

    repository: str
    revision: str = ""
    files: tuple[PinnedFile, ...] = ()
    unpinned: str = ""

    @model_validator(mode="after")
    def _identifies_content(self) -> Self:
        if self.revision and not _REVISION_RE.match(self.revision):
            msg = f"revision must be a 40-character commit sha, not {self.revision!r}"
            raise ValueError(msg)
        if not (self.files or self.revision or self.unpinned):
            msg = f"{self.repository}: a model needs its files, a 40-character revision or an unpinned path"
            raise ValueError(msg)
        return self

    @classmethod
    def from_hub_files(
        cls, repository: str, revision: str, names: list[str], *, cache_dir: Path | None = None
    ) -> Self:
        """The file set ``names`` of ``repository`` at ``revision`` as the Hub cache holds it.

        Raises ``ValueError`` for a file whose blob name is not a SHA-256 (a file the Hub stores outside LFS).
        """
        root = snapshot_dir(repository, revision, cache_dir or _default_hub_cache())
        files = tuple(PinnedFile(path=name, sha256=_repo_blob_name(root / name)) for name in names)
        return cls(repository=repository, revision=revision, files=files)

    @classmethod
    def from_path(cls, repository: str, path: Path) -> Self:
        """A file with no content identity: its path, size and modification time. Announced; prefer a hash."""
        stat = path.stat()
        return cls(repository=repository, unpinned=f"{path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}")

    @property
    def key(self) -> str:
        """A short hash of what identifies the model: its files if it has them, else revision or path."""
        identity: dict[str, Any] = (
            {"files": sorted((f.path, f.sha256) for f in self.files)}
            if self.files
            else {"revision": self.revision, "unpinned": self.unpinned}
        )
        text = json.dumps({"repository": self.repository, **identity}, sort_keys=True)
        return hashlib.sha256(text.encode()).hexdigest()[:_KEY_CHARS]


class SettingsRecord(BaseModel):
    """The stored result of one tune on one machine (schema 1). Unknown fields are ignored."""

    model: ModelIdentity
    kind: WorkloadKind
    hardware: str
    measured_at: datetime
    chosen: Settings
    candidate: str = ""
    quality: str = ""
    stack: dict[str, str] | None = None  # ``stack.fingerprint()``; None: made before stamps
    build: str | None = None  # the llama.cpp build it was measured on
    libraries: dict[str, str] = Field(default_factory=dict)
    recipe_project: str | None = None
    engine: Engine | None = None  # a record made through another engine is not used
    agreement: Agreement | None = None  # the chosen candidate's, against the reference
    reference: str = ""  # the name of the reference candidate
    # PyTorch tune measurements. Another engine's go in a field of their own: readers ignore unknown fields,
    # but one that met a different shape here would drop the whole record.
    measurements: list[TuneMeasurement] = Field(default_factory=list)
    previous: str | None = None  # the replaced record's candidate, or its settings described
    tune_job: str | None = None
    adopted_at: datetime | None = None
    # llama.cpp tune measurements (``llama_tune``), one JSON object per step. Untyped on purpose and apart
    # from ``measurements``: a reader that met another shape there would drop the whole record.
    llama_measurements: list[dict[str, Any]] = Field(default_factory=list)


class TuneRecipe(BaseModel):
    """How a project's tune is run (design 5.4); ``lookup`` queues ``command`` through slurm when one is due.

    ``command`` measures with ``run_tune`` and ends by adopting its record (``record_from_tune``, ``adopt``).
    The reference candidate must be timed at least twice (``repeats >= 2``): with one timed pass the speed
    margin falls back to 5 % and the spread of the timings is unmeasured.
    """

    model_config = ConfigDict(frozen=True)

    project: str
    # Run by slurm in the project's own environment, as its installed command (``<repo>/.venv/bin/<tool>``,
    # never ``uv run``, which syncs the environment and can remove what the lock does not hold); it ends by
    # adopting its record.
    command: tuple[str, ...]
    workdir: Path | None = None
    reference: str = ""  # the name of the reference candidate (listed first)
    quality: str = ""  # the name of the quality check; a recipe without one adopts nothing
    prompt_tokens: int | None = None  # llama.cpp: the typical prompt length of the project's inputs
    text: Path | None = None  # llama.cpp: a text for the teacher-forced check


class RecordRef(BaseModel):
    """Where an answer's record is and how it was made."""

    path: str
    measured_at: datetime
    hardware: str
    candidate: str
    quality: str
    stack: dict[str, str] | None
    build: str | None
    recipe_project: str | None
    changed_from: str | None = None  # the previous choice's name when a re-tune changed it


class RuleRef(BaseModel):
    """The rule of the kind, the models it was measured on, and whether its values were applied."""

    id: str
    models: list[str]
    applied: bool
    values: Settings


class PinEntry(BaseModel):
    """One pinned answer: the settings a study was given for a model and kind, and how they were decided.

    ``status``, ``rule``, ``stale``, ``record_id`` and ``recipe_project`` are those of the answer that was
    pinned, so a result of a pinned study can still say that its settings were untuned or stale.
    """

    model: ModelIdentity
    kind: WorkloadKind
    engine: Engine | None = None  # an entry without one matches any engine
    settings: Settings  # as resolved, before the caller's ``explicit`` was applied
    status: SettingsStatus
    source: SettingsSource
    record_id: str | None
    rule: str | None = None
    stale: list[str] = Field(default_factory=list)
    recipe_project: str | None = None
    stack: dict[str, str]  # ``fingerprint()`` when the entry was made
    pinned_at: datetime


class _Rule(NamedTuple):
    id: str
    values: Settings


#: Only values live here. The models a rule was measured on and its stack are read from rules.md at run time.
#: The slot count, shared pool and context of K1 and K2 stay with ``ServingRole`` and are not settings.
RULE_VALUES: Final[Mapping[WorkloadKind, _Rule]] = {
    WorkloadKind.K1: _Rule("K1.layout", Settings(n_ubatch=512)),
    WorkloadKind.K2: _Rule("K2.layout", Settings(n_ubatch=512)),
    WorkloadKind.K5: _Rule("K5.precision", Settings(dtype=Precision.BFLOAT16)),
    WorkloadKind.K6: _Rule("K6.precision", Settings(dtype=Precision.BFLOAT16)),
}

#: Sentences already written this process, by (model, kind): ``announce`` speaks once.
_announced: set[tuple[str, WorkloadKind]] = set()


def _record_id(record: RecordRef | None) -> str | None:
    return f"{record.path}@{record.measured_at.isoformat()}" if record else None


def _describe(settings: Settings) -> str:
    fields = settings.model_dump(mode="json", exclude_defaults=True)
    return ", ".join(f"{name}={value}" for name, value in fields.items()) or "no opinion"


class SettingsAnswer(BaseModel):
    """What to load a model with, how far to trust it, and where that came from."""

    model: ModelIdentity
    engine: Engine = Engine.PYTORCH
    kind: WorkloadKind
    settings: Settings
    status: SettingsStatus
    source: SettingsSource
    record: RecordRef | None = None
    stale: list[str] = Field(default_factory=list)
    rule: RuleRef | None = None
    tune: TuneState = TuneState.NOT_NEEDED
    tune_plan: TunePlan | None = None
    overridden: list[str] = Field(default_factory=list)
    store: StoreState = StoreState.OK
    notes: list[str] = Field(default_factory=list)
    pinned: str | None = None  # the pin file the answer came from or was written to
    pinned_vs_store: str | None = None  # what the store would answer now, for a pinned answer
    pin_entry: PinEntry | None = None  # the entry a pinned answer was taken from (it has no ``record``)
    pin_problem: str | None = None  # why a pin that was asked for could not be used
    tune_detail: str | None = None  # the job id, log path or reason behind ``tune``

    def stamp(self, applied: Settings | None = None) -> SettingsStamp:
        """The part of a result's stamp that says how its settings were decided.

        ``applied`` is what the result actually ran with when that differs from the answer (a caller that
        loaded something else); by default the answer's settings.
        """
        record, pin = self.record, self.pin_entry
        return SettingsStamp(
            status=self.status,
            source=self.source,
            applied=self.settings if applied is None else applied,
            record_id=pin.record_id if pin else _record_id(record),
            rule=pin.rule if pin else self.rule.id if self.rule else None,
            stale=pin.stale if pin else self.stale,
            tune=self.tune,
            store=self.store,
            overridden=self.overridden,
            pinned=self.pinned,
            pinned_status=pin.status if pin else None,
            pin_problem=self.pin_problem,
            tune_detail=self.tune_detail,
            changed_from=record.changed_from if record else None,
            recipe_project=pin.recipe_project if pin else record.recipe_project if record else None,
        )

    def server_stamp(
        self, model_path: str, build: str, model_sha256: str | None = None, applied: Settings | None = None
    ) -> ServerStamp:
        """The stamp of an in-process load made with this answer; ``build`` is the library's version.

        ``model_sha256`` defaults to the hash of the identity's file when it has exactly one. ``applied`` is
        what the load really used when the caller changed it after the lookup; its precision is the stamp's.
        """
        if model_sha256 is None and len(self.model.files) == 1:
            model_sha256 = self.model.files[0].sha256
        used = self.settings if applied is None else applied
        load = LoadOptions(dtype=used.dtype, options=dict(used.options)) if used.dtype else None
        return ServerStamp(
            engine=self.engine,
            build=build,
            model_path=model_path,
            model_sha256=model_sha256,
            load=load,
            host=fingerprint(),
            settings=self.stamp(applied),
        )

    def announce(self, logger: logging.Logger) -> None:
        """Write the notes once per process per model and kind: info when tuned, else a warning."""
        key = (self.model.key, self.kind)
        if key in _announced:
            return
        _announced.add(key)
        tuned = self.status is SettingsStatus.TUNED and self.store is StoreState.OK and not self.pin_problem
        for note in self.notes:
            logger.log(logging.INFO if tuned else logging.WARNING, "%s", note)

    def lines(self) -> list[str]:
        """The answer as plain lines, for ``localmodels settings show``."""
        out = [
            f"model: {self.model.repository} ({self.model.key})",
            f"kind: {self.kind.value}",
            f"status: {self.status.value} (from {self.source.value})",
            f"settings: {_describe(self.settings)}",
        ]
        if self.record:
            out.append(f"record: {self.record.path} (measured {self.record.measured_at:%Y-%m-%d %H:%M})")
        if self.rule:
            applied = "applied" if self.rule.applied else f"NOT applied: {_describe(self.rule.values)}"
            measured = ", ".join(self.rule.models) or "no named model"
            out.append(f"rule: {self.rule.id}, measured on {measured}; {applied}")
        out.extend(f"stale: {line}" for line in self.stale)
        if self.overridden:
            out.append(f"overridden by explicit: {', '.join(self.overridden)}")
        plan = f" ({self.tune_plan.value} tune wanted)" if self.tune_plan else ""
        detail = f" [{self.tune_detail}]" if self.tune_detail else ""
        out.append(f"tune: {self.tune.value}{detail}{plan}")
        out.append(f"store: {self.store.value}")
        out.extend(f"note: {note}" for note in self.notes)
        return out


def state_dir() -> Path:
    """The state directory: ``$LOCALMODELS_STATE_DIR``, else ``~/.local/state/localmodels``."""
    return Path(os.environ.get(STATE_DIR_ENV) or DEFAULT_STATE_DIR).expanduser()


def settings_dir() -> Path:
    """The store: ``$LOCALMODELS_STATE_DIR``/settings, else ``~/.local/state/localmodels/settings``."""
    return state_dir() / "settings"


@contextmanager
def _locked(path: Path) -> Iterator[None]:
    """Hold an exclusive ``flock`` on ``path``'s sibling ``.lock`` file for the block (created if absent)."""
    lock = path.with_name(path.name + ".lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


def _slug(repository: str) -> str:
    return _SLUG_UNSAFE.sub("_", repository.replace("/", "--"))


def record_path(model: ModelIdentity, kind: WorkloadKind) -> Path:
    """The file holding the records of one model and kind: ``records/<repository>--<key>/<kind>.json``."""
    return settings_dir() / "records" / f"{_slug(model.repository)}--{model.key}" / f"{kind.value}.json"


def stored_identities(repository: str, kind: WorkloadKind) -> list[tuple[Path, ModelIdentity]]:
    """Every identity of ``repository`` with a stored file for ``kind``: (the file, the identity it holds)."""
    found = []
    for path in sorted((settings_dir() / "records").glob(f"{_slug(repository)}--*/{kind.value}.json")):
        records = read_store(path).records
        if records and records[0].model.repository == repository:
            found.append((path, records[0].model))
    return found


class StoreRead(NamedTuple):
    """The records of one store file that can be used, and what was wrong with the rest."""

    records: list[SettingsRecord]
    state: StoreState
    problem: str  # "" when the file is fine or absent


def _first_error(exc: ValidationError) -> str:
    error = exc.errors()[0]
    return f"{'.'.join(map(str, error['loc']))}: {error['msg']}"


def _load_items(path: Path) -> tuple[list[Any], StoreState, str]:
    """The raw entries of a store file, its state and what was wrong with it; never raises.

    The schema number is read first: a file whose schema is newer than ``SCHEMA_VERSION`` is
    ``NEWER_SCHEMA`` whatever else it holds. Without an integer schema the file is ``UNREADABLE``.
    """
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        schema = payload["schema"]
        if not isinstance(schema, int):
            msg = "schema is not a number"
            raise TypeError(msg)
        if schema > SCHEMA_VERSION:
            return [], StoreState.NEWER_SCHEMA, f"{path}: schema {schema}, written by a newer localmodels"
        items = payload["records"]
        if not isinstance(items, list):
            msg = "records is not a list"
            raise TypeError(msg)
    except FileNotFoundError:
        return [], StoreState.OK, ""
    except _BAD_FILE as exc:
        return [], StoreState.UNREADABLE, f"{path}: {exc!r}"
    return items, StoreState.OK, ""


def read_store(path: Path) -> StoreRead:
    """Read one store file; never raises. A bad file, a newer schema or an invalid record is absent."""
    items, state, problem = _load_items(path)
    if state is not StoreState.OK:
        return StoreRead([], state, problem)
    records, problems = [], []
    for index, item in enumerate(items):
        try:
            records.append(SettingsRecord.model_validate(item))
        except ValidationError as exc:
            problems.append(f"record {index}: {_first_error(exc)}")
    state = StoreState.UNREADABLE if problems else StoreState.OK
    return StoreRead(records, state, f"{path}: {'; '.join(problems)}" if problems else "")


def _read_legacy(path: Path, model: ModelIdentity, kind: WorkloadKind) -> StoreRead:
    """A caller's own tune file (``serving.tune``'s layout) as records of this model and kind; no raise."""
    try:
        legacy: list[TuneRecord] = _read_records(path)  # no records when the file is absent
    except _BAD_FILE as exc:
        return StoreRead([], StoreState.UNREADABLE, f"{path}: {exc!r}")
    records = [
        SettingsRecord(
            model=model,
            kind=kind,
            hardware=r.hardware,
            measured_at=r.measured_at,
            chosen=Settings(dtype=r.chosen.dtype, options=r.chosen.options),
            candidate=r.chosen.name,
            quality=r.quality,
            stack=r.stack,
        )
        for r in legacy
    ]
    return StoreRead(records, StoreState.OK, "")


def _stale_lines(
    record: SettingsRecord,
    host: Mapping[str, str],
    build: str | None,
    libraries: Mapping[str, str] | None = None,
) -> list[str]:
    """What changed since ``record`` was made.

    Stamped host components, the llama.cpp ``build`` if named, and the caller's ``libraries`` that the record
    also names (one it does not name is not a reason). localmodels' own version is recorded, not compared
    (``stale_components``).
    """
    lines = [NO_STACK_STAMP] if record.stack is None else stale_components(record.stack, host)
    if build is not None and record.build != build:
        lines.insert(0, f"{StampKey.LLAMA_CPP}: {record.build or UNKNOWN} -> {build}")
    lines.extend(
        f"{name}: {record.libraries[name]} -> {version}"
        for name, version in (libraries or {}).items()
        if name in record.libraries and record.libraries[name] != version
    )
    return lines


def _pick(
    records: list[SettingsRecord], hardware: str, build: str | None, engine: Engine
) -> SettingsRecord | None:
    """This machine's newest record for ``engine``, preferring the one made on the caller's ``build``."""
    mine = [r for r in records if r.hardware == hardware and r.engine in (None, engine)]
    return max(
        [r for r in mine if build is None or r.build == build] or mine,
        key=lambda r: r.measured_at.timestamp(),
        default=None,
    )


def _rule(kind: WorkloadKind, repository: str) -> RuleRef | None:
    """The rule of ``kind`` with its values and the models it was measured on (rules.md); None: no values."""
    entry = RULE_VALUES.get(kind)
    if entry is None:
        return None
    stamp = next((pairs for name, pairs in stamps(rules_text()) if name == kind.value), {})
    models = [m for m in stamp.get(StampKey.MODEL, "").split(",") if m]
    return RuleRef(
        id=entry.id, models=models, applied=repository in models or _ANY_MODEL in models, values=entry.values
    )


def lookup(  # noqa: PLR0913 - the call signature of design 3.1
    model: ModelIdentity,
    kind: WorkloadKind,
    *,
    engine: Engine = Engine.PYTORCH,
    build: str | None = None,
    explicit: Mapping[str, Any] | None = None,
    default: Settings | None = None,
    libraries: Mapping[str, str] | None = None,
    recipe: TuneRecipe | None = None,
    retune: RetunePolicy = RetunePolicy.AUTO,
    legacy_path: Path | None = None,
    pin: Path | None = None,
) -> SettingsAnswer:
    """What to load ``model`` with for ``kind`` work on this machine; it never fails a load.

    ``build`` is the llama.cpp build the caller will run (``Engine.LLAMA_CPP``): a record made on another
    build is stale; without it the record's build is compared with the current one. ``explicit`` holds
    settings the caller has decided; they replace the answer's for those fields (``options`` key by key) and
    are listed in ``overridden``, so the status still shows that a measured alternative exists. ``default``
    is what the caller runs when nothing measured applies. ``libraries`` are the versions of the caller's own
    libraries (``{"torch": torch.__version__}``): a record that names one at another version is stale; one it
    does not name is reported, not a reason. ``recipe`` is how a tune is run: when the answer calls for one
    (``tune_plan``) and nothing rules it out, the lookup queues one slurm job for the model and change
    (``retune.py``) and ``tune`` and ``tune_detail`` say what happened. ``retune=RetunePolicy.OFF``, or
    ``LOCALMODELS_RETUNE`` set to anything but ``on``, queues nothing (``DISABLED``). Under pytest nothing is
    queued either unless ``LOCALMODELS_RETUNE=on``. ``LOCALMODELS_TUNING=1`` marks a tune job's own loads,
    which queue nothing.
    ``legacy_path`` is the caller's own tune
    file, read when the store has no record. ``pin`` is a study's pin file (``_pin``): the first lookup of a
    model, kind and engine writes the answer into it, later ones return that copy as PINNED whatever the
    store says now; ``explicit`` still wins over it. A ``.lock`` file stays beside the pin file; leave it.
    A pin that cannot be used leaves the answer unpinned, says why in ``pin_problem`` and never raises. A
    record made through llama-server is never applied to ``Engine.LLAMA_CPP_PYTHON``, which has no tune path
    yet. ``ValueError`` (pydantic's included) for an
    ``explicit`` key that is not a settings field or a value that is not valid: a caller's bug, not a store
    problem.
    """
    unknown = set(explicit or ()) - set(Settings.model_fields)
    if unknown:
        msg = f"explicit names no such settings field: {', '.join(sorted(unknown))}"
        raise ValueError(msg)
    default = default or Settings()
    name = f"{model.repository} ({kind.value})"
    notes: list[str] = []
    if model.unpinned:
        notes.append(f"{name} is identified by its path, size and modification time, not its content.")
    if engine is Engine.LLAMA_CPP and not model.files:
        notes.append(f"{name} is identified by revision only: every file of this revision shares a record.")
    identity_notes = len(notes)
    answer = SettingsAnswer(
        model=model,
        engine=engine,
        kind=kind,
        settings=default,
        status=SettingsStatus.UNTUNED,
        source=SettingsSource.DEFAULT,
    )

    report = None  # the hardware probed for the answer
    if engine is Engine.LLAMA_CPP_PYTHON:
        answer.tune = TuneState.NOT_AVAILABLE
        notes.append(f"{name}: llama-cpp-python has no tune path yet; it runs the caller's default settings.")
    else:
        report = _from_store(answer, (engine, build, libraries), legacy_path, notes)

    if pin is not None:
        # the store's problem note, if any, follows the identity notes and still holds when pinned
        _pin(answer, pin, notes, identity_notes + (answer.store is not StoreState.OK))
    # the tune is decided after the pin, which wants none; a tune is submitted from here, never before
    if answer.tune_plan and report is not None:
        answer.tune = TuneState.NOT_NEEDED
        if note := _decide_tune(answer, answer.tune_plan, (recipe, retune, explicit, build), report):
            notes.append(f"A {answer.tune_plan.value} tune is wanted; {note}.")
    if explicit:
        answer.settings, answer.overridden = _apply_explicit(answer.settings, explicit)
        notes.append(
            f"{name}: explicit settings replace the measured ones for {', '.join(answer.overridden)}."
        )
    answer.notes = notes
    return answer


class PinFile(NamedTuple):
    """A pin file's valid entries, its items as they stand, and what is wrong with it ("" if nothing).

    A file that cannot be read has no entries. A file with an invalid entry has the valid ones and the
    problem: its entries can be used, but it must not be written.
    """

    entries: list[PinEntry]
    items: list[Any]
    problem: str


def read_pins(path: Path) -> PinFile:
    """Read a pin file; never raises. An absent file has no entries; a bad one is reported, never replaced."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        schema, items = payload["schema"], payload["entries"]
        if not isinstance(schema, int) or not isinstance(items, list):
            msg = "schema is not a number or entries is not a list"
            raise TypeError(msg)
        if schema > SCHEMA_VERSION:
            return PinFile([], [], f"{path}: schema {schema}, written by a newer localmodels")
    except FileNotFoundError:
        return PinFile([], [], "")
    except _BAD_FILE as exc:
        return PinFile([], [], f"{path}: {exc!r}")
    entries, problems = [], []
    for index, item in enumerate(items):
        try:
            entries.append(PinEntry.model_validate(item))
        except ValidationError as exc:
            problems.append(f"entry {index}: {_first_error(exc)}")
    return PinFile(entries, items, f"{path}: {'; '.join(problems)}" if problems else "")


def _pin(answer: SettingsAnswer, pin: Path, notes: list[str], keep: int) -> None:
    """Pin ``answer`` (the store's resolution, before ``explicit``), or replace it with the pin's copy.

    A model, kind and engine not in ``pin`` yet is added and the answer is returned as it is. An entry
    already in the file is returned as PINNED with no tune wanted, its status, rule, stale lines and record
    kept for the stamp, and ``pinned_vs_store`` says what the store would answer now. A file that cannot be
    read or written, or holds an invalid entry, is never replaced: the answer stays unpinned and
    ``pin_problem`` says why. The lock file ``<pin>.lock`` stays beside the pin file. ``keep`` is how many of
    ``notes`` still hold when pinned (the model's identity, the store's problem).
    """
    name = f"{answer.model.repository} ({answer.kind.value})"
    try:
        pin = pin.resolve()  # one path to read, lock and stamp, whatever the working folder or links
        entry, problem = _pin_entry(pin, answer)
    except (OSError, RuntimeError, ValueError) as exc:  # unwritable folder, link loop, non-JSON settings
        entry, problem = None, f"{pin}: cannot pin: {exc!r}"
    if problem:
        answer.pin_problem = problem
        notes.append(f"{name}: pin file not used and left as it is, so this answer is not pinned: {problem}")
        return
    answer.pinned = str(pin)
    if entry is None:  # this lookup made the entry
        return
    now = _store_now(answer, entry)
    answer.pinned_vs_store = now or "the store agrees"
    del notes[keep:]
    if now:
        notes.append(
            f"{name}: settings pinned in {pin} on {entry.pinned_at:%Y-%m-%d} stay in use; "
            f"the store would now answer {now}."
        )
    answer.settings, answer.status, answer.source = entry.settings, SettingsStatus.PINNED, SettingsSource.PIN
    answer.record, answer.rule, answer.stale = None, None, []
    answer.tune, answer.tune_plan = TuneState.NOT_NEEDED, None
    answer.pin_entry = entry


def _pin_entry(pin: Path, answer: SettingsAnswer) -> tuple[PinEntry | None, str]:
    """The pin entry of the answer: (entry, ""); (None, "") if added now; (None, why it cannot be used).

    The file is read again under the lock before an entry is added, so when two processes of a study pin
    one model, the first to write wins and the other uses its entry. The new entry is built before the lock
    is taken: it reads the host's stack, which can be slow, and fails on settings that are not JSON.
    """
    found = read_pins(pin)
    entry = _pinned_entry(found, answer)
    if entry is not None or found.problem:
        return entry, "" if entry else found.problem
    item = _new_entry(answer).model_dump(mode="json")
    with _locked(pin):
        found = read_pins(pin)
        entry = _pinned_entry(found, answer)
        if entry is not None or found.problem:
            return entry, "" if entry else found.problem
        text = json.dumps({"schema": SCHEMA_VERSION, "entries": [*found.items, item]}, indent=2)
        write_text_atomically(pin, text + "\n")
    return None, ""


def _pinned_entry(found: PinFile, answer: SettingsAnswer) -> PinEntry | None:
    return next(
        (
            e
            for e in found.entries
            if (e.model.key, e.kind) == (answer.model.key, answer.kind) and e.engine in (None, answer.engine)
        ),
        None,
    )


def _new_entry(answer: SettingsAnswer) -> PinEntry:
    """The entry that pins ``answer`` as resolved from the store, before ``explicit``."""
    return PinEntry(
        model=answer.model,
        kind=answer.kind,
        engine=answer.engine,
        settings=answer.settings,
        status=answer.status,
        source=answer.source,
        record_id=_record_id(answer.record),
        rule=answer.rule.id if answer.rule else None,
        stale=answer.stale,
        recipe_project=answer.record.recipe_project if answer.record else None,
        stack=fingerprint(),
        pinned_at=datetime.now(UTC),
    )


def _store_now(answer: SettingsAnswer, entry: PinEntry) -> str | None:
    """What the store answers now (status, settings, stale) if that differs from the pin; else None."""
    if (answer.status, answer.settings) == (entry.status, entry.settings):
        return None
    parts = [f"{answer.status.value} ({_describe(answer.settings)})", *answer.stale]
    if answer.store is not StoreState.OK:
        parts.append(f"store {answer.store.value}")
    return "; ".join(parts)


def _from_store(
    answer: SettingsAnswer,
    asked: tuple[Engine, str | None, Mapping[str, str] | None],
    legacy_path: Path | None,
    notes: list[str],
) -> HardwareReport:
    """The record for this machine (the store's, else ``legacy_path``'s), else the rule or the default.

    Returns the hardware it probed, for the tune decision.
    """
    engine, build, libraries = asked
    model, kind = answer.model, answer.kind
    name = f"{model.repository} ({kind.value})"
    report = inspect_hardware()
    hardware = hardware_key(report)
    path = record_path(model, kind)
    store = read_store(path)
    record = _pick(store.records, hardware, build, engine)
    if record is None and legacy_path is not None:
        legacy = _read_legacy(legacy_path, model, kind)
        record, path = _pick(legacy.records, hardware, build, engine), legacy_path
        store = store if store.state is not StoreState.OK else legacy
    answer.store = store.state
    if store.problem:
        notes.append(f"Settings records for {name} unusable, treated as absent: {store.problem}")
    if record is None:
        _from_rule(answer, notes, hardware)
        return report
    compared = build if engine is Engine.LLAMA_CPP else None
    if engine is Engine.LLAMA_CPP and build is None and record.build:
        compared = _current_build()
        notes.append(f"{name}: no llama.cpp build named; the record's build is compared with {compared}.")
    _from_record(answer, record, path, (compared, libraries), notes)
    return report


def _apply_explicit(chosen: Settings, explicit: Mapping[str, Any]) -> tuple[Settings, list[str]]:
    """``chosen`` with the caller's fields on top (``options`` key by key), and the names it replaced."""
    fields = {**chosen.model_dump(), **{name: value for name, value in explicit.items() if name != "options"}}
    overridden = [name for name in explicit if name != "options"]
    if "options" in explicit:
        options = Settings.model_validate({"options": explicit["options"]}).options
        fields["options"] = {**chosen.options, **options}
        overridden.extend(f"options.{key}" for key in options)
    return Settings.model_validate(fields), overridden


def _current_build(home: str | None = None) -> str:
    """The commit of the ``current`` llama.cpp build, read from its library (never run)."""
    from localmodels.llama_builds import (  # noqa: PLC0415 - only a lookup that needs it
        current_server,
        llama_home,
    )

    return llama_cpp_build(current_server(llama_home(home)))


def _refused_tune(
    answer: SettingsAnswer,
    recipe: TuneRecipe | None,
    retune: RetunePolicy,
    explicit: Mapping[str, Any] | None,
    device: DeviceType,
) -> tuple[TuneState, str] | None:
    """Why no tune is queued, checked in the order of design 5.1: (state, reason); None: nothing rules it out.

    ``retune`` is compared by value, so a plain ``"off"`` works. ``LOCALMODELS_RETUNE`` set to anything but
    ``on`` turns the re-tune off (a switch meant to stop GPU jobs fails closed). Under pytest (the variable it
    sets while a test runs, or its module already imported, as while it collects) a tune is queued only if
    that variable is ``on``: a test that loads a model would otherwise submit a real job wherever slurm's
    ``submit.py`` exists.
    """
    fields = _TUNED_FIELDS.get(answer.engine)
    env = os.environ.get(RETUNE_ENV)
    env_on = env is not None and env.strip().lower() == RETUNE_ON
    switched_off = env is not None and not env_on
    under_pytest = (PYTEST_ENV in os.environ or "_pytest" in sys.modules) and not env_on
    why_off = "the automatic re-tune is off"
    if switched_off and env is not None and env.strip().lower() != RetunePolicy.OFF:
        why_off = f"{RETUNE_ENV}={env!r} is not 'on', so the automatic re-tune is off"
    checks = (
        (os.environ.get(TUNING_ENV) == TUNING_ON, TuneState.NOT_NEEDED, ""),  # a tune job's own load
        (retune == RetunePolicy.OFF or switched_off, TuneState.DISABLED, why_off),
        (
            bool(fields and explicit and fields <= set(explicit)),
            TuneState.NOTHING_TO_TUNE,
            "the caller's explicit settings cover every field a tune sets",
        ),
        (recipe is None, TuneState.NO_RECIPE, "the caller gave no recipe"),
        (
            fields is None or (answer.engine is Engine.LLAMA_CPP and not LLAMA_AUTO_RETUNE),
            TuneState.NOT_AVAILABLE,
            "this engine's automatic re-tune is not enabled",
        ),
        (
            device is DeviceType.CPU,
            TuneState.NOT_APPLICABLE,
            "a tune job runs on the GPU; its record is one this CPU lookup never finds",
        ),
        (
            under_pytest,
            TuneState.DISABLED,
            f"automatic re-tune is off under pytest; set {RETUNE_ENV}={RETUNE_ON} to test it",
        ),
    )
    return next(((state, reason) for hit, state, reason in checks if hit), None)


def _decide_tune(
    answer: SettingsAnswer,
    plan: TunePlan,
    asked: tuple[TuneRecipe | None, RetunePolicy, Mapping[str, Any] | None, str | None],
    report: HardwareReport,
) -> str:
    """Set ``answer.tune`` and ``tune_detail`` for the tune ``plan`` calls for; the sentence for the notes.

    ``asked`` is the lookup's recipe, retune policy, explicit settings and build; ``report`` the hardware
    probed for the answer. Queues the job (``retune.request``, imported only here) when nothing in
    ``_refused_tune`` rules it out. Never raises: anything that goes wrong is ``FAILED_BEFORE``.
    """
    recipe, retune, explicit, build = asked
    refused = _refused_tune(answer, recipe, retune, explicit, report.device)
    if refused is not None:
        answer.tune = refused[0]
        return refused[1]
    assert recipe is not None  # noqa: S101 - narrowing for mypy: no recipe was refused above
    from localmodels import retune as retune_module  # noqa: PLC0415 - only a lookup that queues a tune

    try:
        outcome = retune_module.request(
            answer.model,
            answer.kind,
            hardware=hardware_key(report),
            build=build,
            plan=plan,
            stale=answer.stale,
            recipe=recipe,
        )
    except Exception as exc:  # noqa: BLE001 - the re-tune must never fail a model load
        answer.tune, answer.tune_detail = TuneState.FAILED_BEFORE, f"{type(exc).__name__}: {exc}"
        return f"the re-tune failed: {answer.tune_detail}"
    answer.tune, answer.tune_detail = outcome.state, outcome.detail
    return outcome.note


def _from_record(
    answer: SettingsAnswer,
    record: SettingsRecord,
    path: Path,
    compared: tuple[str | None, Mapping[str, str] | None],
    notes: list[str],
) -> None:
    answer.settings, answer.source = record.chosen, SettingsSource.RECORD
    answer.record = RecordRef(
        path=str(path),
        measured_at=record.measured_at,
        hardware=record.hardware,
        candidate=record.candidate,
        quality=record.quality,
        stack=record.stack,
        build=record.build,
        recipe_project=record.recipe_project,
        changed_from=record.previous,
    )
    build, libraries = compared
    answer.stale = _stale_lines(record, fingerprint(), build, libraries)
    name = f"{answer.model.repository} ({answer.kind.value})"
    notes.extend(
        f"{name}: {lib} not recorded; not compared."
        for lib in (libraries or {})
        if lib not in record.libraries
    )
    made = f"{record.candidate or _describe(record.chosen)} on {record.measured_at:%Y-%m-%d}"
    if answer.stale:
        answer.status, answer.tune_plan = SettingsStatus.STALE, TunePlan.CONFIRM
        notes.append(f"{name}: settings {made} stay in use but are stale ({'; '.join(answer.stale)}).")
    else:
        answer.status = SettingsStatus.TUNED
        notes.append(f"{name}: tuned settings {made}.")


def _from_rule(answer: SettingsAnswer, notes: list[str], hardware: str) -> None:
    """No record: the rule of the kind if it names this model, else the caller's default. A tune is wanted."""
    name = f"{answer.model.repository} ({answer.kind.value})"
    answer.tune_plan = TunePlan.FULL
    answer.rule = _rule(answer.kind, answer.model.repository)
    if answer.rule is None:
        notes.append(untuned_message(name, hardware))
    elif answer.rule.applied:
        answer.status, answer.source = SettingsStatus.RULE, SettingsSource.RULE
        answer.settings = answer.rule.values
        notes.append(
            f"{name}: rule {answer.rule.id}, measured on this model; not tuned for this file and machine."
        )
    else:
        answer.status = SettingsStatus.RULE_OTHER_MODEL
        measured = ", ".join(answer.rule.models) or "no named model"
        notes.append(
            f"{name}: rule {answer.rule.id} was measured on {measured}, not on this model; its values "
            f"({_describe(answer.rule.values)}) are not applied."
        )


def stale_records(host: Mapping[str, str], current_build: str | None) -> tuple[list[str], list[str], int]:
    """Every stored record that is stale on this host: (one line each, unusable-file problems, records read).

    A record that names a llama.cpp build is compared with ``current_build``.
    """
    stale, problems, total = [], [], 0
    for path in sorted((settings_dir() / "records").glob("*/*.json")):
        store = read_store(path)
        if store.problem:
            problems.append(store.problem)
        for record in store.records:
            total += 1
            changed = _stale_lines(record, host, current_build if record.build else None)
            if changed:
                who = f"{record.model.repository} {record.kind.value} ({record.hardware})"
                stale.append(f"{who}: {', '.join(changed)} [{path}]")
    return stale, problems, total


class AdoptError(ValueError):
    """A record that must not become current, or a store file that must not be written."""


def _label(record: SettingsRecord) -> str:
    return record.candidate or _describe(record.chosen)


def _to_history(path: Path, kind: WorkloadKind, files: list[bytes]) -> None:
    """Keep ``files`` (one record each, as stored) in the history of ``path``; the newest few stay."""
    folder = settings_dir() / "history" / path.parent.name / kind.value
    folder.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    for index, content in enumerate(files):
        (folder / f"{stamp}-{index}.json").write_bytes(content)
    for old in sorted(folder.glob("*.json"))[:-HISTORY_KEPT]:
        old.unlink()


def _unusable_measurement(record: SettingsRecord) -> str:
    """Why ``record``'s own measurements do not support its candidate; "" if they do or there are none."""
    if not record.measurements:
        return ""
    named = [m for m in record.measurements if m.candidate.name == record.candidate]
    if not named or not all(_eligible(m) for m in named):
        return f"the measurements show {record.candidate!r} as missing, failed, not fitting or not agreeing"
    return ""


class _Split(NamedTuple):
    """A store's entries sorted for an adopt: the ones that stay, the ones that go to history."""

    entries: list[Any]  # raw entries that stay; the new record goes at ``slot`` (None: at the end)
    slot: int | None
    history: list[bytes]
    newest: SettingsRecord | None  # of this hardware and engine, any build


def _raw(item: Any) -> bytes:
    return json.dumps(item, indent=2).encode() + b"\n"


def _split(items: list[Any], record: SettingsRecord) -> _Split:
    """Sort ``items`` for adopting ``record``: the entry of its slot is replaced, every other one stays."""
    slot_key = (record.hardware, record.build, record.engine)
    entries: list[Any] = []
    history: list[bytes] = []
    here: list[tuple[SettingsRecord, Any]] = []  # this hardware and engine, any build
    slot = None
    for item in items:
        try:
            old = SettingsRecord.model_validate(item)
        except ValidationError:
            entries.append(item)  # another version's entry: kept as it is
            continue
        if old.hardware == record.hardware and old.engine in (None, record.engine):
            here.append((old, item))
        if (old.hardware, old.build, old.engine) != slot_key:
            entries.append(item)
            continue
        history.append(_raw(item))
        if slot is None:
            slot = len(entries)
            entries.append(None)
    newest = max(here, key=lambda pair: pair[0].measured_at.timestamp(), default=None)
    if newest is not None and (newest[0].hardware, newest[0].build, newest[0].engine) != slot_key:
        history.append(_raw(newest[1]))  # another build's entry stays in the store, and is kept here too
    return _Split(entries, slot, history, newest[0] if newest else None)


def adopt(record: SettingsRecord) -> SettingsRecord:
    """Make ``record`` the current one for its hardware, build and engine; what was stored comes back.

    The entry it replaces goes to ``history/``, as does the newest entry of the same hardware and engine
    (any build) if that is another one; ``previous`` is that newest entry's candidate when the choice changed,
    else the ``previous`` it already carried. Every other entry stays exactly as it was, including one this
    version cannot read (a newer localmodels wrote it). An unreadable file is copied byte for byte to
    ``history/`` and replaced. An exclusive lock on the file means two adopts lose nothing.

    Raises:
        AdoptError: the record has no quality check, its chosen candidate's agreement did not pass, its own
            measurements (if it carries any) do not support its candidate, or the file was written by a
            newer localmodels.
        OSError: the record folder or its history cannot be written (a read-only folder, a full disk).
    """
    change = os.environ.get(TUNE_CHANGE_ENV)
    try:
        stored = _adopt(record)
    except AdoptError as exc:
        if change:
            _tell_retune(change, record, str(exc))
        raise
    if change:
        _tell_retune(change, stored, None)
    return stored


def _tell_retune(change: str, record: SettingsRecord, refusal: str | None) -> None:
    """In a tune job (``LOCALMODELS_TUNE_CHANGE``): tell the change's pending file how the adopt went.

    The file acts only if it was made for this record's model and kind.
    """
    from localmodels import retune  # noqa: PLC0415 - only in a tune job

    adopted = f"{_label(record)} measured {record.measured_at.isoformat()}"
    retune.job_finished(change, key=record.model.key, kind=record.kind, refusal=refusal, record=adopted)


def _adopt(record: SettingsRecord) -> SettingsRecord:
    who = f"{record.model.repository} ({record.kind.value})"
    if not record.quality:
        msg = f"{who}: a record with no quality check adopts nothing"
        raise AdoptError(msg)
    if record.agreement is None or not record.agreement.passed:
        msg = f"{who}: the chosen candidate's agreement did not pass"
        raise AdoptError(msg)
    if problem := _unusable_measurement(record):
        msg = f"{who}: {problem}"
        raise AdoptError(msg)
    path = record_path(record.model, record.kind)
    with _locked(path):
        items, state, problem = _load_items(path)
        if state is StoreState.NEWER_SCHEMA:
            msg = f"{problem}; it is not overwritten"
            raise AdoptError(msg)
        entries, slot, history, newest = _split(items, record)
        if state is StoreState.UNREADABLE:
            history.append(path.read_bytes())
        changed = newest is not None and newest.chosen != record.chosen
        previous = None
        if newest is not None:
            previous = _label(newest) if changed else newest.previous
        stored = record.model_copy(update={"previous": previous, "adopted_at": datetime.now(UTC)})
        entry = stored.model_dump(mode="json")
        if slot is None:
            entries.append(entry)
        else:
            entries[slot] = entry
        if history:
            _to_history(path, record.kind, history)
        write_text_atomically(
            path, json.dumps({"schema": SCHEMA_VERSION, "records": entries}, indent=2) + "\n"
        )
    log.log(
        logging.WARNING if changed else logging.INFO,
        "settings for %s (%s) %s: %s%s, agreement %.3f (%s)",
        record.model.repository,
        record.kind.value,
        "changed" if changed else "adopted",
        f"{_label(newest)} -> " if changed and newest else "",
        _label(stored),
        record.agreement.score,
        record.quality,
    )
    return stored


def _eligible(measurement: TuneMeasurement | None) -> TuneMeasurement | None:
    """``measurement`` if its candidate ran, fits and agreed with the reference; else None."""
    if (
        measurement is not None
        and measurement.error is None
        and measurement.fits
        and measurement.agreement is not None
        and measurement.agreement.passed
        and measurement.infer_s_median is not None
    ):
        return measurement
    return None


def _timing_spread(reference: TuneMeasurement) -> float:
    """How far the reference's timed passes differ, as a fraction of the quickest; 0 with fewer than two."""
    if len(reference.pass_s) < MIN_PASSES or min(reference.pass_s) <= 0:
        return 0.0
    return (max(reference.pass_s) - min(reference.pass_s)) / min(reference.pass_s)


def record_from_tune(
    tune: TuneRecord,
    *,
    model: ModelIdentity,
    kind: WorkloadKind,
    libraries: Mapping[str, str] | None = None,
    recipe_project: str | None = None,
    tune_job: str | None = None,
    legacy_path: Path | None = None,
) -> SettingsRecord:
    """The store record for a PyTorch ``tune``, after the speed margin of design 5.5; adopt it with ``adopt``.

    The incumbent is the candidate the store holds for this hardware (else ``legacy_path``'s, as ``lookup``
    reads it) if this tune measured it and it is eligible, else the reference. A candidate other than the
    reference or the incumbent is chosen only if its median inference time beats the incumbent's by more than
    the larger of ``SPEED_MARGIN`` and the reference's timing spread (whole passes, as a fraction of the
    quickest); otherwise the incumbent stays, so timing noise cannot change the settings. ``libraries`` are
    the caller's versions, added for names the tune's server did not report.

    Raises:
        AdoptError: the tune's chosen candidate is not an eligible measurement (it failed, does not fit or
            did not agree).
    """
    by_name = {m.candidate.name: m for m in tune.measurements}
    chosen = next((m for m in tune.measurements if m.candidate == tune.chosen), None)
    if _eligible(chosen) is None or chosen is None:
        msg = f"{model.repository} ({kind.value}): the tune's choice {tune.chosen.name!r} is not eligible"
        raise AdoptError(msg)
    records = read_store(record_path(model, kind)).records
    stored = _pick(records, tune.hardware, None, Engine.PYTORCH)
    if stored is None and legacy_path is not None:
        stored = _pick(_read_legacy(legacy_path, model, kind).records, tune.hardware, None, Engine.PYTORCH)
    incumbent = _eligible(by_name.get(stored.candidate) if stored else None) or _eligible(
        by_name.get(tune.reference)
    )
    if incumbent is not None and chosen.candidate.name not in (tune.reference, incumbent.candidate.name):
        assert chosen.infer_s_median is not None  # noqa: S101 - narrowing for mypy; the tune chose an eligible one
        assert incumbent.infer_s_median is not None  # noqa: S101
        faster = incumbent.infer_s_median / chosen.infer_s_median - 1
        reference = by_name[tune.reference]
        if len(reference.pass_s) < MIN_PASSES:
            log.info(
                "%s (%s): the reference was timed once, so its spread is not measured; only %.0f %% applies",
                model.repository,
                kind.value,
                SPEED_MARGIN * 100,
            )
        if faster <= max(SPEED_MARGIN, _timing_spread(reference)):
            chosen = incumbent
    return SettingsRecord(
        model=model,
        kind=kind,
        hardware=tune.hardware,
        measured_at=tune.measured_at,
        chosen=Settings(dtype=chosen.candidate.dtype, options=chosen.candidate.options),
        candidate=chosen.candidate.name,
        quality=tune.quality,
        stack=tune.stack,
        engine=Engine.PYTORCH,
        libraries={**(libraries or {}), **tune.libraries},
        recipe_project=recipe_project,
        agreement=chosen.agreement,
        reference=tune.reference,
        measurements=tune.measurements,
        tune_job=tune_job,
    )


def cmd_settings_show(args: argparse.Namespace) -> int:
    """`localmodels settings show`: print one lookup answer; nothing is loaded and nothing is queued."""
    kind, engine = WorkloadKind(args.kind), Engine(args.engine)
    try:
        model = (
            ModelIdentity.from_hub_files(args.model, args.revision or "", args.file)
            if args.file
            else ModelIdentity(repository=args.model, revision=args.revision or "")
        )
    except ValueError as exc:  # no --revision or --file, or a file whose blob is not a hash
        sys.stderr.write(f"cannot identify the model (give --revision or --file): {exc}\n")
        return 1
    answer = lookup(
        model, kind, engine=engine, build=args.build, legacy_path=Path(args.legacy) if args.legacy else None
    )
    sys.stdout.writelines(f"{line}\n" for line in answer.lines())
    for path, other in stored_identities(model.repository, kind):
        if other.key != model.key:
            names = ", ".join(f.path for f in other.files) or other.revision or other.unpinned
            sys.stdout.write(
                f"note: records exist for another identity of this repository: {names} [{path}]\n"
            )
    return 0


def cmd_settings_check(args: argparse.Namespace) -> int:
    """`localmodels settings check`: list the stored records that are stale on this host; exit 1 if any."""
    stale, problems, total = stale_records(fingerprint(), _current_build(args.home))
    out = sys.stdout
    out.writelines(f"{line}\n" for line in stale)
    out.writelines(f"unusable: {line}\n" for line in problems)
    out.write(
        f"{len(stale)} of {total} records stale.\n" if stale else f"No record is stale ({total} read).\n"
    )
    return EXIT_STALE if stale else 0
