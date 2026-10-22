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

import hashlib
import json
import logging
import os
import re
import sys
from datetime import datetime  # noqa: TC003 - pydantic needs it at runtime
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, NamedTuple, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from localmodels.enums import Precision, WorkloadKind
from localmodels.hardware import inspect_hardware
from localmodels.pinned import PinnedFile, snapshot_dir
from localmodels.pull import _default_hub_cache, _repo_blob_name
from localmodels.rules import EXIT_STALE, StampKey, rules_text, stamps
from localmodels.serving.tune import TuneRecord, _read_records, hardware_key, untuned_message
from localmodels.stack import UNKNOWN, Engine, fingerprint, stale_components

if TYPE_CHECKING:
    import argparse
    from collections.abc import Mapping

STATE_DIR_ENV: Final = "LOCALMODELS_STATE_DIR"
DEFAULT_STATE_DIR: Final = Path("~/.local/state/localmodels")
SCHEMA_VERSION: Final = 1
NO_STACK_STAMP: Final = "recorded without a stack stamp"
_KEY_CHARS: Final = 16
_SLUG_UNSAFE = re.compile(r"[^A-Za-z0-9._-]")
_ANY_MODEL: Final = "any"  # a rules.md stamp's model= for a training method

log = logging.getLogger(__name__)


class SpecType(StrEnum):
    """llama.cpp speculative decoding (``--spec-type``)."""

    NONE = "none"
    DRAFT_MTP = "draft-mtp"
    DRAFT_SIMPLE = "draft-simple"
    NGRAM_MOD = "ngram-mod"


class SettingsStatus(StrEnum):
    """How far an answer can be trusted."""

    TUNED = "tuned"  # a record made on this machine and the current stack
    STALE = "stale"  # a record exists but a stamped part of the stack changed since
    RULE = "rule"  # no record; the rule was measured on this model, but not on this file and machine
    RULE_OTHER_MODEL = "rule-other-model"  # no record; the rule was measured on other models only
    UNTUNED = "untuned"  # nothing measured applies


class SettingsSource(StrEnum):
    """Where the settings of an answer came from."""

    RECORD = "record"
    RULE = "rule"
    DEFAULT = "default"


class TuneState(StrEnum):
    """What the lookup did about a tune."""

    NOT_NEEDED = "not-needed"
    NOT_AVAILABLE = (
        "not-available"  # a tune is wanted and a recipe is given, but automatic re-tune is not built
    )
    NO_RECIPE = "no-recipe"  # a tune is wanted and the caller gave no recipe


class TunePlan(StrEnum):
    """The tune an answer calls for: every candidate (first use) or the reference and the current choice."""

    FULL = "full"
    CONFIRM = "confirm"


class StoreState(StrEnum):
    """The state of the files the answer was read from."""

    OK = "ok"
    UNREADABLE = "unreadable"
    NEWER_SCHEMA = "newer-schema"


class Settings(BaseModel):
    """What to load a model with. A missing field is "no opinion": the caller keeps its own."""

    model_config = ConfigDict(frozen=True)

    dtype: Precision | None = None
    options: dict[str, Any] = Field(
        default_factory=dict
    )  # free-form: token_budget, matmul_precision, batch_size
    n_ubatch: int | None = None
    spec: SpecType | None = None
    spec_draft_file: str | None = None
    model_file: str | None = None  # the quantised file to load, by SHA-256, when the tune chose another


class ModelIdentity(BaseModel):
    """What a model is: its repository plus the files loaded, so two quantisations never share a record.

    A GGUF model is its file set (names and SHA-256 per file, from the Hub cache's blob names, never hashed
    on demand). A PyTorch snapshot loaded whole is its repository and revision. ``unpinned`` marks the
    fallback for a bare path with no content identity.
    """

    model_config = ConfigDict(frozen=True)

    repository: str
    revision: str = ""
    files: tuple[PinnedFile, ...] = ()
    unpinned: str = ""

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


class RuleRef(BaseModel):
    """The rule of the kind, the models it was measured on, and whether its values were applied."""

    id: str
    models: list[str]
    applied: bool
    values: Settings


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

_TUNE_REASON: Final = {
    TuneState.NOT_AVAILABLE: "automatic re-tune is not built yet",
    TuneState.NO_RECIPE: "the caller gave no recipe",
}

#: Sentences already written this process, by (model, kind): ``announce`` speaks once.
_announced: set[tuple[str, WorkloadKind]] = set()


def _describe(settings: Settings) -> str:
    fields = settings.model_dump(mode="json", exclude_defaults=True)
    return ", ".join(f"{name}={value}" for name, value in fields.items()) or "no opinion"


class SettingsAnswer(BaseModel):
    """What to load a model with, how far to trust it, and where that came from."""

    model: ModelIdentity
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

    def announce(self, logger: logging.Logger) -> None:
        """Write the notes once per process per model and kind: info when tuned, else a warning."""
        key = (self.model.key, self.kind)
        if key in _announced:
            return
        _announced.add(key)
        tuned = self.status is SettingsStatus.TUNED and self.store is StoreState.OK
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
        out.append(f"tune: {self.tune.value}{plan}")
        out.append(f"store: {self.store.value}")
        out.extend(f"note: {note}" for note in self.notes)
        return out


def settings_dir() -> Path:
    """The store: ``$LOCALMODELS_STATE_DIR``/settings, else ``~/.local/state/localmodels/settings``."""
    return Path(os.environ.get(STATE_DIR_ENV) or DEFAULT_STATE_DIR).expanduser() / "settings"


def record_path(model: ModelIdentity, kind: WorkloadKind) -> Path:
    """The file holding the records of one model and kind: ``records/<repository>--<key>/<kind>.json``."""
    folder = f"{_SLUG_UNSAFE.sub('_', model.repository.replace('/', '--'))}--{model.key}"
    return settings_dir() / "records" / folder / f"{kind.value}.json"


class StoreRead(NamedTuple):
    """The records of one store file that can be used, and what was wrong with the rest."""

    records: list[SettingsRecord]
    state: StoreState
    problem: str  # "" when the file is fine or absent


def _first_error(exc: ValidationError) -> str:
    error = exc.errors()[0]
    return f"{'.'.join(map(str, error['loc']))}: {error['msg']}"


def read_store(path: Path) -> StoreRead:
    """Read one store file; never raises. A bad file, a newer schema or an invalid record is absent."""
    if not path.is_file():
        return StoreRead([], StoreState.OK, "")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        schema, items = payload["schema"], payload["records"]
        if not isinstance(schema, int) or not isinstance(items, list):
            msg = "schema is not a number or records is not a list"
            raise TypeError(msg)  # noqa: TRY301 - one handler for every shape of a bad file
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return StoreRead([], StoreState.UNREADABLE, f"{path}: {exc!r}")
    if schema > SCHEMA_VERSION:
        return StoreRead(
            [], StoreState.NEWER_SCHEMA, f"{path}: schema {schema}, written by a newer localmodels"
        )
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
    if not path.is_file():
        return StoreRead([], StoreState.OK, "")
    try:
        legacy: list[TuneRecord] = _read_records(path)
    except (OSError, ValueError, KeyError, TypeError) as exc:
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


def _stale_lines(record: SettingsRecord, host: Mapping[str, str], build: str | None) -> list[str]:
    """What changed since ``record`` was made: stamped host components, and the llama.cpp ``build`` if named.

    localmodels' own version is recorded, not compared (``stale_components``).
    """
    lines = [NO_STACK_STAMP] if record.stack is None else stale_components(record.stack, host)
    if build is not None and record.build != build:
        lines.insert(0, f"{StampKey.LLAMA_CPP}: {record.build or UNKNOWN} -> {build}")
    return lines


def _pick(records: list[SettingsRecord], hardware: str, build: str | None) -> SettingsRecord | None:
    """This machine's newest record, preferring the one made on the caller's llama.cpp ``build``."""
    mine = [r for r in records if r.hardware == hardware]
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


def lookup(
    model: ModelIdentity,
    kind: WorkloadKind,
    *,
    engine: Engine = Engine.PYTORCH,
    build: str | None = None,
    explicit: Mapping[str, Any] | None = None,
    default: Settings | None = None,
    recipe: object | None = None,
    legacy_path: Path | None = None,
) -> SettingsAnswer:
    """What to load ``model`` with for ``kind`` work on this machine; it never fails a load.

    ``build`` is the llama.cpp build the caller will run (``Engine.LLAMA_CPP``): a record made on another
    build is stale. ``explicit`` holds settings the caller has decided; they replace the answer's for those
    fields and are listed in ``overridden``, so the status still shows that a measured alternative exists.
    ``default`` is what the caller runs when nothing measured applies. ``recipe`` is how a tune is run (the
    automatic re-tune is not built: a tune the answer calls for reports ``NOT_AVAILABLE``). ``legacy_path`` is
    the caller's own tune file, read when the store has no record. A record made through llama-server is never
    applied to ``Engine.LLAMA_CPP_PYTHON``, which has no tune path yet. ``ValueError`` for an ``explicit`` key
    that is not a settings field (a caller's bug, not a store problem).
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
    answer = SettingsAnswer(
        model=model, kind=kind, settings=default, status=SettingsStatus.UNTUNED, source=SettingsSource.DEFAULT
    )

    if engine is Engine.LLAMA_CPP_PYTHON:
        answer.tune = TuneState.NOT_AVAILABLE
        notes.append(f"{name}: llama-cpp-python has no tune path yet; it runs the caller's default settings.")
    else:
        hardware = hardware_key(inspect_hardware())
        path = record_path(model, kind)
        store = read_store(path)
        record = _pick(store.records, hardware, build)
        if record is None and legacy_path is not None:
            legacy = _read_legacy(legacy_path, model, kind)
            record, path = _pick(legacy.records, hardware, build), legacy_path
            store = store if store.state is not StoreState.OK else legacy
        answer.store = store.state
        if store.problem:
            notes.append(f"Settings records for {name} unusable, treated as absent: {store.problem}")
        if record is not None:
            _from_record(answer, record, path, build if engine is Engine.LLAMA_CPP else None, notes)
        else:
            _from_rule(answer, notes, hardware)
        answer.tune = _tune_state(answer.tune_plan, recipe)
        if answer.tune_plan:
            notes.append(f"A {answer.tune_plan.value} tune is wanted; {_TUNE_REASON[answer.tune]}.")

    if explicit:
        answer.settings = Settings.model_validate({**answer.settings.model_dump(), **explicit})
        answer.overridden = list(explicit)
        notes.append(
            f"{name}: explicit settings replace the measured ones for {', '.join(answer.overridden)}."
        )
    answer.notes = notes
    return answer


def _tune_state(plan: TunePlan | None, recipe: object | None) -> TuneState:
    if plan is None:
        return TuneState.NOT_NEEDED
    return TuneState.NOT_AVAILABLE if recipe is not None else TuneState.NO_RECIPE


def _from_record(
    answer: SettingsAnswer, record: SettingsRecord, path: Path, build: str | None, notes: list[str]
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
    )
    answer.stale = _stale_lines(record, fingerprint(), build)
    name = f"{answer.model.repository} ({answer.kind.value})"
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


def cmd_settings_show(args: argparse.Namespace) -> int:
    """`localmodels settings show`: print one lookup answer; nothing is loaded and nothing is queued."""
    kind, engine = WorkloadKind(args.kind), Engine(args.engine)
    try:
        model = (
            ModelIdentity.from_hub_files(args.model, args.revision or "", args.file)
            if args.file
            else ModelIdentity(repository=args.model, revision=args.revision or "")
        )
    except ValueError as exc:
        sys.stderr.write(f"cannot identify the model files: {exc}\n")
        return 1
    answer = lookup(
        model, kind, engine=engine, build=args.build, legacy_path=Path(args.legacy) if args.legacy else None
    )
    sys.stdout.writelines(f"{line}\n" for line in answer.lines())
    return 0


def cmd_settings_check(args: argparse.Namespace) -> int:
    """`localmodels settings check`: list the stored records that are stale on this host; exit 1 if any."""
    from localmodels.llama_builds import current_server, llama_home  # noqa: PLC0415 - read only for `check`
    from localmodels.stack import llama_cpp_build  # noqa: PLC0415

    build = llama_cpp_build(current_server(llama_home(args.home)))
    stale, problems, total = stale_records(fingerprint(), build)
    out = sys.stdout
    out.writelines(f"{line}\n" for line in stale)
    out.writelines(f"unusable: {line}\n" for line in problems)
    out.write(
        f"{len(stale)} of {total} records stale.\n" if stale else f"No record is stale ({total} read).\n"
    )
    return EXIT_STALE if stale else 0
