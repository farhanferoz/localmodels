"""The measured rules by workload kind, shipped with the package as ``rules.md``.

``localmodels rules show K2`` prints one kind's section, then the newest re-measurement record of that kind
and its lines from the newest monthly digest (``localmodels digest``) when they exist.
``localmodels rules check`` compares each section's ``stamp:`` line with this host's stack and the current
llama.cpp build. A changed stamp is answered by a record that holds on this stack (``localmodels refresh
measure`` writes them, one file per kind under ``$LOCALMODELS_STATE_DIR/rules/``), by a stated reason why
localmodels cannot re-measure the kind, or not at all; it exits 1 only for the last. The tracked ``rules.md``
is never rewritten by a job.
"""

from __future__ import annotations

import os
import re
import sys
from enum import StrEnum
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING, Final, NamedTuple

from pydantic import BaseModel, Field, ValidationError

from .atomic import write_text_atomically
from .digest import DEFAULT_OUT_DIR as DEFAULT_DIGEST_DIR
from .digest import LINE_RE
from .enums import WorkloadKind

if TYPE_CHECKING:
    import argparse
    from collections.abc import Mapping, Sequence

RULES_FILE: Final = "rules.md"
_SECTION = re.compile(r"^## (K\d+)\b", re.MULTILINE)
_DIGEST_NAME = re.compile(r"^\d{4}-\d{2}-\d{2}\.md$")
_STAMP_LINE = re.compile(r"^- stamp: (?P<pairs>.+)$")
EXIT_STALE: Final = 1
RECORDS_DIRNAME: Final = "rules"
#: Design D14: a figure more than this far from the previous record's is listed as moved.
MOVED_FRACTION: Final = 0.10
DATE_LENGTH: Final = 10  # "YYYY-MM-DD", the front of an ISO 8601 time

_REASON_PER_MODEL_TUNE: Final = (
    "re-measured per model by its project's own tune, which runs on the model's first load after the change "
    "(D2); localmodels runs no PyTorch"
)
_REASON_NEEDS_RECIPE: Final = (
    "the recipe runs another project's model code in that project's environment, and localmodels runs no "
    "PyTorch; a self-contained recipe is needed first"
)
#: The stamped kinds that ``localmodels refresh`` cannot re-measure, with the reason `rules check` prints.
NOT_MEASURED_HERE: Final[dict[str, str]] = {
    WorkloadKind.K5.value: _REASON_PER_MODEL_TUNE,
    WorkloadKind.K6.value: _REASON_PER_MODEL_TUNE,
    WorkloadKind.K8.value: _REASON_NEEDS_RECIPE,
    WorkloadKind.K9.value: _REASON_NEEDS_RECIPE,
}


class RuleState(StrEnum):
    """What answers a stamped rule whose stamp no longer matches this host."""

    REMEASURED = "re-measured"
    DOES_NOT_HOLD = "does not hold"
    REASON = "not re-measured here"
    NEEDS_REMEASURE = "needs re-measuring"


class RuleRecord(BaseModel):
    """One kind's newest re-measurement on this host (``$LOCALMODELS_STATE_DIR/rules/<kind>.json``)."""

    kind: str
    measured_at: str
    slurm_job: str = ""
    run_dir: str = ""
    stamp: dict[str, str]  # the host fingerprint, ``llama_cpp`` (the build measured) and ``model``
    figures: dict[str, float]
    holds: bool
    detail: str = ""
    moved: list[str] = Field(default_factory=list)  # figures more than ``MOVED_FRACTION`` from ``previous``
    previous: dict[str, float] = Field(default_factory=dict)  # the figures of the record this replaced


class StampKey(StrEnum):
    """The entries of a stamp line that are not the host's ``stack.Component`` names."""

    LLAMA_CPP = "llama_cpp"
    TORCH = "torch"
    MODEL = "model"  # repository ids the rule was measured on, comma-separated; "any" for a training method


def rules_text() -> str:
    """The whole rules document."""
    return resources.files("localmodels").joinpath(RULES_FILE).read_text(encoding="utf-8")


def section(kind: WorkloadKind, text: str | None = None) -> str:
    """One kind's section of the rules document, from its heading to the next."""
    text = rules_text() if text is None else text
    starts = [(m.group(1), m.start()) for m in _SECTION.finditer(text)]
    for i, (name, start) in enumerate(starts):
        if name == kind.value:
            end = starts[i + 1][1] if i + 1 < len(starts) else len(text)
            return text[start:end].rstrip() + "\n"
    msg = f"{RULES_FILE} has no section for {kind.value}"
    raise LookupError(msg)


def digest_lines(kind: WorkloadKind, digests_dir: Path = DEFAULT_DIGEST_DIR) -> tuple[str, list[str]] | None:
    """The newest digest's date and its finding lines tagged with ``kind``; None when there is no digest."""
    if not digests_dir.is_dir():
        return None
    notes = sorted(p for p in digests_dir.glob("*.md") if _DIGEST_NAME.match(p.name))
    if not notes:
        return None
    newest = notes[-1]
    lines = []
    for line in newest.read_text(encoding="utf-8").splitlines():
        match = LINE_RE.match(line)
        if match and kind.value in match.group("kinds").split(","):
            lines.append(line)
    return newest.stem, lines


def cmd_rules_show(args: argparse.Namespace) -> int:
    """Print one kind's rules, then its lines from the newest digest."""
    kind = WorkloadKind(args.kind)
    out = sys.stdout
    out.write(section(kind))
    record = read_record(kind.value)
    if record is not None:
        stamp = record.stamp
        out.write(
            f"\nRe-measured {record.measured_at[:DATE_LENGTH]} (slurm {record.slurm_job}, "
            f"llama.cpp {stamp.get(StampKey.LLAMA_CPP, 'unknown')}, "
            f"driver {stamp.get('driver', 'unknown')}): "
            f"{figures_text(record.figures)}\n"
        )
    digest = digest_lines(kind, Path(args.digests))
    if digest is not None:
        date, lines = digest
        out.write(f"\nDigest {date}:\n" if lines else f"\nDigest {date}: nothing for {kind.value}.\n")
        out.writelines(f"{line}\n" for line in lines)
    return 0


def stamps(text: str) -> list[tuple[str, dict[str, str]]]:
    """Each ``- stamp: key=value ...`` line of the rules document with the kind (``K1``) of its section."""
    found: list[tuple[str, dict[str, str]]] = []
    kind = ""
    for line in text.splitlines():
        heading = _SECTION.match(line)
        if heading:
            kind = heading.group(1)
        stamp = _STAMP_LINE.match(line)
        if stamp:
            found.append(
                (kind, dict(pair.split("=", 1) for pair in stamp.group("pairs").split() if "=" in pair))
            )
    return found


def _changes(stamp: Mapping[str, str], current: Mapping[str, str]) -> list[str]:
    """``driver: was -> now`` per component of ``stamp`` that differs from ``current`` (llama_cpp first)."""
    from .stack import UNKNOWN, stale_components  # noqa: PLC0415 - rules show does not need the stack

    changed = stale_components(stamp, current)
    build = stamp.get(StampKey.LLAMA_CPP)
    now = current.get(StampKey.LLAMA_CPP, UNKNOWN)
    if build is not None and build != now:
        changed.insert(0, f"{StampKey.LLAMA_CPP}: {build} -> {now}")
    return changed


def stale_rules(text: str, current: Mapping[str, str]) -> list[str]:
    """One ``K1: driver: was -> now, llama_cpp: was -> now`` line per stamp whose components changed.

    ``current`` maps the host fingerprint's names and ``llama_cpp`` to what they are now. ``torch`` is not
    compared (localmodels has no PyTorch); a component the stamp does not name is not compared.
    """
    lines = []
    for kind, stamp in stamps(text):
        if changed := _changes(stamp, current):
            lines.append(f"{kind}: {', '.join(changed)}")
    return lines


def records_dir() -> Path:
    """Where the re-measurement records are kept: ``rules/`` under the state directory."""
    from .settings import DEFAULT_STATE_DIR, STATE_DIR_ENV  # noqa: PLC0415 - settings imports this module

    return Path(os.environ.get(STATE_DIR_ENV) or DEFAULT_STATE_DIR).expanduser() / RECORDS_DIRNAME


def read_record(kind: str) -> RuleRecord | None:
    """The newest record of ``kind``; None when there is none or it cannot be read."""
    try:
        return RuleRecord.model_validate_json((records_dir() / f"{kind}.json").read_text(encoding="utf-8"))
    except (OSError, ValidationError):
        return None


def _moved(figures: Mapping[str, float], previous: Mapping[str, float]) -> list[str]:
    """The figures that are more than ``MOVED_FRACTION`` away from the same figure of ``previous``."""
    return [
        name
        for name, value in figures.items()
        if name in previous and abs(value - previous[name]) > MOVED_FRACTION * abs(previous[name])
    ]


def write_record(record: RuleRecord) -> RuleRecord:
    """Store ``record`` as its kind's newest; ``previous`` and ``moved`` come from the one it replaces."""
    before = read_record(record.kind)
    if before is not None:
        record = record.model_copy(
            update={"previous": before.figures, "moved": _moved(record.figures, before.figures)}
        )
    write_text_atomically(records_dir() / f"{record.kind}.json", record.model_dump_json(indent=2) + "\n")
    return record


def figures_text(figures: Mapping[str, float]) -> str:
    """``name=value`` pairs, four significant digits."""
    return ", ".join(f"{name}={value:.4g}" for name, value in figures.items())


class Assessment(NamedTuple):
    """One stamped rule whose stamp differs from this host: how it is answered, and the line to print."""

    kind: str
    state: RuleState
    line: str


def assess(text: str, current: Mapping[str, str]) -> list[Assessment]:
    """Answer each stamp that no longer matches ``current``: a holding record, a reason, or nothing."""
    found = []
    for kind, stamp in stamps(text):
        if not (changed := _changes(stamp, current)):
            continue
        record = read_record(kind)
        if record is not None and not _changes(record.stamp, current):
            date = record.measured_at[:DATE_LENGTH]
            if record.holds:
                moved = f"; moved: {', '.join(record.moved)}" if record.moved else ""
                line = (
                    f"{kind}: re-measured {date} on this stack (slurm {record.slurm_job}): "
                    f"{figures_text(record.figures)}{moved}"
                )
                found.append(Assessment(kind, RuleState.REMEASURED, line))
            else:
                line = f"{kind}: re-measured {date}: does not hold: {record.detail}"
                found.append(Assessment(kind, RuleState.DOES_NOT_HOLD, line))
        elif kind in NOT_MEASURED_HERE:
            line = f"{kind}: {', '.join(changed)}; not re-measured here: {NOT_MEASURED_HERE[kind]}"
            found.append(Assessment(kind, RuleState.REASON, line))
        else:
            found.append(Assessment(kind, RuleState.NEEDS_REMEASURE, f"{kind}: {', '.join(changed)}"))
    return found


def host_stamp(binary: Path) -> dict[str, str]:
    """This host's stack and the llama.cpp commit ``binary`` was built from, as ``assess`` compares them."""
    from . import stack  # noqa: PLC0415 - rules show does not need the stack

    return {**stack.fingerprint(), StampKey.LLAMA_CPP.value: stack.llama_cpp_build(binary)}


def summary_line(found: Sequence[Assessment], total: int) -> str:
    """The last line of `rules check`: how many changed and how each is answered."""
    count = {state: sum(a.state is state for a in found) for state in RuleState}
    answered = count[RuleState.REMEASURED]
    return (
        f"{len(found)} of {total} stamped rules changed: {answered} re-measured, "
        f"{count[RuleState.REASON]} with a reason, "
        f"{count[RuleState.DOES_NOT_HOLD] + count[RuleState.NEEDS_REMEASURE]} need re-measuring.\n"
    )


def cmd_rules_check(args: argparse.Namespace) -> int:
    """Compare every stamp with this host; exit 1 when a changed one has no holding record and no reason."""
    from .llama_builds import current_server, llama_home  # noqa: PLC0415

    text = Path(args.rules).read_text(encoding="utf-8") if args.rules else rules_text()
    binary = current_server(llama_home(args.home))
    found = assess(text, host_stamp(binary))
    out = sys.stdout
    out.writelines(f"{a.line}\n" for a in found)
    torch_kinds = [kind for kind, stamp in stamps(text) if StampKey.TORCH in stamp]
    if torch_kinds:
        out.write(f"torch stamps ({', '.join(torch_kinds)}): not checked here, localmodels has no PyTorch.\n")
    unnamed = [kind for kind, stamp in stamps(text) if StampKey.MODEL not in stamp]
    if unnamed:
        out.write(
            f"stamps naming no model ({', '.join(unnamed)}): add model=<repository id>, or model=any for a "
            "training method.\n"
        )
    out.write(f"llama.cpp build read from {binary}\n")
    out.write(summary_line(found, len(stamps(text))) if found else "No stamp differs.\n")
    needs = [a for a in found if a.state in {RuleState.DOES_NOT_HOLD, RuleState.NEEDS_REMEASURE}]
    return EXIT_STALE if needs else 0


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Add ``rules show`` and ``rules check`` to the ``localmodels`` command."""
    rules = subparsers.add_parser("rules", help="the measured rules by workload kind")
    commands = rules.add_subparsers(dest="rules_command", required=True)
    show = commands.add_parser("show", help="print one kind's rules and its newest digest lines")
    show.add_argument("kind", choices=[k.value for k in WorkloadKind], help="workload kind, K1 to K10")
    show.add_argument("--digests", default=str(DEFAULT_DIGEST_DIR), help="folder of monthly digest notes")
    show.set_defaults(fn=cmd_rules_show)
    check = commands.add_parser(
        "check", help="compare each rule's stamp with this host; exit 1 when one changed"
    )
    check.add_argument("--rules", help="rules document to check (default: the shipped rules.md)")
    check.add_argument("--home", help="llama home, where the `current` build is (see `localmodels llama`)")
    check.set_defaults(fn=cmd_rules_check)
