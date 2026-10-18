"""The measured rules by workload kind, shipped with the package as ``rules.md``.

``localmodels rules show K2`` prints one kind's section, then that kind's lines from the newest monthly digest
(``localmodels digest``) when one exists.
"""

from __future__ import annotations

import re
import sys
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING, Final

from .enums import WorkloadKind

if TYPE_CHECKING:
    import argparse

RULES_FILE: Final = "rules.md"
DEFAULT_DIGEST_DIR: Final = Path.home() / "cache" / "localmodels" / "digests"
_SECTION = re.compile(r"^## (K\d+)\b", re.MULTILINE)
_DIGEST_LINE = re.compile(r"^- \[([^\]]+)\] ")
_DIGEST_NAME = re.compile(r"^\d{4}-\d{2}-\d{2}\.md$")


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
        match = _DIGEST_LINE.match(line)
        if match and kind.value in {k.strip() for k in match.group(1).split(",")}:
            lines.append(line)
    return newest.stem, lines


def cmd_rules_show(args: argparse.Namespace) -> int:
    """Print one kind's rules, then its lines from the newest digest."""
    kind = WorkloadKind(args.kind)
    out = sys.stdout
    out.write(section(kind))
    digest = digest_lines(kind, Path(args.digests))
    if digest is not None:
        date, lines = digest
        out.write(f"\nDigest {date}:\n" if lines else f"\nDigest {date}: nothing for {kind.value}.\n")
        out.writelines(f"{line}\n" for line in lines)
    return 0


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Add ``rules show`` to the ``localmodels`` command."""
    rules = subparsers.add_parser("rules", help="the measured rules by workload kind")
    commands = rules.add_subparsers(dest="rules_command", required=True)
    show = commands.add_parser("show", help="print one kind's rules and its newest digest lines")
    show.add_argument("kind", choices=[k.value for k in WorkloadKind], help="workload kind, K1 to K10")
    show.add_argument("--digests", default=str(DEFAULT_DIGEST_DIR), help="folder of monthly digest notes")
    show.set_defaults(fn=cmd_rules_show)
