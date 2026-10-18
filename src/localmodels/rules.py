"""The measured rules by workload kind, shipped with the package as ``rules.md``.

``localmodels rules show K2`` prints one kind's section, then that kind's lines from the newest monthly digest
(``localmodels digest``) when one exists. ``localmodels rules check`` compares each section's ``stamp:`` line
with this host's stack and the current llama.cpp build, and exits 1 when a stamped component changed.
"""

from __future__ import annotations

import re
import sys
from enum import StrEnum
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING, Final

from .digest import DEFAULT_OUT_DIR as DEFAULT_DIGEST_DIR
from .digest import LINE_RE
from .enums import WorkloadKind

if TYPE_CHECKING:
    import argparse
    from collections.abc import Mapping

RULES_FILE: Final = "rules.md"
_SECTION = re.compile(r"^## (K\d+)\b", re.MULTILINE)
_DIGEST_NAME = re.compile(r"^\d{4}-\d{2}-\d{2}\.md$")
_STAMP_LINE = re.compile(r"^- stamp: (?P<pairs>.+)$")
EXIT_STALE: Final = 1


class StampKey(StrEnum):
    """The engine entries of a stamp line (the other keys are the host's ``stack.Component`` names)."""

    LLAMA_CPP = "llama_cpp"
    TORCH = "torch"


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


def stale_rules(text: str, current: Mapping[str, str]) -> list[str]:
    """One ``K1: driver: was -> now, llama_cpp: was -> now`` line per stamp whose components changed.

    ``current`` maps the host fingerprint's names and ``llama_cpp`` to what they are now. ``torch`` is not
    compared (localmodels has no PyTorch); a component the stamp does not name is not compared.
    """
    from .stack import UNKNOWN, stale_components  # noqa: PLC0415 - rules show does not need the stack

    lines = []
    for kind, stamp in stamps(text):
        changed = stale_components(stamp, current)
        build = stamp.get(StampKey.LLAMA_CPP)
        now = current.get(StampKey.LLAMA_CPP, UNKNOWN)
        if build is not None and build != now:
            changed.insert(0, f"{StampKey.LLAMA_CPP}: {build} -> {now}")
        if changed:
            lines.append(f"{kind}: {', '.join(changed)}")
    return lines


def cmd_rules_check(args: argparse.Namespace) -> int:
    """Compare every stamp with this host; exit 1 when a stamped component changed."""
    from . import stack  # noqa: PLC0415 - rules show does not need the stack
    from .llama_builds import current_server, llama_home  # noqa: PLC0415

    text = Path(args.rules).read_text(encoding="utf-8") if args.rules else rules_text()
    binary = current_server(llama_home(args.home))
    current = {**stack.fingerprint(), StampKey.LLAMA_CPP.value: stack.llama_cpp_build(binary)}
    out = sys.stdout
    stale = stale_rules(text, current)
    out.writelines(f"{line}\n" for line in stale)
    torch_kinds = [kind for kind, stamp in stamps(text) if StampKey.TORCH in stamp]
    if torch_kinds:
        out.write(f"torch stamps ({', '.join(torch_kinds)}): not checked here, localmodels has no PyTorch.\n")
    out.write(f"llama.cpp build read from {binary}\n")
    out.write(
        f"{len(stale)} of {len(stamps(text))} stamped rules changed.\n" if stale else "No stamp differs.\n"
    )
    return EXIT_STALE if stale else 0


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
