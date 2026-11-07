"""Refuse a commit that carries machine-private details.

Private network addresses, tailnet host names and secret-shaped strings are refused everywhere:

    python3 tools/check_private.py            # every tracked file
    python3 tools/check_private.py --staged   # only what is staged (the pre-commit hook)

Your own names (user name, host names, domains) go one regular expression per line in `.private-patterns`
at the repository root: that file is ignored by git, so the names it holds never reach a commit. See
`.gitignore` for the other files kept local to a machine.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCAL_PATTERNS = ROOT / ".private-patterns"
# The guard and its test hold the sample lines the rules must catch.
SELF = {"tools/check_private.py", "tests/test_check_private.py"}
# Each part of an address is 0-255, so a version number such as 10.0.22621.0 is not one; 10.0.0.x is the
# documentation example used in the tests; 172.17.0.1 is Docker's default bridge, the same on every machine.
_OCTET = r"(25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)"
GENERIC = (
    (
        "private network address",
        (
            rf"\b(192\.168\.{_OCTET}\.{_OCTET}|10\.(?!0\.0\.){_OCTET}\.{_OCTET}\.{_OCTET}"
            rf"|172\.(?!17\.0\.1\b)(1[6-9]|2\d|3[01])\.{_OCTET}\.{_OCTET})\b"
        ),
    ),
    ("tailnet host name", r"\b[\w-]+\.(?!tailnet-example\.)[\w-]+\.ts\.net\b"),
    (
        "secret-shaped string",
        (
            r"ghp_[A-Za-z0-9]{20}|github_pat_|sk-[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{12}"
            r"|BEGIN (RSA|OPENSSH|EC|PRIVATE)"
        ),
    ),
)
PLACEHOLDERS = "192.0.2.x, 100.64.0.x, example.com, /home/user"


def patterns() -> list[tuple[str, re.Pattern[str]]]:
    """Return the generic rules, plus one rule per line of `.private-patterns` when that file exists."""
    found = [(name, re.compile(rx)) for name, rx in GENERIC]
    if LOCAL_PATTERNS.exists():
        lines = [x.strip() for x in LOCAL_PATTERNS.read_text().splitlines()]
        found += [
            ("a name in .private-patterns", re.compile(x)) for x in lines if x and not x.startswith("#")
        ]
    return found


def hits(text: str, rules: list[tuple[str, re.Pattern[str]]]) -> list[tuple[int, str]]:
    """Return (line number, rule name) for every line of `text` that a rule matches."""
    return [(n, name) for n, line in enumerate(text.splitlines(), 1) for name, rx in rules if rx.search(line)]


def _git(*args: str) -> bytes:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, check=True).stdout


def main(argv: list[str]) -> int:
    """Check every tracked file (or with --staged, the staged blobs); print each hit; 1 if any, else 0."""
    staged = "--staged" in argv
    listing = ("diff", "--cached", "--name-only", "--diff-filter=ACM", "-z") if staged else ("ls-files", "-z")
    names = _git(*listing).decode().split("\0")
    rules, bad = patterns(), 0
    for name in filter(None, names):
        if name in SELF:
            continue
        if staged:
            blob = _git("show", f":{name}")
        else:
            path = ROOT / name
            blob = path.read_bytes() if path.is_file() else b""
        try:
            text = blob.decode()
        except UnicodeDecodeError:
            continue
        for line_no, what in hits(text, rules):
            print(f"{name}:{line_no}: {what}")
            bad += 1
    if bad:
        print(f"check_private: {bad} private-looking line(s); use a placeholder ({PLACEHOLDERS})")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
