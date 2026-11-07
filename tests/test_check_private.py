"""The commit guard: the tracked tree is clean, and each generic rule fires on a line it should catch."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import check_private  # noqa: E402

GENERIC_ONLY = [(name, re.compile(rx)) for name, rx in check_private.GENERIC]


def test_the_tracked_tree_is_clean() -> None:
    assert check_private.main([]) == 0


def test_each_generic_rule_catches_its_line_and_lets_the_placeholders_through() -> None:
    bad = [
        "host 192.168.4.7",
        "addr 172.20.1.1",
        "ssh 10.1.2.3",
        "box.tail1234.ts.net",
        "AKIAABCDEFGHIJKLMNOP",
        "token ghp_" + "a" * 20,
    ]
    ok = [
        "192.0.2.8",
        "100.64.0.2",
        "10.0.0.2",
        "box.tailnet-example.ts.net",
        "/home/user/dev",
        "Lib\\10.0.22621.0",
        "172.17.0.1",
    ]
    assert all(check_private.hits(line, GENERIC_ONLY) for line in bad)
    assert not any(check_private.hits(line, GENERIC_ONLY) for line in ok)
