"""`localmodels rules show`: one kind's section of the shipped rules, then its newest digest lines."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from localmodels.cli import main
from localmodels.enums import WorkloadKind
from localmodels.rules import rules_text

if TYPE_CHECKING:
    from pathlib import Path


def test_show_prints_one_section_and_the_newest_digest_lines_for_that_kind(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "2026-09-01.md").write_text("- [K2] D1 llama.cpp: old finding (x)\n", encoding="utf-8")
    (tmp_path / "2026-10-01.md").write_text(
        "# Digest\n- [K1,K2,K3] D1 llama.cpp: new build (x)\n- [K5] D4 PyTorch: 2.11 (y)\n", encoding="utf-8"
    )
    assert main(["rules", "show", "K2", "--digests", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert out.startswith("## K2 bulk text generation")
    assert "## K3" not in out
    assert "Digest 2026-10-01:\n- [K1,K2,K3] D1 llama.cpp: new build (x)\n" in out
    assert "old finding" not in out
    assert "PyTorch" not in out


def test_every_kind_has_a_section_and_an_unknown_kind_is_refused(capsys: pytest.CaptureFixture[str]) -> None:
    text = rules_text()
    assert all(f"\n## {kind.value} " in text for kind in WorkloadKind)
    with pytest.raises(SystemExit) as exc:
        main(["rules", "show", "K11"])
    assert exc.value.code != 0
    assert "K10" in capsys.readouterr().err
