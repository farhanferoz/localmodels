"""`localmodels rules show`: one kind's section of the shipped rules, then its newest digest lines."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from localmodels import stack
from localmodels.cli import main
from localmodels.enums import WorkloadKind
from localmodels.rules import StampKey, rules_text, stamps

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
    assert all(StampKey.MODEL in stamp for _kind, stamp in stamps(text))  # each rule names its models
    with pytest.raises(SystemExit) as exc:
        main(["rules", "show", "K11"])
    assert exc.value.code != 0
    assert "K10" in capsys.readouterr().err


STAMPED = """\
## K1 interactive text generation
- stamp: llama_cpp=ccc3646 driver=580.1 cuda=13.0 kernel=6.17 dgx_os=7.6.0 date=2026-10-18 model=org/m-GGUF
## K5 vision-language
- stamp: torch=2.10.0 driver=580.1 cuda=13.0 kernel=6.17 dgx_os=7.6.0 date=2026-10-18
"""
HOST = {"driver": "580.1", "cuda": "13.0", "kernel": "6.17", "dgx_os": "7.6.0", "localmodels": "0.6.0"}


def test_check_exits_1_only_when_a_stamped_component_changed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    rules = tmp_path / "rules.md"
    rules.write_text(STAMPED, encoding="utf-8")
    host, build = dict(HOST), ["ccc3646"]
    monkeypatch.setattr(stack, "fingerprint", lambda: host)
    monkeypatch.setattr(stack, "llama_cpp_build", lambda _binary: build[0])
    args = ["rules", "check", "--rules", str(rules), "--home", str(tmp_path)]
    assert main(args) == 0
    out = capsys.readouterr().out
    assert "No stamp differs." in out
    assert "torch stamps (K5): not checked here" in out
    assert "stamps naming no model (K5): add model=" in out  # reported, but not a reason to exit 1
    host["driver"], build[0] = "590.2", "abc1234"
    assert main(args) == 1
    assert "K1: llama_cpp: ccc3646 -> abc1234, driver: 580.1 -> 590.2\n" in capsys.readouterr().out
