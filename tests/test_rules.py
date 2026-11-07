"""`localmodels rules show`: one kind's section of the shipped rules, then its newest digest lines."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from localmodels import rules, stack
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


TWO_RULES = """\
## K1 interactive text generation
- stamp: llama_cpp=ccc3646 driver=580.1 cuda=13.0 kernel=6.17 dgx_os=7.6.0 date=2026-10-18 model=org/m-GGUF
## K5 vision-language
- stamp: torch=2.10.0 driver=580.1 cuda=13.0 kernel=6.17 dgx_os=7.6.0 date=2026-10-18 model=org/v
"""
NEW_HOST = {**HOST, "driver": "590.2", "llama_cpp": "abc1234"}


def _record(kind: str, *, holds: bool, figures: dict[str, float] | None = None) -> rules.RuleRecord:
    return rules.RuleRecord(
        kind=kind,
        measured_at="2026-11-01T09:00:00+00:00",
        slurm_job="77",
        stamp=dict(NEW_HOST),
        figures=figures or {"small_tg128": 65.0},
        holds=holds,
        detail="mean KL 0.01 above 1e-4",
    )


def test_check_answers_a_changed_stamp_by_a_record_or_a_reason_and_exits_1_only_for_neither(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "rules.md").write_text(TWO_RULES, encoding="utf-8")
    monkeypatch.setenv("LOCALMODELS_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setattr(stack, "fingerprint", lambda: NEW_HOST)
    monkeypatch.setattr(stack, "llama_cpp_build", lambda _binary: "abc1234")
    args = ["rules", "check", "--rules", str(tmp_path / "rules.md"), "--home", str(tmp_path)]
    assert main(args) == 1  # K1 has no record and no reason
    out = capsys.readouterr().out
    assert "K1: llama_cpp: ccc3646 -> abc1234" in out
    assert "K5: driver: 580.1 -> 590.2; not re-measured here: re-measured per model by its project" in out
    assert "2 of 2 stamped rules changed: 0 re-measured, 1 with a reason, 1 need re-measuring." in out
    rules.write_record(_record("K1", holds=False))  # a record that does not hold is no answer
    assert main(args) == 1
    assert "K1: re-measured 2026-11-01: does not hold: mean KL 0.01 above 1e-4" in capsys.readouterr().out
    assert main(["rules", "show", "K1"]) == 0
    assert "; does not hold: mean KL 0.01 above 1e-4" in capsys.readouterr().out
    rules.write_record(_record("K1", holds=True, figures={"small_tg128": 80.0}))  # moved 23 % from 65
    assert main(args) == 0  # a holding record and a reason remain
    out = capsys.readouterr().out
    assert "K1: re-measured 2026-11-01 on this stack (slurm 77): small_tg128=80; moved: small_tg128" in out
    assert "2 of 2 stamped rules changed: 1 re-measured, 1 with a reason, 0 need re-measuring." in out
    assert main(["rules", "show", "K1"]) == 0
    assert "Re-measured 2026-11-01 (slurm 77, llama.cpp abc1234, driver 590.2): small_tg128=80" in (
        capsys.readouterr().out
    )
    (rules.records_dir() / "K1.json").write_bytes(b"\xff\xfe not utf-8")  # unreadable: no record, no crash
    assert rules.read_record("K1") is None
