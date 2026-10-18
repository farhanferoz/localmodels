"""The monthly digest: two parsers from recorded fixtures, a failed source, and the note's line format."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from localmodels import digest
from localmodels.digest import Context, Source

FIXTURES = Path(__file__).parent / "fixtures" / "digest"
SINCE = "2026-10-01T00:00:00Z"
FULL = "ccc3646a1b2c3d4e5f60718293a4b5c6d7e8f901"
NOW = datetime(2026, 11, 1, 7, tzinfo=UTC)


def _fixture(name: str) -> Any:
    return json.loads((FIXTURES / name).read_text())


class FixtureNet:
    """Answers only the requests it was given a recording for; anything else fails like a dead network."""

    def __init__(self, *, forum: bool = False, llama: bool = False) -> None:
        self.json: dict[str, Any] = {}
        self.pages: dict[str, str] = {}
        if llama:
            base = "repos/ggml-org/llama.cpp"
            self.json[f"{base}/compare/{FULL}...master?per_page=1"] = _fixture("llama_compare.json")
            self.json[f"{base}/commits?since={SINCE}&per_page=100&path=ggml/src/ggml-cuda"] = _fixture(
                "llama_cuda_commits.json"
            )
            for path in digest.LLAMA_PATHS[1:]:
                self.json[f"{base}/commits?since={SINCE}&per_page=100&path={path}"] = []
        if forum:
            self.pages[digest.FORUM_URL] = (FIXTURES / "forum.json").read_text()

    def gh_json(self, endpoint: str) -> Any:
        return self.json[endpoint]

    def gh_text(self, endpoint: str) -> str:
        raise ConnectionError(endpoint)

    def get(self, url: str, *, browser: bool = False) -> str:
        assert browser or url != digest.FORUM_URL
        return self.pages[url]

    def head(self, url: str) -> dict[str, str]:
        raise ConnectionError(url)

    def read(self, path: str) -> str:
        raise FileNotFoundError(path)

    def run(self, argv: list[str]) -> str:
        if argv[:1] == ["git"]:
            return FULL + "\n"
        raise FileNotFoundError(argv[0])


def test_llama_cpp_counts_ahead_and_keeps_keyword_titles_only() -> None:
    found = digest.check_llama_cpp(Context(FixtureNet(llama=True), SINCE))
    texts = [f.text for f in found]
    assert texts[0].startswith(
        f"master is {_fixture('llama_compare.json')['ahead_by']} commits ahead of the installed ccc3646"
    )
    assert any("rms_norm: add variants that load float4" in t for t in texts)
    assert not any("tautological" in t for t in texts)


def test_forum_keeps_new_topics_matching_the_keywords() -> None:
    found = digest.check_forum(Context(FixtureNet(forum=True), SINCE))
    titles = [f.text for f in found]
    assert len(titles) == 2
    assert any(t.startswith("GB10 kernel regression") for t in titles)
    assert not any("RAIDFLASH" in t for t in titles)
    assert all(f.link.startswith("https://forums.developer.nvidia.com/t/") for f in found)


def test_failed_source_is_listed_and_does_not_stop_the_run(tmp_path: Path) -> None:
    assert digest.run_digest(FixtureNet(forum=True), tmp_path, NOW) == digest.EXIT_OK
    note = (tmp_path / "2026-11-01.md").read_text()
    assert "## Failed sources" in note
    assert "- D1 llama.cpp: " in note.split("## Failed sources")[1]
    assert "D8 NVIDIA forum" in note.split("## Findings")[1].split("## ")[0]
    raw = json.loads((tmp_path / ".raw" / "2026-11-01.json").read_text())
    assert raw["sources"]["D1"]["failed"]
    assert digest.run_digest(FixtureNet(), tmp_path / "none", NOW) == digest.EXIT_ERROR


def test_note_lines_follow_the_format_the_rules_command_parses(tmp_path: Path) -> None:
    net = FixtureNet(forum=True, llama=True)
    state = {"last_run": SINCE}
    (tmp_path / ".state.json").write_text(json.dumps(state))
    digest.run_digest(net, tmp_path, NOW)
    lines = [ln for ln in (tmp_path / "2026-11-01.md").read_text().splitlines() if ln.startswith("- [")]
    parsed = [digest.LINE_RE.match(ln) for ln in lines]
    assert parsed
    assert all(m is not None for m in parsed)
    by_source = {m.group("source"): m.group("kinds") for m in parsed if m}
    assert by_source[Source.D1] == "K1,K2,K3"
    assert by_source[Source.D8] == "K1,K2"
