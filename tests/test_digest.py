"""The monthly digest: two parsers from recorded fixtures, a failed source, and the note's line format."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from localmodels import digest
from localmodels.digest import Context, Source

FIXTURES = Path(__file__).parent / "fixtures" / "digest"
SINCE = "2026-10-01T00:00:00Z"
FULL = "ccc3646a1b2c3d4e5f60718293a4b5c6d7e8f901"
NOW = datetime(2026, 11, 1, 7, tzinfo=UTC)


@pytest.fixture(autouse=True)
def isolated_llama_home(tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> Path:
    """No `current` link and no refresh run: the checks read an empty llama home, never this machine's."""
    home = tmp_path_factory.mktemp("llama-home")
    monkeypatch.setenv("LOCALMODELS_LLAMA_HOME", str(home))
    monkeypatch.setenv("LOCALMODELS_REFRESH_DIR", str(home / "refresh"))
    monkeypatch.setenv("HF_HUB_CACHE", str(home / "hub"))  # an empty Hub cache, never this machine's
    return home


def _fixture(name: str) -> Any:
    return json.loads((FIXTURES / name).read_text())


class FixtureNet:
    """Answers only the requests it was given a recording for; anything else fails like a dead network."""

    def __init__(self, *, forum: bool = False, llama: bool = False, installed: str = FULL) -> None:
        self.json: dict[str, Any] = {}
        self.pages: dict[str, str] = {}
        if llama:
            base = "repos/ggml-org/llama.cpp"
            self.json[f"{base}/compare/{installed}...master?per_page=1"] = _fixture("llama_compare.json")
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
        if url not in self.pages:
            raise ConnectionError(url)
        return self.pages[url]

    def head(self, url: str) -> dict[str, str]:
        raise ConnectionError(url)

    def read(self, path: str) -> str:
        raise FileNotFoundError(path)

    def run(self, argv: list[str]) -> str:
        if argv[:1] == ["git"]:
            return FULL + "\n"
        raise FileNotFoundError(argv[0])


def test_llama_cpp_counts_ahead_and_keeps_keyword_titles_only(isolated_llama_home: Path) -> None:
    found = digest.check_llama_cpp(Context(FixtureNet(llama=True), SINCE))
    texts = [f.text for f in found]
    ahead = f"master is {_fixture('llama_compare.json')['ahead_by']} commits ahead of the installed "
    assert texts[0].startswith(f"{ahead}ccc3646")  # no `current` link: the hand-made build's HEAD
    builds = isolated_llama_home / "builds"
    (builds / "5aeac37a3").mkdir(parents=True)
    (builds / "current").symlink_to("5aeac37a3")
    linked = digest.check_llama_cpp(Context(FixtureNet(llama=True, installed="5aeac37a3"), SINCE))
    assert linked[0].text.startswith(f"{ahead}5aeac37")  # the build projects run, not the reference
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
    net.pages[digest.PYPI_LLAMA_CPP_PYTHON] = json.dumps({"info": {"version": "0.3.36"}})
    state = {"last_run": SINCE, Source.D15: {"version": "0.3.35"}}  # a release since the last run
    (tmp_path / ".state.json").write_text(json.dumps(state))
    digest.run_digest(net, tmp_path, NOW)
    lines = [ln for ln in (tmp_path / "2026-11-01.md").read_text().splitlines() if ln.startswith("- [")]
    parsed = [digest.LINE_RE.match(ln) for ln in lines]
    assert parsed
    assert all(m is not None for m in parsed)
    by_source = {m.group("source"): m.group("kinds") for m in parsed if m}
    assert by_source[Source.D1] == "K1,K2,K3"
    assert by_source[Source.D8] == "K1,K2"
    assert by_source[Source.D15] == "K1,K2,K3"
    d15 = next(m for m in parsed if m and m.group("source") == Source.D15)
    assert d15.group("link") == "https://pypi.org/project/llama-cpp-python/0.3.36/"  # a page, not the API
    again = Context(net, SINCE, {"version": "0.3.36"})
    assert digest.check_llama_cpp_python(again) == []  # the same release is reported once


def test_cached_models_reports_files_replaced_on_main_once_and_names_unanswered_repos(
    isolated_llama_home: Path,
) -> None:
    old, kept, new = "a" * 64, "b" * 64, "c" * 64
    snapshot = isolated_llama_home / "hub" / "models--unsloth--M-GGUF" / "snapshots" / ("1" * 40)
    blobs = snapshot.parents[1] / "blobs"
    (snapshot / "Q4").mkdir(parents=True)
    blobs.mkdir()
    for name, sha in (("Q4/m-00001-of-00002.gguf", old), ("Q4/m-00002-of-00002.gguf", kept)):
        (blobs / sha).write_text("x")
        (snapshot / name).symlink_to(Path("../../..") / "blobs" / sha)
    (blobs / ("f" * 40)).write_text("readme")  # a small file's blob is named by its git hash: never compared
    (snapshot / "README.md").symlink_to(Path("../..") / "blobs" / ("f" * 40))
    gated = isolated_llama_home / "hub" / "models--google--gated" / "snapshots" / ("2" * 40)
    gated.mkdir(parents=True)
    converted = isolated_llama_home / "hub" / "models--facebook--opt" / "snapshots" / ("3" * 40)
    converted.mkdir(parents=True)  # fetched from a conversion pull request only: never compared, never asked
    (converted.parents[1] / "refs" / "pr").mkdir(parents=True)
    (converted.parents[1] / "refs" / "pr" / "29").write_text("3" * 40)
    net = FixtureNet()
    siblings = [
        {"rfilename": "README.md", "blobId": "e" * 40},
        {"rfilename": "Q4/m-00001-of-00003.gguf", "lfs": {"sha256": new}},  # re-quantised: 3 parts now
        {"rfilename": "Q4/m-00002-of-00002.gguf", "lfs": {"sha256": kept}},
    ]
    net.pages[digest.HF_MODEL_URL.format(repo="unsloth/M-GGUF")] = json.dumps(
        {"sha": "9" * 40, "siblings": siblings}
    )
    ctx = Context(net, SINCE)
    found = digest.check_cached_models(ctx)
    replaced = "1 of the 2 large files on disk (snapshot 1111111) replaced or removed on main 9999999"
    assert [f.text for f in found] == [
        f"unsloth/M-GGUF: {replaced}, e.g. Q4/m-00001-of-00002.gguf",
        "not checked, the Hub did not answer: google/gated (ConnectionError)",
    ]
    again = digest.check_cached_models(Context(net, SINCE, ctx.mem))
    assert [f.text for f in again][:-1] == []  # the same state of main is reported once
