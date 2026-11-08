"""`localmodels engines ...`: version records, tag choice, switch refusals and prune. No network or Docker."""

from __future__ import annotations

import json
import subprocess
from typing import TYPE_CHECKING

import pytest

from localmodels.atomic import write_text_atomically
from localmodels.cli import main
from localmodels.engines import (
    SOURCES,
    GateVerdict,
    Publisher,
    Response,
    VersionRecord,
    VersionSource,
    list_tags,
    matching_tags,
    newest_untried,
    prune,
    read_record,
    record_path,
    sources_for,
    switch,
    tried_tags,
    versions_dir,
    write_record,
)
from localmodels.stack import Engine

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

DIGEST = "sha256:" + "ab" * 32
CHALLENGE = 'Bearer realm="https://auth.example/token",service="registry.example",scope="repository:a/b:pull"'


def source(engine: Engine, publisher: Publisher = Publisher.OFFICIAL) -> VersionSource:
    return next(s for s in SOURCES if s.engine == engine and s.publisher == publisher)


@pytest.fixture(autouse=True)
def state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("LOCALMODELS_STATE_DIR", str(tmp_path))
    return tmp_path


def make(
    engine: Engine,
    tag: str,
    gate: GateVerdict = GateVerdict.PASSED,
    publisher: Publisher = Publisher.OFFICIAL,
    *,
    pulled: bool = True,
) -> VersionRecord:
    record = VersionRecord(
        engine=engine,
        tag=tag,
        image=f"repo/{engine.value}-{tag}@{DIGEST}",
        publisher=publisher,
        gate=gate,
        pulled=pulled,
    )
    write_record(record)
    return record


def test_record_round_trips_and_ignores_unknown_fields(state: Path) -> None:
    record = VersionRecord(
        engine=Engine.SGLANG,
        tag="v0.5.23-cu130",
        image=f"lmsysorg/sglang@{DIGEST}",
        publisher=Publisher.OFFICIAL,
        sparkrun_version="0.3.1",
        recipes={"nvidia/Qwen3-30B-A3B-NVFP4": "name: x\n"},
        results=[{"phase": "16 at once", "tokens_per_s": 1010.5}],
        gate=GateVerdict.RECORD_ONLY,
    )
    path = write_record(record)
    assert path == state / "engines" / "sglang" / "versions" / "v0.5.23-cu130" / "record.json"
    back = read_record(Engine.SGLANG, "v0.5.23-cu130")
    assert back is not None
    assert back.model_dump(exclude={"recorded_at"}) == record.model_dump(exclude={"recorded_at"})
    assert back.recorded_at  # set when written
    stored = json.loads(path.read_text(encoding="utf-8"))
    write_text_atomically(path, json.dumps({**stored, "added_by_a_newer_writer": [1]}))
    assert read_record(Engine.SGLANG, "v0.5.23-cu130") == back
    for bad in ({"image": "repo:latest"}, {"tag": "../x"}):
        write_text_atomically(path, json.dumps({**stored, **bad}))
        assert read_record(Engine.SGLANG, "v0.5.23-cu130") is None  # refused, not half-read
    assert read_record(Engine.SGLANG, "../v0.5.23-cu130") is None


def test_tag_order_forms_and_the_newest_never_tried() -> None:
    trt = source(Engine.TRTLLM)
    tags = ["1.3.0rc13", "1.3.0rc3", "1.4.0rc3", "1.4.0", "1.4.0rc13", "1.4.0-runtime", "latest", "1.4"]
    assert matching_tags(trt, tags) == ["1.3.0rc3", "1.3.0rc13", "1.4.0rc3", "1.4.0rc13", "1.4.0"]
    variants = ["v0.5.23-cu130", "v0.5.23-cu130-runtime", "v0.5.23-cu130-efa", "v0.5.23", "v0.5.9-cu130"]
    sglang = source(Engine.SGLANG)
    assert matching_tags(sglang, [*variants, "v0.5.10.post1-cu130"]) == ["v0.5.9-cu130", "v0.5.23-cu130"]
    official = source(Engine.VLLM)
    community = source(Engine.VLLM, Publisher.COMMUNITY)
    mixed = ["v0.33.0", "v0.33.0-gb10.0", "v0.33.0rc1", "v0.9.0", "v0.32.0-gb10.10", "v0.32.0-gb10.2"]
    assert matching_tags(official, mixed) == ["v0.9.0", "v0.33.0"]
    assert matching_tags(community, mixed) == ["v0.32.0-gb10.2", "v0.32.0-gb10.10", "v0.33.0-gb10.0"]
    assert newest_untried(trt, tags, tried={"1.3.0rc13"}) == "1.4.0"
    assert newest_untried(trt, tags, tried={"1.4.0"}) is None  # the newest failed: wait for a newer tag
    assert newest_untried(trt, ["latest"], tried=set()) is None
    assert [s.publisher for s in sources_for(Engine.VLLM)] == [Publisher.OFFICIAL, Publisher.COMMUNITY]
    assert sglang.reference == "lmsysorg/sglang"
    assert community.reference == "ghcr.io/timothystewart6/vllm-gb10"


class FakeRegistry:
    """Docker Hub's shape: a 401 with a bearer challenge, a token endpoint, then pages joined by Link."""

    def __init__(self, pages: Sequence[Sequence[str]], *, other_host: bool = False) -> None:
        self.pages, self.other_host = pages, other_host
        self.calls: list[tuple[str, dict[str, str]]] = []

    def __call__(self, url: str, headers: Mapping[str, str], timeout: float) -> Response:
        self.calls.append((url, dict(headers)))
        if url.startswith("https://auth.example/token"):
            return Response(200, {}, json.dumps({"token": "t0k"}).encode())
        if headers.get("authorization") != "Bearer t0k":
            return Response(401, {"www-authenticate": CHALLENGE}, b"{}")
        index = 1 if "last=" in url else 0
        link = {}
        if index + 1 < len(self.pages):
            host = "https://elsewhere.example" if self.other_host else ""
            link = {"link": f'<{host}/v2/a/b/tags/list?n=1000&last=x>; rel="next"'}
        return Response(200, link, json.dumps({"tags": list(self.pages[index])}).encode())


def test_tag_listing_follows_the_token_challenge_and_the_next_link() -> None:
    src = VersionSource(Engine.SGLANG, "registry.example", "a/b", SOURCES[2].tag_pattern, Publisher.OFFICIAL)
    fake = FakeRegistry([["latest", "v0.5.1-cu130"], ["v0.5.23-cu130", "v0.5.1-cu130"]])
    listing = list_tags(src, fetch=fake)
    assert listing.reason is None
    assert listing.tags == ("latest", "v0.5.1-cu130", "v0.5.23-cu130")  # page 2 read; duplicates once
    assert newest_untried(src, listing.tags, tried=set()) == "v0.5.23-cu130"
    token_call = next(url for url, _ in fake.calls if url.startswith("https://auth.example"))
    assert "service=registry.example" in token_call
    assert all(headers.get("authorization") == "Bearer t0k" for url, headers in fake.calls[2:])
    # a page that points at another host is not followed (the token stays with the registry)
    leaky = FakeRegistry([["a"], ["b"]], other_host=True)
    listing = list_tags(src, fetch=leaky)
    assert listing.tags == ()
    assert "another host" in (listing.reason or "")
    assert not any(url.startswith("https://elsewhere.example") for url, _ in leaky.calls)


def test_tag_listing_returns_the_reason_instead_of_raising() -> None:
    def down(url: str, headers: Mapping[str, str], timeout: float) -> Response:
        raise TimeoutError("timed out")

    def garbage(url: str, headers: Mapping[str, str], timeout: float) -> Response:
        return Response(200, {}, b"<html>")

    def refused(url: str, headers: Mapping[str, str], timeout: float) -> Response:
        return Response(401, {}, b"")

    for fetch, words in ((down, "timed out"), (garbage, "JSONDecodeError"), (refused, "no usable token")):
        listing = list_tags(SOURCES[0], fetch=fetch)
        assert listing.tags == ()
        assert words in (listing.reason or "")


def test_switch_refuses_an_unpassed_or_community_version_unless_overridden() -> None:
    folder = versions_dir(Engine.VLLM)
    make(Engine.VLLM, "v0.31.0")
    assert switch(Engine.VLLM, "v0.31.0") is None
    make(Engine.VLLM, "v0.32.0", GateVerdict.FAILED)
    problem = switch(Engine.VLLM, "v0.32.0")
    assert problem is not None
    assert "gate verdict is failed" in problem
    make(Engine.VLLM, "v0.33.0", GateVerdict.NOT_RUN)
    assert switch(Engine.VLLM, "v0.33.0", override="owner: go") is not None  # no gate was run
    assert switch(Engine.VLLM, "v9.9.9", override="owner: go") is not None  # no record
    sibling = VersionRecord(
        engine=Engine.VLLM, tag="evil", publisher=Publisher.OFFICIAL, gate=GateVerdict.PASSED
    )
    write_text_atomically(folder.parent / "evil" / "record.json", sibling.model_dump_json())
    assert switch(Engine.VLLM, "../evil") is not None  # a tag is a plain folder name, not a path
    assert (folder / "current").readlink().name == "v0.31.0"

    assert switch(Engine.VLLM, "v0.32.0", override="owner: 2 (slower but fine)") is None
    assert (folder / "current").readlink().name == "v0.32.0"
    assert (folder / "previous").readlink().name == "v0.31.0"
    kept = json.loads((folder / "v0.32.0" / "override.json").read_text(encoding="utf-8"))
    assert kept["override"] == "owner: 2 (slower but fine)"
    assert "gate verdict is failed" in kept["refused"]
    failed = read_record(Engine.VLLM, "v0.32.0")
    assert failed is not None
    assert failed.gate is GateVerdict.FAILED  # the verdict stays as measured

    make(Engine.VLLM, "v0.33.0-gb10.0", GateVerdict.PASSED, Publisher.COMMUNITY)
    community = switch(Engine.VLLM, "v0.33.0-gb10.0")
    assert community is not None
    assert "community image" in community
    assert switch(Engine.VLLM, "v0.33.0-gb10.0", override="owner: use the machine build") is None
    assert (folder / "current").readlink().name == "v0.33.0-gb10.0"
    assert switch(Engine.PYTORCH, "x") is not None  # not a container engine


def pulled(tag: str) -> bool | None:
    record = read_record(Engine.TRTLLM, tag)
    return None if record is None else record.pulled


def test_prune_removes_only_what_the_keep_list_does_not_name(capsys: pytest.CaptureFixture[str]) -> None:
    folder = versions_dir(Engine.TRTLLM)
    for tag in ("1.3.0", "1.4.0rc3", "1.4.0", "1.2.0", "1.1.0", "1.0.0"):
        make(Engine.TRTLLM, tag)
    make(Engine.TRTLLM, "0.9.0", pulled=False)
    assert switch(Engine.TRTLLM, "1.3.0") is None
    assert switch(Engine.TRTLLM, "1.4.0rc3") is None  # 1.3.0 is now previous
    users = '[projects]\nstudy = "1.4.0rc3"\nlong = "1.2.0"\nfollower = "current"\n'
    (folder / "users.toml").write_text(users, encoding="utf-8")
    run: list[Sequence[str]] = []

    def docker(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        run.append(command)
        busy = "1.0.0" in command[2]
        return subprocess.CompletedProcess(command, 1 if busy else 0, "", "daemon is busy" if busy else "")

    dry = prune(Engine.TRTLLM, pins=["1.1.0"], dry_run=True, runner=docker)
    assert run == []
    assert dry.kept == {
        "1.1.0": ["pinned"],
        "1.2.0": ["used by long"],
        "1.4.0rc3": ["current", "used by follower", "used by study"],
    }  # 1.4.0 is not named by "1.4.0rc3"; previous 1.3.0 is its record and digest only
    assert sorted(item.split()[0] for item in dry.removed) == ["1.0.0", "1.3.0", "1.4.0"]

    report = prune(Engine.TRTLLM, pins=["1.1.0"], runner=docker)
    assert sorted(item.split()[0] for item in report.removed) == ["1.3.0", "1.4.0"]
    assert [c[:2] for c in run] == [["docker", "rmi"]] * 3
    assert len(report.failed) == 1
    assert report.failed[0].startswith("1.0.0")
    assert "daemon is busy" in report.failed[0]
    assert [pulled(t) for t in ("1.3.0", "1.4.0", "1.0.0", "1.4.0rc3", "1.1.0", "1.2.0")] == [
        False, False, True, True, True, True,
    ]  # fmt: skip
    assert tried_tags(Engine.TRTLLM) >= {
        "1.0.0",
        "1.3.0",
        "1.4.0",
        "0.9.0",
    }  # records stay: they are the tried list

    gone = subprocess.CompletedProcess([], 1, "", "Error response from daemon: No such image: x")
    again = prune(Engine.TRTLLM, pins=["1.1.0"], runner=lambda _c: gone)
    assert [item.split()[0] for item in again.removed] == ["1.0.0"]  # already gone counts as removed
    assert pulled("1.0.0") is False

    assert main(["engines", "prune", "--dry-run"]) == 0
    assert "kept trtllm 1.4.0rc3: current, used by follower, used by study" in capsys.readouterr().out


def test_engines_command_switches_and_lists(capsys: pytest.CaptureFixture[str]) -> None:
    make(Engine.SGLANG, "v0.5.22-cu130", GateVerdict.FAILED)
    assert main(["engines", "switch", "sglang", "v0.5.22-cu130"]) == 1
    assert "gate verdict is failed" in capsys.readouterr().err
    assert main(["engines", "switch", "sglang", "v0.5.22-cu130", "--override", "owner: 1"]) == 0
    assert main(["engines", "list"]) == 0
    line = capsys.readouterr().out.splitlines()[-1]
    assert line == (
        "sglang  v0.5.22-cu130  official  gate: failed  current, image on disk, switched by override  "
        "projects: none"
    )
    assert record_path(Engine.SGLANG, "v0.5.22-cu130").is_file()
