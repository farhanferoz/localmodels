"""`localmodels engines ...`: which version of each container engine is in use, and where newer come from.

vLLM, SGLang and TensorRT-LLM run as container images started through sparkrun inside localmodels' own
wrapper. Each engine has a folder of versions under ``$LOCALMODELS_STATE_DIR/engines/<engine>/versions/``,
laid out like ``llama_builds``' builds folder (one set of switch rules, D15/D17):

- ``<tag>/record.json``: the ``VersionRecord`` of one image (its digest, publisher, realized recipes, results
  and gate verdict); ``<tag>/override.json``: the owner's words when they switched although the gate refused.
- ``current`` and ``previous``: symlinks to a tag folder. ``previous`` is kept as its record and digest only:
  its image is removed by ``engines prune`` and pulled again on a rollback (an image is 26-37 GB).
- ``users.toml``: ``[projects]`` with ``<project> = "<tag>"`` or ``"current"``, as for llama.cpp.

A record stays after its image is removed: a tag that has a record is "tried", and a failed candidate is not
tried again until a newer tag appears (D12). Design: docs/research/engine-registry-design-2026-11-08.md
(part 2).

Nothing here starts a container or pulls an image. ``import localmodels`` does not load this module.
"""

from __future__ import annotations

import http.client
import json
import logging
import re
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Final, NamedTuple

from pydantic import BaseModel, field_validator

from .atomic import read_model, write_model
from .llama_builds import (
    CURRENT,
    EXIT_ERROR,
    EXIT_OK,
    OVERRIDE_FILE,
    PREVIOUS,
    link_target,
    named_by,
    read_users_in,
    switch_version,
    unreferenced,
    version_folders,
)
from .stack import Engine

if TYPE_CHECKING:
    import argparse
    import email.message
    from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
    from pathlib import Path
    from typing import IO

log = logging.getLogger(__name__)

RECORD_FILE: Final = "record.json"
VERSIONS_DIRNAME: Final = "versions"
ENGINES_DIRNAME: Final = "engines"
DOCKER_HUB: Final = "registry-1.docker.io"
#: A registry's tag list is asked for in pages of this many (a hint: a registry may send fewer).
PAGE_SIZE: Final = 1000
#: Docker Hub's lmsysorg/sglang had 2 pages on 2026-11-08; a registry that sends more is stopped here.
MAX_PAGES: Final = 50
MAX_BODY_BYTES: Final = 16 * 1024 * 1024
TIMEOUT_SECONDS: Final = 30.0
HTTP_OK: Final = 200
HTTP_UNAUTHORIZED: Final = 401
#: ``docker rmi`` of an image that is already gone says this; the record then reads "not on disk".
NO_SUCH_IMAGE: Final = "No such image"
DOCKER_TIMEOUT_SECONDS: Final = 600.0
RUNNER_NOT_FOUND: Final = 127
RUNNER_TIMEOUT: Final = 124

_IMAGE: Final = re.compile(r"[^@\s]+@sha256:[0-9a-f]{64}")
#: A tag is a folder name: no separator, no leading dot.
_TAG: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_CHALLENGE_PARAM: Final = re.compile(r'(\w+)="([^"]*)"')
_LINK_NEXT: Final = re.compile(r'<([^>]+)>\s*;\s*rel="?next"?', re.IGNORECASE)


class Publisher(StrEnum):
    """Who builds the image: only an official one is ever switched to without the owner's override."""

    OFFICIAL = "official"
    COMMUNITY = "community"  # one person's account


class GateVerdict(StrEnum):
    """What the gates said about a version, as ``engines switch`` reads it; only ``PASSED`` goes freely."""

    NOT_RUN = "not-run"
    PASSED = "passed"
    FAILED = "failed"
    #: Passed the speed gate but the engine's quality limits are not calibrated yet (design 4.3).
    RECORD_ONLY = "record-only"


CONTAINER_ENGINES: Final = (Engine.VLLM, Engine.SGLANG, Engine.TRTLLM)


class VersionRecord(BaseModel):
    """One image of a container engine. Fields it does not know are ignored on read."""

    engine: Engine
    tag: str
    #: ``repo@sha256:...``; empty for a candidate whose pull failed before a digest was known.
    image: str = ""
    publisher: Publisher
    sparkrun_version: str = ""
    #: Model (Hub repository) -> the recipe ``sparkrun export recipe --realize`` made for it.
    recipes: dict[str, str] = {}
    #: The measured result records, newest last (written by the measuring harness).
    results: list[dict[str, object]] = []
    gate: GateVerdict = GateVerdict.NOT_RUN
    #: Why the gate said what it said, or why the pull or start failed.
    reason: str = ""
    #: The image is on disk now (``engines prune`` clears it).
    pulled: bool = False
    recorded_at: str = ""

    @field_validator("image")
    @classmethod
    def _digest_form(cls, image: str) -> str:
        if image and not _IMAGE.fullmatch(image):
            msg = f"image must be repo@sha256:<64 hex digits>, got {image!r}"
            raise ValueError(msg)
        return image

    @field_validator("tag")
    @classmethod
    def _folder_name(cls, tag: str) -> str:
        if not _TAG.fullmatch(tag):
            msg = f"tag {tag!r} is not a plain folder name"
            raise ValueError(msg)
        return tag


@dataclass(frozen=True)
class VersionSource:
    """Where an engine's versions come from: a registry repository, the tag form it uses and who builds it."""

    engine: Engine
    registry: str
    repository: str
    #: The whole tag; groups ``major``, ``minor``, ``patch`` and maybe ``rc``, ``build`` or ``post`` order it.
    tag_pattern: re.Pattern[str]
    publisher: Publisher

    @property
    def reference(self) -> str:
        """The repository as ``docker pull`` names it (Docker Hub has no host part)."""
        return self.repository if self.registry == DOCKER_HUB else f"{self.registry}/{self.repository}"


#: Checked 2026-11-08 (docs/research/engine-registry-search-2026-11-08.md). SGLang is exactly ``-cu130``: not
#: ``-runtime``, ``-efa`` or another variant; its ``.postN`` hotfix releases count, after their base
#: release, and its release candidates do not. TensorRT-LLM release candidates sort before their release.
#: One version core for every tag form (matched whole, ``fullmatch``): ASCII digits, at most nine each.
_NUM: Final = "[0-9]{1,9}"
_VERSION: Final = rf"(?P<major>{_NUM})\.(?P<minor>{_NUM})\.(?P<patch>{_NUM})"


SOURCES: Final = (
    VersionSource(
        Engine.VLLM, DOCKER_HUB, "vllm/vllm-openai", re.compile(rf"v{_VERSION}"), Publisher.OFFICIAL
    ),
    VersionSource(
        Engine.VLLM,
        "ghcr.io",
        "timothystewart6/vllm-gb10",
        re.compile(rf"v{_VERSION}-gb10\.(?P<build>{_NUM})"),
        Publisher.COMMUNITY,
    ),
    VersionSource(
        Engine.SGLANG,
        DOCKER_HUB,
        "lmsysorg/sglang",
        re.compile(rf"v{_VERSION}(?:\.post(?P<post>{_NUM}))?-cu130"),
        Publisher.OFFICIAL,
    ),
    VersionSource(
        Engine.TRTLLM,
        "nvcr.io",
        "nvidia/tensorrt-llm/release",
        re.compile(rf"{_VERSION}(?:rc(?P<rc>{_NUM}))?"),
        Publisher.OFFICIAL,
    ),
)


def sources_for(engine: Engine) -> list[VersionSource]:
    """The version sources of ``engine``, official first."""
    return sorted(
        (s for s in SOURCES if s.engine == engine), key=lambda s: s.publisher is not Publisher.OFFICIAL
    )


# --- tag order ------------------------------------------------------------------------------------


def tag_key(source: VersionSource, tag: str) -> tuple[int, ...] | None:
    """The sort key of ``tag`` (numbers, not text: rc3 < rc13 < release); None if it is not the form."""
    match = source.tag_pattern.fullmatch(tag) if isinstance(tag, str) else None
    if match is None:
        return None
    groups = match.groupdict()
    release = 1 if groups.get("rc") is None else 0
    extra = int(groups.get("rc") or groups.get("build") or groups.get("post") or 0)
    return (int(groups["major"]), int(groups["minor"]), int(groups["patch"]), release, extra)


def matching_tags(source: VersionSource, tags: Iterable[str]) -> list[str]:
    """The tags that have the source's exact form (variants such as ``-runtime`` do not), oldest first."""
    keyed = {tag: key for tag in set(tags) if (key := tag_key(source, tag)) is not None}
    return sorted(keyed, key=lambda tag: keyed[tag])


def newest_untried(source: VersionSource, tags: Iterable[str], tried: Collection[str]) -> str | None:
    """The newest tag of the source's form when it has not been tried; None when it has or there is none.

    An older untried tag is not offered: it would be a step back, and a failed newest tag waits for a newer.
    """
    found = matching_tags(source, tags)
    return found[-1] if found and found[-1] not in tried else None


# --- tag listing ----------------------------------------------------------------------------------


class Response(NamedTuple):
    """One HTTP answer: status, headers with lower-case names, body."""

    status: int
    headers: Mapping[str, str]
    body: bytes


type Fetch = Callable[[str, Mapping[str, str], float], Response]


class _ListingError(Exception):
    """A reason a tag list could not be read; ``list_tags`` turns it into a ``TagListing``."""


@dataclass(frozen=True)
class TagListing:
    """The tags a registry listed, or why it could not (``tags`` is then empty: a part may miss one)."""

    tags: tuple[str, ...] = ()
    reason: str | None = None


class NoRedirect(urllib.request.HTTPRedirectHandler):
    """Refuse every redirect: urllib would carry the registry token to the new address, whatever its host."""

    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: IO[bytes],
        code: int,
        msg: str,  # noqa: ARG002 - urllib's signature
        headers: email.message.Message,
        newurl: str,
    ) -> urllib.request.Request | None:
        """Raise instead of building the next request."""
        raise urllib.error.HTTPError(req.full_url, code, f"redirect to {newurl} refused", headers, fp)


_OPENER: Final = urllib.request.build_opener(NoRedirect)


def urllib_fetch(url: str, headers: Mapping[str, str], timeout: float) -> Response:
    """One GET over https with the standard library; an HTTP error status is an answer, not an exception."""
    if urllib.parse.urlsplit(url).scheme != "https":
        msg = f"not an https URL: {url}"
        raise ValueError(msg)
    request = urllib.request.Request(url, headers=dict(headers))  # noqa: S310 - https checked above
    try:
        with _OPENER.open(request, timeout=timeout) as reply:
            return Response(reply.status, _lower(reply.headers.items()), reply.read(MAX_BODY_BYTES + 1))
    except urllib.error.HTTPError as err:
        return Response(err.code, _lower(err.headers.items()), err.read(MAX_BODY_BYTES + 1))


def _lower(items: Iterable[tuple[str, str]]) -> dict[str, str]:
    return {name.lower(): value for name, value in items}


def _token(challenge: str, repository: str, fetch: Fetch, timeout: float) -> str | None:
    """The anonymous bearer token a ``WWW-Authenticate: Bearer realm=...,service=...,scope=...`` asks for."""
    if not challenge.lower().startswith("bearer "):
        return None
    params = dict(_CHALLENGE_PARAM.findall(challenge))
    realm = params.get("realm", "")
    if not realm:
        return None
    query = {"service": params["service"]} if "service" in params else {}
    query["scope"] = params.get("scope", f"repository:{repository}:pull")
    reply = fetch(f"{realm}?{urllib.parse.urlencode(query)}", {}, timeout)
    if reply.status != HTTP_OK or len(reply.body) > MAX_BODY_BYTES:
        return None
    body = json.loads(reply.body)
    token = body.get("token") or body.get("access_token")
    return token if isinstance(token, str) and token else None


def list_tags(
    source: VersionSource, *, fetch: Fetch = urllib_fetch, timeout: float = TIMEOUT_SECONDS
) -> TagListing:
    """Every tag of the source's repository through ``/v2/<repo>/tags/list``; never raises.

    A ``401`` with a bearer challenge is answered with an anonymous token; the ``Link: rel="next"`` header
    is followed (Docker Hub's first page of lmsysorg/sglang held none of the matching tags). A token is only
    sent to the registry's own host.
    """
    host = source.registry
    url = f"https://{host}/v2/{source.repository}/tags/list?n={PAGE_SIZE}"
    headers: dict[str, str] = {}
    seen: set[str] = set()
    try:
        for _ in range(MAX_PAGES):
            reply = fetch(url, headers, timeout)
            if reply.status == HTTP_UNAUTHORIZED and "authorization" not in headers:
                challenge = reply.headers.get("www-authenticate", "")
                token = _token(challenge, source.repository, fetch, timeout)
                if token is None:
                    msg = f"{host} refused the tag list and gave no usable token challenge"
                    raise _ListingError(msg)  # noqa: TRY301 - one exit for every reason
                headers["authorization"] = f"Bearer {token}"
                continue
            if reply.status != HTTP_OK or len(reply.body) > MAX_BODY_BYTES:
                msg = f"{url}: HTTP {reply.status}, {len(reply.body)} bytes"
                raise _ListingError(msg)  # noqa: TRY301
            seen.update(tag for tag in json.loads(reply.body).get("tags") or [] if isinstance(tag, str))
            link = _LINK_NEXT.search(reply.headers.get("link", ""))
            if link is None:
                return TagListing(tags=tuple(sorted(seen)))
            url = urllib.parse.urljoin(url, link.group(1))
            if urllib.parse.urlsplit(url).netloc != host:
                msg = f"{host} sent the next page to another host: {url}"
                raise _ListingError(msg)  # noqa: TRY301
        msg = f"more than {MAX_PAGES} pages"
        raise _ListingError(msg)  # noqa: TRY301
    except (OSError, ValueError, TypeError, AttributeError, http.client.HTTPException, _ListingError) as exc:
        return TagListing(reason=f"{host}/{source.repository}: {type(exc).__name__}: {exc}")


# --- records and the switch rules -----------------------------------------------------------------


def engines_dir() -> Path:
    """``engines/`` under the state directory (``$LOCALMODELS_STATE_DIR``, as ``settings.state_dir`` says)."""
    from .settings import state_dir  # noqa: PLC0415 - settings is large and needs nothing from here

    return state_dir() / ENGINES_DIRNAME


def versions_dir(engine: Engine) -> Path:
    """The folder of ``engine``'s versions, ``current``, ``previous`` and ``users.toml``."""
    return engines_dir() / engine.value / VERSIONS_DIRNAME


def record_path(engine: Engine, tag: str) -> Path:
    """Where the record of ``tag`` is kept."""
    return versions_dir(engine) / tag / RECORD_FILE


def read_record(engine: Engine, tag: str) -> VersionRecord | None:
    """The record of ``tag``; None when it is missing or unreadable."""
    if not _TAG.fullmatch(tag):
        return None
    return read_model(record_path(engine, tag), VersionRecord)


def write_record(record: VersionRecord) -> Path:
    """Store ``record`` atomically; its time is set when it has none."""
    stamped = (
        record
        if record.recorded_at
        else record.model_copy(update={"recorded_at": datetime.now(UTC).isoformat(timespec="seconds")})
    )
    return write_model(record_path(record.engine, record.tag), stamped)


def records(engine: Engine) -> list[VersionRecord]:
    """Every readable record of ``engine``, by tag."""
    found = (
        read_model(folder / RECORD_FILE, VersionRecord) for folder in version_folders(versions_dir(engine))
    )
    return [record for record in found if record is not None]


def tried_tags(engine: Engine) -> set[str]:
    """The tags that have a record: tried, whatever came of it."""
    return {record.tag for record in records(engine)}


def switch_refusals(record: VersionRecord, path: Path) -> list[str]:
    """Why ``record`` may not become ``current`` without the owner's override; empty when none."""
    reasons = []
    if record.gate is not GateVerdict.PASSED:
        reasons.append(f"{path}: gate verdict is {record.gate.value}, not passed")
    if record.publisher is Publisher.COMMUNITY:
        reasons.append(
            f"{record.image or record.tag} is a community image: it is switched to only by override"
        )
    return reasons


def switch(engine: Engine, tag: str, *, override: str = "") -> str | None:
    """Move ``engine``'s ``current`` to ``tag``; None, or why it was not moved.

    Refused when the gate did not pass (D17) or the image is a community build; ``override`` (the owner's
    words, kept in ``override.json``) allows either, but never stands in for a gate that was not run.
    """
    if engine not in CONTAINER_ENGINES:
        return f"{engine.value} is not a container engine: llama.cpp has `localmodels llama switch`"
    record = read_record(engine, tag)
    path = record_path(engine, tag)
    if record is None:
        return f"{path} is missing or not a version record"
    return switch_version(
        versions_dir(engine) / tag,
        unfinished=None if record.image else f"{path} names no image digest: there is nothing to run",
        refused="; ".join(switch_refusals(record, path)) or None,
        measured=record.gate is not GateVerdict.NOT_RUN,
        override=override,
    )


class Pin(NamedTuple):
    """A study's pin of one engine version: the version and the pin file that holds the pin."""

    version: str
    file: Path


def keep_list(engine: Engine, pins: Collection[Pin] = ()) -> dict[str, list[str]]:
    """The tags whose image stays on disk, each with why: ``current``, a pin or a ``users.toml`` name.

    A pin keeps its version only while its pin file exists (checked now; a dropped pin is logged).
    ``previous``
    is not here: it is kept as its record and digest only.
    """
    folder = versions_dir(engine)
    current, users = link_target(folder), read_users_in(folder)
    pinned: dict[str, list[str]] = {}
    for pin in pins:
        if pin.file.exists():
            pinned.setdefault(pin.version, []).append(f"pinned by {pin.file}")
        else:
            log.info(
                "%s %s: pin file %s is gone, the pin no longer keeps the version",
                engine,
                pin.version,
                pin.file,
            )
    kept: dict[str, list[str]] = {}
    for version in version_folders(folder):
        why = [
            *([CURRENT] if version.name == current else []),
            *pinned.get(version.name, []),
            *(f"used by {project}" for project in named_by(version.name, current, users, by_prefix=False)),
        ]
        if why:
            kept[version.name] = why
    return kept


# --- prune ----------------------------------------------------------------------------------------

type Runner = Callable[[Sequence[str]], subprocess.CompletedProcess[str]]


def docker_runner(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    """Run a docker command; a missing binary or a timeout is an exit status, not an exception."""
    try:
        return subprocess.run(  # noqa: S603 - argv list, no shell
            list(command), check=False, capture_output=True, text=True, timeout=DOCKER_TIMEOUT_SECONDS
        )
    except FileNotFoundError as exc:
        return subprocess.CompletedProcess(list(command), RUNNER_NOT_FOUND, "", str(exc))
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(list(command), RUNNER_TIMEOUT, "", "timed out")


@dataclass
class PruneReport:
    """What a prune did to one engine: images removed, what stays and why, and what could not be removed."""

    removed: list[str]
    kept: dict[str, list[str]]
    failed: list[str]


def prune(
    engine: Engine, *, pins: Collection[Pin] = (), dry_run: bool = False, runner: Runner = docker_runner
) -> PruneReport:
    """Remove the images of every version that is not ``current``, pinned or named in ``users.toml``.

    ``previous`` loses its image too (its record and digest stay). No record is deleted: the records say which
    tags were tried. ``docker rmi`` goes through ``runner``.
    """
    kept = keep_list(engine, pins)
    report = PruneReport(removed=[], kept=kept, failed=[])
    for folder in unreferenced(versions_dir(engine), keep=kept, by_prefix=False, keep_previous=False):
        record = read_model(folder / RECORD_FILE, VersionRecord)
        if record is None or not record.pulled or not record.image:
            continue
        label = f"{record.tag} ({record.image})"
        if dry_run:
            report.removed.append(label)
            continue
        done = runner(["docker", "rmi", record.image])
        if done.returncode == 0 or NO_SUCH_IMAGE in done.stderr:
            latest = read_model(folder / RECORD_FILE, VersionRecord) or record  # a job may have added results
            write_record(latest.model_copy(update={"pulled": False}))
            report.removed.append(label)
        else:
            report.failed.append(f"{label}: {done.stderr.strip() or f'exit {done.returncode}'}")
    return report


# --- command line ---------------------------------------------------------------------------------


def cmd_list(_args: argparse.Namespace) -> int:
    """Print each version of each container engine: its publisher, gate verdict and where it stands."""
    for engine in CONTAINER_ENGINES:
        folder = versions_dir(engine)
        current, previous, users = link_target(folder), link_target(folder, PREVIOUS), read_users_in(folder)
        for record in records(engine):
            notes = [
                *([CURRENT] if record.tag == current else []),
                *([PREVIOUS] if record.tag == previous else []),
                *(["image on disk"] if record.pulled else []),
                *(["switched by override"] if (folder / record.tag / OVERRIDE_FILE).is_file() else []),
            ]
            projects = named_by(record.tag, current, users, by_prefix=False)
            sys.stdout.write(
                f"{engine.value}  {record.tag}  {record.publisher.value}  gate: {record.gate.value}  "
                f"{', '.join(notes) or '-'}  projects: {', '.join(projects) or 'none'}\n"
            )
    return EXIT_OK


def cmd_switch(args: argparse.Namespace) -> int:
    """Move an engine's ``current`` to a version whose gate passed (or by the owner's override)."""
    engine = Engine(args.engine)
    problem = switch(engine, args.version, override=args.override or "")
    if problem is not None:
        sys.stderr.write(f"not switching: {problem}\n")
        return EXIT_ERROR
    sys.stdout.write(f"{engine.value} {CURRENT} -> {args.version}\n")
    return EXIT_OK


def cmd_prune(args: argparse.Namespace) -> int:
    """Remove the images that are not on any engine's keep list; print what went and what stays."""
    status = EXIT_OK
    for engine in CONTAINER_ENGINES:
        report = prune(engine, dry_run=args.dry_run)
        out = sys.stdout
        out.writelines(
            f"{'would remove' if args.dry_run else 'removed'} {engine.value} {item}\n"
            for item in report.removed
        )
        out.writelines(f"kept {engine.value} {tag}: {', '.join(why)}\n" for tag, why in report.kept.items())
        for item in report.failed:
            sys.stderr.write(f"could not remove {engine.value} {item}\n")
            status = EXIT_ERROR
    return status


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Add ``engines list|switch|prune`` to the ``localmodels`` command."""
    engines = subparsers.add_parser("engines", help="container engine versions (vLLM, SGLang, TensorRT-LLM)")
    commands = engines.add_subparsers(dest="engines_command", required=True)

    commands.add_parser("list", help="the versions, their gate verdicts and `current`").set_defaults(
        fn=cmd_list
    )

    switch_p = commands.add_parser("switch", help="move `current` to a version whose gate passed")
    switch_p.add_argument("engine", choices=[e.value for e in CONTAINER_ENGINES])
    switch_p.add_argument("version", help="tag, as `engines list` prints it")
    switch_p.add_argument(
        "--override",
        metavar="WORDS",
        help="the owner's decision to switch although the gate did not pass, or to use a community image "
        f"(kept in {OVERRIDE_FILE})",
    )
    switch_p.set_defaults(fn=cmd_switch)

    prune_p = commands.add_parser(
        "prune", help="remove the images that are not current, pinned or named in users.toml"
    )
    prune_p.add_argument("--dry-run", action="store_true", help="list what would be removed")
    prune_p.set_defaults(fn=cmd_prune)
