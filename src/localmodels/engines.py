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
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, NamedTuple

from pydantic import BaseModel, ValidationError, field_validator

from .atomic import read_model, write_model, write_text_atomically
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
RUNNER_CANNOT_RUN: Final = 126  # found, but not executable (the shell's code)
RUNNER_TIMEOUT: Final = 124

APPROVALS_FILE: Final = "approvals.json"
PINS_FILE: Final = "pins.json"
STORE_SCHEMA: Final = 1
_REVISION: Final = re.compile(r"[0-9a-f]{40}")
_GGUF_SPEC: Final = re.compile(r"(?P<repository>[^@:\s]+)@(?P<revision>[^@:\s]+):(?P<file>[^:\s]+)")
_FILE_SPEC: Final = re.compile(r"(?P<repository>[^@:\s]+)@(?P<revision>[^@:\s]+)")

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
CONTAINER_NAMES: Final = ", ".join(e.value for e in CONTAINER_ENGINES)


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
    return sorted(keyed, key=lambda tag: (keyed[tag], tag))  # the text breaks a tie (v0.05.1 and v0.5.1)


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


def keep_list(engine: Engine, pins: Collection[Pin] | None = None) -> dict[str, list[str]]:
    """The tags whose image stays on disk, each with why: ``current``, a pin or a ``users.toml`` name.

    ``pins`` None reads the pins registered for the engine (``pins_for``), so every caller keeps a pinned
    version without passing them. A pin keeps its version only while its pin file exists (checked now; a
    dropped pin is logged). ``previous`` is not here: it is kept as its record and digest only.
    """
    pins = (pins_for(engine, tidy=False) or []) if pins is None else pins
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


# --- pins: the versions studies hold (design 6) -----------------------------------------------------


def pins_path(engine: Engine) -> Path:
    """``engines/<engine>/pins.json``: ``{"<pin file>": ["<version>", ...]}``, the versions each file pins.

    A list, since one study's pin file can pin two models to different versions of one engine.
    """
    return engines_dir() / engine.value / PINS_FILE


def _pin_map(path: Path) -> dict[str, list[str]] | None:
    """The register as it is: {} when absent; None (logged) when unreadable, so it is never overwritten."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as exc:
        log.warning("%s cannot be read, so no pin is registered or kept by it: %r", path, exc)
        return None
    if not isinstance(raw, dict) or not all(
        isinstance(v, list) and all(isinstance(t, str) for t in v) for v in raw.values()
    ):
        log.warning("%s is not a map of pin file to versions, so no pin is registered or kept by it", path)
        return None
    return raw


def _write_pins(path: Path, pins: Mapping[str, list[str]]) -> None:
    write_text_atomically(path, json.dumps(dict(pins), indent=2) + "\n")


def register_pin(engine: Engine, version: str, pin_file: Path) -> bool:
    """Record that ``pin_file`` pins ``version`` of ``engine``; False (logged) when it could not be recorded.

    Idempotent and cheap when already recorded (read without the lock first), so every pinned lookup calls
    it: a registration dropped while the pin file was out of sight comes back with the next use.
    """
    from .settings import _locked  # noqa: PLC0415 - the pin file's own lock, as settings.py takes it

    path = pins_path(engine)
    if version in (_pin_map(path) or {}).get(str(pin_file), []):
        return True
    try:
        with _locked(path):
            pins = _pin_map(path)
            if pins is None:
                return False
            versions = {*pins.get(str(pin_file), []), version}  # a set: a race cannot write a duplicate
            _write_pins(path, {**pins, str(pin_file): sorted(versions)})
    except OSError as exc:
        log.warning("%s: cannot register the pin of %s %s by %s: %r", path, engine, version, pin_file, exc)
        return False
    return True


def pins_for(engine: Engine, *, tidy: bool = True) -> list[Pin] | None:
    """The registered pins of ``engine`` whose pin file still exists; None when the register is unreadable.

    With ``tidy`` a pin whose file is gone is dropped from the register (logged); without it (a dry run)
    it is only left out of the answer.
    """
    from .settings import _locked  # noqa: PLC0415 - as above

    path = pins_path(engine)
    pins = _pin_map(path)
    if pins is None:
        return None
    gone = [file for file in pins if not Path(file).exists()]
    if gone and tidy:
        with _locked(path):
            pins = _pin_map(path)
            if pins is None:
                return None
            gone = [file for file in pins if not Path(file).exists()]
            for file in gone:
                log.info("%s: pin file %s is gone, so its pins of %s are dropped", engine, file, pins[file])
            _write_pins(path, {file: versions for file, versions in pins.items() if file not in gone})
    return [Pin(v, Path(file)) for file, versions in pins.items() if file not in gone for v in versions]


# --- approvals: a project accepts an engine running a file for a model (design 6) ----------------


class Approval(BaseModel):
    """A project's acceptance of one container engine serving one file in place of one llama.cpp model.

    ``model_key`` is the llama.cpp ``ModelIdentity.key`` of the GGUF (``gguf_*``); ``file_*`` is the
    container engine's file. ``evidence`` is the project's quality check, kept by path and SHA-256.
    """

    project: str
    model_key: str
    gguf_repository: str
    gguf_revision: str
    gguf_file: str
    engine: Engine
    file_repository: str
    file_revision: str
    evidence: str
    evidence_sha256: str
    approved_at: str

    @field_validator("gguf_revision", "file_revision")
    @classmethod
    def _commit(cls, revision: str) -> str:
        if not _REVISION.fullmatch(revision):
            msg = f"a revision is a 40-character commit sha, not {revision!r}"
            raise ValueError(msg)
        return revision

    @field_validator("engine")
    @classmethod
    def _container(cls, engine: Engine) -> Engine:
        if engine not in CONTAINER_ENGINES:
            msg = f"{engine.value} is not a container engine ({CONTAINER_NAMES})"
            raise ValueError(msg)
        return engine

    @field_validator("evidence")
    @classmethod
    def _absolute(cls, evidence: str) -> str:
        if not Path(evidence).is_absolute():
            msg = f"the evidence path must be absolute: {evidence!r}"
            raise ValueError(msg)
        return evidence

    @property
    def file_pin(self) -> str:
        """The container file as ``repository@revision``."""
        return f"{self.file_repository}@{self.file_revision}"


class ApprovalFile(NamedTuple):
    """The store's valid approvals, its items as they stand, and what is wrong with it ("" if nothing)."""

    approvals: list[Approval]
    items: list[Any]
    problem: str


def approvals_path() -> Path:
    """``engines/approvals.json``: ``{"schema": 1, "approvals": [...]}``."""
    return engines_dir() / APPROVALS_FILE


def read_approvals(path: Path | None = None) -> ApprovalFile:
    """Read the store; never raises. A bad file or item is reported in ``problem`` and never replaced."""
    path = path or approvals_path()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        schema, items = payload["schema"], payload["approvals"]
        if not isinstance(schema, int) or not isinstance(items, list):
            msg = "schema is not a number or approvals is not a list"
            raise TypeError(msg)  # noqa: TRY301 - one exit for every damaged shape
        if schema > STORE_SCHEMA:
            return ApprovalFile([], [], f"{path}: schema {schema}, written by a newer localmodels")
    except FileNotFoundError:
        return ApprovalFile([], [], "")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return ApprovalFile([], [], f"{path}: {exc!r}")
    found, problems = [], []
    for index, item in enumerate(items):
        try:
            found.append(Approval.model_validate(item))
        except ValidationError as exc:
            problems.append(f"approval {index}: {exc.errors()[0]['msg']}")
    return ApprovalFile(found, items, f"{path}: {'; '.join(problems)}" if problems else "")


def register_approval(approval: Approval) -> str | None:
    """Store ``approval``, replacing the one for the same project, model and engine; None, or why not."""
    from .settings import _locked  # noqa: PLC0415 - the pin file's own lock, as settings.py takes it

    path = approvals_path()
    with _locked(path):
        found = read_approvals(path)
        if found.problem:
            return found.problem  # reported, never overwritten
        same = (approval.project, approval.model_key, approval.engine)
        kept = [a for a in found.approvals if (a.project, a.model_key, a.engine) != same]
        items = [a.model_dump(mode="json") for a in (*kept, approval)]
        write_text_atomically(path, json.dumps({"schema": STORE_SCHEMA, "approvals": items}, indent=2) + "\n")
    return None


class ApprovalError(ValueError):
    """Why an approval was refused; the message is the refusal text."""


def make_approval(project: str, gguf: str, engine: str, file: str, evidence: str) -> Approval:
    """Check the command's inputs and build the approval; ``ApprovalError`` says what is wrong.

    ``gguf`` is ``<repo>@<revision>:<file>`` (the Hub cache must hold it: its blob hashes make the model's
    key), ``file`` is ``<repo>@<revision>``.
    """
    from .pinned import sha256_file  # noqa: PLC0415 - settings and pinned are loaded only when used
    from .settings import ModelIdentity  # noqa: PLC0415

    if engine not in {e.value for e in CONTAINER_ENGINES}:
        msg = f"{engine!r} is not a container engine ({CONTAINER_NAMES})"
        raise ApprovalError(msg)
    gguf_match, file_match = _GGUF_SPEC.fullmatch(gguf), _FILE_SPEC.fullmatch(file)
    if gguf_match is None:
        msg = f"--gguf must be <repo>@<revision>:<file>, not {gguf!r}"
        raise ApprovalError(msg)
    if file_match is None:
        msg = f"--file must be <repo>@<revision>, not {file!r}"
        raise ApprovalError(msg)
    for what, revision in (("--gguf", gguf_match["revision"]), ("--file", file_match["revision"])):
        if not _REVISION.fullmatch(revision):
            msg = f"{what}: the revision must be a 40-character commit sha, not {revision!r}"
            raise ApprovalError(msg)
    evidence_path = Path(evidence).expanduser().resolve()
    if not evidence_path.is_file():
        msg = f"the evidence file {evidence_path} does not exist"
        raise ApprovalError(msg)
    try:
        identity = ModelIdentity.from_hub_files(
            gguf_match["repository"], gguf_match["revision"], [gguf_match["file"]]
        )
    except ValueError as exc:
        msg = (
            f"the Hub cache does not hold {gguf} (its blob names the model's key): download it first "
            f"(`hf download {gguf_match['repository']} {gguf_match['file']} --revision "
            f"{gguf_match['revision']}`, into the shared cache), then approve; the check said: "
            f"{str(exc).splitlines()[0]}"
        )
        raise ApprovalError(msg) from exc
    return Approval(
        project=project,
        model_key=identity.key,
        gguf_repository=gguf_match["repository"],
        gguf_revision=gguf_match["revision"],
        gguf_file=gguf_match["file"],
        engine=Engine(engine),
        file_repository=file_match["repository"],
        file_revision=file_match["revision"],
        evidence=str(evidence_path),
        evidence_sha256=sha256_file(evidence_path),
        approved_at=datetime.now(UTC).isoformat(timespec="seconds"),
    )


# --- prune ----------------------------------------------------------------------------------------

type Runner = Callable[[Sequence[str]], subprocess.CompletedProcess[str]]


def run_command(
    command: Sequence[str], *, timeout: float = DOCKER_TIMEOUT_SECONDS, env: Mapping[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    """Run one external command (docker, squeue, sparkrun, git, uv) and never raise.

    A missing or unrunnable binary or a timeout is an exit status, not an exception: the caller always gets
    an answer (a stop path that raised would leave its container behind).
    """
    try:
        return subprocess.run(  # noqa: S603 - argv list, no shell
            list(command), check=False, capture_output=True, text=True, timeout=timeout, env=env
        )
    except OSError as exc:  # not found, not executable, ...
        code = RUNNER_NOT_FOUND if isinstance(exc, FileNotFoundError) else RUNNER_CANNOT_RUN
        return subprocess.CompletedProcess(list(command), code, "", str(exc))
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(
            list(command), RUNNER_TIMEOUT, "", f"timed out after {timeout:g} s"
        )


def docker_runner(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    """``run_command`` with Docker's timeout: the default runner of everything here."""
    return run_command(command)


@dataclass
class PruneReport:
    """What a prune did to one engine: images removed, what stays and why, and what could not be removed."""

    removed: list[str]
    kept: dict[str, list[str]]
    failed: list[str]


def _image_id(reference: str, runner: Runner) -> str | None:
    """The local image ID ``reference`` names; None when Docker has no such image."""
    done = runner(["docker", "image", "inspect", "--format", "{{.Id}}", reference])
    return (done.stdout.strip() or None) if done.returncode == 0 else None


def _image_references(record: VersionRecord, runner: Runner) -> list[str]:
    """The names to remove for ``record``'s image: its digest, and its tag while that names the same image.

    A pull by tag (the refresh's first pull) leaves the image under both names, and ``docker rmi`` of the
    digest alone only drops that name: the image stays on disk under the tag (Docker 29.6.2, 2026-11-09:
    ``Untagged: <repo>@sha256:...`` and nothing deleted; job 2220's prune left vllm v0.33.0's 26.3 GB so).
    The tag goes too only while it still names this image, so a tag since moved to another is left alone.
    While a container uses the image Docker removes neither name (checked on 29.6.2), so the digest the
    record holds stays for the next prune to find.
    """
    tagged = f"{record.image.split('@', 1)[0]}:{record.tag}"
    image = _image_id(record.image, runner)
    if image is None:
        return [record.image]  # already gone (or unreadable): ``docker rmi`` says which
    other = _image_id(tagged, runner)
    if other != image:
        names = "no image" if other is None else f"another image ({other})"
        log.info("prune %s: %s names %s, so only the digest is removed", record.tag, tagged, names)
        return [record.image]
    return [record.image, tagged]


def prune(
    engine: Engine,
    *,
    pins: Collection[Pin] | None = None,
    dry_run: bool = False,
    runner: Runner = docker_runner,
) -> PruneReport:
    """Remove the images of every version that is not ``current``, pinned or named in ``users.toml``.

    ``pins`` None reads the registered pins (``keep_list``).

    ``previous`` loses its image too (its record and digest stay). No record is deleted: the records say which
    tags were tried. ``docker rmi`` goes through ``runner``.
    """
    if pins is None:
        pins = pins_for(engine, tidy=not dry_run)
        if pins is None:  # which versions studies pin is unknown: removing any image could break one
            why = f"{pins_path(engine)} cannot be read, so nothing was removed: repair it (it names pins)"
            return PruneReport(removed=[], kept={}, failed=[why])
    kept = keep_list(engine, pins)
    report = PruneReport(removed=[], kept=kept, failed=[])
    kept_images = {r.image for r in records(engine) if r.tag in kept and r.image}
    for folder in unreferenced(versions_dir(engine), keep=kept, by_prefix=False, keep_previous=False):
        record = read_model(folder / RECORD_FILE, VersionRecord)
        if record is None or not record.pulled or not record.image or record.image in kept_images:
            continue  # nothing on disk, or the same image is a kept version's too
        label = f"{record.tag} ({record.image})"
        if dry_run:
            report.removed.append(label)
            continue
        done = runner(["docker", "rmi", *_image_references(record, runner)])
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
    switched = read_record(engine, args.version)
    if switched is not None and not switched.pulled:
        sys.stdout.write(
            f"{engine.value} {args.version}: its image is not on disk; pull it with "
            f"`localmodels engines pull {engine.value} {args.version}` (a CPU job) before a job uses it\n"
        )
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


def cmd_measure(args: argparse.Namespace) -> int:
    """`engines measure`: the measuring harness against a running server (``engine_bench``, loaded on use)."""
    from .engine_bench import cmd_measure as run  # noqa: PLC0415 - the harness is loaded only when used

    return run(args)


def cmd_prompts_install(args: argparse.Namespace) -> int:
    """`engines prompts install`: the pinned measuring inputs into the state directory."""
    from .engine_bench import cmd_prompts_install as run  # noqa: PLC0415 - as above

    return run(args)


def cmd_reap(args: argparse.Namespace) -> int:
    """`engines reap`: the container reaper (``container_run``, loaded on use)."""
    from .container_run import cmd_reap as run  # noqa: PLC0415 - loaded only when used

    return run(args)


def cmd_pull(args: argparse.Namespace) -> int:
    """`engines pull`: submit the CPU job that pulls one version (``engine_refresh``, loaded on use)."""
    from .engine_refresh import cmd_engines_pull as run  # noqa: PLC0415 - loaded only when used

    return run(args)


def cmd_calibrate(args: argparse.Namespace) -> int:
    """`engines calibrate`: submit the pull and calibration jobs for an engine's `current`."""
    from .engine_refresh import cmd_engines_calibrate as run  # noqa: PLC0415 - as above

    return run(args)


def cmd_approve(args: argparse.Namespace) -> int:
    """`engines approve`: store a project's approval, install its recipe and measure the file on `current`."""
    try:
        approval = make_approval(args.project, args.gguf, args.engine, args.file, args.evidence)
    except ApprovalError as exc:
        sys.stderr.write(f"not approving: {exc}\n")
        return EXIT_ERROR
    from .engine_refresh import approve_measurement, recipe_conflict  # noqa: PLC0415 - loaded only when used

    conflict = recipe_conflict(approval)
    if conflict:
        sys.stderr.write(f"not approving: {conflict}\n")
        return EXIT_ERROR
    if not args.dry_run:
        problem = register_approval(approval)
        if problem:
            sys.stderr.write(f"not approving: the approvals store cannot be written: {problem}\n")
            return EXIT_ERROR
    sys.stdout.write(
        f"{'would approve' if args.dry_run else 'approved'} {approval.engine.value} on {approval.file_pin} "
        f"for {approval.project} in place of {approval.gguf_repository}@{approval.gguf_revision}:"
        f"{approval.gguf_file} (model {approval.model_key})\n"
    )
    sys.stdout.writelines(f"{line}\n" for line in approve_measurement(approval, dry_run=args.dry_run))
    return EXIT_OK


def cmd_approvals(_args: argparse.Namespace) -> int:
    """`engines approvals`: every approval, one line each."""
    found = read_approvals()
    if found.problem:
        sys.stderr.write(f"approvals store: {found.problem}\n")
    sys.stdout.writelines(
        f"{a.project}  {a.engine.value}  {a.file_pin}  for {a.gguf_repository}@{a.gguf_revision}:"
        f"{a.gguf_file} ({a.model_key})  evidence {a.evidence}  approved {a.approved_at}\n"
        for a in found.approvals
    )
    return EXIT_ERROR if found.problem else EXIT_OK


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Add the ``engines`` commands (list, switch, pull, calibrate, approve, approvals, prune, ...)."""
    engines = subparsers.add_parser(
        "engines", help="serving engines: container engine versions and the measuring harness"
    )
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

    pull_p = commands.add_parser("pull", help="submit the CPU job that pulls one version's image")
    pull_p.add_argument("engine", choices=[e.value for e in CONTAINER_ENGINES])
    pull_p.add_argument("version", help="tag, as `engines list` prints it")
    pull_p.set_defaults(fn=cmd_pull)

    calibrate_p = commands.add_parser(
        "calibrate", help="submit the jobs that calibrate an engine's quality limits on its `current`"
    )
    calibrate_p.add_argument("engine", choices=[e.value for e in CONTAINER_ENGINES])
    calibrate_p.set_defaults(fn=cmd_calibrate)

    approve_p = commands.add_parser(
        "approve", help="accept a container engine serving a file in place of a llama.cpp model"
    )
    approve_p.add_argument("project", help="the project that accepts it")
    approve_p.add_argument("--gguf", required=True, metavar="REPO@REVISION:FILE", help="the llama.cpp model")
    approve_p.add_argument("--engine", required=True, help=CONTAINER_NAMES)
    approve_p.add_argument("--file", required=True, metavar="REPO@REVISION", help="the container's file")
    approve_p.add_argument("--evidence", required=True, help="the project's quality check (a file)")
    approve_p.add_argument("--dry-run", action="store_true", help="check and print; store and submit nothing")
    approve_p.set_defaults(fn=cmd_approve)
    commands.add_parser("approvals", help="the approvals, one per project, model and engine").set_defaults(
        fn=cmd_approvals
    )

    prune_p = commands.add_parser(
        "prune", help="remove the images that are not current, pinned or named in users.toml"
    )
    prune_p.add_argument("--dry-run", action="store_true", help="list what would be removed")
    prune_p.set_defaults(fn=cmd_prune)

    measure_p = commands.add_parser("measure", help="run the measuring phases against a running server")
    measure_p.add_argument("--url", required=True, help="the server, e.g. http://127.0.0.1:18101")
    measure_p.add_argument(
        "--engine", required=True, help="llama.cpp, vllm, sglang, ... (picks the scoring path)"
    )
    measure_p.add_argument("--out", required=True, help="where to write the record (JSON)")
    measure_p.add_argument("--model", default="auto", help="served model id, or auto to read /v1/models")
    measure_p.add_argument(
        "--extra-json", default="", help="extra request fields, e.g. '{\"cache_prompt\": false}'"
    )
    measure_p.add_argument(
        "--stamp", action="append", default=[], metavar="KEY=VALUE", help="a stamp field (repeat)"
    )
    measure_p.add_argument(
        "--startup-s", type=float, default=None, help="start until ready, measured by the caller"
    )
    measure_p.add_argument(
        "--phases", default="", help="comma list: warm,prefill,decode,conc,judge,quality,..."
    )
    measure_p.add_argument("--tokenizer", default=None, help="tokenizer for llama-benchy")
    measure_p.add_argument("--tiny", action="store_true", help="tiny sizes, for testing against a stub")
    measure_p.set_defaults(fn=cmd_measure)
    prompts_p = commands.add_parser("prompts", help="the pinned measuring inputs")
    prompts_commands = prompts_p.add_subparsers(dest="prompts_command", required=True)
    install_p = prompts_commands.add_parser(
        "install", help="copy D16's inputs into the state directory, by hash"
    )
    install_p.add_argument("--dir", default=None, help="target folder (default: <state dir>/engines/prompts)")
    install_p.set_defaults(fn=cmd_prompts_install)
    commands.add_parser(
        "reap", help="remove engine containers whose slurm job is no longer running"
    ).set_defaults(fn=cmd_reap)
