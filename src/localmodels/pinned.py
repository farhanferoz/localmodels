"""Pinned Hugging Face snapshots: a commit revision plus every file's SHA-256, checked before use.

Weights are fetched once at a recorded commit and loaded offline from the local cache, so a model never
changes underneath its results. ``huggingface_hub`` is imported only by ``fetch_snapshot``; checking and
hashing read the local disk alone. Moved here from service_b's model catalogue.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Final

from pydantic import BaseModel, ConfigDict, field_validator

if TYPE_CHECKING:
    from collections.abc import Iterable

_REVISION_RE: Final = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE: Final = re.compile(r"^[0-9a-f]{64}$")
_CHUNK_BYTES: Final = 1 << 20


class PinnedFile(BaseModel):
    """One file of a snapshot: its path inside the snapshot and its SHA-256."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str
    sha256: str

    @field_validator("path")
    @classmethod
    def _relative(cls, value: str) -> str:
        parts = PurePosixPath(value).parts
        if not parts or PurePosixPath(value).is_absolute() or ".." in parts:
            msg = f"file path must be relative and stay inside the snapshot: {value!r}"
            raise ValueError(msg)
        return value

    @field_validator("sha256")
    @classmethod
    def _digest(cls, value: str) -> str:
        if not _SHA256_RE.match(value):
            msg = "sha256 must be 64 lower-case hex characters"
            raise ValueError(msg)
        return value


class PinnedModel(BaseModel):
    """A Hub repository at a pinned commit with the files a model needs; catalogue entries extend it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    repo_id: str
    revision: str = ""
    """Full commit sha; empty until pinned."""
    files: tuple[PinnedFile, ...] = ()

    @field_validator("revision")
    @classmethod
    def _sha(cls, value: str) -> str:
        if value and not _REVISION_RE.match(value):
            msg = "revision must be empty or a 40-character commit sha"
            raise ValueError(msg)
        return value

    @property
    def is_pinned(self) -> bool:
        """Whether a revision and file hashes are recorded."""
        return bool(self.revision) and bool(self.files)

    def snapshot_dir(self, cache_dir: Path) -> Path:
        """Where the pinned snapshot lives in the Hub cache ``cache_dir``."""
        return snapshot_dir(self.repo_id, self.revision, cache_dir)

    def mismatching(self, cache_dir: Path) -> list[str]:
        """Paths of the pinned files missing or differing in SHA-256; empty: every file verified."""
        return verify_files(self.snapshot_dir(cache_dir), self.files)

    def present(self, cache_dir: Path) -> bool:
        """Whether every pinned file is on disk (presence only; ``mismatching`` checks the hashes)."""
        root = self.snapshot_dir(cache_dir)
        return all((root / file.path).is_file() for file in self.files)


def sha256_file(path: Path) -> str:
    """Hex SHA-256 of a file, read in chunks so a large weight file never sits in memory."""
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def repo_dir(repo_id: str, cache_dir: Path) -> Path:
    """The folder of ``repo_id`` in the Hub cache ``cache_dir`` (``huggingface_hub``'s layout)."""
    return cache_dir / ("models--" + repo_id.replace("/", "--"))


def snapshot_dir(repo_id: str, revision: str, cache_dir: Path) -> Path:
    """The snapshot folder of ``repo_id`` at commit ``revision`` in the Hub cache ``cache_dir``."""
    return repo_dir(repo_id, cache_dir) / "snapshots" / revision


def verify_files(root: Path, files: Iterable[PinnedFile]) -> list[str]:
    """Paths of ``files`` missing under ``root`` or differing in SHA-256; empty means every file verified."""
    return [f.path for f in files if not (root / f.path).is_file() or sha256_file(root / f.path) != f.sha256]


def hash_snapshot(snapshot: Path) -> list[PinnedFile]:
    """Every file of a downloaded snapshot with its SHA-256, in path order (what a pin records)."""
    paths = sorted(p for p in snapshot.rglob("*") if p.is_file())
    return [PinnedFile(path=p.relative_to(snapshot).as_posix(), sha256=sha256_file(p)) for p in paths]


def fetch_snapshot(
    repo_id: str,
    revision: str | None,
    cache_dir: Path,
    *,
    patterns: list[str] | None = None,
    token: str | None = None,
    repo_type: str = "model",
) -> Path:
    """Download ``repo_id`` at ``revision`` (None: default branch) into the Hub cache; its snapshot folder.

    ``patterns`` limits the download to matching files (a pin's file list).
    """
    from huggingface_hub import snapshot_download  # noqa: PLC0415 - network path only

    return Path(
        snapshot_download(
            repo_id=repo_id,
            repo_type=repo_type,
            revision=revision,
            cache_dir=cache_dir,
            token=token,
            allow_patterns=patterns,
        )
    )
