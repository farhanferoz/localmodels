"""Model download and catalogue status over the Hugging Face Hub, torch-free.

``huggingface_hub`` is imported lazily inside each function that touches the network, so a caller
who only wants ``model_status`` never loads it. ``model_status`` itself never imports it — it only
reads the local disk.
"""

from __future__ import annotations

import os
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .pinned import PinnedFile, fetch_snapshot, repo_dir, sha256_file, snapshot_dir, verify_files
from .process import LocalModelsError

if TYPE_CHECKING:
    from .catalogue import ModelSpec


class UpdateState(StrEnum):
    """The ``status`` of a ``check_updates`` entry; a str, so callers reading the dicts are unchanged."""

    UPDATE_AVAILABLE = "update_available"
    UP_TO_DATE = "up_to_date"
    NOT_CACHED = "not_cached"
    CHECK_FAILED = "check_failed"


class ModelFileError(LocalModelsError):
    """A catalogue file cannot be resolved from the Hub cache: unpinned, nameless or failing its hash."""


def _default_hub_cache() -> Path:
    return Path(os.environ.get("HF_HUB_CACHE") or Path.home() / ".cache" / "huggingface" / "hub")


def _default_hf_home() -> Path:
    return Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))


def _repo_to_cache_dir(repo_id: str, hf_home: Path) -> Path:
    return repo_dir(repo_id, hf_home / "hub")


def _dir_size_mb(path: Path) -> float:
    if path.is_file():
        return path.stat().st_size / (1024 * 1024)
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            file_path = Path(root) / name
            if not file_path.is_symlink():
                total += file_path.stat().st_size
    return total / (1024 * 1024)


def model_status(spec: ModelSpec, *, models_dir: Path, hf_home: Path | None = None) -> dict[str, Any]:
    """Report whether ``spec`` is cached locally, and where — reads the filesystem only."""
    hf_home_given = hf_home is not None
    hf_home = hf_home or _default_hf_home()
    is_cached = False
    local_path: str | None = None
    size_mb: float | None = None
    local_sha: str | None = None

    if spec.filename:
        gguf_path = models_dir / spec.filename
        if not gguf_path.exists() and spec.revision:
            hub = hf_home / "hub" if hf_home_given else _default_hub_cache()
            gguf_path = snapshot_dir(spec.repo_id, spec.revision, hub) / spec.filename
        if gguf_path.exists():
            is_cached = True
            local_path = str(gguf_path)
            size_mb = gguf_path.stat().st_size / (1024 * 1024)
    else:
        cache_dir = _repo_to_cache_dir(spec.repo_id, hf_home)
        if cache_dir.exists():
            snapshots = cache_dir / "snapshots"
            if snapshots.exists() and any(snapshots.iterdir()):
                is_cached = True
                local_path = str(cache_dir)
                size_mb = _dir_size_mb(cache_dir)
                ref_file = cache_dir / "refs" / "main"
                if ref_file.exists():
                    local_sha = ref_file.read_text().strip()

    return {
        "alias": spec.alias,
        "category": spec.category.value,
        "repo_id": spec.repo_id,
        "filename": spec.filename,
        "cached": is_cached,
        "local_path": local_path,
        "size_mb": round(size_mb, 1) if size_mb is not None else None,
        "local_sha": local_sha[:8] if local_sha else None,
    }


def pull_model(spec: ModelSpec, *, models_dir: Path, filename: str | None = None) -> Path:
    """Download ``spec`` (or one file of it) into ``models_dir``, resuming an interrupted pull.

    This copies into ``models_dir``; to share one pinned, verified file through the Hugging Face cache
    instead, use ``resolve_model_file``.
    """
    from huggingface_hub import hf_hub_download, snapshot_download  # noqa: PLC0415 — optional dep

    fn = filename or spec.filename
    if fn:
        models_dir.mkdir(parents=True, exist_ok=True)
        path = hf_hub_download(
            repo_id=spec.repo_id, filename=fn, revision=spec.revision, local_dir=models_dir
        )
        return Path(path)
    return Path(snapshot_download(repo_id=spec.repo_id, revision=spec.revision))


def resolve_model_file(spec: ModelSpec, *, cache_dir: Path | None = None, token: str | None = None) -> Path:
    """Return the path of ``spec.filename`` at ``spec.revision`` in the Hub cache, fetching it only if absent.

    ``cache_dir`` defaults to ``HF_HUB_CACHE``, else ``~/.cache/huggingface/hub``. A file already in the
    cache is served with no network call. When ``spec.sha256`` is recorded, a fresh download is hashed in
    full; a cached file is checked by its blob's name, which the Hub cache sets to a large (LFS) file's
    SHA-256, and hashed in full only when the name differs, so a server start never re-reads gigabytes.
    """
    if not spec.filename:
        msg = f"{spec.alias}: no filename to resolve; only single-file entries come from the cache"
        raise ModelFileError(msg)
    if not spec.revision:
        msg = f"{spec.alias}: no pinned revision; an unpinned model is not served"
        raise ModelFileError(msg)
    cache_dir = cache_dir or _default_hub_cache()
    root = snapshot_dir(spec.repo_id, spec.revision, cache_dir)
    fetched = not (root / spec.filename).is_file()
    if fetched:
        root = fetch_snapshot(spec.repo_id, spec.revision, cache_dir, patterns=[spec.filename], token=token)
    path = root / spec.filename
    checked_by_name = not fetched and path.resolve().name == spec.sha256
    if (
        spec.sha256
        and not checked_by_name
        and verify_files(root, [PinnedFile(path=spec.filename, sha256=spec.sha256)])
    ):
        actual = sha256_file(path) if path.is_file() else "missing"
        msg = f"{spec.filename}: sha256 {actual} does not match the expected {spec.sha256}"
        raise ModelFileError(msg)
    return path


def check_updates(
    specs: list[ModelSpec], *, models_dir: Path, hf_home: Path | None = None
) -> list[dict[str, Any]]:
    """Compare each cached entry's local state against the Hub's current revision."""
    from huggingface_hub import HfApi  # noqa: PLC0415 — optional dep

    api = HfApi()
    results: list[dict[str, Any]] = []
    for spec in specs:
        status = model_status(spec, models_dir=models_dir, hf_home=hf_home)
        try:
            info = api.model_info(spec.repo_id, files_metadata=True, timeout=10.0)
            remote_sha = info.sha
            update_available = False
            if status["cached"]:
                if spec.filename:
                    sibling = next((s for s in (info.siblings or []) if s.rfilename == spec.filename), None)
                    if sibling is not None:
                        lfs = getattr(sibling, "lfs", None)
                        expected_size = getattr(lfs, "size", None) or getattr(sibling, "size", None)
                        local_file = models_dir / spec.filename
                        size_mismatch = (
                            expected_size
                            and local_file.exists()
                            and local_file.stat().st_size != expected_size
                        )
                        if size_mismatch:
                            update_available = True
                elif status.get("local_sha") and remote_sha:
                    update_available = not remote_sha.startswith(status["local_sha"])
                state = UpdateState.UPDATE_AVAILABLE if update_available else UpdateState.UP_TO_DATE
            else:
                state = UpdateState.NOT_CACHED
            results.append(
                {
                    "alias": spec.alias,
                    "repo_id": spec.repo_id,
                    "cached": status["cached"],
                    "local_sha": status.get("local_sha"),
                    "remote_sha": remote_sha[:8] if remote_sha else None,
                    "status": state,
                }
            )
        except Exception as exc:  # noqa: BLE001 — one bad Hub call must not abort the whole scan
            results.append(
                {
                    "alias": spec.alias,
                    "repo_id": spec.repo_id,
                    "cached": status["cached"],
                    "status": UpdateState.CHECK_FAILED,
                    "error": str(exc)[:120],
                }
            )
    return results


def search_hub(query: str, *, limit: int = 5) -> list[dict[str, Any]]:
    """Search the Hub for candidate repositories."""
    from huggingface_hub import HfApi  # noqa: PLC0415 — optional dep

    api = HfApi()
    return [
        {
            "id": m.id,
            "downloads": m.downloads,
            "pipeline_tag": getattr(m, "pipeline_tag", None),
            "last_modified": str(getattr(m, "last_modified", None)),
        }
        for m in api.list_models(search=query, limit=limit, sort="downloads")
    ]
