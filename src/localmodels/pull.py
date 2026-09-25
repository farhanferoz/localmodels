"""Model download and catalogue status over the Hugging Face Hub, torch-free.

``huggingface_hub`` is imported lazily inside each function that touches the network: this
package's own dependency list does not carry it (B12.4 adds it to the package that runs
``localmodels setup``/``serve``), so a caller who only wants ``model_status`` never needs it
installed. ``model_status`` itself never imports it — it only reads the local disk.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .catalogue import ModelSpec


def _default_hf_home() -> Path:
    return Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))


def _repo_to_cache_dir(repo_id: str, hf_home: Path) -> Path:
    return hf_home / "hub" / ("models--" + repo_id.replace("/", "--"))


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
    hf_home = hf_home or _default_hf_home()
    is_cached = False
    local_path: str | None = None
    size_mb: float | None = None
    local_sha: str | None = None

    if spec.filename:
        gguf_path = models_dir / spec.filename
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
    """Download ``spec`` (or one file of it) into ``models_dir``, resuming an interrupted pull."""
    from huggingface_hub import hf_hub_download, snapshot_download  # noqa: PLC0415 — optional dep

    fn = filename or spec.filename
    if fn:
        models_dir.mkdir(parents=True, exist_ok=True)
        return Path(hf_hub_download(repo_id=spec.repo_id, filename=fn, local_dir=models_dir))
    return Path(snapshot_download(repo_id=spec.repo_id))


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
                state = "update_available" if update_available else "up_to_date"
            else:
                state = "not_cached"
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
                    "status": "check_failed",
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
