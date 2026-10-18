import hashlib
from pathlib import Path
from typing import Any

import pytest

from localmodels.catalogue import ModelCategory, ModelSpec
from localmodels.enums import ModelSizeClass
from localmodels.pinned import snapshot_dir
from localmodels.process import LocalModelsError
from localmodels.pull import model_status, resolve_model_file

_GGUF_SPEC = ModelSpec(
    alias="qwen3-moe",
    category=ModelCategory.GENERATOR,
    repo_id="Qwen/Qwen3-30B-A3B-GGUF",
    filename="Qwen3-30B-A3B-Q4_K_M.gguf",
    description="workhorse generator",
    licence="apache-2.0",
    osi_approved=True,
    size_class=ModelSizeClass.LARGE,
)

_SNAPSHOT_SPEC = ModelSpec(
    alias="bge-large",
    category=ModelCategory.EMBED,
    repo_id="BAAI/bge-large-en-v1.5",
    description="embedding model",
    licence="mit",
    osi_approved=True,
    size_class=ModelSizeClass.LARGE,
)


def test_model_status_gguf_not_cached(tmp_path: Path) -> None:
    status = model_status(_GGUF_SPEC, models_dir=tmp_path)
    assert status["cached"] is False
    assert status["local_path"] is None


def test_model_status_gguf_cached(tmp_path: Path) -> None:
    gguf = tmp_path / _GGUF_SPEC.filename
    gguf.write_bytes(b"x" * 1024)
    status = model_status(_GGUF_SPEC, models_dir=tmp_path)
    assert status["cached"] is True
    assert status["local_path"] == str(gguf)
    assert status["size_mb"] == round(1024 / (1024 * 1024), 1)


def test_model_status_snapshot_cached_reads_sha(tmp_path: Path) -> None:
    hf_home = tmp_path / "hf"
    cache_dir = hf_home / "hub" / "models--BAAI--bge-large-en-v1.5"
    snapshots = cache_dir / "snapshots" / "abc123"
    snapshots.mkdir(parents=True)
    (snapshots / "config.json").write_bytes(b"{}")
    refs = cache_dir / "refs"
    refs.mkdir()
    (refs / "main").write_text("abcdef0123456789\n")

    status = model_status(_SNAPSHOT_SPEC, models_dir=tmp_path, hf_home=hf_home)
    assert status["cached"] is True
    assert status["local_sha"] == "abcdef01"


def test_model_status_snapshot_not_cached(tmp_path: Path) -> None:
    status = model_status(_SNAPSHOT_SPEC, models_dir=tmp_path, hf_home=tmp_path / "hf")
    assert status["cached"] is False


_PINNED = "a" * 40


def _fake_fetch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, content: bytes = b"weights") -> list[Any]:
    calls: list[Any] = []

    def fake(repo_id: str, revision: str | None, cache_dir: Path, **kwargs: Any) -> Path:
        calls.append((repo_id, revision, cache_dir, kwargs))
        root = snapshot_dir(repo_id, revision or "", cache_dir)
        root.mkdir(parents=True)
        (root / "Qwen3-30B-A3B-Q4_K_M.gguf").write_bytes(content)
        return root

    monkeypatch.setattr("localmodels.pull.fetch_snapshot", fake)
    return calls


def test_resolve_model_file_returns_cached_snapshot_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = _fake_fetch(monkeypatch, tmp_path)
    digest = hashlib.sha256(b"weights").hexdigest()
    spec = _GGUF_SPEC.model_copy(update={"revision": _PINNED, "sha256": digest})
    path = resolve_model_file(spec, cache_dir=tmp_path)
    assert path == snapshot_dir(spec.repo_id, _PINNED, tmp_path) / "Qwen3-30B-A3B-Q4_K_M.gguf"
    assert calls[0][3]["patterns"] == ["Qwen3-30B-A3B-Q4_K_M.gguf"]


def test_resolve_model_file_names_both_hashes_on_mismatch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _fake_fetch(monkeypatch, tmp_path)
    spec = _GGUF_SPEC.model_copy(update={"revision": _PINNED, "sha256": "b" * 64})
    with pytest.raises(
        LocalModelsError, match=r"Q4_K_M.gguf.*" + hashlib.sha256(b"weights").hexdigest() + ".*" + "b" * 64
    ):
        resolve_model_file(spec, cache_dir=tmp_path)


def test_resolve_model_file_refuses_unpinned_and_nameless(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("localmodels.pull.fetch_snapshot", lambda *_a, **_k: pytest.fail("fetched"))
    with pytest.raises(LocalModelsError, match="no pinned revision"):
        resolve_model_file(_GGUF_SPEC, cache_dir=tmp_path)
    with pytest.raises(LocalModelsError, match="no filename"):
        resolve_model_file(_SNAPSHOT_SPEC.model_copy(update={"revision": _PINNED}), cache_dir=tmp_path)


def test_model_status_finds_a_file_in_the_hub_cache(tmp_path: Path) -> None:
    spec = _GGUF_SPEC.model_copy(update={"revision": _PINNED})
    target = snapshot_dir(spec.repo_id, _PINNED, tmp_path / "hub") / "Qwen3-30B-A3B-Q4_K_M.gguf"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"x")
    status = model_status(spec, models_dir=tmp_path / "none", hf_home=tmp_path)
    assert status["cached"] is True
    assert status["local_path"] == str(target)
