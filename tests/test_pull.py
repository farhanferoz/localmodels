from pathlib import Path

from localmodels.catalogue import ModelCategory, ModelSpec
from localmodels.pull import model_status

_GGUF_SPEC = ModelSpec(
    alias="qwen3-moe",
    category=ModelCategory.GENERATOR,
    repo_id="Qwen/Qwen3-30B-A3B-GGUF",
    filename="Qwen3-30B-A3B-Q4_K_M.gguf",
    description="workhorse generator",
    licence="apache-2.0",
    osi_approved=True,
)

_SNAPSHOT_SPEC = ModelSpec(
    alias="bge-large",
    category=ModelCategory.EMBED,
    repo_id="BAAI/bge-large-en-v1.5",
    description="embedding model",
    licence="mit",
    osi_approved=True,
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
