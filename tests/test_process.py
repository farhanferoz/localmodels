from pathlib import Path

import pytest

from localmodels.enums import ModelSizeClass, QuantLevel
from localmodels.params import ServingParams
from localmodels.process import BinaryNotFoundError, ServerSpec, build_args, resolve_llama_server_bin


def test_build_args_cuda_and_cpu() -> None:
    spec = ServerSpec(host="127.0.0.1", port=8091, model_path=Path("/models/qwen.gguf"))

    params_cuda = ServingParams(
        n_gpu_layers=-1,
        n_ctx=8192,
        n_parallel=2,
        recommended_quant=QuantLevel.Q4_K_M,
        recommended_size_class=ModelSizeClass.LARGE,
        assumptions=[],
    )
    args_cuda = build_args("/bin/llama-server", spec, params_cuda)
    idx = args_cuda.index("-ngl")
    assert args_cuda[idx + 1] == "99"

    params_cpu = ServingParams(
        n_gpu_layers=0,
        n_ctx=4096,
        n_parallel=1,
        recommended_quant=None,
        recommended_size_class=ModelSizeClass.SMALL,
        assumptions=[],
    )
    args_cpu = build_args("/bin/llama-server", spec, params_cpu)
    idx_cpu = args_cpu.index("-ngl")
    assert args_cpu[idx_cpu + 1] == "0"


def test_build_args_appends_extra_flags() -> None:
    spec = ServerSpec(host="127.0.0.1", port=8091, model_path=Path("/models/qwen.gguf"))
    params = ServingParams(
        n_gpu_layers=-1,
        n_ctx=8192,
        n_parallel=2,
        recommended_quant=QuantLevel.Q4_K_M,
        recommended_size_class=ModelSizeClass.LARGE,
        assumptions=[],
    )
    args = build_args("/bin/llama-server", spec, params, extra_flags=("--spec-type", "draft"))
    assert args[-2:] == ["--spec-type", "draft"]


def test_binary_chain_resolution(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # 1. env var
    env_bin = tmp_path / "env_bin"
    env_bin.write_text("#!/bin/sh\n")
    env_bin.chmod(0o755)
    monkeypatch.setenv("LLAMA_SERVER_BIN", str(env_bin))
    assert resolve_llama_server_bin() == env_bin
    monkeypatch.delenv("LLAMA_SERVER_BIN")

    # 2. explicit path
    explicit_bin = tmp_path / "explicit_bin"
    explicit_bin.write_text("#!/bin/sh\n")
    explicit_bin.chmod(0o755)
    assert resolve_llama_server_bin(explicit=explicit_bin) == explicit_bin

    # 3. lib_dir
    lib_dir = tmp_path / "lib"
    lib_dir.mkdir()
    lib_bin = lib_dir / "llama-server"
    lib_bin.write_text("#!/bin/sh\n")
    lib_bin.chmod(0o755)
    assert resolve_llama_server_bin(lib_dir=lib_dir) == lib_bin

    # 4. none exists -> BinaryNotFoundError
    with pytest.raises(BinaryNotFoundError, match="could not resolve llama-server"):
        resolve_llama_server_bin(explicit=tmp_path / "absent")
