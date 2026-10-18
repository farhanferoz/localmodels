from pathlib import Path

import pytest

from localmodels.enums import DeviceType, ServingRole
from localmodels.params import derive_serving_params
from localmodels.process import BinaryNotFoundError, ServerSpec, build_args, resolve_llama_server_bin


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


def test_build_args_offloads_by_device_adds_kvu_only_for_a_role_and_keeps_extra_flags_last() -> None:
    spec = ServerSpec(host="127.0.0.1", port=8091, model_path=Path("/models/qwen.gguf"))
    cpu = build_args("/bin/llama-server", spec, derive_serving_params(DeviceType.CPU, 8.0))
    assert cpu[cpu.index("-ngl") + 1] == "0"
    plain = derive_serving_params(DeviceType.CUDA, 100.0)
    assert build_args("/bin/llama-server", spec, plain) == [
        "/bin/llama-server", "-m", "/models/qwen.gguf", "-ngl", "99", "-c", str(plain.n_ctx),
        "-np", str(plain.n_parallel), "--jinja", "--host", "127.0.0.1", "--port", "8091",
    ]  # fmt: skip

    for role, slots in ((ServingRole.INTERACTIVE, "4"), (ServingRole.BULK, "16")):
        params = derive_serving_params(DeviceType.CUDA, 100.0, role=role)
        args = build_args("/bin/llama-server", spec, params, extra_flags=("--spec-type", "draft-mtp"))
        assert args[args.index("-np") : args.index("-np") + 3] == ["-np", slots, "-kvu"]
        assert "-ub" not in args
        assert args[-2:] == ["--spec-type", "draft-mtp"]
