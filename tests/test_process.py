from pathlib import Path

import pytest

from localmodels.enums import DeviceType, ServingRole
from localmodels.params import derive_serving_params
from localmodels.process import (
    BinaryNotFoundError,
    ServerSpec,
    build_args,
    llama_flags,
    llama_option,
    resolve_llama_server_bin,
)
from localmodels.stack import Settings, SpecType


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


def test_llama_flags_leave_a_group_out_whole_when_the_callers_own_flags_or_env_carry_any_of_it() -> None:
    chosen = Settings(n_ubatch=2048, spec=SpecType.DRAFT_MTP, spec_draft_file="/d.gguf")
    ub, spec = ("-ub", "2048"), ("--spec-type", "draft-mtp", "--spec-draft-model", "/d.gguf")
    assert llama_flags(chosen, (), None) == (*ub, *spec)
    assert llama_flags(Settings(spec=SpecType.NONE), (), None) == llama_flags(Settings(), (), None) == ()
    for flags, env in [
        (("-ub", "512"), None),
        (("--ubatch-size=512",), None),
        ((), {"LLAMA_ARG_UBATCH": "5"}),
    ]:
        assert llama_flags(chosen, flags, env) == spec
    # any speculation option keeps all of them out: an alias, a tuning knob, a variable of either family
    for flags, env in [
        (("-md", "/x.gguf"), None), (("--draft-max=8",), None), (("--spec-draft-n-min", "1"), None),
        ((), {"LLAMA_ARG_SPEC_TYPE": "ngram-mod"}), ((), {"LLAMA_ARG_DRAFT_MIN": "1"}),
    ]:  # fmt: skip
        assert llama_flags(chosen, flags, env) == ub
    # the value in force: the last flag, either form, over the environment
    names = ("-ub", "--ubatch-size")
    assert (
        llama_option(("-ub", "1", "--ubatch-size=2"), {"LLAMA_ARG_UBATCH": "3"}, names, "LLAMA_ARG_UBATCH")
        == "2"
    )
    assert llama_option((), {"LLAMA_ARG_UBATCH": "3"}, names, "LLAMA_ARG_UBATCH") == "3"
    assert llama_option(("-c", "8"), None, names, "LLAMA_ARG_UBATCH") is None
