"""A keyed llama-server (ported from service_a's G10 carry-over item B15).

llama-server at ``ccc3646`` checks ``Authorization: Bearer`` on every route except ``/health`` and
``/v1/health`` (``tools/server/server-http.cpp``), so ``/props`` probes carry the key too.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from localmodels.params import ServingParams
from localmodels.process import (
    ApiKeyRequiredError,
    ServerSpec,
    auth_headers,
    build_args,
    require_key_off_loopback,
)


def _spec(host: str, key_file: Path | None) -> ServerSpec:
    return ServerSpec(host=host, port=8091, model_path=Path("m.gguf"), api_key_file=key_file)


def _params() -> ServingParams:
    return ServingParams.model_construct(n_ctx=4096, n_parallel=1, n_gpu_layers=-1)


def test_argv_carries_api_key_file_when_configured(tmp_path: Path) -> None:
    key = tmp_path / "k"
    key.write_text("k" * 40 + "\n")
    args = build_args("llama-server", _spec("172.17.0.1", key), _params())
    assert args[args.index("--api-key-file") + 1] == str(key)
    assert "--api-key-file" not in build_args("llama-server", _spec("127.0.0.1", None), _params())


@pytest.mark.parametrize("host", ["172.17.0.1", "0.0.0.0", "192.0.2.52"])  # noqa: S104 -- the case under test
def test_non_loopback_host_without_key_is_refused(host: str) -> None:
    with pytest.raises(ApiKeyRequiredError):
        require_key_off_loopback(_spec(host, None))


def test_loopback_or_keyed_is_allowed(tmp_path: Path) -> None:
    key = tmp_path / "k"
    key.write_text("x" * 40)
    require_key_off_loopback(_spec("127.0.0.1", None))
    require_key_off_loopback(_spec("localhost", None))
    require_key_off_loopback(_spec("172.17.0.1", key))


def test_auth_headers_send_the_first_key_and_skip_comments(tmp_path: Path) -> None:
    key = tmp_path / "k"
    key.write_text("# rotated\n" + "a" * 40 + "\n")
    assert auth_headers(key) == {"Authorization": "Bearer " + "a" * 40}
    assert auth_headers(None) == {}
