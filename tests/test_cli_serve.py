"""Unit tests for `localmodels serve`'s CLI wiring.

`run_serve`'s own behaviour (the restart-after-yield loop) is covered by
`tests/test_service.py`; this only checks that `main(["serve", "--config", ...])` calls
`service._cmd_serve` with the right path.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import localmodels.cli as cli_mod
from localmodels.cli import EXIT_OK, main

if TYPE_CHECKING:
    from pathlib import Path


def test_cmd_serve_forwards_config_path(monkeypatch, tmp_path: Path) -> None:
    captured = []

    def fake_cmd_serve(config_path: Path) -> int:
        captured.append(config_path)
        return EXIT_OK

    monkeypatch.setattr(cli_mod, "_cmd_serve", fake_cmd_serve)

    config_path = tmp_path / "serve.toml"
    code = main(["serve", "--config", str(config_path)])
    assert code == EXIT_OK
    assert captured == [config_path]


def test_serve_requires_config_argument() -> None:
    parser = cli_mod.build_parser()
    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(["serve"])
    assert exc_info.value.code != 0
