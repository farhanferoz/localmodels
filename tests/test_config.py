"""The tracked settings files: they load, a broken one is refused naming it, and jobs use what they hold."""

from __future__ import annotations

from pathlib import Path

import pytest

from localmodels import config, digest, engine_bench, engine_refresh, refresh, retune
from localmodels.digest import Finding, Outcome, Source
from localmodels.engines import Publisher
from localmodels.settings import TuneRecipe
from localmodels.stack import Engine


def test_the_tracked_files_load_and_a_broken_one_is_refused_naming_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # the shipped files validate: the timers read them
    assert config.job_settings().refresh.measure_mem == "52G"
    assert config.digest_sources().forum_words.search("A GB10 driver regression")  # (?i) kept
    shipped = (Path(config.__file__).parent / config.JOB_SETTINGS_FILE).read_text()
    (tmp_path / "data").mkdir()
    edited = tmp_path / config.JOB_SETTINGS_FILE
    monkeypatch.setattr(config.resources, "files", lambda _package: tmp_path)
    edited.write_text(shipped.replace('measure_mem = "52G"', 'measure_mem = "52"'))  # no unit
    with pytest.raises(
        ValueError, match=r"(?s)data/job_settings\.toml is not valid: .*measure_mem.*fix that file"
    ):
        config._read(config.JOB_SETTINGS_FILE, config.JobSettings)
    edited.write_text(shipped.replace("[retune]", "[retunes]"))  # a misspelt section is not silently ignored
    with pytest.raises(ValueError, match="retunes"):
        config._read(config.JOB_SETTINGS_FILE, config.JobSettings)
    edited.unlink()
    with pytest.raises(ValueError, match=rf"cannot read {tmp_path}/data/job_settings\.toml: .*reinstall"):
        config._read(config.JOB_SETTINGS_FILE, config.JobSettings)


def test_a_changed_setting_is_what_each_job_and_the_digest_use(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shipped = config.job_settings()
    changed = shipped.model_copy(
        update={
            "refresh": shipped.refresh.model_copy(
                update={"build_cpus": 4, "measure_mem": "60G", "measure_limit_s": 1500, "kill_after_s": 45}
            ),
            "engine_refresh": shipped.engine_refresh.model_copy(
                update={"pull_cpus": 3, "kill_after_s": 200, "same_client_limit_s": 700}
            ),
            "engine_bench": shipped.engine_bench.model_copy(update={"runs": 5}),
            "retune": shipped.retune.model_copy(update={"kill_after_s": 40}),
        }
    )
    monkeypatch.setattr(config, "job_settings", lambda: changed)
    script, run = tmp_path / "submit.py", tmp_path / "run"

    def after(argv: list[str], flag: str) -> str:
        return argv[argv.index(flag) + 1]

    assert after(refresh.build_argv(script, run, "b1", None), "--cpus") == "4"
    measure = refresh.measure_argv(script, run, None, None)
    assert after(measure, "--mem") == "60G"
    assert measure[measure.index("timeout") : measure.index("timeout") + 3] == [
        "timeout",
        "--kill-after=45",
        "1500",
    ]
    assert after(refresh.urgent_argv(script, run, None), "--mem") == "60G"
    assert (
        after(engine_refresh.pull_argv(script, run, Engine.VLLM, "v1", Publisher.OFFICIAL, None), "--cpus")
        == "3"
    )
    same = engine_refresh.same_client_argv(script, run, home=None)
    assert same[same.index("timeout") : same.index("timeout") + 3] == ["timeout", "--kill-after=200", "700"]
    assert after(engine_bench.benchy_command("http://host", "m", tmp_path / "r.json"), "--runs") == "5"
    recipe = TuneRecipe(project="p", command=("tune",))
    assert "--kill-after=40" in retune._submit_argv(script, recipe, "c", "w", tmp_path / "log", "n")

    watched = config.digest_sources()
    monkeypatch.setattr(
        config, "digest_sources", lambda: watched.model_copy(update={"max_lines_per_source": 2})
    )
    note = digest.render_note(
        "2026-12-01", "2026-11-01", {Source.D1: Outcome([Finding(str(i), "u") for i in range(5)])}
    )
    assert "3 more in the raw file" in note
