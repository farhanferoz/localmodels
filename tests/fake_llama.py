"""A stand-in for llama.cpp's programs in a fake build folder.

Called as ``fake_llama.py <program> <path of the wrapper> <the program's own arguments>``. It prints the
recorded output of the real program (``bench.json``, ``batched.jsonl`` and ``perplexity.txt`` in
``fixtures/refresh``, from a real run with paths removed), scaled by the build folder's ``fake.json``:

- ``speed``: a factor for every throughput figure, or a list of factors used in turn, one per call
- ``tg``: a further factor for the generation figures
- ``ppl_ratio``, ``mean_kl``, ``same_top`` (percent): the figures llama-perplexity reports against the base
  of another build; against its own base it reports a repeat (``repeat_kl``, ratio 1, 100 % same top token)
- ``fail``: programs that exit 3
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).parent / "fixtures" / "refresh"


def _flag(args: list[str], name: str) -> str:
    return args[args.index(name) + 1]


def _speed(build: Path, cfg: dict, program: str) -> float:
    counter = build / f"calls-{program}"
    calls = int(counter.read_text()) if counter.exists() else 0
    counter.write_text(str(calls + 1))
    speed = cfg.get("speed", 1.0)
    return speed[calls % len(speed)] if isinstance(speed, list) else speed


def _bench(args: list[str], speed: float, tg: float) -> None:
    prompts = {int(p) for p in _flag(args, "-p").split(",")}
    rows = []
    for row in json.loads((HERE / "bench.json").read_text()):
        if row["n_prompt"] in prompts or row["n_gen"] == int(_flag(args, "-n")):
            factor = speed * (tg if row["n_gen"] else 1.0)
            row["avg_ts"] *= factor
            row["samples_ts"] = [s * factor for s in row["samples_ts"]]
            rows.append(row)
    sys.stdout.write(json.dumps(rows) + "\n")


def _batched(args: list[str], speed: float, tg: float) -> None:
    slots = {int(p) for p in _flag(args, "-npl").split(",")}
    for line in filter(None, (HERE / "batched.jsonl").read_text().splitlines()):
        row = json.loads(line)
        if row["pl"] in slots:
            row["speed_pp"] *= speed
            row["speed_tg"] *= speed * tg
            sys.stdout.write(json.dumps(row) + "\n")


def _perplexity(args: list[str], cfg: dict, build: Path) -> None:
    base = Path(_flag(args, "--kl-divergence-base"))
    if "--kl-divergence" not in args:
        base.write_text(build.name)  # whose base it is
        sys.stderr.write("saved base\n")
        return
    own = base.read_text() == build.name
    ratio, kl, top = (
        (1.0, cfg.get("repeat_kl", 0.00005), 100.0)
        if own
        else (cfg.get("ppl_ratio", 1.0003), cfg.get("mean_kl", 0.012), cfg.get("same_top", 98.7))
    )
    text = (HERE / "perplexity.txt").read_text()
    text = re.sub(r"(Mean PPL\(Q\)/PPL\(base\)\s*:\s*)[\d.]+", rf"\g<1>{ratio:.6f}", text)
    text = re.sub(r"(Mean\s+KLD:\s*)[\d.]+", rf"\g<1>{kl:.6f}", text)
    text = re.sub(r"(Same top p:\s*)[\d.]+", rf"\g<1>{top:.3f}", text)
    sys.stdout.write(text)  # llama-perplexity prints its statistics through LOG(), on stdout


def main() -> int:
    program, wrapper, *args = sys.argv[1:]
    build = Path(wrapper).resolve().parent.parent
    cfg = json.loads((build / "fake.json").read_text())
    if program in cfg.get("fail", []):
        sys.stderr.write("fake failure\n")
        return 3
    if program == "llama-perplexity":
        _perplexity(args, cfg, build)
        return 0
    speed, tg = _speed(build, cfg, program), cfg.get("tg", 1.0)
    (_bench if program == "llama-bench" else _batched)(args, speed, tg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
