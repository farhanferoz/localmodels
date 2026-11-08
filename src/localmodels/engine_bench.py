"""One measuring harness for every serving engine: ``engines measure`` and ``engines prompts install``.

Imported lazily (never by ``import localmodels``). It is the D16 client (``bench_client.py``, proven on
llama-server, vLLM and SGLang in slurm 2160) moved into the package, kept to the phases the gates and the
chooser read (design 4.1): prompt reading, one request writing, 16 at once writing, the judge batch (16 real
prompts at once), the 48 fixed prompts answered greedily (reported only, never a gate) and the teacher-forced
scoring of the pinned text, plus the pinned ``llama-benchy`` figure. The start until ready is measured by the
caller and passed in. Every completion phase goes through ``/v1/completions`` on one code path; request fields
an engine needs beyond the OpenAI schema (llama-server's ``cache_prompt: false``) are passed as data
(``extra``), never chosen by engine name. The one exception is teacher-forced scoring, whose request and
response differ per engine.

A phase that fails is recorded as not measured with its reason; ``measure`` never raises for a phase.

Teacher-forced scoring, fields ASSUMED FROM DOCUMENTATION, NOT YET SEEN FROM A REAL ENGINE (E4's bootstrap run
is where they meet one; a mismatch fails the phase and is recorded as such):

- vLLM, ``POST /v1/completions`` with ``max_tokens: 1`` and ``prompt_logprobs: 1`` (docs.vllm.ai
  "OpenAI-Compatible Server", extra parameters). Response ``choices[0].prompt_logprobs``: a list with one
  entry per prompt token; entry 0 is ``null`` (nothing precedes the first token); every other entry is a dict
  keyed by token id (as a string) whose values carry ``logprob``, ``rank`` and ``decoded_token`` (vllm's
  ``Logprob`` object). The dict holds the top token and the prompt's own token (one key when they are the
  same). Which key is the prompt's own token is not marked, so it is taken as the entry of highest ``rank``
  (the top token has rank 1): an assumption about ``rank``.
- SGLang, ``POST /generate`` with ``text``, ``sampling_params: {max_new_tokens: 1, temperature: 0}``,
  ``return_logprob: true``, ``logprob_start_len: 0`` and ``top_logprobs_num: 1`` (docs.sglang.ai sampling
  parameters). Response ``meta_info.input_token_logprobs``: one ``[logprob, token_id, text]`` per prompt
  token, the first logprob ``null``; ``meta_info.input_top_logprobs``: per position a list of
  ``[logprob, token_id, text]`` (the first position ``null`` or empty). The entry layout and the ``meta_info``
  nesting are from memory of the documented examples, not from the parameter page the request fields come
  from.

Seen from a real file, not assumed: ``llama-benchy`` 0.4.0's JSON (``--format json --save-result``), read from
sparkrun's run of 2026-11-08 (``benchmarks[].pp_throughput.mean`` and ``tg_throughput.mean`` in tokens per
second, ``ttfr.mean`` in milliseconds, ``context_size``, ``concurrency``, ``is_context_prefill_phase``).
"""

from __future__ import annotations

import concurrent.futures as cf
import hashlib
import json
import logging
import math
import statistics
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import httpx
from pydantic import BaseModel, Field

from .atomic import write_model, write_text_atomically
from .settings import state_dir

if TYPE_CHECKING:
    import argparse
    from collections.abc import Callable, Sequence

log = logging.getLogger(__name__)

NOTHINK: Final = "<think>\n\n</think>\n\n"
END_OF_TURN: Final = ["<|im_end|>", "<|im_start|>"]
CHARS_PER_TOKEN: Final = 4
MIN_CHUNKS_FOR_RATE: Final = 3
REQUEST_TIMEOUT_S: Final = 900.0
BENCHY_TIMEOUT_S: Final = 900.0
BENCHY_PACKAGE: Final = "llama-benchy==0.4.0"
BENCHY_PROMPT_TOKENS: Final = 2048
BENCHY_WRITE_TOKENS: Final = 32
BENCHY_RUNS: Final = 3
TF_TOP_K: Final = 1  # top tokens asked for per position
VARIANT_STRIDE: Final = 7919  # a prime, so successive variants start far apart in the text

# Where D16 read its inputs from, and what they hash to. The pinned text is the text of llama.cpp's docs/*.md
# (sorted, read with errors ignored, joined by a blank line) at commit ccc3646 (unchanged since 2026-09-07,
# before D16 ran); the judge prompts are tracecraft's rendered judge prompts of hl5.
D16_DOCS_DIR: Final = Path("/home/ff235/dev/llama.cpp/docs")
D16_JUDGE_PROMPTS: Final = Path("/home/ff235/cache/localmodels/hl5/prompts.jsonl")
JUDGE_FILE: Final = "judge_prompts.jsonl"
TEXT_FILE: Final = "pinned_text.txt"
JUDGE_SHA256: Final = "30a293c19cfb15e580dd3e757b18f3b25d9e089d95fac445df9ba4a877d4236f"
TEXT_SHA256: Final = "053646c45414166c473768fb1a9cf34037e084117f6944f23732cdb576881e77"


class PromptFileError(RuntimeError):
    """A pinned input is missing, or its bytes are not the ones that were pinned."""


class PhaseError(RuntimeError):
    """A phase could not be measured; the message is the reason recorded."""


class ScoringError(PhaseError):
    """A teacher-forced response did not have the documented shape."""


class Phase(StrEnum):
    """The measured phases, in the order they run."""

    WARM = "warm"
    PREFILL = "prefill"
    DECODE = "decode"
    CONC = "conc"
    JUDGE = "judge"
    QUALITY = "quality"
    TEACHER_FORCED = "teacher_forced"
    BENCHY = "benchy"


class PhaseStatus(StrEnum):
    """Whether a phase produced figures."""

    MEASURED = "measured"
    NOT_MEASURED = "not_measured"


class ScoringPath(StrEnum):
    """How an engine scores a given text token by token."""

    VLLM_PROMPT_LOGPROBS = "vllm_prompt_logprobs"
    SGLANG_GENERATE = "sglang_generate"


# Engine names (``engines measure --engine``) that have a scoring path; llama.cpp keeps its own.
SCORING_PATHS: Final = {"vllm": ScoringPath.VLLM_PROMPT_LOGPROBS, "sglang": ScoringPath.SGLANG_GENERATE}


@dataclass(frozen=True)
class Sizes:
    """Everything that sets how much work a phase does. ``TINY`` is only for testing against a stub."""

    prefill_tokens: tuple[int, ...] = (512, 2048, 8192)  # approximate; measured about 550 / 3,100 / 11,000
    prefill_reps: int = 3
    decode_reps: int = 5
    decode_tokens: int = 256
    conc_levels: tuple[int, ...] = (1, 16)
    conc_reps: int = 2
    conc_tokens: int = 128
    conc_prompt_tokens: int = 200
    judge_requests: int = 16
    judge_reps: int = 2
    judge_tokens: int = 128
    quality_n: int = 48
    quality_tokens: int = 96
    tf_chunks: int = 8
    tf_chunk_chars: int = 2400  # about 600 tokens


FULL: Final = Sizes()
TINY: Final = Sizes(
    prefill_tokens=(64, 128, 256),
    prefill_reps=2,
    decode_reps=2,
    decode_tokens=12,
    conc_levels=(1, 2),
    conc_reps=2,
    conc_tokens=8,
    conc_prompt_tokens=20,
    judge_requests=4,
    judge_reps=2,
    judge_tokens=8,
    quality_n=4,
    quality_tokens=12,
    tf_chunks=3,
    tf_chunk_chars=120,
)

# 48 fixed quality prompts: short, varied, deterministic (greedy, 96 tokens). Copied from D16's client.
QUALITY_PROMPTS: Final = (
    "Explain in two sentences why the sky is blue.",
    "Write a Python function that returns the n-th Fibonacci number iteratively.",
    "What is the capital of Australia, and why is it not Sydney?",
    "Summarise the plot of Romeo and Juliet in three sentences.",
    "List five differences between TCP and UDP.",
    "Translate to French: 'The library closes at nine o'clock tonight.'",
    "What is 17 * 23? Show the working.",
    "Give a one-paragraph explanation of how a hash table handles collisions.",
    "Write a haiku about autumn rain.",
    "Explain the difference between a process and a thread.",
    "What causes the seasons on Earth?",
    (
        "Write a SQL query that returns the top 3 customers by total order value from tables "
        "customers(id, name) and orders(id, customer_id, total)."
    ),
    "Describe the main idea of gradient descent to a high-school student.",
    "What are the three laws of thermodynamics? One line each.",
    "Rewrite this sentence to be more formal: 'Hey, can you send me that report asap?'",
    "Solve for x: 3x + 7 = 25.",
    "Explain what a mutex is and when you would use one.",
    "Name four causes of the First World War.",
    "Write a bash one-liner that counts the lines in every .txt file in a directory.",
    "What is the difference between precision and recall?",
    "Describe how photosynthesis works in plain language.",
    "Give three tips for writing a clear email subject line.",
    "What is the time complexity of binary search and why?",
    "Explain what an API rate limit is.",
    "Write a short polite message declining a meeting invitation.",
    "What does the Python keyword 'yield' do?",
    "Why do we use cross-validation in machine learning?",
    "Summarise the difference between stocks and bonds.",
    "Convert 98.6 degrees Fahrenheit to Celsius and show the formula.",
    "Explain the CAP theorem in simple terms.",
    "Write a regular expression that matches a UK postcode loosely, and explain it.",
    "What is the difference between 'affect' and 'effect'?",
    "Describe what DNS does when you type a web address.",
    "Give a short definition of entropy in information theory.",
    "Write a JavaScript function that debounces another function.",
    "What are the main symptoms of dehydration?",
    "Explain what a Docker container is compared to a virtual machine.",
    "How does a binary heap support a priority queue?",
    "Write two sentences describing a quiet harbour at dawn.",
    "What is the Pythagorean theorem? Give an example.",
    "Explain overfitting and one way to reduce it.",
    "Why is the Moon always showing the same face to Earth?",
    "Write a polite reminder about an overdue invoice.",
    "Explain what the 'git rebase' command does.",
    "What is inflation, and how do central banks respond to it?",
    "Give a brief comparison of REST and GraphQL.",
    "Describe the water cycle in four steps.",
    "What is the derivative of x^3 * sin(x)?",
)


# --- the record -----------------------------------------------------------------------------------


class PhaseOutcome(BaseModel):
    """Whether one phase was measured, why not if it was not, and how long it took."""

    status: PhaseStatus
    reason: str | None = None
    seconds: float = 0.0


class TeacherForced(BaseModel):
    """The pinned text scored token by token."""

    path: ScoringPath
    chunks: int
    positions: int
    mean_logprob: float
    perplexity: float
    top_tokens: list[list[int]]  # per chunk, the id of the most likely token at each scored position


class BenchyFigures(BaseModel):
    """``llama-benchy``'s figures at prompt 2,048, writing 32 tokens (tokens/s; first response in ms)."""

    package: str
    prompt_tokens: int
    write_tokens: int
    read_tps: float | None
    write_tps: float | None
    first_response_ms: float | None
    runs: int


class EngineRun(BaseModel):
    """Everything one ``measure`` run found, written as one file. ``stamp`` is filled by the caller."""

    version: int = 1
    engine: str
    model: str
    url: str
    started: str
    tiny: bool = False
    extra: dict[str, Any] | None = None
    stamp: dict[str, str] = Field(default_factory=dict)
    startup_s: float | None = None
    phases: dict[Phase, PhaseOutcome] = Field(default_factory=dict)
    details: dict[str, Any] = Field(default_factory=dict)
    greedy_outputs: list[str] = Field(default_factory=list)
    teacher_forced: TeacherForced | None = None
    benchy: BenchyFigures | None = None
    figures: dict[str, float | None] = Field(default_factory=dict)


# --- pinned inputs --------------------------------------------------------------------------------


@dataclass(frozen=True)
class PromptPins:
    """The SHA-256 each pinned input must have."""

    judge: str = JUDGE_SHA256
    text: str = TEXT_SHA256


PINS: Final = PromptPins()


@dataclass(frozen=True)
class Prompts:
    """The pinned inputs, loaded and checked."""

    judge: tuple[str, ...]
    text: str


def prompts_dir() -> Path:
    """Where the pinned inputs live: ``$LOCALMODELS_STATE_DIR/engines/prompts``."""
    return state_dir() / "engines" / "prompts"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_pinned(path: Path, expected: str) -> bytes:
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise PromptFileError(
            f"{path} cannot be read ({exc}); run `localmodels engines prompts install`"
        ) from exc
    got = _sha256(data)
    if got != expected:
        raise PromptFileError(
            f"{path} is not the pinned file: its SHA-256 is {got}, the pinned one is {expected}; "
            "a changed input would make this run incomparable with earlier records"
        )
    return data


def load_prompts(folder: Path | None = None, pins: PromptPins | None = None) -> Prompts:
    """The judge prompts and the pinned text from ``folder``, each checked by its SHA-256."""
    folder = folder or prompts_dir()
    pins = pins or PINS
    judge = _read_pinned(folder / JUDGE_FILE, pins.judge).decode()
    text = _read_pinned(folder / TEXT_FILE, pins.text).decode()
    lines = tuple(json.loads(line)["prompt"] for line in judge.splitlines() if line.strip())
    return Prompts(judge=lines, text=text)


def install_prompts(
    folder: Path | None = None,
    *,
    docs_dir: Path = D16_DOCS_DIR,
    judge_source: Path = D16_JUDGE_PROMPTS,
    pins: PromptPins | None = None,
) -> list[Path]:
    """Copy what D16 used into ``folder``, refusing a source whose SHA-256 is not the pinned one."""
    folder = folder or prompts_dir()
    pins = pins or PINS
    try:
        judge = judge_source.read_bytes()
        text = "\n\n".join(p.read_text(errors="ignore") for p in sorted(docs_dir.glob("*.md"))).encode()
    except OSError as exc:
        raise PromptFileError(f"cannot read the D16 inputs: {exc}") from exc
    for name, data, expected in ((judge_source, judge, pins.judge), (docs_dir, text, pins.text)):
        if _sha256(data) != expected:
            raise PromptFileError(
                f"{name} is not what D16 used: SHA-256 {_sha256(data)}, pinned {expected}; nothing installed"
            )
    written = [
        write_text_atomically(folder / JUDGE_FILE, judge.decode()),
        write_text_atomically(folder / TEXT_FILE, text.decode()),
    ]
    load_prompts(folder, pins)  # what was written reads back as the pinned bytes
    return written


# --- one streamed completion ----------------------------------------------------------------------


@dataclass(frozen=True)
class Target:
    """A running server: the HTTP client, the served model id and the extra request fields."""

    client: httpx.Client
    model: str
    extra: dict[str, Any] | None = None


@dataclass(frozen=True)
class Completion:
    """One streamed ``/v1/completions`` request: timings, text and token counts."""

    ttft: float | None
    total: float
    stamps: list[float]
    completion_tokens: int
    prompt_tokens: int
    prompt_tokens_estimated: bool
    text: str
    finish_reason: str | None


def served_model(client: httpx.Client) -> str:
    """The model id the server reports at ``GET /v1/models``."""
    response = client.get("/v1/models")
    response.raise_for_status()
    return str(response.json()["data"][0]["id"])


def stream_completion(target: Target, prompt: str, max_tokens: int, *, ignore_eos: bool = True) -> Completion:
    """One streamed ``/v1/completions`` request (greedy). Raises ``httpx.HTTPError`` on a failed request."""
    body: dict[str, Any] = {
        "model": target.model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if ignore_eos:
        body["ignore_eos"] = True
    else:
        body["stop"] = END_OF_TURN
    if target.extra:
        body.update(target.extra)
    t0 = time.perf_counter()
    stamps: list[float] = []
    parts: list[str] = []
    usage: dict[str, Any] = {}
    finish: str | None = None
    with target.client.stream("POST", "/v1/completions", json=body) as response:
        response.raise_for_status()
        for raw in response.iter_lines():
            line = raw.strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            chunk = json.loads(payload)
            usage = chunk.get("usage") or usage
            for choice in chunk.get("choices", []):
                if choice.get("text"):
                    stamps.append(time.perf_counter())
                    parts.append(choice["text"])
                finish = choice.get("finish_reason") or finish
    t_end = time.perf_counter()
    n_prompt = usage.get("prompt_tokens")
    return Completion(
        ttft=(stamps[0] - t0) if stamps else None,
        total=t_end - t0,
        stamps=[s - t0 for s in stamps],
        completion_tokens=usage.get("completion_tokens", len(stamps)),
        prompt_tokens=n_prompt if n_prompt is not None else len(prompt) // CHARS_PER_TOKEN,
        prompt_tokens_estimated=n_prompt is None,
        text="".join(parts),
        finish_reason=finish,
    )


def decode_rate(res: Completion) -> float | None:
    """Tokens per second after the first, from the server's token count and the first and last chunk times."""
    s = res.stamps
    if len(s) < MIN_CHUNKS_FOR_RATE or s[-1] <= s[0]:
        return None
    return (res.completion_tokens - 1) / (s[-1] - s[0])


def med(values: Sequence[float | None]) -> float | None:
    """The median of the values that exist; None when none does."""
    kept = [x for x in values if x is not None]
    return statistics.median(kept) if kept else None


def chatml(user: str) -> str:
    """A raw ChatML prompt in Qwen3's no-think form, so every engine sees the same tokens."""
    return f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n{NOTHINK}"


def _slice(text: str, start: int, chars: int) -> str:
    """``chars`` characters from ``start``; a text shorter than asked wraps around."""
    text *= chars // max(1, len(text)) + 2
    return text[start % max(1, len(text) - chars - 1) :][:chars]


def long_prompt(approx_tokens: int, variant: int, text: str) -> str:
    """A prompt of about ``approx_tokens`` tokens whose FIRST line differs per variant (no cache hit)."""
    chars = approx_tokens * CHARS_PER_TOKEN
    body = _slice(text, variant * VARIANT_STRIDE, chars)
    return chatml(
        f"[document id {variant:04d}-{approx_tokens}]\n{body}\n\nSummarise the text above in one sentence."
    )


def judge_prompts(lines: Sequence[str], n: int, rep: int) -> list[str]:
    """The first ``n`` pinned judge prompts, each with its own first line (defeats a prefix cache)."""
    return [f"[judge batch {rep}-{i:02d}]\n{p}" for i, p in enumerate(lines[:n])]


def _batch(
    target: Target, prompts: Sequence[str], max_tokens: int, workers: int | None = None
) -> tuple[float, list[Completion]]:
    """All prompts through ``workers`` threads (default: all at once); the wall time and the completions."""
    t0 = time.perf_counter()
    with cf.ThreadPoolExecutor(max_workers=workers or len(prompts)) as pool:
        results = list(pool.map(lambda p: stream_completion(target, p, max_tokens), prompts))
    return time.perf_counter() - t0, results


# --- the completion phases ------------------------------------------------------------------------


def phase_warm(target: Target, _sizes: Sizes) -> dict[str, Any]:
    """Two short requests, so the first measured one does not pay for a cold path."""
    for _ in range(2):
        r = stream_completion(target, chatml("Say hello."), 16)
    return {"warm_ttft": r.ttft, "warm_total": r.total}


def phase_prefill(target: Target, sizes: Sizes, text: str) -> dict[str, Any]:
    """Prompt reading: one token written after prompts of several sizes."""
    out: dict[str, Any] = {}
    for approx in sizes.prefill_tokens:
        reps = []
        for v in range(sizes.prefill_reps):
            r = stream_completion(target, long_prompt(approx, 100 * approx + v, text), 1)
            tps = (r.prompt_tokens / r.ttft) if r.ttft else None
            reps.append({"prompt_tokens": r.prompt_tokens, "ttft_s": r.ttft, "prefill_tps": tps})
        out[str(approx)] = {
            "reps": reps,
            "median_prompt_tokens": med([x["prompt_tokens"] for x in reps]),
            "median_prefill_tps": med([x["prefill_tps"] for x in reps]),
            "median_ttft_s": med([x["ttft_s"] for x in reps]),
        }
    return out


def phase_decode(target: Target, sizes: Sizes) -> dict[str, Any]:
    """One request writing. The first repetition warms the decode path: recorded, left out of the median."""
    reps = []
    for i in range(sizes.decode_reps):
        r = stream_completion(
            target, chatml(f"Write a long story about a lighthouse keeper, variant {i}."), sizes.decode_tokens
        )
        reps.append({"ttft_s": r.ttft, "decode_tps": decode_rate(r), "tokens": r.completion_tokens})
    keep = reps[1:] or reps
    return {
        "reps": reps,
        "median_decode_tps": med([x["decode_tps"] for x in keep]),
        "median_ttft_s": med([x["ttft_s"] for x in keep]),
    }


def phase_conc(target: Target, sizes: Sizes, text: str) -> dict[str, Any]:
    """Several requests at once, all writing together."""
    out: dict[str, Any] = {}
    for c in sizes.conc_levels:
        n_req = 4 if c == 1 else 2 * c
        runs = []
        for rep in range(sizes.conc_reps):
            prompts = [
                long_prompt(sizes.conc_prompt_tokens, 5000 + 97 * rep + i + 1000 * c, text)
                for i in range(n_req)
            ]
            wall, results = _batch(target, prompts, sizes.conc_tokens, c)
            tokens = sum(r.completion_tokens for r in results)
            ttfts = sorted(r.ttft or 0.0 for r in results)
            runs.append(
                {
                    "requests": n_req,
                    "wall_s": wall,
                    "output_tokens": tokens,
                    "total_tps": tokens / wall,
                    "per_request_decode_tps_median": med([decode_rate(r) for r in results]),
                    "ttft_p50_s": ttfts[len(ttfts) // 2],
                    "ttft_max_s": ttfts[-1],
                }
            )
        out[str(c)] = {
            "runs": runs,
            "total_tps_median": med([r["total_tps"] for r in runs]),
            "ttft_p50_s_median": med([r["ttft_p50_s"] for r in runs]),
            "ttft_max_s_median": med([r["ttft_max_s"] for r in runs]),
        }
    return out


def phase_judge(target: Target, sizes: Sizes, lines: Sequence[str]) -> dict[str, Any]:
    """The judge batch: real judge prompts, all at once."""
    runs = []
    for rep in range(sizes.judge_reps):
        prompts = judge_prompts(lines, sizes.judge_requests, rep)
        wall, results = _batch(target, prompts, sizes.judge_tokens)
        tokens = sum(r.completion_tokens for r in results)
        prompt_tokens = sum(r.prompt_tokens for r in results)
        ttfts = sorted(r.ttft or 0.0 for r in results)
        runs.append(
            {
                "requests": len(prompts),
                "wall_s": wall,
                "prompt_tokens": prompt_tokens,
                "output_tokens": tokens,
                "write_tps": tokens / wall,
                "read_tps_est": prompt_tokens / ttfts[-1] if ttfts[-1] else None,
                "ttft_p50_s": ttfts[len(ttfts) // 2],
                "ttft_max_s": ttfts[-1],
                "prompt_tokens_estimated": any(r.prompt_tokens_estimated for r in results),
            }
        )
    keys = ("wall_s", "write_tps", "read_tps_est", "ttft_p50_s", "ttft_max_s")
    return {"runs": runs, **{f"median_{k}": med([r[k] for r in runs]) for k in keys}}


def phase_quality(target: Target, sizes: Sizes) -> list[dict[str, Any]]:
    """The fixed prompts answered greedily, stopping at the end of turn. Reported only, never a gate."""
    recs = []
    for i, question in enumerate(QUALITY_PROMPTS[: sizes.quality_n]):
        r = stream_completion(target, chatml(question), sizes.quality_tokens, ignore_eos=False)
        recs.append(
            {
                "i": i,
                "prompt": question,
                "text": r.text,
                "completion_tokens": r.completion_tokens,
                "finish_reason": r.finish_reason,
            }
        )
    return recs


# --- teacher-forced scoring -----------------------------------------------------------------------


@dataclass(frozen=True)
class ChunkScores:
    """One scored chunk: each prompt token's log-probability after the first, and the likeliest token."""

    logprobs: list[float]
    top_tokens: list[int]


def parse_vllm_prompt_logprobs(response: dict[str, Any]) -> ChunkScores:
    """Read ``choices[0].prompt_logprobs`` (shape in the module docstring; assumed from documentation)."""
    try:
        entries = response["choices"][0]["prompt_logprobs"]
        scores = ChunkScores([], [])
        for entry in entries[1:]:  # entry 0 is null: nothing precedes the first token
            by_rank = sorted(entry.items(), key=lambda kv: kv[1]["rank"])
            scores.top_tokens.append(int(by_rank[0][0]))
            scores.logprobs.append(
                float(by_rank[-1][1]["logprob"])
            )  # the prompt's own token has the highest rank
    except (KeyError, IndexError, TypeError, ValueError, AttributeError) as exc:
        raise ScoringError(f"vLLM prompt_logprobs response not in the documented shape: {exc!r}") from exc
    if not scores.logprobs:
        raise ScoringError("vLLM prompt_logprobs response holds no scored position")
    return scores


def parse_sglang_input_logprobs(response: dict[str, Any]) -> ChunkScores:
    """Read ``meta_info.input_token_logprobs`` and ``input_top_logprobs`` (assumed from documentation)."""
    try:
        meta = response["meta_info"]
        own, top = meta["input_token_logprobs"], meta["input_top_logprobs"]
        scores = ChunkScores([], [])
        # position 0 has no logprob; strict: a length mismatch is a shape error, not a shorter text
        for (logprob, _token_id, *_), candidates in zip(own[1:], top[1:], strict=True):
            scores.logprobs.append(float(logprob))
            scores.top_tokens.append(int(max(candidates, key=lambda c: c[0])[1]))
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise ScoringError(f"SGLang /generate response not in the documented shape: {exc!r}") from exc
    if not scores.logprobs:
        raise ScoringError("SGLang response holds no scored position")
    return scores


def scoring_path(engine: str) -> ScoringPath:
    """The teacher-forced path of an engine, by its name; engines without one are not scored here."""
    try:
        return SCORING_PATHS[engine.lower()]
    except KeyError:
        raise PhaseError(
            f"no teacher-forced scoring for engine {engine!r} (llama.cpp keeps its own gate; "
            "TensorRT-LLM has none yet)"
        ) from None


def chunk_prompts(text: str, sizes: Sizes) -> list[str]:
    """The pinned text cut into chunks, each starting with its own unique first line (no prefix cache hit)."""
    stride = max(1, len(text) // sizes.tf_chunks)
    return [
        f"[teacher-forced chunk {i:02d}]\n{_slice(text, i * stride, sizes.tf_chunk_chars)}"
        for i in range(sizes.tf_chunks)
    ]


def _score_chunk(target: Target, path: ScoringPath, prompt: str) -> ChunkScores:
    if path is ScoringPath.VLLM_PROMPT_LOGPROBS:
        body = {
            "model": target.model,
            "prompt": prompt,
            "max_tokens": 1,
            "temperature": 0.0,
            "prompt_logprobs": TF_TOP_K,
        }
        response = target.client.post("/v1/completions", json=body)
        response.raise_for_status()
        return parse_vllm_prompt_logprobs(response.json())
    body = {
        "text": prompt,
        "sampling_params": {"max_new_tokens": 1, "temperature": 0.0},
        "return_logprob": True,
        "logprob_start_len": 0,
        "top_logprobs_num": TF_TOP_K,
    }
    response = target.client.post("/generate", json=body)
    response.raise_for_status()
    return parse_sglang_input_logprobs(response.json())


def phase_teacher_forced(target: Target, sizes: Sizes, text: str, engine: str) -> TeacherForced:
    """Score the pinned text: perplexity over every scored position and the top token at each."""
    path = scoring_path(engine)
    scored = [_score_chunk(target, path, prompt) for prompt in chunk_prompts(text, sizes)]
    logprobs = [lp for chunk in scored for lp in chunk.logprobs]
    mean = statistics.fmean(logprobs)
    return TeacherForced(
        path=path,
        chunks=len(scored),
        positions=len(logprobs),
        mean_logprob=mean,
        perplexity=math.exp(-mean),
        top_tokens=[chunk.top_tokens for chunk in scored],
    )


# --- llama-benchy ---------------------------------------------------------------------------------


def benchy_command(url: str, model: str, result_file: Path, *, tokenizer: str | None = None) -> list[str]:
    """``uvx llama-benchy==0.4.0`` against ``url``: prompt 2,048, writing 32 tokens, JSON saved to a file."""
    cmd = [
        "uvx",
        BENCHY_PACKAGE,
        *("--base-url", url),
        *("--model", model),
        *("--pp", str(BENCHY_PROMPT_TOKENS)),
        *("--tg", str(BENCHY_WRITE_TOKENS)),
        *("--runs", str(BENCHY_RUNS)),
        *("--format", "json"),
        *("--save-result", str(result_file)),
    ]
    if tokenizer:
        cmd += ["--tokenizer", tokenizer]
    return cmd


def run_command(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    """The real runner: the command, its output captured, stopped after ``BENCHY_TIMEOUT_S``."""
    return subprocess.run(  # noqa: S603 - argv built by benchy_command from fixed words and the caller's URL
        cmd, capture_output=True, text=True, timeout=BENCHY_TIMEOUT_S, check=False
    )


def parse_benchy(data: dict[str, Any]) -> BenchyFigures:
    """The no-context, one-request row of llama-benchy's JSON (fields seen in a real 0.4.0 file)."""
    try:
        rows = [
            b
            for b in data["benchmarks"]
            if not b.get("is_context_prefill_phase")
            and not b.get("context_size")
            and b.get("concurrency") == 1
        ]
        row = rows[0]
        return BenchyFigures(
            package=BENCHY_PACKAGE,
            prompt_tokens=int(row["prompt_size"]),
            write_tokens=int(row["response_size"]),
            read_tps=row["pp_throughput"]["mean"],
            write_tps=row["tg_throughput"]["mean"],
            first_response_ms=row["ttfr"]["mean"],
            runs=len(row["tg_throughput"].get("values") or []),
        )
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise PhaseError(f"llama-benchy JSON not in the expected shape: {exc!r}") from exc


def phase_benchy(
    url: str,
    model: str,
    runner: Callable[[list[str]], subprocess.CompletedProcess[str]] = run_command,
    *,
    tokenizer: str | None = None,
) -> BenchyFigures:
    """Run llama-benchy (through ``runner``) and read its saved JSON."""
    with tempfile.TemporaryDirectory(prefix="benchy-") as folder:
        result_file = Path(folder) / "result.json"
        done = runner(benchy_command(url, model, result_file, tokenizer=tokenizer))
        if done.returncode != 0:
            tail = (done.stderr or done.stdout or "").strip().splitlines()[-3:]
            raise PhaseError(f"llama-benchy exited {done.returncode}: {' | '.join(tail)}")
        try:
            data = json.loads(result_file.read_text())
        except (OSError, ValueError) as exc:
            raise PhaseError(f"llama-benchy wrote no readable result: {exc}") from exc
    return parse_benchy(data)


# --- the run --------------------------------------------------------------------------------------


def figures(details: dict[str, Any]) -> dict[str, float | None]:
    """The flat set of numbers the gates and the chooser read; absent phases leave their keys out."""
    f: dict[str, float | None] = {}
    if "decode" in details:
        f["write_tps_1_decode_phase"] = details["decode"]["median_decode_tps"]
    for c, v in details.get("conc", {}).items():
        f[f"write_tps_{c}_at_once"] = v["total_tps_median"]
    if "16" in details.get("conc", {}):
        f["ttft_p50_s_at_16"] = details["conc"]["16"]["ttft_p50_s_median"]
        f["ttft_max_s_at_16"] = details["conc"]["16"]["ttft_max_s_median"]
    for approx, v in details.get(
        "prefill", {}
    ).items():  # keyed by the nominal size: tokenizers count differently
        f[f"read_tps_prefill_{approx}"] = v["median_prefill_tps"]
        f[f"prompt_tokens_prefill_{approx}"] = v["median_prompt_tokens"]
    if "judge" in details:
        j = details["judge"]
        f.update(
            {
                "judge_total_s": j["median_wall_s"],
                "judge_write_tps": j["median_write_tps"],
                "judge_read_tps_est": j["median_read_tps_est"],
                "judge_ttft_p50_s": j["median_ttft_p50_s"],
                "judge_ttft_max_s": j["median_ttft_max_s"],
            }
        )
    return f


# A phase's failure is a recorded reason, never a raise: these are the ways a phase can fail.
PHASE_ERRORS: Final = (
    PhaseError,
    PromptFileError,
    httpx.HTTPError,
    OSError,
    ValueError,
    KeyError,
    IndexError,
)


@dataclass
class _Run:
    """What the phases share: the server, the sizes, the pinned inputs (or why missing), the record."""

    target: Target
    url: str
    engine: str
    sizes: Sizes
    prompts: Prompts | None
    prompts_error: str | None
    runner: Callable[[list[str]], subprocess.CompletedProcess[str]]
    tokenizer: str | None
    record: EngineRun
    warm_failed: str | None = field(default=None)

    def pinned(self) -> Prompts:
        if self.prompts is None:
            raise PhaseError(self.prompts_error or "pinned inputs not loaded")
        return self.prompts

    def run(self, phase: Phase) -> Any:  # noqa: PLR0911 - one return per phase
        match phase:
            case Phase.WARM:
                return phase_warm(self.target, self.sizes)
            case Phase.PREFILL:
                return phase_prefill(self.target, self.sizes, self.pinned().text)
            case Phase.DECODE:
                return phase_decode(self.target, self.sizes)
            case Phase.CONC:
                return phase_conc(self.target, self.sizes, self.pinned().text)
            case Phase.JUDGE:
                return phase_judge(self.target, self.sizes, self.pinned().judge)
            case Phase.QUALITY:
                return phase_quality(self.target, self.sizes)
            case Phase.TEACHER_FORCED:
                return phase_teacher_forced(self.target, self.sizes, self.pinned().text, self.engine)
            case Phase.BENCHY:
                return phase_benchy(self.url, self.target.model, self.runner, tokenizer=self.tokenizer)

    def store(self, phase: Phase, data: Any) -> None:
        if phase is Phase.QUALITY:
            self.record.greedy_outputs = [x["text"] for x in data]
            self.record.details[phase.value] = data
        elif phase is Phase.TEACHER_FORCED:
            self.record.teacher_forced = data
        elif phase is Phase.BENCHY:
            self.record.benchy = data
        else:
            self.record.details[phase.value] = data


def _measure_phase(run: _Run, phase: Phase) -> None:
    if run.warm_failed and phase is not Phase.WARM:
        run.record.phases[phase] = PhaseOutcome(
            status=PhaseStatus.NOT_MEASURED, reason=f"skipped: the warm-up request failed ({run.warm_failed})"
        )
        return
    t0 = time.perf_counter()
    try:
        data = run.run(phase)
    except PHASE_ERRORS as exc:
        reason = f"{type(exc).__name__}: {exc}"
        log.warning("phase %s not measured: %s", phase.value, reason)
        run.record.phases[phase] = PhaseOutcome(
            status=PhaseStatus.NOT_MEASURED, reason=reason, seconds=time.perf_counter() - t0
        )
        if phase is Phase.WARM:
            run.warm_failed = reason
        return
    run.store(phase, data)
    run.record.phases[phase] = PhaseOutcome(status=PhaseStatus.MEASURED, seconds=time.perf_counter() - t0)
    log.info("phase %s done in %.1fs", phase.value, time.perf_counter() - t0)


def measure(  # noqa: PLR0913 - one entry point; every argument is a distinct input of the run
    url: str,
    engine: str,
    *,
    model: str = "auto",
    extra: dict[str, Any] | None = None,
    stamp: dict[str, str] | None = None,
    startup_s: float | None = None,
    phases: Sequence[Phase] = tuple(Phase),
    sizes: Sizes = FULL,
    prompts_folder: Path | None = None,
    pins: PromptPins | None = None,
    runner: Callable[[list[str]], subprocess.CompletedProcess[str]] = run_command,
    tokenizer: str | None = None,
) -> EngineRun:
    """Run ``phases`` against the server at ``url``; one record. A failed phase is recorded, not raised."""
    base = url.rstrip("/")
    with httpx.Client(base_url=base, timeout=REQUEST_TIMEOUT_S) as client:
        unreachable: str | None = None
        served = model
        try:
            served = served_model(client) if model == "auto" else model
        except (httpx.HTTPError, KeyError, IndexError, ValueError) as exc:
            unreachable = f"{type(exc).__name__}: {exc}"
        record = EngineRun(
            engine=engine,
            model=served,
            url=base,
            started=time.strftime("%Y-%m-%d %H:%M:%S"),
            tiny=sizes is TINY,
            extra=extra,
            stamp=dict(stamp or {}),
            startup_s=startup_s,
        )
        prompts, prompts_error = None, None
        try:
            prompts = load_prompts(prompts_folder, pins)
        except PromptFileError as exc:
            prompts_error = str(exc)
        run = _Run(
            Target(client, served, extra),
            base,
            engine,
            sizes,
            prompts,
            prompts_error,
            runner,
            tokenizer,
            record,
        )
        if unreachable:
            run.warm_failed = f"the server's model list could not be read ({unreachable})"
        for phase in phases:
            _measure_phase(run, phase)
    record.figures = figures(record.details)
    return record


# --- command line ---------------------------------------------------------------------------------


def _parse_stamp(items: Sequence[str]) -> dict[str, str]:
    pairs = [item.split("=", 1) for item in items]
    if any(len(p) != 2 for p in pairs):  # noqa: PLR2004 - a key and a value
        raise SystemExit("--stamp takes key=value")
    return dict(pairs)


def cmd_measure(args: argparse.Namespace) -> int:
    """``localmodels engines measure``: run the phases against a running server and write the record."""
    phases = [Phase(p) for p in args.phases.split(",")] if args.phases else list(Phase)
    record = measure(
        args.url,
        args.engine,
        model=args.model,
        extra=json.loads(args.extra_json) if args.extra_json else None,
        stamp=_parse_stamp(args.stamp),
        startup_s=args.startup_s,
        phases=phases,
        sizes=TINY if args.tiny else FULL,
        tokenizer=args.tokenizer,
    )
    write_model(Path(args.out), record)
    for phase, outcome in record.phases.items():
        reason = f": {outcome.reason}" if outcome.reason else ""
        print(f"{phase.value}: {outcome.status.value} in {outcome.seconds:.1f}s{reason}")
    print(f"record: {args.out}")
    return 0 if all(o.status is PhaseStatus.MEASURED for o in record.phases.values()) else 1


def cmd_prompts_install(args: argparse.Namespace) -> int:
    """``localmodels engines prompts install``: copy D16's inputs into the state directory, by hash."""
    folder = Path(args.dir) if args.dir else prompts_dir()
    try:
        written = install_prompts(folder)
    except PromptFileError as exc:
        print(f"refused: {exc}")
        return 1
    for path in written:
        print(f"installed {path}")
    return 0
