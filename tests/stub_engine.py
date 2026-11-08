"""A stub engine for the measuring harness's tests: no model, no GPU, no network beyond 127.0.0.1.

From D16's ``stub_server.py`` (deterministic text from a hash of the prompt, a natural end of turn after 8
words unless ``ignore_eos``), plus the two teacher-forced shapes the harness reads, as documented (not seen
from a real engine; see the ``engine_bench`` docstring):

- ``/v1/completions`` with ``prompt_logprobs``: vLLM's ``choices[0].prompt_logprobs``;
- ``/generate`` with ``return_logprob``: SGLang's ``meta_info.input_token_logprobs`` / ``input_top_logprobs``.

A "token" is a whitespace-separated word. Position ``i`` (from 1) has the token id ``100 + i``; for even
``i`` it is the most likely token (log-probability ``-0.5``), for odd ``i`` another token ``900 + i`` is
(``-0.1``) and the prompt's own token has ``-1.5``. So the log-probabilities of ``n`` words are known: see
``expected_logprobs``.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Self

WORDS = "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike nov oscar".split()  # noqa: SIM905
MODEL = "stub-model"
OWN_EVEN, OWN_ODD, TOP_ODD = -0.5, -1.5, -0.1


def expected_logprobs(n_words: int) -> list[float]:
    """The log-probabilities (positions 1 to n-1) the stub reports for an ``n_words`` prompt."""
    return [OWN_EVEN if i % 2 == 0 else OWN_ODD for i in range(1, n_words)]


def expected_top_tokens(n_words: int) -> list[int]:
    return [100 + i if i % 2 == 0 else 900 + i for i in range(1, n_words)]


def words_for(prompt: str, n: int) -> list[str]:
    h = hashlib.sha256(prompt.encode()).digest()
    return [WORDS[h[i % len(h)] % len(WORDS)] for i in range(n)]


def vllm_prompt_logprobs(prompt: str) -> list[Any]:
    """vLLM's list: null first, then a dict per position keyed by token id."""
    out: list[Any] = [None]
    for i in range(1, len(prompt.split())):
        own = {
            "logprob": OWN_EVEN if i % 2 == 0 else OWN_ODD,
            "rank": 1 if i % 2 == 0 else 7,
            "decoded_token": "w",
        }
        if i % 2 == 0:
            out.append({str(100 + i): own})
        else:  # the own token is listed first on purpose: the parser must not rely on order
            out.append(
                {str(100 + i): own, str(900 + i): {"logprob": TOP_ODD, "rank": 1, "decoded_token": "t"}}
            )
    return out


def sglang_meta_info(prompt: str) -> dict[str, Any]:
    n = len(prompt.split())
    own: list[Any] = [[None, 99, None]]
    top: list[Any] = [None]
    for i in range(1, n):
        own.append([OWN_EVEN if i % 2 == 0 else OWN_ODD, 100 + i, None])
        top.append([[OWN_EVEN, 100 + i, None]] if i % 2 == 0 else [[TOP_ODD, 900 + i, None]])
    return {"input_token_logprobs": own, "input_top_logprobs": top, "completion_tokens": 1}


class _Handler(BaseHTTPRequestHandler):
    server: StubEngine

    def log_message(self, *args: Any) -> None:
        pass

    def send_json(self, obj: Any, status: int = 200) -> None:
        data = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        if self.path in self.server.fail_paths:
            self.send_error(404)
        elif self.path == "/v1/models":
            self.send_json({"data": [{"id": MODEL}]})
        else:
            self.send_error(404)

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.requests.append((self.path, body))
        if self.path in self.server.fail_paths or (
            body.get("prompt_logprobs") is not None and "logprobs" in self.server.fail_paths
        ):
            self.send_json({"error": "stub failure"}, 400)
        elif self.path == "/v1/completions" and body.get("prompt_logprobs") is not None:
            self.send_json(
                {"choices": [{"text": "x", "prompt_logprobs": vllm_prompt_logprobs(body["prompt"])}]}
            )
        elif self.path == "/v1/completions":
            self.stream(body)
        elif self.path == "/generate":
            self.send_json({"text": "x", "meta_info": sglang_meta_info(body["text"])})
        else:
            self.send_error(404)

    def stream(self, body: dict[str, Any]) -> None:
        n = body["max_tokens"]
        n_out = n if body.get("ignore_eos") else min(n, 8)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for w in words_for(body["prompt"], n_out):
            time.sleep(self.server.delay)
            chunk = {"choices": [{"text": w + " ", "finish_reason": None}]}
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.flush()
        end = {"choices": [{"text": "", "finish_reason": "length" if n_out == n else "stop"}]}
        usage = {
            "choices": [],
            "usage": {"prompt_tokens": len(body["prompt"]) // 4, "completion_tokens": n_out},
        }
        for obj in (end, usage):
            self.wfile.write(f"data: {json.dumps(obj)}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


class StubEngine(ThreadingHTTPServer):
    """``with StubEngine() as stub:`` serves on ``stub.url``; ``stub.requests`` lists ``(path, json body)``.

    ``fail_paths``: request paths answered with HTTP 400; ``"logprobs"`` fails only ``prompt_logprobs`` ones.
    """

    daemon_threads = True

    def __init__(self, fail_paths: tuple[str, ...] = (), delay: float = 0.0) -> None:
        super().__init__(("127.0.0.1", 0), _Handler)
        self.fail_paths = fail_paths
        self.delay = delay
        self.requests: list[tuple[str, dict[str, Any]]] = []

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server_address[1]}"

    def __enter__(self) -> Self:
        threading.Thread(target=self.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.shutdown()
        self.server_close()
