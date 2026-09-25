# localmodels

Hardware-aware local LLM serving, GPU-lease coordination and a model catalogue shape, extracted
from [service_a](https://github.com/farhanferoz/service_a) and
[service_c](https://github.com/farhanferoz/service_c) so both projects — and anything else
that serves a local `llama-server` — share one implementation instead of two.

**Torch-free.** Hardware is measured through NVML, `nvidia-smi` and `sysconf`/`sysctl`, never
through `torch.cuda`. A caller can depend on this package without pulling in a CUDA build meant
for a different accelerator.

## What it does

- `localmodels.hardware` — measures the host's device type and memory (never keyed on a card
  name) and reports it as a `HardwareReport`.
- `localmodels.params` — derives `llama-server` serving parameters (`n_gpu_layers`, `n_ctx`,
  `n_parallel`, recommended quantisation and size class) as a pure function of measured hardware,
  and classifies a serving tier with an operator-facing warning/recommendation.
- `localmodels.preflight` — one call answering "can this host run local models at all, and how
  should it be configured", with a `PreflightVerdict` distinguishing *runs slowly* from *cannot
  run any local GGUF at all*.
- `localmodels.lease` — a file-based, cooperative GPU lease: one JSON file per project in a
  shared directory, written atomically, so two projects sharing one GPU on a host can see who
  holds it and ask each other to yield. The file format is a contract shared with service_c's
  bash queue (`scripts/run_gpu_queue.sh`), which writes and reads it without depending on this
  package.
- `localmodels.process` / `localmodels.launcher` — resolves the `llama-server` binary, builds its
  argv, and starts/stops/supervises one server instance per `LocalServerManager`. `Contention`
  lets a caller choose to refuse when another project holds the GPU, or wait for it.
- `localmodels.catalogue` / `localmodels.pull` — the shared shape of a model catalogue entry
  (`ModelSpec`), a `LicencePolicy` gate a caller applies to its own catalogue, and download/status
  helpers over the Hugging Face Hub. Catalogue *entries* are each caller's own data; this package
  never ships a model list of its own.

## Installing

```
uv add localmodels
```

`huggingface_hub` is an optional extra (`localmodels[hub]`) — only `pull_model`, `check_updates`
and `search_hub` need it; `model_status` reads the local disk only.

## Development

```
uv sync
uv run pytest -q
uv run ruff check .
uv run mypy src
```
