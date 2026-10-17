# localmodels

Hardware-aware local model serving, GPU-lease coordination and a model catalogue shape, extracted
from [service_a](https://github.com/farhanferoz/service_a),
[service_c](https://github.com/farhanferoz/service_c) and service_b so every project that runs
local models — a `llama-server` text model, or a PyTorch model of any kind in its own process —
shares one implementation.

**Torch-free on the managing side.** Hardware is measured through NVML, `nvidia-smi` and
`sysconf`/`sysctl`, never through `torch.cuda`, and the serving manager only starts processes and
exchanges JSON. A caller can depend on this package without pulling in a CUDA build meant for a
different accelerator; only a model server's own process imports PyTorch.

## What it does

- `localmodels.hardware` — measures the host's device type and memory (never keyed on a card
  name) and reports it as a `HardwareReport`.
- `localmodels.hardware.available_memory(device)` — how much a new allocation on a device can get now. On a GPU
  that shares the host's RAM (such as a DGX Spark's GB10) it is the host's `MemAvailable`, by NVIDIA's guidance, because
  the driver's own free count (`cudaMemGetInfo`, `torch.cuda.mem_get_info`, nvidia-smi) leaves out the page cache and
  after a large file read reports far less than an allocation can get. Decide whether something fits from this, never
  from the driver's count.
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
  package. `GpuLeaseClient` is the waiting side: it publishes its request, takes its turn in
  request order (holders and earlier waiters block, later waiters do not, so two waiters can never
  wait for each other), holds the lease while working and releases it.
- `localmodels.serving` — any local model in its own process and Python environment:
  `ModelServerProcess` starts a model server (one JSON message per line on its stdin/stdout,
  `protocol`), with deadlines and cancellation that kill its process group; `server` is the child's
  loop (`run`, `block_network`, `open_load`); `LazyRegistry` imports a model's implementation only
  when asked for, so each environment needs only its own model's libraries; `ModelHost` keeps one
  server warm on a shared GPU, holding the lease only while a model is loaded, unloading after an
  idle time and at once when another project waits. A server announces its localmodels version first and the
  manager refuses another version with a plain message (rebuild that model's environment); `require_device`
  refuses, in plain words, a model the machine's hardware cannot run, before anything starts. A model file on a
  network filesystem (NFS, SMB) is reported, since every cold start then reads it at network speed.
- `localmodels.serving.tune` — tuning a model: `run_tune` loads one fresh model server per candidate (a
  precision plus load options), times the same inputs on each, and compares every candidate's outputs with
  the first (the reference) through a quality check (`text_agreement`, `score_agreement`,
  `embedding_cosine`). It chooses the fastest candidate that agrees and fits in memory. `save_record` keeps
  the outcome per machine (`hardware_key`), `load_choice` reads it back, and `untuned_message` is the notice
  for a machine with no record. A tune is GPU work: submit it through the scheduler.
- `localmodels.pinned` — Hugging Face snapshots pinned to a commit with every file's SHA-256
  (`PinnedModel`, `PinnedFile`), fetched once (`fetch_snapshot`), hashed when pinned
  (`hash_snapshot`) and checked before use (`verify_files`); weights then load offline.
- `localmodels.atomic` — writes a reader never sees half-done: a file (`write_text_atomically`) or a
  whole directory (`replace_dir_atomically`).
- `localmodels.process` / `localmodels.launcher` — resolves the `llama-server` binary, builds its
  argv, and starts/stops/supervises one server instance per `LocalServerManager`. `Contention`
  lets a caller choose to refuse when another project holds the GPU, or wait for it.
- `localmodels.catalogue` / `localmodels.pull` — the shared shape of a model catalogue entry
  (`ModelSpec`), a `LicencePolicy` gate a caller applies to its own catalogue, and download/status
  helpers over the Hugging Face Hub. Catalogue *entries* are each caller's own data; this package
  never ships a *production* model list of its own (service_a's OSI-gated list and service_c's,
  including non-commercial detectors, each stay in their own repository). The one exception is
  `localmodels.setup.DEFAULT_CATALOGUE`, a small bootstrapping default so `localmodels setup`
  works standalone with no `--catalogue` given — pass `--catalogue` to override it entirely.
- `localmodels.backend` / `localmodels.bench` — detects which compute backend (`CUDA`/`Metal`/
  `Vulkan`/`CPU`) a `llama-server` build actually serves on (`--list-devices`, falling back to
  `ldd`), and runs/parses `llama-bench` for measured prompt/decode throughput.
- `localmodels.setup` / `localmodels.service` — the `localmodels setup`/`localmodels serve` CLI
  commands (below).

## Installing

```
uv add localmodels
```

`huggingface_hub` is a core dependency (Apache-2.0): `localmodels setup` needs it unconditionally
to download a model, not just the callers who use `pull_model` directly. It is declared as a range
(`>=0.26`, the oldest release carrying every call `pull.py` makes), never an exact pin: each
consumer's own lockfile pins it, and an exact pin here made the package unusable beside
`transformers`, whose every release requires `huggingface-hub<2.0` (found adopting it in service_c).

## CLI

```
localmodels setup [--role chat] [--models-dir DIR] [--catalogue FILE] [--yes]
                   [--llama-server-bin PATH] [--device OVERRIDE]
localmodels serve --config <toml>
localmodels lease {status,acquire,wait,begin-wait,admitted,release} --project NAME ...
```

`localmodels setup` inspects the host, picks the largest OSI-licensed catalogue model for the
measured size class, downloads it (pinned to its recorded revision), checks the `llama-server`
backend, and benchmarks it under the shared GPU lease — one command, hardware-aware, idempotent,
and safe to run repeatedly. Without `--yes` (or under `--dry-run`) it stops before downloading or
benchmarking and shows the plan; every other step (hardware, model choice, backend detection) is
always real, so the plan is accurate.

**Pointing `setup`/`serve` at your `llama-server` build:** this package never guesses another
project's binary location (torch-free/host-agnostic by design — no card-name tables, no
hardcoded paths belonging to service_a or service_c). Resolution order: `--llama-server-bin
PATH` (setup) / a server binary in your `ServeConfig`, then the `$LLAMA_SERVER_BIN` environment
variable, then `PATH`. On a host where `llama-server` isn't on `PATH` (the common case — it is
usually a project-local build, not a system package), set one of the first two or the backend
check reports `FAILED` and `localmodels setup` exits non-zero, even under `--dry-run` — a
`FAILED` step means the command could not truthfully complete what it was asked, and `--dry-run`
does not change that meaning (matching service_c's own `setup.py`).

`localmodels serve --config <toml>` runs one configured server (`[server]` table: `host`, `port`,
`model_path`, `project`, `reserved_ports`, `contention`) under `LocalServerManager`/`supervise`,
restarting after a yield once the other project's lease clears. `deploy/localmodels.service` is a
systemd user-unit template.

`localmodels lease ...` gives the shared GPU-lease protocol's operations as a command with
documented exit codes (`0` ok, `2` another project's lease is live, `3` `wait` timed out; the lease directory
is `~/.local/state/gpu-lease` unless `GPU_LEASE_DIR` says otherwise), so a
caller that cannot import this package (e.g. a bash queue) can still speak the protocol. After
`begin-wait`, `wait` takes its turn in request order (`GpuLeaseClient`); without a waiting lease of
its own it waits for every other live lease, as before.

## Development

```
uv sync
uv run pytest -q
uv run ruff check .
uv run mypy src
```
