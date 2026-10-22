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
  manager refuses another version with a plain message (rebuild that model's environment; update the
  managing process before the model environments, because a newer server's `loaded` reply carries library versions that
  an older manager reads only if the version number differs, which 0.7.0 does); `require_device`
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
- `localmodels.pull.resolve_model_file(spec)` — one model file, pinned to `spec.revision`, served from the shared
  Hugging Face cache (`HF_HUB_CACHE`, default `~/.cache/huggingface/hub`): one copy per repository and revision for every
  project, never a copy on a network share. A download is hashed in full against `spec.sha256`; a cached file is checked
  by its blob's name, so a server start never re-reads gigabytes. `pull_model` (a copy in a models folder) is unchanged.
- `localmodels.stack` — what a result was made with, read without starting CUDA: `fingerprint()` (NVIDIA driver, CUDA
  version the driver reports, kernel, DGX OS, localmodels version) and `llama_cpp_build(binary)` (the commit llama.cpp
  embeds in `libllama-common.so`, read without running it). `LocalServerManager.stamp()` and `ModelHost.stamp()` return
  a `ServerStamp` (engine, build, model file and hash, layout or load options, host) to store with each result. A tune
  record carries the stack it was measured on, and `load_choice` warns when the driver, CUDA, kernel or DGX OS changed
  since (the choice still applies).
- `localmodels.atomic` — writes a reader never sees half-done: a file (`write_text_atomically`) or a
  whole directory (`replace_dir_atomically`).
- `localmodels.process` / `localmodels.launcher` — resolves the `llama-server` binary, builds its
  argv, and starts/stops/supervises one server instance per `LocalServerManager`. `Contention`
  lets a caller choose to refuse when another project holds the GPU, or wait for it.
  `MemoryGuards` (all off unless passed) adds a kill floor on available memory, a host reserve checked
  against `llama-fit-params`' projection before the load and the memory left after it, and the server's
  `oom_score_adj`.
- Layouts by role (opt-in): `derive_serving_params(device, total_mem_gb, role=ServingRole.INTERACTIVE)` (or `BULK`)
  lays `llama-server` out as one shared pool (`-kvu`) with 4 (interactive) or 16 (bulk) slots and a fixed context
  (`n_ctx_cap`, default 32,768), never derived from free memory (`localmodels rules show K1`, `K2`). Without a role
  the parameters, and so the argv, are exactly as before. Under the shared pool `min_slot_tokens` is checked as
  `n_ctx >= n_parallel * min_slot_tokens` before the start, since llama.cpp then reports the whole pool per slot.
- Stay-up mode (opt-in, default off): `Contention.STAY_UP` for a `LocalServerManager` (and `localmodels serve`), or
  `ModelHost(stay_up=StayUp(floor_gib, allowance_gib, oom_score_adj))` for a PyTorch model, keeps a model loaded while
  another project or slurm holds the GPU, instead of stepping aside. It never takes the lease; it registers in the
  lease directory's `resident/` folder (one file per service), which readers of the top-level lease files never open.
  It unloads when available memory falls below its floor (the kill floor and the OOM score are required) and when a
  quiet card is held, and loads again once memory is back to floor plus allowance and no card remains.
  `lease.QuietCard` is that card: a measurement (`run_tune(quiet_card=...)`, `run_bench(quiet_card=...)`) writes a
  request in `quiet/` and waits until every resident has unloaded. Records left by a dead process, or written before
  the last boot, are ignored and deleted. `localmodels lease status` lists residents and quiet cards when there are
  any.
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
localmodels rules {show KIND,check}
localmodels settings {show --model REPO (--revision REV | --file NAME) --kind KIND,check}
localmodels llama {build --commit SHA,list,switch COMMIT,prune}
localmodels digest run
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

## Rules and llama.cpp builds

`localmodels rules show K2` prints the measured rule for one kind of workload (K1 interactive text generation to K10
fine-tuning; `src/localmodels/rules.md`): the rule, its evidence, the stack it was measured on (a `stamp:` line) and
a benchmark recipe, then the newest digest lines for that kind. `localmodels rules check` compares every stamp with
this host and the current llama.cpp build and exits 1 when a stamped component changed, naming the rules to re-measure;
it also lists stamps that do not name the models they were measured on (`model=`).

`localmodels llama build --commit SHA` compiles one llama.cpp commit on the CPU into `~/cache/llama.cpp/builds/<commit>`
(`LOCALMODELS_LLAMA_HOME` moves it), with the CMake options of the build in `~/dev/llama.cpp/build`. `llama list` shows
the builds, which is `current` and which projects name each (`builds/users.toml`); `llama switch COMMIT` moves `current`
only when that build's `comparison.json` (from the K1-K3 recipes against the current build) records a pass; `llama
prune` deletes builds that are neither current nor named.

## Measured settings

`localmodels.settings.lookup(model, kind, ...)` answers what to load a model with on this machine and how far to trust
it, from a tune record in `~/.local/state/localmodels/settings` (`$LOCALMODELS_STATE_DIR` moves it), else the written
rule of the kind, else the caller's default. Only a record is `tuned`; a record whose stack or llama.cpp build changed
is `stale` and stays in use; a rule measured on this model is `rule`; a rule measured on other models only is
`rule-other-model` and changes nothing (the caller's default stays). `explicit=` settings are never overridden (`options` key by key). A model is identified by its files, or a 40-character revision,
or an announced path; a repository alone is refused. The caller's `libraries={"torch": ...}` are compared with the
ones the record names. The store is never a reason for a load to fail: a bad or newer file is treated as absent and announced
(`answer.announce(log)`). The module is imported on demand; `import localmodels` does not load it.
`localmodels settings show --model REPO --revision REV --kind K6` (or `--file NAME`) prints one answer and loads nothing; `settings check` lists the
stored records that are stale on this host and exits 1 when any is. Automatic re-tuning is not built yet: an answer
that calls for a tune reports `tune: not-available`.

## Monthly digest

`localmodels digest run [--out DIR]` checks the 13 sources that can change a serving or training rule (llama.cpp
commits and discussions, PyTorch and its libraries, vLLM/SGLang/TRT-LLM, NVIDIA's Spark pages, the NVIDIA forum,
playbooks, Spark Arena, new models, and this host's driver, kernel and DGX OS) with fixed local checks and no
language model. It writes `DIR/<date>.md` (default `DIR` is `~/cache/localmodels/digests`), the raw findings in
`DIR/.raw/<date>.json`, and `DIR/.state.json` (what the next run compares with). A finding line reads
`- [K1,K2] D1 llama.cpp: <finding> (<link>)`, tagged with the workload kinds the source touches. A source whose
check fails is listed under "Failed sources" and never stops the run; the exit code is 1 only if every source failed.
`deploy/localmodels-digest.timer` runs it on the 1st of each month at 07:00 (install steps in the `.service` file).

## Development

```
uv sync
uv run pytest -q
uv run ruff check .
uv run mypy src
```
