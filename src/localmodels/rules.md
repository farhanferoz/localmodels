# Measured rules by workload kind

The fastest way to run each kind of model work on gpu-node-1 (NVIDIA DGX Spark: GB10, compute capability 12.1,
aarch64, 128 GB of memory shared by CPU and GPU at 273 GB/s) that keeps its output quality, and how that is known.
Print one kind with `localmodels rules show K1`. A rule holds on the stack in its `stamp:` line; when a stamped
component changes, re-run the kind's recipe before trusting it (`localmodels rules check` reports such rules).

Labels: **MH** measured here, **PG** published by others on a GB10, **E** estimated, **DOC** documentation or source.
Evidence: the research guide of 2026-10-18 (localmodels docs/research, local), its strand reports in
~/cache/localmodels/research-2026-10-18/, the gpu-job-settings skill and service_b's docs/measurements.md.

For every kind:
- GPU work goes through slurm (`python3 ~/dev/turing/preflight/slurm/submit.py ...`); a measurement needs the card to
  itself (a benchmark beside another job read about 4x slow, MH).
- Model files live on local disk: the NAS reads at 0.04 GB/s against 1.1 GB/s locally (MH).
- Never run `llama-server --help` or `--version` on the shared card: both start CUDA. Read flags from llama.cpp's
  `common/arg.cpp`.
- A Docker container started from a slurm job is outside the job: slurm's audit and footprint cannot see it.
- Record with every result how it was made: engine, build, model file, format, layout (`localmodels` stamps).

## K1 interactive text generation (one person waiting)

- Rule K1.engine: llama.cpp `llama-server`, Q4_K_M GGUF, f16 cache, flash attention and CUDA graphs at their
  defaults, mmap load, no speculative decoding by default.
- Rule K1.layout (`ServingRole.INTERACTIVE`): shared pool (`-kvu`), 4 slots, a fixed context cap (32,768 unless the
  project sets one), `-ub` 512.
- Evidence: Qwen3-30B-A3B writes 92.3-92.8 tokens/s for one request (MH), about 63 % of a ~146 tokens/s memory
  bandwidth ceiling (E); vLLM with NVFP4 wrote 73 tokens/s and took 223-257 s to start against 18 s (MH).
- Quality check after a change: greedy outputs on a fixed prompt set (text agreement at least 0.9) and
  `llama-perplexity --kl-divergence` against the previous build or format.
- Stale when: the llama.cpp build, the driver or the model changes.
- stamp: llama_cpp=ccc3646 driver=580.173.02 cuda=13.0 kernel=6.17.0-1031-nvidia dgx_os=7.2.3 date=2026-10-18
- Recipe (about 4 GPU-minutes): `llama-bench -m <Qwen3.5-4B Q4_K_M> -p 512,2048 -n 128 -fa 1 -r 3 -o json` through
  `localmodels.bench.run_bench`, and llama-benchy against a K1-layout server at depth 0, 4096 and 8192 (prompt 2048,
  output 128, one request, 3 runs). Production tier: the same on Qwen3-30B-A3B.

## K2 bulk text generation (many requests)

- Rule K2.engine: llama.cpp first. vLLM (NVFP4) only for a bulk job well past its 8-10 minute break-even, run per job
  with `vllm run-batch` inside slurm, installed natively, after a repeatability control and a measurement on the
  project's real workload (engines design C5).
- Rule K2.layout (`ServingRole.BULK`): shared pool (`-kvu`), 16 slots, a fixed context cap set by the project, `-ub`
  512; the project's own flags (for example `--spec-type`) pass through.
- Evidence: 16 slots with the shared pool against today's 4 slots: +55 % total output at 16 requests at once and the
  first token after 1.7 s instead of 12.1 s, same memory, on short prompts (a few hundred tokens in, 128 out) (MH).
  vLLM: a further +77 % at 16 at once, 0.79x for one request, 35 GiB held at memory fraction 0.30, greedy output not
  repeated across restarts (0 of 48 prompts identical) (MH). The slot layout itself changes greedy output (33 of 48
  identical between the two llama.cpp layouts, MH): record it with results.
- Quality check after a change: the project's own quality metric on its real data.
- Stale when: llama.cpp or vLLM is upgraded, or the driver changes.
- stamp: llama_cpp=ccc3646 driver=580.173.02 cuda=13.0 kernel=6.17.0-1031-nvidia dgx_os=7.2.3 date=2026-10-18
- Recipe (about 5 GPU-minutes): `llama-batched-bench -m <Qwen3.5-4B Q4_K_M> -fa 1 -npp 512 -ntg 32 -npl 1,4,16`,
  and llama-benchy at 1, 4 and 16 requests at once against a K2-layout server.

## K3 text scoring (per-token log-probabilities)

- Rule K3.engine: llama.cpp or llama-cpp-python only, at a fixed slot layout; `logits_all=True` only where scoring,
  one handle per model, float32 batched log-softmax (gpu-job-settings section 3c).
- Evidence: log-probabilities bit-identical across a restart at a fixed layout; vLLM's were not repeatable (MH).
- Quality check after a change: KL divergence and same-top-token rate against the previous build.
- Stale when: the llama.cpp build changes.
- stamp: llama_cpp=ccc3646 driver=580.173.02 cuda=13.0 kernel=6.17.0-1031-nvidia dgx_os=7.2.3 date=2026-10-18
- Recipe (about 4 GPU-minutes): `llama-perplexity` on one text file pinned by SHA-256, with `--kl-divergence-base`
  saved from the previous build; the pass threshold comes from two repeat runs.

## K4 embeddings and reranking

- No measured rule. Today: SentenceTransformer and CrossEncoder in fp32 at their default batch (service_c,
  corpus-search). Thresholds calibrated in fp32 must be re-checked after any change of precision or batch.
- Recipe (about 3 GPU-minutes): encode 2,000 fixed sentences with BAAI/bge-small-en-v1.5 at batch 32 and 256 in fp32,
  TF32, bf16 and fp16; a setting passes when every vector's cosine with fp32 is at least 0.999 and the top-10
  neighbours agree.

## K5 vision-language (reads images, writes text)

- Rule K5.precision: PyTorch, bf16, SDPA attention, `device_map="cuda"`; a model's own precision comes from its tune
  record (`localmodels.serving.tune`).
- Evidence: bf16 was the best of MedGemma's precision and quantization trials (MH); FP8 through torchao was 2.4x
  slower than bf16 on torch 2.10 (MH). Open levers: the cold load (49-51 s of a 71-76 s run, MH), a static cache with
  compiled decoding (57-62 ms per token against a 31 ms floor, E), prompt reading (13.8-14.0 s for 15,388 tokens, MH).
- Quality check after a change: text agreement with the reference precision's output.
- Stale when: PyTorch, transformers or the driver changes, or the model revision changes.
- stamp: torch=2.10.0+cu130 driver=580.173.02 cuda=13.0 kernel=6.17.0-1031-nvidia dgx_os=7.2.3 date=2026-10-18
- Recipe (about 6 GPU-minutes): `service_b tune --model medgemma-1-5-4b --study=<one small study>` through slurm.

## K6 image, 3D and text encoders, classifiers

- Rule K6.precision: PyTorch with bf16 autocast (NeuroVFM, Pillar-0); a model's own precision comes from its tune
  record. No compile, FP8 or channels-last: the forward takes 0.4-5 s and CPU preparation about as long (MH).
- Evidence: service_b spikes and measurements (MH); NeuroVFM's scores vary by up to 0.014 (CT) and 0.021 (MRI)
  between runs on identical inputs (MH).
- Quality check after a change: score agreement (Spearman at least 0.99, largest difference at most 0.05).
- Stale when: PyTorch or the driver changes, or the model revision changes.
- stamp: torch=2.10.0+cu130 driver=580.173.02 cuda=13.0 kernel=6.17.0-1031-nvidia dgx_os=7.2.3 date=2026-10-18
- Recipe (about 4 GPU-minutes): `service_b tune --model <entry> --study=<one small study>` through slurm.

## K7 speech (text-to-speech, recognition, audio scoring)

- No measured rule; no speech workload runs on this machine.
- Recipe when one does: microsoft/wavlm-base-plus on a fixed 30 s clip, 20 times, in fp32, TF32 and bf16; real-time
  factor and peak memory; a setting passes when the hidden states' cosine with fp32 is at least 0.999.

## K8 training, small models from scratch

- Rule K8.precision: fp32, no compile, no bf16 below about 2 M parameters. TF32 only where scores are not compared
  across settings: switching TF32 moved one system's published scores by 16-20 % (project_a D173, measured on
  Isambard, not here). Batch size is the lever (10.3x throughput at 16x the batch, MH), but it changes gradient
  noise, so the learning rate must be re-derived.
- Evidence: gpu-job-settings (MH).
- Quality check after a change: loss after 200 steps against fp32, same seed.
- Stale when: PyTorch or the driver changes.
- stamp: torch=2.10.0+cu130 driver=580.173.02 cuda=13.0 kernel=6.17.0-1031-nvidia dgx_os=7.2.3 date=2026-10-18
- Recipe (about 4 GPU-minutes): `launch_bound_bench.py` and `crossover_sweep.py` from
  /home/user1/dev/workspace-sync/dgx/scripts/bench at batch 128 and 2048, fp32 with TF32 against bf16, compile off
  and on.

## K9 training, larger models and heads on frozen encoders

- Rule K9.precision: torch.compile and bf16 together above about 10 M parameters (1.92x at 53 M parameters, MH);
  neither alone is worth much; compile warm-up takes 25-30 s.
- Evidence: gpu-job-settings and PERFORMANCE_TUNING.md (MH).
- Quality check after a change: the validation metric against fp32.
- Stale when: PyTorch or the driver changes.
- stamp: torch=2.10.0+cu130 driver=580.173.02 cuda=13.0 kernel=6.17.0-1031-nvidia dgx_os=7.2.3 date=2026-10-18
- Recipe (about 5 GPU-minutes): `compile_at_scale.py` from the same folder, and a small MLP head on 10,000 cached
  1,024-dimensional features at batch 256 to 4,096.

## K10 fine-tuning (full or LoRA)

- No measured rule; no fine-tuning runs on this machine.
- Recipe when one does: LoRA rank 16 on Qwen3-4B, sequence 512, batch 4, 30 steps, bf16 against an fp32 master copy;
  steps/s, tokens/s, peak memory; loss at step 30 within 2 % of the reference. Whether peft and trl install here is
  unverified.
