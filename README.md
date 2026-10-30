# Lockstep

**Guarding Cross-Instance KV Reuse with Numerical Identities**

Research prototype for guarding cross-instance KV cache reuse in LLM serving with
compiler-derived numerical identities, runtime admission checks, and divergence
diagnostics.

LLM instances can select different compiled kernels even when their visible
configurations match. Lockstep records the numerical choices observed during
execution, combines them with configuration and software/hardware information,
and uses the resulting identity to check cross-instance KV reuse.

The vLLM integration adds checks to the NIXL connector without editing vLLM's
source. The final v7 guard includes coverage checks, invalidation after new
decisions, and per-request transfer tags. When a transfer is blocked, the
configured vLLM recompute policy performs prefill on the consumer.

This is research software. Matching identities describe the decisions the tracer
covers; they are not a proof of numerical equivalence for arbitrary kernels or
deployments. See [scope and limitations](docs/guard.md).

## Install and test

Use Python 3.10 or newer. The CPU regression tests do not require vLLM or a GPU.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest -q
python scripts/verify_source.py
```

GPU experiments additionally require Linux, NVIDIA GPUs, and a compatible
PyTorch/Triton/vLLM installation. The recorded final experiments used vLLM
0.32.0 and a specific development revision; see the recorded
[environment matrix](docs/reproduction.md#recorded-environments). Engine
dependencies are installed separately because their CUDA builds must match the
host. Installing Lockstep registers its vLLM general-plugin entry point.

## Try the examples

The launcher accepts a local model snapshot and supplies writable capture,
registry, and GPU-lock directories. The four example prompts are original smoke
test inputs, separate from the paper's workloads.

First inspect the plan; `--dry-run` does not load a model or launch servers:

```bash
python scripts/run_demo.py explore --model /path/to/model --gpus 0 --dry-run
```

Run a configuration comparison on one GPU:

```bash
python scripts/run_demo.py explore --model /path/to/model --gpus 0
```

Run a guarded prefill/decode pair on two GPUs:

```bash
python scripts/run_demo.py pd --model /path/to/model --gpus 0,1
```

Collect direct K/V digests from two independent instances:

```bash
python scripts/run_demo.py kv --model /path/to/model --gpus 0,1
python -m lockstep.kv_truth --compare runs/demo-kv
```

Use `--out DIR`, `--prompts FILE`, and `--max-model-len N` to adjust a run.
The model, batch sizes, and context length must fit the selected GPUs. The
prefill/decode example requires NIXL. Complete commands and expected outputs are
in [reproduction](docs/reproduction.md).

## Repository layout

| Path | Contents |
| --- | --- |
| `lockstep/` | Numerical identities, tracing, live guard, inference harnesses, and report modules |
| `tests/` | Existing CPU regression tests for v7 identity and guard decisions |
| `scripts/` | Portable demo launcher and source-integrity check |
| `examples/` | Original smoke test prompts |
| `experiments/analysis/` | Earlier standalone analysis programs |
| `experiments/legacy/specs/` | Historical study configurations, generators, and experiment launchers |
| `experiments/preregistrations/` | Final-stage experiment plans copied from the research tree |
| `experiments/probes/` | Earlier determinism and contention probes |
| `sitecustom/` | SGLang capture bootstrap |
| `docs/` | Guard semantics, reproduction instructions, and source manifest |

Historical experiment launchers retain their recorded filesystem layout and
require adaptation. Use `scripts/run_demo.py` for a first run on a new host.

## Main modules

| Module | Purpose |
| --- | --- |
| `live_guard` | Online v7 admission checks and lifecycle handling in NIXL |
| `guard` | Identity construction and historical offline evaluations |
| `decisions` | Compiler/autotuning decision tracing and comparison |
| `vllm_plugin`, `sglang_adapter` | Engine instrumentation |
| `explore`, `runner`, `runner_sgl` | Configuration sweeps and individual inference runs |
| `attribute`, `align`, `codediff` | Divergence localization and compiler-output comparison |
| `pd_harness`, `kv_truth`, `kv_digest_connector` | Live deployments and direct K/V ground truth |
| `pool_demo`, `pool_connector` | Shared KV-pool experiments |
| `fh_report`, `fn_report`, `fc_report`, `rw_report` | Final-stage experiment reports |

**Identity versions:** the live guard defaults to v7. The historical
`guard.identity(run_dir)` helper defaults to v6 and does not implement a v7
dispatch. Do not pass `version=7` to that helper expecting the live v7 result.
Use the v7 parts and verdicts recorded by the live guard; see
[guard semantics](docs/guard.md#identity-versions).

## Source provenance

The 72 Python package modules are copied byte for byte from the project's
working tree. The original regression tests, SGLang bootstrap, and selected
experiment sources are also preserved. [The source manifest](docs/source-manifest.json)
records their paths and SHA-256 digests. Repository documentation, packaging,
CI, the demo launcher, and the integrity script are added for this distribution.

Raw captures, model weights, compiler caches, server mirrors, manuscript drafts,
SSH configuration, and the original Git history are outside this code release.
The manifest's commit identifies the source repository HEAD; it does not assert
that every working-tree file matched that commit. Historical tag names in the
experiment plans refer to the original research repository.

See [local validation](docs/validation.md) for the checks performed on this
distribution and their limits.

## License

No redistribution license has been selected for this source distribution.
Third-party engines, models, and datasets retain their own terms.

