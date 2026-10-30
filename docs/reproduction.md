# Reproduction guide

## CPU validation

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
python scripts/verify_source.py
python -m build
```

The existing regression suite checks identity inputs, coverage, strict/canary
decisions, and invalidation predicates. It does not validate a running NIXL
deployment or execute the complete paper workload.

## Engine requirements

GPU examples require Linux, NVIDIA GPUs, and compatible PyTorch, Triton, and
vLLM installations. Prefill/decode transfer also requires NIXL. Select engine
versions and CUDA builds together for the target host; this prototype uses
engine internals and is not tested against every release. SGLang diagnostics
need a separate compatible SGLang installation. CPU CI installs no GPU engine.

## Portable examples

Run commands from the repository root after installation. Pass a local model
directory containing a complete model snapshot. Defaults are small smoke
workloads; use a model and memory settings that fit your GPUs.

```bash
python scripts/run_demo.py explore --model /path/to/model --gpus 0 --out runs/explore
python scripts/run_demo.py pd --model /path/to/model --gpus 0,1 --out runs/pd
python scripts/run_demo.py kv --model /path/to/model --gpus 0,1 --out runs/kv
python -m lockstep.kv_truth --compare runs/kv
```

All modes accept `--dry-run`, which prints the command, settings, and any
generated spec without creating output files or starting subprocesses. Actual
runs refuse a nonempty output directory to avoid mixing results.

The explorer writes `spec.json` and `runs/spec/` under its output directory.
The latter contains `base`, `base_rep`, configuration variants, and `report.json`.
Re-analyze it with:

```bash
python -m lockstep.explore --spec runs/explore/spec.json --out-root runs/explore/runs --analyze-only
python -m lockstep.attribute runs/explore/runs/spec/base runs/explore/runs/spec/budget
python -m lockstep.decisions runs/explore/runs/spec/base runs/explore/runs/spec/budget
```

The PD harness records `outputs.json`, `local_a.json`, `local_b.json`, and
`inst_a`/`inst_b` capture directories. The KV harness records `kv_a`/`kv_b`,
prompt log-probabilities, and local generations. Its `--compare` command writes
`kv_truth.json`. Inspect coverage and live events even when a run completes.

The launcher sets `LOCKSTEP_LOCK_DIR` to `.cache/gpu-locks` unless the operator
already supplied it. Explorers sharing a host should use the same lock directory.
The direct two-instance examples use the explicitly selected GPUs and do not
reserve them through the explorer's scheduler.

### Canary policy

```bash
python scripts/run_demo.py pd --model /path/to/model --gpus 0,1 --policy canary --out runs/canary
```

The launcher enables on-demand canaries over up to the first four supplied
prompts. Keep this policy distinct from strict identity admission when
reporting results.

### SGLang diagnostics

Use a separate SGLang environment and add the bootstrap directory to
`PYTHONPATH` before starting its worker processes:

```bash
export PYTHONPATH="$PWD/sitecustom${PYTHONPATH:+:$PYTHONPATH}"
export LOCKSTEP_ENGINE=sglang
python -m lockstep.runner_sgl --help
```

The SGLang explorer uses the current interpreter by default. Set
`LOCKSTEP_SGLANG_PYTHON` to a different interpreter when needed, and
`LOCKSTEP_SGLANG_BOOTSTRAP` to the directory containing `sitecustomize.py`
when running from an installed wheel. Compiler and cache settings are inherited
from the caller or supplied through the experiment spec's `env` object.

## Analysis inputs

Raw captures and full prompt datasets are not bundled. The four example prompts
are smoke inputs and do not reproduce paper statistics. Report modules consume
capture directories supplied by the operator:

```bash
python -m lockstep.fh_report /path/to/fh_runs > fh_report.json
python -m lockstep.fn_report /path/to/fn_runs > fn_report.json
python -m lockstep.rw_report /path/to/rw_runs > rw_report.json
python -m lockstep.timing_margin --root /path/to/runs --same-host "*" --out timing_margin.json
```

For timing margins, the glob selects run directories below the root; each run
contains deployment directories with `inst_a` and `inst_b` captures. Use
`--cross-host PRODUCER CONSUMER` for paired relative capture directories.
That analysis retains its vLLM 0.32.0 eligibility filter. Earlier report modules
may expect historical directory names within the supplied data tree.
`lockstep.fixed_env2` and `lockstep.v7_families` take one or more positional capture
roots. Offline O3 analysis recognizes the single-host cohort under `single/`.
`experiments/analysis/analyze_tier1.py` accepts the capture root through
`LOCKSTEP_RUNS_ROOT` (default `runs`).

Dataset prompt builders require `--data` with a local JSONL file.
`lockstep.loadgen` requires `--model` and `--prompts` (a JSON list of strings).
Direct `lockstep.pd_harness` calls also require `--model`. GPU lock files default
to a user-specific directory under the system temporary directory; set
`LOCKSTEP_LOCK_DIR` consistently across cooperating processes to override it.

## Publication scope

This distribution contains runtime modules, tests, analysis programs, probes,
portable examples, and setup documentation. It excludes historical operational
scripts, deployment configurations, private environment records, internal path
maps, raw captures, model weights, full datasets, and original Git history.
The SHA-256 manifest describes the public release contents.
