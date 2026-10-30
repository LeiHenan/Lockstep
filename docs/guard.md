# Guard semantics

## Identity and coverage

The v7 identity combines the effective configuration, the environment under a
default-deny rule, local model metadata, observed per-subgraph kernel choices,
lowering structure, and the software/hardware stack. Tensor-parallel execution
uses per-rank metadata and traces.

Model identification hashes JSON metadata and weight filenames/sizes; it does
not checksum all weight bytes. Equal identities therefore do not establish
weight integrity by themselves.

Coverage is recorded separately from the identity. Missing traces, incomplete
stack/configuration information, untracked user kernels, quantization, or
tracing errors can make coverage partial. Strict admission requires both an
identity match and acceptable coverage.

| Policy | Behavior |
| --- | --- |
| `strict` | Admit on matching identity and coverage; otherwise block |
| `canary` | Use the identity when eligible; otherwise admit only if available canary results match |
| `off` | Experimental control that disables the guard |

A canary checks only the requests and positions it exercises. It does not
establish equivalence for untested inputs.

## Lifecycle and fallback

The NIXL compatibility check incorporates the numerical identity. The guard
also tracks newly observed choices after the handshake. In the final version,
per-request transfer tags check the identity associated with the producer's
KV state. The strict demo sets `LOCKSTEP_XFER_TAG=enforce` and the connector's
`kv_load_failure_policy` to `recompute`.

`LOCKSTEP_XFER_TAG=observe` and `LOCKSTEP_TEST_*` variables belong to experimental
controls and fault injection. They are not enabled by the demo launcher.

## Settings used by the guarded demo

| Setting | Value |
| --- | --- |
| `LOCKSTEP_LIVE_GUARD` | `1` |
| `LOCKSTEP_GUARD_VERSION` | `7` |
| `LOCKSTEP_GUARD_POLICY` | `strict` unless `--policy canary` is requested |
| `LOCKSTEP_XFER_TAG` | `enforce` |
| `LOCKSTEP_CAPTURE` | `1` |
| `LOCKSTEP_DEC_EAGER_FLUSH` | `1` |
| `LOCKSTEP_MODE` | `off` for decision tracing without boundary fingerprints |
| `VLLM_BATCH_INVARIANT` | `1` |
| `LOCKSTEP_HYBRID_DIR` | A registry directory under the selected run output |
| `kv_load_failure_policy` | `recompute` |

The harness creates an instance directory containing `cfg.json` and `capture/`
and sets each instance's `LOCKSTEP_DIR`. Setting environment variables alone on
an arbitrary `vllm serve` command does not create that required layout.

## Identity versions

Online v7 uses `live_guard.identity_v7_now()` or per-rank
`live_guard.identity_rank()`. The underlying pure constructor is
`guard.identity_v7(...)`, which returns an identity and coverage reasons.

The older `guard.identity(run_dir, version=None)` helper defaults to v6.
It handles historical versions and has no explicit v7 branch. An integer `7`
falls through to an older identity calculation, so it must not be used as a v7
offline evaluator. Setting `LOCKSTEP_GUARD_VERSION=7` does not change this
offline helper's default.

For actual live runs, inspect each instance's `capture/live_guard.jsonl` and
`capture/v7_parts*.json` files. Final experiment report modules read live
verdicts and, where appropriate, compute reduced identities from traces.
Those reduced identities omit some live fields and are labeled as such in
their source; they are not interchangeable with a live admission verdict.

## Scope and limitations

- The guard was evaluated on recorded engine revisions. Its hooks depend on
  engine and compiler internals, so compatibility must be checked when those
  dependencies change.
- Compiler tracing observes generated Triton kernels and lowering to external
  libraries. It does not inspect every decision inside vendor/custom kernels.
- Boundary localization requires suitable piecewise compilation. Other graph
  modes and execution paths may expose only output/log-probability comparisons.
- The SGLang adapter supports diagnostic experiments; this repository does not
  claim an online SGLang KV-transfer guard.
- Matching identities can be conservative: numerically equal instances may be
  blocked. Report unnecessary blocks separately from unsafe admissions.
- Shared filesystem registry paths and recorded environment files belong to
  the deployment's trust boundary. Keep generated captures out of source control.

