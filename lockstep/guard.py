"""Provenance guard: reuse KV / prefix state across instances only if their numerical identities match.

An instance's identity combines (1) the contract-relevant configuration (model, dtype, quantization,
parallelism, determinism mode) and (2) its hidden decisions: every compiler-generated kernel's
source digest and the launch configuration it actually runs with (lockstep.decisions snapshot).
Two instances with equal identity execute the same kernels with the same configurations on the
same model shards.

python -m lockstep.guard RUN_DIR [...]          # print identities
python -m lockstep.guard --eval RUNS_ROOT       # pairwise soundness/completeness vs ground truth
"""
import hashlib
import itertools
import json
import os
import sys
import time

from lockstep import decisions

# Default-deny: every engine option and engine environment variable is part of the identity, except
# options with evidence that they do not change numerics, or whose effect is fully visible in the
# traced decisions. Evidence (04_results.md): prefix caching, max_num_seqs, and CPU KV offloading
# leave outputs bit-identical under batch invariance (E1, E13c); the token budget acts only through
# compiled-kernel specialization and tuning, which the trace records (E10, E11, E12).
NEUTRAL_KEYS = {"max_num_batched_tokens", "gpu_memory_utilization", "max_num_seqs", "enable_prefix_caching",
                "num_gpu_blocks_override", "kv_transfer_config", "seed", "served_model_name", "port",
                "log_level", "disable_log_stats", "enforce_eager_note"}
NEUTRAL_ENV_PREFIXES = ("LOCKSTEP_", "CUDA_VISIBLE", "VLLM_NIXL_SIDE_CHANNEL", "VLLM_HOST_IP")


def _args_to_kwargs(args):
    """`vllm serve` CLI args (as recorded by pd_harness) -> engine keyword names."""
    out, i = {}, 0
    while i < len(args):
        a = str(args[i])
        if a.startswith("--"):
            key = a[2:].replace("-", "_")
            if i + 1 < len(args) and not str(args[i + 1]).startswith("--"):
                out[key] = args[i + 1]
                i += 2
                continue
            out[key] = True
        i += 1
    return out


# Neutral only because the trace shows their effect; if no compiled kernel was traced (e.g. tensor
# parallelism, where kernels run in worker processes the adapter does not observe), they count.
TRACE_DEPENDENT_KEYS = {"max_num_batched_tokens"}
# Compilation-cache locations: whether a process compiles cold or loads cached artifacts shows up in
# the traced decisions (v3), so the location itself is neutral when a trace exists.
TRACE_DEPENDENT_ENV = {"TORCHINDUCTOR_CACHE_DIR", "VLLM_DISABLE_COMPILE_CACHE", "VLLM_CACHE_ROOT", "TRITON_CACHE_DIR"}


def config_part(cfg, traced=True):
    ek = dict(cfg.get("engine_kwargs", {}))
    ek.update(_args_to_kwargs(ek.pop("args", []) or []))
    env = cfg.get("env", {}) or {}
    neutral = NEUTRAL_KEYS if traced else NEUTRAL_KEYS - TRACE_DEPENDENT_KEYS
    return {"ek": {k: ek[k] for k in sorted(ek) if k not in neutral},
            "env": {k: env[k] for k in sorted(env) if env[k] is not None and not k.startswith(NEUTRAL_ENV_PREFIXES)
                    and not (traced and k in TRACE_DEPENDENT_ENV)}}


IDENTITY_VERSION = 6  # offline default (frozen); the live guard picks its version by LOCKSTEP_GUARD_VERSION


def canonical_kernels(dec):
    """v5 kernel part: position-preserving but independent of generated kernel names.
    Decisions are grouped by the piecewise subgraph they ran in; within a subgraph they form a multiset
    of (name-normalized source digest, chosen launch configuration), so the names Inductor gives a
    kernel (which differ between cold compilation and loading from the cache, and enter the plain
    source digest) do not matter, while a configuration that moves to another layer still changes the
    identity. A cold process runs the kernels of the subgraph it compiles first outside any subgraph
    context; if exactly one subgraph index is missing from the trace, those unlabeled records are
    placed there. Records of run-time choices (e.g. the attention backend) are kept by key."""
    groups, floating, runtime = {}, [], []
    for k, v in dec.items():
        if k.startswith("__runtime__"):
            runtime.append((k, json.dumps(v.get("chosen"), sort_keys=True, default=str)))
            continue
        item = (v.get("srcn"), json.dumps(v.get("chosen"), sort_keys=True, default=str))
        sub = v.get("subgraph")
        (floating if sub is None else groups.setdefault(int(sub), [])).append(item)
    if floating and groups:
        present = set(groups)
        missing = [i for i in range(min(present), max(present) + 1) if i not in present]
        if len(missing) == 1:
            groups[missing[0]] = floating
            floating = []
    out = [[i, sorted(groups[i])] for i in sorted(groups)]
    return {"subgraphs": out, "unplaced": sorted(floating), "runtime": sorted(runtime)}


def usage_kernels(use, dec, pointwise_config=True):
    """v6 kernel part: for every piecewise subgraph, the multiset of (name-normalized source digest,
    launch configuration) of the kernel objects that ran in it (lockstep.decisions usage records).
    Kernels that only ran while compiling (outside any subgraph) do not enter; the names Inductor gives a
    kernel do not matter; a configuration that moves to another layer still changes the identity.
    Run-time choices outside kernels (e.g. the attention backend) are kept by key.
    pointwise_config=False (version 7 since HB2): a pointwise kernel (Inductor's triton_poi_*) computes every
    element independently, so its block size and warps decide only how work is split, not any result; its
    source still counts. Cold processes pick these from the budget's shape hints (HB, 2026-10-23)."""
    groups = {}
    for v in use.values():
        cfg = v.get("chosen")
        if not pointwise_config and str(v.get("kernel", "")).startswith("triton_poi_"):
            cfg = "pointwise"
        groups.setdefault(int(v["subgraph"]), []).append(
            (v.get("srcn"), json.dumps(cfg, sort_keys=True, default=str)))
    runtime = sorted((k, json.dumps(v.get("chosen"), sort_keys=True, default=str)) for k, v in dec.items()
                     if k.startswith("__runtime__"))
    return {"subgraphs": [[i, sorted(groups[i])] for i in sorted(groups)], "runtime": runtime}


def identity_v6(cfg, use, dec, stack=None):
    """stack: GPU model, driver, and library versions (decisions.stack); two instances on different GPU models
    or software stacks never share an identity, since equal source digests can compile to different code."""
    traced = bool(use)
    blob = json.dumps({"cfg": config_part(cfg, traced), "kernels6": usage_kernels(use, dec), "stack": stack or {}},
                      sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


# --- version 7 (review round 7): v6 + the process environment, resolved engine configuration, model files,
# and the lowering structure as a set, plus an explicit coverage verdict. v6 above stays frozen.

# environment entries that name places, ports, devices or users, not computations
NEUTRAL_ENV_V7 = ("LOCKSTEP_", "CUDA_VISIBLE", "VLLM_NIXL_SIDE_CHANNEL", "VLLM_HOST_IP", "VLLM_PORT", "XDG_", "SSH_",
                  "LC_", "LANG", "TERM", "HOME", "PWD", "OLDPWD", "SHLVL", "PATH", "HOSTNAME", "USER", "LOGNAME",
                  "MAIL", "SHELL", "TMPDIR", "_", "LS_COLORS", "MOTD_SHOWN", "HF_HOME", "UV_CACHE_DIR", "PIP_",
                  "FLASHINFER_WORKSPACE_BASE", "SGLANG_CACHE_DIR", "VIRTUAL_ENV", "CONDA_", "PYTHONPATH",
                  "MASTER_ADDR", "MASTER_PORT", "TMUX", "DISPLAY", "OLDPATH", "NVIDIA_VISIBLE_DEVICES",
                  "NVIDIA_DRIVER_CAPABILITIES", "NVIDIA_REQUIRE_CUDA", "KUBERNETES_",
                  "TOKENIZERS_PARALLELISM", "HF_HUB_OFFLINE",
                  "VLLM_OBJECT_STORAGE_SHM_BUFFER_NAME")  # random per process (K1/L1 smoke test, addendum 2)


def env_part(env):
    """Default-deny over the whole process environment: every entry counts unless it is neutral above or
    acts only through decisions that the trace records (compilation-cache locations). LOCKSTEP_NEUTRAL_ENV
    (comma-separated prefixes) lets an operator declare a launcher's bookkeeping neutral, e.g. "SLURM_,POD_"
    (ENV experiment); it is itself neutral, as a LOCKSTEP_ variable."""
    extra = tuple(p for p in env.get("LOCKSTEP_NEUTRAL_ENV", "").split(",") if p)
    neutral = NEUTRAL_ENV_V7 + extra
    return {k: v for k, v in sorted(env.items())
            if not k.startswith(neutral) and k not in TRACE_DEPENDENT_ENV}


# user (non-Inductor) Triton kernels whose launch configuration is fixed by the engine, not tuned at run
# time: vLLM's batch-invariant persistent matmul picks its tiles from a static table by dtype
FIXED_USER_KERNELS = {"matmul_kernel_persistent"}


def lowering_set(structure):
    """Per-module lowering records (vendor-library calls, number of generated kernels, user kernels) as a
    set: a cold process records modules that a warm one loads only once, so multiplicities differ although
    the set does not (04_results.md, review round 6: as a multiset, 37 unneeded cold/warm blocks)."""
    return sorted({json.dumps({"extern": v.get("extern", {}), "n_triton": v.get("n_triton"),
                               "user": sorted(v.get("user_kernels", []))}, sort_keys=True)
                   for v in (structure or {}).values()})


STACK_FIELDS = ("gpu", "cc", "sms", "driver", "cuda", "torch", "triton", "vllm")


def coverage_v7(use, structure, stack, resolved, health=None):
    """Reasons why the trace may miss a decision that changes numerics; empty if covered. The guard does not
    admit on identity alone when this is non-empty. `health` (live only; None offline, where it was not recorded)
    is the tracer's record of its own failures: any failure makes the identity partial (fail closed), so that a
    hook that raised cannot leave an empty or incomplete part that compares equal."""
    why = []
    if not use:
        why.append("no-trace")  # nothing compiled (e.g. tensor parallelism in batch-invariant mode) or tracer failed
    if any(set(v.get("user_kernels", [])) - FIXED_USER_KERNELS for v in (structure or {}).values()):
        why.append("user-kernels")  # Triton kernels called from compiled code but tuned outside Inductor
    if (resolved or {}).get("quantization"):
        why.append("quantized")  # custom GEMM kernels whose internals the trace cannot see
    if not stack or not stack.get("gpu") or stack.get("driver") is None:
        why.append("no-stack")
    elif health is not None and any(stack.get(k) is None for k in STACK_FIELDS):
        why.append("stack-incomplete")
    if resolved is None:
        why.append("no-resolved-config")
    if health:
        why.append("trace-error:" + ",".join(sorted(health)))
    return why


def identity_v7(cfg, use, dec, stack, env, resolved, model_files, structure, health=None):
    why = coverage_v7(use, structure, stack, resolved, health)
    pw = os.environ.get("LOCKSTEP_V7_POINTWISE_CONFIG") == "1"  # 1: the first version-7 runs (before HB2)
    blob = json.dumps({"cfg": config_part(cfg, bool(use)), "kernels6": usage_kernels(use, dec, pointwise_config=pw),
                       "stack": stack or {}, "env": env_part(env), "resolved": resolved, "model": model_files,
                       "lowering": lowering_set(structure)}, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16], why


def identity(run_dir, version=None):
    """Returns (identity hex, number of kernel objects covered) for a finished run.
    v2 (frozen for E19/E26): snapshot of in-process tuner objects + run-time trace. The snapshot
    differs between a cold-compiling and a warm process (deserialized objects skip the constructor),
    which caused unnecessary blocks on the H800 (g1).
    v3: run-time trace only. Since the tracer keys records per object and re-keys kernels first seen
    outside a subgraph, the trace covers every launched kernel instance in both cold and warm
    processes; kernels that never run cannot affect numerics."""
    version = version or IDENTITY_VERSION
    cfg = json.load(open(os.path.join(run_dir, "cfg.json")))
    if version == 6:
        use, dec = decisions.load_usage(run_dir), decisions.load(run_dir)
        return identity_v6(cfg, use, dec, decisions.load_stack(run_dir)), len(use)
    if version == "6p":  # offline: version 6 without pointwise launch configurations (the HB2 rule)
        use, dec = decisions.load_usage(run_dir), decisions.load(run_dir)
        blob = json.dumps({"cfg": config_part(cfg, bool(use)), "kernels6": usage_kernels(use, dec, pointwise_config=False),
                           "stack": decisions.load_stack(run_dir) or {}}, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:16], len(use)
    if version == 5:
        dec = decisions.load(run_dir)
        traced = any(not k.startswith("__runtime__") for k in dec)
        blob = json.dumps({"cfg": config_part(cfg, traced), "kernels5": canonical_kernels(dec)}, sort_keys=True,
                          default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:16], len(dec)
    snap = decisions.load_snapshot(run_dir) if version == 2 else {}
    dec = decisions.load(run_dir)
    items = sorted((k, json.dumps(v.get("chosen"), sort_keys=True)) for k, v in snap.items())
    items += sorted(("run:" + k, json.dumps(v.get("chosen"), sort_keys=True)) for k, v in dec.items())
    traced = any(not k.startswith("__runtime__") for k in dec) or bool(snap)
    blob = json.dumps({"cfg": config_part(cfg, traced), "kernels": items}, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16], len(snap) + len(dec)


def evaluate(root):
    """Ground truth: Lockstep equivalence classes in each spec's report.json (same class = bitwise
    equal consumed values and log-probabilities). Pairs are taken within and across specs that
    share a prompt set."""
    runs = {}
    for spec in sorted(os.listdir(root)):
        rep = os.path.join(root, spec, "report.json")
        if not os.path.exists(rep):
            continue
        for ci, cls in enumerate(json.load(open(rep)).get("classes", [])):
            for name in cls:
                d = os.path.join(root, spec, name)
                if os.path.exists(os.path.join(d, "outputs.json")):
                    runs[f"{spec}/{name}"] = {"dir": d, "cls": f"{spec}#{ci}"}
    t0 = time.time()
    for r in runs.values():
        r["id"], r["n_kernels"] = identity(r["dir"])
    t_id = (time.time() - t0) / max(len(runs), 1)
    # cross-spec ground truth: compare outputs' logprobs bitwise (all specs share prompts)
    outs = {k: json.load(open(os.path.join(v["dir"], "outputs.json"))) for k, v in runs.items()}
    eq = lambda a, b: all(x["logprobs"] == y["logprobs"] for x, y in zip(outs[a], outs[b]))
    tp = fp = tn = fn = 0
    bad = []
    for a, b in itertools.combinations(sorted(runs), 2):
        same_num = runs[a]["cls"] == runs[b]["cls"] if a.split("/")[0] == b.split("/")[0] else eq(a, b)
        allow = runs[a]["id"] == runs[b]["id"]
        if allow and same_num:
            tp += 1
        elif allow and not same_num:
            fp += 1
            bad.append((a, b))
        elif not allow and same_num:
            fn += 1
        else:
            tn += 1
    return {"runs": len(runs), "pairs": tp + fp + tn + fn, "allowed_and_equal": tp,
            "UNSAFE_allowed_but_different": fp, "blocked_but_equal": fn, "blocked_and_different": tn,
            "unsafe_pairs": bad[:10], "identity_ms_per_instance": 1000 * t_id}


def evaluate_pd(root):
    """Live PD deployment: the guard admits the prefill instance's KV into the decode instance iff
    their identities match. Ground truth: after the PD requests, each instance generated every prompt
    on its own (local_a/local_b.json); the transfer is safe iff both instances compute bit-identical
    tokens and log-probabilities."""
    rows = []
    for sc in sorted(os.listdir(root)):
        d = os.path.join(root, sc)
        need = ["outputs.json", "local_a.json", "local_b.json", "inst_a", "inst_b"]
        if not all(os.path.exists(os.path.join(d, n)) for n in need):
            continue
        ia, na = identity(os.path.join(d, "inst_a"))
        ib, nb = identity(os.path.join(d, "inst_b"))
        la, lb = (json.load(open(os.path.join(d, f))) for f in ("local_a.json", "local_b.json"))
        pd = json.load(open(os.path.join(d, "outputs.json")))
        same = sum(x == y for x, y in zip(la, lb))
        pd_eq_b = sum(x == y for x, y in zip(pd, lb))
        rows.append({"scenario": sc, "allow": ia == ib, "instances_identical": same == len(la),
                     "local_a_eq_b": f"{same}/{len(la)}", "pd_eq_local_b": f"{pd_eq_b}/{len(pd)}",
                     "kernels": (na, nb)})
    ok = sum(r["allow"] == r["instances_identical"] for r in rows)
    unsafe = [r["scenario"] for r in rows if r["allow"] and not r["instances_identical"]]
    return {"scenarios": rows, "correct": f"{ok}/{len(rows)}", "unsafe_allows": unsafe}


if __name__ == "__main__":
    if sys.argv[1] == "--pd":
        print(json.dumps(evaluate_pd(sys.argv[2]), indent=1))
        sys.exit(0)
    if sys.argv[1] == "--eval":
        print(json.dumps(evaluate(sys.argv[2]), indent=1))
    else:
        for r in sys.argv[1:]:
            print(*identity(r), r)
