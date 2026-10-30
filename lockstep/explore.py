"""Config Explorer: run configuration variants, compare, cluster into numerical equivalence classes.

python -m lockstep.explore --spec spec.json --out DIR [--gpus 0,1,2] [--min-free-gb 21]

spec.json:
{
  "base": {"engine_kwargs": {...}, "env": {...}, "runner_args": ["--max-tokens","1"]},
  "prompts": "path/to/prompts.json",
  "variants": [ {"name": "mbt2048", "engine_kwargs": {"max_num_batched_tokens": 2048}},
                {"name": "tp2", "engine_kwargs": {"tensor_parallel_size": 2}, "gpus": 2},
                {"name": "serial", "runner_args": ["--order","serial"]}, ... ],
  "dims": {"mbt2048": "max_num_batched_tokens", ...}   # optional: dimension label per variant
}
Every variant is the base with overrides applied. The base itself is run twice ("base", "base_rep")
as a same-config control.
"""
import argparse
import copy
import json
import os
import subprocess
import sys
import time
import tempfile

from lockstep import align


def deep_update(d, u):
    for k, v in u.items():
        if isinstance(v, dict) and isinstance(d.get(k), dict):
            deep_update(d[k], v)
        else:
            d[k] = v
    return d


def materialize(spec):
    base = spec["base"]
    runs = [{"name": "base", "engine": base.get("engine", "vllm"), **copy.deepcopy(base)},
            {"name": "base_rep", "engine": base.get("engine", "vllm"), **copy.deepcopy(base)}]
    for v in spec["variants"]:
        r = copy.deepcopy(base)
        deep_update(r.setdefault("engine_kwargs", {}), v.get("engine_kwargs", {}))
        deep_update(r.setdefault("env", {}), v.get("env", {}))
        r["runner_args"] = base.get("runner_args", []) + v.get("runner_args", [])
        r["engine"] = v.get("engine", base.get("engine", "vllm"))
        r["gpus"] = v.get("gpus", base.get("gpus", 1))
        r["name"] = v["name"]
        runs.append(r)
    return runs


LOCK_DIR = os.environ.get("LOCKSTEP_LOCK_DIR",
                          os.path.join(tempfile.gettempdir(), f"lockstep-{os.getuid()}", "gpu-locks"))


def _pid_alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def locked_gpus():
    """GPUs held by any of our live runs (explorers or manual tests), via $LOCK_DIR/gpu<N>.<pid>."""
    held = set()
    if not os.path.isdir(LOCK_DIR):
        return held
    for f in os.listdir(LOCK_DIR):
        try:
            g, pid = f[3:].split(".")
            if _pid_alive(int(pid)):
                held.add(int(g))
            else:
                os.remove(os.path.join(LOCK_DIR, f))
        except (ValueError, OSError):
            pass
    return held


def reservations():
    """Total GPUs requested by live multi-GPU waiters (files $LOCK_DIR/reserve.<pid>.<n>)."""
    total = 0
    if not os.path.isdir(LOCK_DIR):
        return 0
    for f in os.listdir(LOCK_DIR):
        if not f.startswith("reserve."):
            continue
        try:
            _, pid, n = f.split(".")
            if _pid_alive(int(pid)):
                total += int(n)
            else:
                os.remove(os.path.join(LOCK_DIR, f))
        except (ValueError, OSError):
            pass
    return total


def lock(gpus, pid):
    os.makedirs(LOCK_DIR, exist_ok=True)
    for g in gpus:
        open(os.path.join(LOCK_DIR, f"gpu{g}.{pid}"), "w").close()


def unlock(gpus, pid):
    for g in gpus:
        try:
            os.remove(os.path.join(LOCK_DIR, f"gpu{g}.{pid}"))
        except OSError:
            pass


def free_gpus(candidates, min_free_gb):
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    free = {}
    for line in out.strip().splitlines():
        i, f = [x.strip() for x in line.split(",")]
        free[int(i)] = float(f) / 1024
    held = locked_gpus()
    ph = {}
    try:  # memory held by our own placeholders (lockstep.hold) counts as free for our runs
        from lockstep.hold import held as _placeholders
        ph = _placeholders()
        for g, (_, gib) in ph.items():
            free[g] = free.get(g, 0) + gib
    except Exception:
        pass
    ok = [g for g in candidates if free.get(g, 0) >= min_free_gb and g not in held]
    # GPUs our placeholders hold come first: a run then takes a held GPU instead of an extra one, so runs and
    # placeholders together never exceed the number of GPUs we may occupy
    return sorted(ok, key=lambda g: g not in ph)


def claim(gpus, need):
    """Mark `gpus` as ours, then stop our placeholders on them (if any) and wait for their memory."""
    lock(gpus, os.getpid())
    try:
        from lockstep.hold import held, release
        release(gpus, need)
        # Concurrent placeholder registration may exceed the configured GPU cap.
        cap = int(os.environ.get("LOCKSTEP_MAX_GPUS", "4"))
        extra = sorted(set(held()) - locked_gpus())
        over = len(locked_gpus() | set(held())) - cap
        if over > 0:
            release(extra[:over])
    except Exception:
        pass


def launch(run, out_dir, prompts, gpus):
    rdir = os.path.join(out_dir, run["name"])
    os.makedirs(rdir, exist_ok=True)
    cap = os.path.join(rdir, "capture")
    if os.path.isdir(cap) and os.listdir(cap):  # partial capture from an interrupted/failed attempt: keep, but move aside
        os.rename(cap, f"{cap}.stale_{time.strftime('%Y%m%d_%H%M%S')}")
    for f in ("run.log",):
        if os.path.exists(os.path.join(rdir, f)):
            os.rename(os.path.join(rdir, f), os.path.join(rdir, f"{f}.prev_{time.strftime('%Y%m%d_%H%M%S')}"))
    cfg_path = os.path.join(rdir, "cfg.json")
    json.dump({"engine_kwargs": run["engine_kwargs"], "env": run.get("env", {})}, open(cfg_path, "w"), indent=1)
    env = os.environ.copy()
    py, mod = sys.executable, "lockstep.runner"
    if run.get("engine") == "sglang":
        env["LOCKSTEP_ENGINE"] = "sglang"
        bootstrap = env.get("LOCKSTEP_SGLANG_BOOTSTRAP", os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sitecustom"))
        if os.path.isdir(bootstrap):
            env["PYTHONPATH"] = os.pathsep.join(filter(None, (bootstrap, env.get("PYTHONPATH"))))
        py, mod = env.get("LOCKSTEP_SGLANG_PYTHON", sys.executable), "lockstep.runner_sgl"
    env.update({k: str(v) for k, v in run.get("env", {}).items()})
    env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpus))
    try:  # co-tenant load on the assigned GPUs at launch (timing noise drives autotuning flips)
        q = subprocess.run(["nvidia-smi", "-i", ",".join(map(str, gpus)), "--query-gpu=index,utilization.gpu,memory.used",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20).stdout
        a = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=20).stdout
        open(os.path.join(rdir, "gpu_state.txt"), "w").write(time.strftime("%Y-%m-%dT%H:%M:%S\n") + q + "--\n" + a)
    except Exception:
        pass
    log = open(os.path.join(rdir, "run.log"), "w")
    cmd = [py, "-m", mod, "--config", cfg_path, "--prompts", prompts, "--out", rdir] + run.get("runner_args", [])
    # own process group, so that engine subprocesses can be killed together (no orphans holding GPUs)
    return subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)


def kill_group(p, sig=None):
    import signal as _signal
    try:
        os.killpg(p.pid, sig or _signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


def need_gb(run, total_gb=23.64, margin=1.5):  # GiB; nvidia-smi MiB / 1024
    ek = run["engine_kwargs"]
    frac = ek.get("gpu_memory_utilization", ek.get("mem_fraction_static", 0.9))
    return frac * total_gb + margin


def run_all(runs, gpu_pool, poll=10, max_retries=3):
    """runs: dicts with name, out_dir, prompts, engine_kwargs, env, runner_args, gpus.
    One of our runs per GPU at a time; a GPU is eligible if its free memory covers the run's need."""
    tag = lambda r: f"{os.path.basename(r['out_dir'])}/{r['name']}"
    done = lambda r: os.path.exists(os.path.join(r["out_dir"], r["name"], "outputs.json"))         or os.path.exists(os.path.join(r["out_dir"], r["name"], "HANG"))
    timeout = float(os.environ.get("LOCKSTEP_RUN_TIMEOUT", "1800"))
    started = {}
    pending = [r for r in runs if not done(r)]
    tries = {tag(r): 0 for r in pending}
    active = {}  # tag -> (proc, gpus, run)
    while pending or active:
        for t, (p, g, r) in list(active.items()):
            if p.poll() is None and time.time() - started.get(t, time.time()) > timeout:
                # a hang is a finding, not a transient failure: record it and do not retry
                kill_group(p)
                p.wait()
                with open(os.path.join(r["out_dir"], r["name"], "HANG"), "w") as fh:
                    fh.write(f"killed after {timeout:.0f}s without finishing" + chr(10))
                print(f"[explore] HANG {t} after {timeout:.0f}s gpus={g}", flush=True)
                del active[t]
                unlock(g, p.pid)
                continue
            if p.poll() is not None:
                ok = p.returncode == 0 and done(r)
                print(f"[explore] {'done' if ok else 'FAILED'} {t} rc={p.returncode} gpus={g}", flush=True)
                del active[t]
                unlock(g, p.pid)
                if not ok and tries[t] < max_retries:
                    tries[t] += 1
                    pending.append(r)
        busy = {x for _, g, _ in active.values() for x in g}
        reserved = reservations()
        if reserved:
            # a multi-GPU job is waiting: let GPUs drain to it instead of refilling them
            if len([g for g in free_gpus(gpu_pool, 18.0) if g not in busy]) <= reserved:
                time.sleep(poll)
                continue
        for r in list(pending):
            if r["name"] != "base" and not os.path.exists(os.path.join(r["out_dir"], "base", "outputs.json"))                     and any(p["name"] == "base" and p["out_dir"] == r["out_dir"] for p in pending + [x[2] for x in active.values()]):
                continue  # warm-up: let the base run populate the compile caches first
            need = r.get("gpus", 1)
            cap = int(os.environ.get("LOCKSTEP_MAX_GPUS", "0") or 0)  # cap on GPUs that are ours at once
            avail = [g for g in free_gpus(gpu_pool, need_gb(r)) if g not in busy]
            if cap:  # our runs and our placeholders together; take held GPUs first, so a claim keeps the total
                from lockstep.hold import held
                hset = set(held())
                avail = sorted(avail, key=lambda x: x not in hset)
                take = avail[:need]
                if len(busy | hset | set(take)) > cap:
                    break
            if len(avail) >= need:
                g = avail[:need]
                busy |= set(g)
                claim(g, need_gb(r))
                proc = launch(r, r["out_dir"], r["prompts"], g)
                lock(g, proc.pid)
                unlock(g, os.getpid())
                started[tag(r)] = time.time()
                active[tag(r)] = (proc, g, r)
                pending.remove(r)
                print(f"[explore] start {tag(r)} on {g}", flush=True)
        time.sleep(poll)


def compile_state(run_dir):
    """'hit' if vLLM loaded all compiled graphs from its cache, 'compiled' if it compiled any, else '?'."""
    f = os.path.join(run_dir, "run.log")
    if not os.path.exists(f):
        return "?"
    txt = open(f, errors="ignore").read()
    if "Compiling a graph for compile range" in txt:
        return "compiled"
    if "Directly load the compiled graph" in txt:
        return "hit"
    return "?"


def analyze(runs, out_dir):
    tabs = {}
    for r in runs:
        rdir = os.path.join(out_dir, r["name"])
        if not os.path.exists(os.path.join(rdir, "outputs.json")):
            continue
        keep = align.prompt_keys(rdir)
        tabs[r["name"]], _ = align.kappa_table(align.load_records(rdir), keep)
    names = list(tabs)
    pair = {}
    for a in names:
        for b in names:
            if a < b:
                c = align.compare(tabs[a], tabs[b])
                pair[(a, b)] = c
    # token-level agreement (covers decode steps replayed from CUDA graphs, which bypass capture)
    toks = {}
    for n in names:
        outs = json.load(open(os.path.join(out_dir, n, "outputs.json")))
        toks[n] = {o["pkey"]: o["token_ids"] for o in outs}
    lps = {}
    for n in names:
        outs = json.load(open(os.path.join(out_dir, n, "outputs.json")))
        lps[n] = {o["pkey"]: o.get("logprobs") for o in outs}
    for (a, b), c in pair.items():
        ks = toks[a].keys() & toks[b].keys()
        c["token_identical_frac"] = sum(toks[a][k] == toks[b][k] for k in ks) / max(1, len(ks))
        has_lp = all(lps[a].get(k) is not None and lps[b].get(k) is not None for k in ks) and len(ks) > 0
        c["logprob_bitwise_frac"] = (sum(lps[a][k] == lps[b][k] for k in ks) / len(ks)) if has_lp else None
        # signal used for equivalence: kappa fingerprints when captured, else exact logprobs + tokens
        if c["common"] > 0:
            c["equal"] = c["mismatched_keys"] == 0
        elif c["logprob_bitwise_frac"] is not None:
            c["equal"] = c["logprob_bitwise_frac"] == 1.0 and c["token_identical_frac"] == 1.0
        else:
            c["equal"] = None
    # equivalence classes: union of pairs with zero mismatches on a non-trivial common key set
    parent = {n: n for n in names}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for (a, b), c in pair.items():
        if c.get("equal"):
            parent[find(a)] = find(b)
    classes = {}
    for n in names:
        classes.setdefault(find(n), []).append(n)
    cstate = {n: compile_state(os.path.join(out_dir, n)) for n in names}
    for (a, b), c in pair.items():
        c["compile_state"] = (cstate[a], cstate[b])
    vs_base = {n: pair.get(tuple(sorted(("base", n)))) for n in names if n != "base"}
    return {"classes": sorted(classes.values(), key=lambda c: ("base" not in c, c)),
            "vs_base": vs_base,
            "pairs": {f"{a}|{b}": v for (a, b), v in pair.items()}}


def ddmin(diff_keys, violates):
    """Classic delta debugging: minimal subset of differing config keys that still violates.
    violates(subset) -> bool runs the base with only `subset` of the differences applied."""
    n = 2
    keys = list(diff_keys)
    while len(keys) >= 2:
        chunk = max(1, len(keys) // n)
        subsets = [keys[i:i + chunk] for i in range(0, len(keys), chunk)]
        reduced = False
        for s in subsets:
            if violates(s):
                keys, n, reduced = s, 2, True
                break
            comp = [k for k in keys if k not in s]
            if comp and violates(comp):
                keys, n, reduced = comp, max(n - 1, 2), True
                break
        if not reduced:
            if n >= len(keys):
                break
            n = min(len(keys), 2 * n)
    return keys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True, action="append", help="repeatable; all specs share one GPU pool")
    ap.add_argument("--out-root", required=True, help="each spec writes to OUT_ROOT/<spec basename>")
    ap.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--analyze-only", action="store_true")
    ap.add_argument("--max-retries", type=int, default=3,
                    help="start-up failures (e.g. other tenants taking GPU memory) are retried this many times")
    a = ap.parse_args()
    specs = [(sp, json.load(open(sp))) for sp in a.spec]
    allruns, per_spec = [], []
    for sp, spec in specs:
        out = os.path.join(a.out_root, os.path.splitext(os.path.basename(sp))[0])
        os.makedirs(out, exist_ok=True)
        runs = materialize(spec)
        for r in runs:
            r["out_dir"], r["prompts"] = out, spec["prompts"]
        allruns += runs
        per_spec.append((out, runs))
    if not a.analyze_only:
        run_all(allruns, [int(x) for x in a.gpus.split(",")], max_retries=a.max_retries)
    for out, runs in per_spec:
        report(runs, out)
    print("EXPLORE_DONE")


def report(runs, out):
    rep = analyze(runs, out)
    json.dump(rep, open(os.path.join(out, "report.json"), "w"), indent=1, default=str)
    print(f"===== {out}")
    print("EQUIVALENCE CLASSES:")
    for c in rep["classes"]:
        print("  ", c)
    print("VS BASE:")
    for n, c in rep["vs_base"].items():
        if c:
            print(f"  {n:28s} common={c['common']:>8} mismatched={c['mismatched_keys']:>8} "
                  f"prompts_div={c['prompts_diverged']}/{c['prompts']} tok_id={c['token_identical_frac']:.2f} "
                  f"lp_bit={c['logprob_bitwise_frac'] if c['logprob_bitwise_frac'] is None else round(c['logprob_bitwise_frac'], 2)} first={c['earliest_site']} cc={c.get('compile_state')}")


if __name__ == "__main__":
    main()
