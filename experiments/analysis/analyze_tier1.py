"""Analysis for the tier-1/2 supplementary experiments (E19-E23). Set LOCKSTEP_RUNS_ROOT to the captures directory:
    python analyze_tier1.py  ->  prints a report and writes runs/tier1_report.json"""
import itertools
import json
import math
import os
import statistics

from lockstep import guard, harm

R = os.environ.get("LOCKSTEP_RUNS_ROOT", "runs")


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(max(0, c - h), 3), round(min(1, c + h), 3))


def classes(spec_dir):
    rep = json.load(open(os.path.join(spec_dir, "report.json")))
    return {n: i for i, c in enumerate(rep["classes"]) for n in c}


def e19():
    out = {}
    root = f"{R}/e19"
    for spec in sorted(os.listdir(root)):
        d = os.path.join(root, spec)
        if not os.path.exists(os.path.join(d, "report.json")):
            continue
        cls = classes(d)
        names = sorted(n for n in cls if os.path.exists(os.path.join(d, n, "outputs.json")))
        ids = {n: guard.identity(os.path.join(d, n))[0] for n in names}
        tp = fp = fn = tn = 0
        unsafe, false_block = [], []
        for a, b in itertools.combinations(names, 2):
            same, allow = cls[a] == cls[b], ids[a] == ids[b]
            if allow and same:
                tp += 1
            elif allow:
                fp += 1
                unsafe.append((a, b))
            elif same:
                fn += 1
                false_block.append((a, b))
            else:
                tn += 1
        out[spec] = {"processes": len(names), "classes": len(set(cls[n] for n in names)),
                     "divergent_pairs": fp + tn, "missed": fp, "miss_rate_ci": wilson(fp, fp + tn),
                     "identical_pairs": tp + fn, "false_blocks": fn, "false_block_ci": wilson(fn, tp + fn),
                     "unsafe_pairs": unsafe[:6], "false_block_pairs": false_block[:6]}
    return out


def e20():
    out = {}
    for d in (f"{R}/e9/e9_score", f"{R}/e20/e20_score_llama"):
        if not os.path.isdir(d):
            continue
        rows = [harm.summarize(os.path.join(d, n)) for n in sorted(os.listdir(d))
                if os.path.exists(os.path.join(d, n, "outputs.json"))]
        out[d] = rows
    return out


def tokens_per_s(run):
    meta = json.load(open(os.path.join(run, "meta.json")))
    outs = json.load(open(os.path.join(run, "outputs.json")))
    return sum(len(o["token_ids"]) for o in outs) / meta["gen_wall_s"]


def e21():
    d = f"{R}/e21/e21_overhead"
    if not os.path.isdir(d):
        return None
    tps = {}
    for n in sorted(os.listdir(d)):
        if os.path.exists(os.path.join(d, n, "outputs.json")) and "_r" in n:
            tps.setdefault(n.split("_r")[0], []).append(tokens_per_s(os.path.join(d, n)))
    med = {k: statistics.median(v) for k, v in tps.items()}
    off = med.get("off")
    return {"tokens_per_s": tps, "median": med,
            "relative_to_off": {k: (v / off if off else None) for k, v in med.items()}}


def e22():
    d = f"{R}/e22/e22_load"
    if not os.path.exists(os.path.join(d, "report.json")):
        return None
    cls = classes(d)
    warm = [n for n in cls if n != "base"]
    major = statistics.mode([cls[n] for n in warm])
    arms = {}
    for n in warm:
        arm = "load" if n.startswith("load") else ("noload" if n.startswith("noload") else "other")
        a = arms.setdefault(arm, [0, 0])
        a[0] += cls[n] != major
        a[1] += 1
    states = {}
    for n in warm:
        f = os.path.join(d, n, "gpu_state.txt")
        if os.path.exists(f):
            states[n] = open(f).read().split("\n")[1]
    return {"flips": {k: {"flipped": v[0], "n": v[1], "ci": wilson(v[0], v[1])} for k, v in arms.items()},
            "gpu_state_at_launch": states}


def e23():
    d = f"{R}/e23/e23_ultra256"
    if not os.path.exists(os.path.join(d, "report.json")):
        return None
    rep = json.load(open(os.path.join(d, "report.json")))
    vs = {k: {kk: v.get(kk) for kk in ("mismatched_keys", "common", "prompts_diverged", "token_identical_frac")}
          for k, v in (rep.get("vs_base") or {}).items() if v}
    return {"classes": rep["classes"], "vs_base": vs}


if __name__ == "__main__":
    res = {"E19": e19(), "E20": e20(), "E21": e21(), "E22": e22(), "E23": e23()}
    json.dump(res, open(f"{R}/tier1_report.json", "w"), indent=1, default=str)
    print(json.dumps(res, indent=1, default=str)[:12000])
