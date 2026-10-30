"""Configuration minimization: which of the differences between two configurations changes numerics?

python -m lockstep.minimize --spec BASE_SPEC.json --target '{"key": value, ...}' --out-root DIR

The base spec's engine configuration is the reference. `target` lists the engine options in which the
other configuration differs. Delta debugging (explore.ddmin) searches for a minimal subset of these
differences that still makes the outputs differ bit for bit from the base. Each trial is one engine
run scheduled on the shared GPU pool. A trial is cached by its subset, so repeated subsets are free.
Use a base configuration that is reproducible across processes (e.g. with the compiler's
batch-invariant setting); otherwise process-level tuning flips are mistaken for culprits.
"""
import argparse
import copy
import json
import os

from lockstep import explore


def outputs(d):
    return json.load(open(os.path.join(d, "outputs.json")))


def same(a, b):
    return all(x["token_ids"] == y["token_ids"] and x["logprobs"] == y["logprobs"] for x, y in zip(a, b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--target", required=True, help="JSON object: engine options that differ")
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    a = ap.parse_args()
    spec = json.load(open(a.spec))
    target = json.loads(a.target)
    out = os.path.join(a.out_root, os.path.splitext(os.path.basename(a.spec))[0])
    os.makedirs(out, exist_ok=True)
    pool = [int(g) for g in a.gpus.split(",")]
    base = spec["base"]
    log = []

    def run(name, subset):
        r = copy.deepcopy(base)
        r.update({"name": name, "out_dir": out, "prompts": spec["prompts"], "engine": base.get("engine", "vllm"),
                  "gpus": base.get("gpus", 1)})
        r.setdefault("engine_kwargs", {}).update({k: target[k] for k in subset})
        explore.run_all([r], pool)
        return os.path.join(out, name)

    ref = outputs(run("base", []))
    cache = {}

    def violates(subset):
        key = tuple(sorted(subset))
        if key not in cache:
            name = "t_" + "+".join(key) if key else "base"
            d = run(name, list(key))
            cache[key] = not same(ref, outputs(d)) if os.path.exists(os.path.join(d, "outputs.json")) else None
            log.append({"subset": list(key), "violates": cache[key]})
            print("[minimize]", list(key), "->", "DIFFERS" if cache[key] else "identical", flush=True)
        return bool(cache[key])

    if not violates(list(target)):
        print("[minimize] the full set of differences does not change numerics; nothing to minimize")
        res = []
    else:
        res = explore.ddmin(list(target), violates)
    summary = {"target": target, "minimal": res, "trials": log, "runs": len(cache) + 1}
    json.dump(summary, open(os.path.join(out, "minimize.json"), "w"), indent=1, default=str)
    print("[minimize] minimal culprit set:", res, "after", len(cache) + 1, "runs")


if __name__ == "__main__":
    main()
