"""End-to-end harm: sampler (rollout) vs trainer-side recompute of the same tokens' log-probabilities.

python -m lockstep.harm SCORED_RUN_DIR [...]
Each run's outputs.json holds, per sequence, the rollout log-probabilities ("rollout_logprobs") and
the recomputed ones ("logprobs"). On-policy RL assumes they are equal; the importance ratio
exp(trainer - sampler) measures how far the update silently becomes off-policy.
"""
import json
import math
import os
import sys


def summarize(run):
    outs = json.load(open(os.path.join(run, "outputs.json")))
    n = exact = seq_exact = 0
    dmax, ratio_dev = 0.0, []
    for o in outs:
        ok = True
        for r, t in zip(o["rollout_logprobs"], o["logprobs"]):
            if r is None or t is None:
                continue
            n += 1
            if r == t:
                exact += 1
            else:
                ok = False
            d = float(t) - float(r)
            dmax = max(dmax, abs(d))
            ratio_dev.append(abs(math.exp(d) - 1.0))
        seq_exact += ok
    ratio_dev.sort()
    q = lambda p: ratio_dev[min(len(ratio_dev) - 1, int(p * len(ratio_dev)))] if ratio_dev else float("nan")
    return {"run": os.path.basename(run.rstrip("/")), "tokens": n, "token_exact": exact / max(n, 1),
            "seq_exact": f"{seq_exact}/{len(outs)}", "max_abs_dlogp": dmax,
            "ratio_dev_p50": q(0.5), "ratio_dev_p99": q(0.99), "ratio_dev_max": q(1.0)}


if __name__ == "__main__":
    rows = [summarize(r) for r in sys.argv[1:] if os.path.exists(os.path.join(r, "outputs.json"))]
    for x in rows:
        print(json.dumps(x))
