"""RW timeline for the results figure: for each RW deployment, every read the consumer checked (seconds since the
producer's decision, verdict) and when the consumer invalidated the path.

python -m lockstep.rw_timeline RUN_DIR > rw_timeline.json
"""
import glob
import json
import os
import sys


def _events(path):
    out = []
    for f in glob.glob(path):
        out += [json.loads(l) for l in open(f)]
    return sorted(out, key=lambda e: e["t"])


def main(run):
    res = {}
    for d in sorted(glob.glob(os.path.join(run, "rw_*"))):
        if not os.path.exists(os.path.join(d, "local_b.json")) or "smoke" in d:
            continue
        ea = _events(os.path.join(d, "inst_a", "capture", "live_guard.jsonl"))
        eb = _events(os.path.join(d, "inst_b", "capture", "live_guard.jsonl"))
        t0 = next((e["t"] for e in ea if e["event"] == "test_retuned"), None)
        if t0 is None:
            continue
        mode = next(e.get("mode") for e in eb if e["event"] == "xfer_tag_installed")
        res[os.path.basename(d)] = {
            "mode": mode, "tp": "_tp_" in d,
            "reads": [[round(e["t"] - t0, 3), e["verdict"]] for e in eb if e["event"] == "xfer_check"],
            "invalidated": [round(e["t"] - t0, 3) for e in eb if e["event"] == "invalidated" and e["t"] >= t0]}
    json.dump(res, sys.stdout, indent=0)


if __name__ == "__main__":
    main(sys.argv[1])
