"""KV transfer rate of a PD run, from vLLM's "KV Transfer metrics" log lines.

python -m lockstep.xfer_rate RUN_DIR [RUN_DIR ...]

Per run: transfers, MB moved, and the per-transfer rate (MB moved / summed transfer time, in GB/s).
Concurrent transfers overlap, so the per-transfer rate is a lower bound on the link's aggregate rate;
vLLM's own logged "Throughput (MB/s)" values are reported alongside (median).
"""
import glob
import json
import os
import re
import statistics as st
import sys

PAT = re.compile(r"KV Transfer metrics: Num successful transfers=(\d+), Avg xfer time \(ms\)=([\d.]+).*?"
                 r"Avg MB per transfer=([\d.]+), Throughput \(MB/s\)=([\d.]+)")


def rate(d):
    n = mb = ms = 0.0
    logged = []
    for f in glob.glob(os.path.join(d, "server_*.log")):
        for line in open(f, errors="replace"):
            m = PAT.search(line)
            if m:
                k, t, size, thr = int(m[1]), float(m[2]), float(m[3]), float(m[4])
                n, mb, ms = n + k, mb + k * size, ms + k * t
                logged.append(thr)
    return {"run": os.path.basename(os.path.normpath(d)), "transfers": int(n), "mb": round(mb, 1),
            "per_transfer_gb_s": round(mb / ms, 3) if ms else None,
            "logged_throughput_mb_s_median": st.median(logged) if logged else None}


if __name__ == "__main__":
    print(json.dumps([rate(d) for d in sys.argv[1:]], indent=1))
