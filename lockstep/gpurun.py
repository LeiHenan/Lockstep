"""Run a command on free GPU(s) while holding the shared Lockstep GPU locks.

python -m lockstep.gpurun --n 1 --need-gb 12 [--pool 0,...,7] [--wait] -- <command ...>
Sets CUDA_VISIBLE_DEVICES for the command; locks are released when it exits.
"""
import argparse
import os
import subprocess
import sys
import time

from lockstep.explore import LOCK_DIR, claim, free_gpus, lock, unlock


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--need-gb", type=float, required=True)
    ap.add_argument("--pool", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--wait", action="store_true", help="wait until enough GPUs are free")
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
    pool = [int(x) for x in a.pool.split(",")]
    resv = None
    if a.wait and a.n > 1:  # ask explorers to let GPUs drain to us
        os.makedirs(LOCK_DIR, exist_ok=True)
        resv = os.path.join(LOCK_DIR, f"reserve.{os.getpid()}.{a.n}")
        open(resv, "w").close()
    import fcntl
    os.makedirs(LOCK_DIR, exist_ok=True)
    sel = open(os.path.join(LOCK_DIR, ".gpurun_select"), "w")
    while True:
        # choosing GPUs and locking them is one critical section: two gpurun processes that chose at the same
        # moment both took GPU 1 and 2 (2026-10-23, CF/KVE) and started four servers on two GPUs
        fcntl.flock(sel, fcntl.LOCK_EX)
        g = free_gpus(pool, a.need_gb)[: a.n]
        if len(g) == a.n:
            claim(g, a.need_gb)  # stop our placeholders on these GPUs (if any) right before the start
            fcntl.flock(sel, fcntl.LOCK_UN)
            break
        fcntl.flock(sel, fcntl.LOCK_UN)
        if not a.wait:
            sys.exit(f"[gpurun] not enough free GPUs (need {a.n} x {a.need_gb} GB)")
        time.sleep(15)
    if resv and os.path.exists(resv):
        os.remove(resv)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, g))
    proc = subprocess.Popen(cmd, env=env, start_new_session=True)
    lock(g, proc.pid)
    unlock(g, os.getpid())
    import signal

    def _stop(*_):  # forward termination to the whole child process group (engines included)
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        unlock(g, proc.pid)
        sys.exit(143)
    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    print(f"[gpurun] gpus={g} pid={proc.pid}", flush=True)
    try:
        rc = proc.wait()
    finally:
        unlock(g, proc.pid)
    sys.exit(rc)


if __name__ == "__main__":
    main()
