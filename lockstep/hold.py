"""Optional GPU memory placeholders for controlled experiments.

A manager keeps up to --max GPUs either running one of our jobs or held by a placeholder process
that allocates the GPU's free memory. Our schedulers (explore.run_all, gpurun) treat held memory as
free and call release() right before they start a run on that GPU, so the run starts on a GPU that
nobody else could take in between. Placeholders never touch other users' processes; they only claim
memory that is free. The manager exits, and frees every placeholder, when all --while-pids are gone.

python -m lockstep.hold --manager --pool 0,...,7 --max 6 --min-free-gb 20 --while-pids P1,P2 [--top-up]
python -m lockstep.hold --holder --gpu G            (started by the manager)
"""
import argparse
import os
import signal
import subprocess
import sys
import time
import tempfile

LOCK_DIR = os.environ.get("LOCKSTEP_LOCK_DIR",
                          os.path.join(tempfile.gettempdir(), f"lockstep-{os.getuid()}", "gpu-locks"))


def _alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def held():
    """{gpu: (pids, gib)} for live placeholders ($LOCK_DIR/hold<g>.<pid>, content: GiB held). Several
    placeholders on one GPU (e.g. from two managers) are summed, so the held memory is never undercounted."""
    out = {}
    if not os.path.isdir(LOCK_DIR):
        return out
    for f in os.listdir(LOCK_DIR):
        if not f.startswith("hold"):
            continue
        try:
            g, pid = f[4:].split(".")
            g, pid = int(g), int(pid)
            if not _alive(pid):
                os.remove(os.path.join(LOCK_DIR, f))
                continue
            gib = float(open(os.path.join(LOCK_DIR, f)).read().strip() or 0)
            pids, tot = out.get(g, ([], 0.0))
            out[g] = (pids + [pid], tot + gib)
        except (ValueError, OSError):
            pass
    return out


def _free_gib():
    q = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True).stdout
    return {int(i): float(f) / 1024 for i, f in (l.split(",") for l in q.strip().splitlines())}


def release(gpus, need_gib=0.0, timeout=90):
    """Stop the placeholders on `gpus` and wait until their memory is free again."""
    h = held()
    pids = [p for g in gpus if g in h for p in h[g][0]]
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass
    t0 = time.time()
    while time.time() - t0 < timeout:
        if not any(_alive(p) for p in pids):
            f = _free_gib()
            if all(f.get(g, 0) >= need_gib for g in gpus):
                return True
        time.sleep(0.5)
    return False


def _holder(gpu, leave_gib=0.6, until_lock=None):
    """Placeholder (default) or top-up (until_lock set): allocate the GPU's free memory minus
    leave_gib. A top-up fills the memory that one of our running jobs leaves unused and exits as
    soon as that job's lock file disappears; its file is top<g>.<pid>, which held() ignores."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    import torch
    parent = os.getppid()
    free, _ = torch.cuda.mem_get_info()
    nbytes = max(0, free - int(leave_gib * 2**30))
    buf = torch.empty(nbytes, dtype=torch.uint8, device="cuda")
    os.makedirs(LOCK_DIR, exist_ok=True)
    path = os.path.join(LOCK_DIR, f"{'top' if until_lock else 'hold'}{gpu}.{os.getpid()}")
    open(path, "w").write(f"{nbytes / 2**30:.2f}")
    stop = []
    signal.signal(signal.SIGTERM, lambda *_: stop.append(1))
    try:
        while not stop and os.getppid() == parent:  # die with the manager
            if until_lock and not os.path.exists(until_lock):
                break
            time.sleep(1)
    finally:
        del buf
        try:
            os.remove(path)
        except OSError:
            pass


def _standby(gpu, min_free_gib, max_n):
    """Standby placeholder: initialise CUDA on `gpu` in advance, then claim the GPU's free memory within
    about half a second once it has >= min_free_gib free, is not running one of our jobs, and fewer than
    max_n GPUs are ours. Placeholders started by a manager otherwise need ~5 s (torch import, CUDA
    context), and other tenants' schedulers took freed GPUs within that time."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    import torch
    torch.cuda.init()
    parent = os.getppid()
    stop = []
    signal.signal(signal.SIGTERM, lambda *_: stop.append(1))
    while not stop and os.getppid() == parent:
        free, _ = torch.cuda.mem_get_info()
        if free >= min_free_gib * 2**30:
            ours = _ours_locked() | set(held())
            if gpu not in ours and len(ours) < max_n:
                nbytes = max(0, free - (600 << 20))
                try:
                    buf = torch.empty(nbytes, dtype=torch.uint8, device="cuda")
                except RuntimeError:
                    time.sleep(0.5)
                    continue
                path = os.path.join(LOCK_DIR, f"hold{gpu}.{os.getpid()}")
                open(path, "w").write(f"{nbytes / 2**30:.2f}")
                try:
                    while not stop and os.getppid() == parent:
                        time.sleep(0.5)
                finally:
                    del buf
                    try:
                        os.remove(path)
                    except OSError:
                        pass
                return
        time.sleep(0.5)


def _ours_locked():
    from lockstep.explore import locked_gpus
    return locked_gpus()


def _run_locks():
    """{gpu: lock path} for our running jobs (gpu<g>.<pid> files of live pids)."""
    out = {}
    if not os.path.isdir(LOCK_DIR):
        return out
    for f in os.listdir(LOCK_DIR):
        if f.startswith("gpu"):
            try:
                g, pid = f[3:].split(".")
                if _alive(int(pid)):
                    out[int(g)] = os.path.join(LOCK_DIR, f)
            except ValueError:
                pass
    return out


def _manager(pool, max_n, min_free, while_pids, poll=5.0, top_up=False, top_leave=2.0, top_after=240.0, standby=False):
    procs, tops, seen = {}, {}, {}  # placeholders, top-ups, (lock path -> first seen, last free)
    log = lambda m: print(f"[hold] {time.strftime('%H:%M:%S')} {m}", flush=True)
    try:
        while any(_alive(p) for p in while_pids):
            for g, p in list(procs.items()):
                if p.poll() is not None:
                    del procs[g]
            for g, p in list(tops.items()):
                if p.poll() is not None:
                    del tops[g]
            if top_up:
                f = _free_gib()
                for g, lk in _run_locks().items():
                    if g not in pool or g in tops:
                        continue
                    now, cur = time.time(), f.get(g, 0)
                    t0, last, changed = seen.get(lk, (now, cur, now))
                    if abs(cur - last) >= 0.3:
                        changed = now
                    seen[lk] = (t0, cur, changed)
                    # only after start-up: running for top_after s and free memory unchanged for 60 s
                    if now - t0 > top_after and now - changed > 60 and cur > top_leave + 1.0:
                        tops[g] = subprocess.Popen([sys.executable, "-m", "lockstep.hold", "--holder", "--gpu", str(g),
                                                    "--leave-gb", str(top_leave), "--until-lock", lk])
                        log(f"top up gpu {g} ({f[g]:.1f} GiB free, leaving {top_leave} GiB to our job)")
            h, mine = held(), _ours_locked()
            # placeholders started but not yet registered count too, or a burst of starts overshoots max_n
            n = len(set(h) | mine | set(procs))
            if standby:
                for g in pool:
                    if g not in procs:  # also on GPUs running our jobs: claims them the moment the job ends
                        procs[g] = subprocess.Popen([sys.executable, "-m", "lockstep.hold", "--standby", "--gpu", str(g),
                                                     "--min-free-gb", str(min_free), "--max", str(max_n)])
                        log(f"standby on gpu {g}")
                time.sleep(poll)
                continue
            if n < max_n:
                f = _free_gib()
                for g in pool:
                    if n >= max_n:
                        break
                    if g in h or g in mine or g in procs or f.get(g, 0) < min_free:
                        continue
                    procs[g] = subprocess.Popen([sys.executable, "-m", "lockstep.hold", "--holder", "--gpu", str(g)])
                    log(f"hold gpu {g} ({f[g]:.1f} GiB free)")
                    n += 1
            time.sleep(poll)
    finally:
        for p in list(procs.values()) + list(tops.values()):
            p.terminate()
        log("manager exit; placeholders released")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manager", action="store_true")
    ap.add_argument("--holder", action="store_true")
    ap.add_argument("--gpu", type=int)
    ap.add_argument("--pool", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--max", type=int, default=6)
    ap.add_argument("--min-free-gb", type=float, default=20.0)
    ap.add_argument("--while-pids", default="")
    ap.add_argument("--poll", type=float, default=5.0, help="seconds between checks")
    ap.add_argument("--top-up", action="store_true", help="also fill memory our running jobs leave unused")
    ap.add_argument("--leave-gb", type=float, default=0.6)
    ap.add_argument("--until-lock", default=None)
    ap.add_argument("--standby", action="store_true", help="manager: keep CUDA-ready standby placeholders; or run one")
    a = ap.parse_args()
    if a.standby and not a.manager:
        _standby(a.gpu, a.min_free_gb, a.max)
    elif a.holder:
        _holder(a.gpu, a.leave_gb, a.until_lock)
    else:
        _manager([int(x) for x in a.pool.split(",")], a.max, a.min_free_gb,
                 [int(x) for x in a.while_pids.split(",") if x], a.poll, a.top_up, standby=a.standby)


if __name__ == "__main__":
    main()
