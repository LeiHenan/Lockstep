"""Hidden-decision tracing: record the numerics-relevant choices the compiler/runtime makes per process.

Installed by the vLLM plugin when LOCKSTEP_DIR is set. For every Inductor-generated Triton kernel it
records, at the first launch, the configuration actually used (block sizes, warps, stages), the
number of candidate configurations, and whether the choice came from run-time benchmarking (with
the measured timings). The piecewise subgraph that was executing is attached, so a decision diff
can be joined with Lockstep's first-divergence site.

Output: $LOCKSTEP_DIR/decisions_rank{R}_pid{P}.json  (rewritten at every flush and at exit)
"""
import atexit
import json
import os

_DEC = {}            # kernel key -> record
_CTX = {"sub": None}  # currently executing piecewise subgraph index
_INST = {}            # (subgraph, kernel name) -> count of distinct kernel objects seen
_KEYS = {}            # id(tuner) -> key
_KEPT = []            # strong refs to keyed tuners, so ids stay unique
_SEEN = set()         # ids of tuners whose first launch was recorded
_EAGER = os.environ.get("LOCKSTEP_DEC_EAGER_FLUSH") == "1"  # flush on every new record
_ALL = []             # every CachingAutotuner constructed in this process (snapshot at flush)
_STRUCT = {}          # compiled module file -> lowering structure (extern library calls, generated kernels)
_HIST = []           # every autotuning call in order (E25b: does a tuner re-tune after its first launch?)
_GEMM_EXTERN = ("mm", "addmm", "bmm", "baddbmm", "_int_mm", "_scaled_mm", "convolution")
_FORCE = None         # kernel key -> str(config) to select instead of the timing winner (LOCKSTEP_FORCE)


def _force_map():
    """Decision forcing (planted-decision experiments): LOCKSTEP_FORCE names a JSON file that maps
    kernel keys to the configuration (as str(_cfg(launcher))) that autotuning must select. Timing still
    runs; only the winner is overridden. Used to replay a reference process's decisions and change one."""
    global _FORCE
    if _FORCE is None:
        p = os.environ.get("LOCKSTEP_FORCE")
        _FORCE = json.load(open(p)) if p else {}
    return _FORCE


def set_subgraph(idx):
    _CTX["sub"] = idx


def _cfg(launcher):
    c = getattr(launcher, "config", None)
    if c is None:
        return None
    return {"kwargs": dict(getattr(c, "kwargs", {}) or {}), "num_warps": getattr(c, "num_warps", None),
            "num_stages": getattr(c, "num_stages", None)}


def _bin(tuner, launcher):
    """Binary-level facts that the config alone does not fix (O-002 investigation, E25): register and
    spill counts of the launcher actually used, the autotune-cache state (a cache hit skips the
    occupancy-driven RBLOCK halving), and how many compiled candidates existed."""
    try:
        info = getattr(tuner, "autotune_cache_info", None) or {}
        return {"n_regs": getattr(launcher, "n_regs", None), "n_spills": getattr(launcher, "n_spills", None),
                "shared": getattr(launcher, "shared", None), "cache_state": info.get("autotune_cache_state"),
                "n_compiled": len(getattr(tuner, "compile_results", None) or []),
                "rblock_scaling": bool(getattr(tuner, "_could_rblock_scale", False))}
    except Exception as e:  # never let tracing break the engine
        return {"error": repr(e)[:80]}


def _key(tuner):
    """Stable across processes: subgraph + kernel name + instance order within that subgraph.
    Each compiled subgraph owns its own autotuner objects, which tune independently.
    Keys are held per object (by id, with a strong reference so ids are never reused), not as an
    attribute on the tuner: vLLM builds the per-layer kernel objects of some models (e.g. Qwen3-4B)
    as copies of one object, so an attribute set on the original was inherited by all 36 copies and
    their decisions overwrote each other under one key (found in E25b; this hid O-002)."""
    k = _KEYS.get(id(tuner))
    if k is not None:
        return k
    try:
        name = getattr(getattr(tuner, "fn", None), "__name__", "?")
    except Exception:
        name = "?"
    sub = _CTX["sub"]
    i = _INST.get((sub, name), 0)
    _INST[(sub, name)] = i + 1
    k = f"sub{sub}/{name}#{i}"
    _KEYS[id(tuner)] = k
    _KEPT.append(tuner)
    return k


def _path():
    d = os.environ.get("LOCKSTEP_DIR")
    if not d:
        return None
    try:
        import torch.distributed as dist
        r = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    except Exception:
        r = 0
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"decisions_rank{r}_pid{os.getpid()}.json")


_HEALTH = {}  # site -> {"n", "first"}: tracing failures. Any failure makes the identity partial (fail closed):
                # the engine keeps serving, but the guard no longer admits on identity
_FAIL_SITE = os.environ.get("LOCKSTEP_TEST_FAIL_SITE")  # test hook: raise at this site (fail-closed acceptance)


def _fail(site, e):
    h = _HEALTH.setdefault(site, {"n": 0, "first": repr(e)[:200]})
    h["n"] += 1


def _test_fail(site):
    """LOCKSTEP_TEST_FAIL_SITE=<site> fails every call at the site, <site>:once only the first, so that the trace is
    complete but for one record (the case an empty-trace check cannot see)."""
    if _FAIL_SITE == site or (_FAIL_SITE == site + ":once" and site not in _HEALTH):
        raise RuntimeError(f"lockstep test: injected failure at {site}")


def _src(tuner):
    """Short digest of the generated kernel source (shape constants included)."""
    try:
        _test_fail("src")
        import hashlib
        return hashlib.sha1(tuner.fn.src.encode()).hexdigest()[:12]
    except Exception as e:
        _fail("src", e)
        return None


def _srcn(tuner):
    """Digest of the kernel source with its generated name removed. Inductor names the same kernel
    differently when it compiles cold and when it loads the module from the cache, and the name is part
    of the source, so the plain digest differs although the code is the same (2026-10-20, LG-M)."""
    try:
        _test_fail("src")
        import hashlib
        return hashlib.sha1(tuner.fn.src.replace(tuner.fn.__name__, "K").encode()).hexdigest()[:12]
    except Exception as e:
        _fail("src", e)
        return None


_SUBKEYS = {}  # (id(tuner), subgraph) -> (key of the per-subgraph usage record, id of its launcher, config)
_ON_USAGE = []  # callbacks run after a new usage record (the live guard republishes / drops stale canaries)
_SUBOBJ = {}    # (id(tuner), subgraph) -> the tuner (strong reference; for retune_for_test)
_VER = [0]      # bumped when a record's configuration changes in place (retune_for_test, adopt): with len(_USE), the
                # live guard's per-request transfer tag notices every change of what the process runs


def _usage(tuner):
    """Record the configuration a kernel object runs with in the current subgraph, once per (object,
    subgraph, configuration). A cold process compiles a subgraph once and runs identical subgraphs with the
    same kernel objects (e.g. layers 1 and 2), while a warm process loads separate objects per subgraph;
    per-object records alone then leave a subgraph without records in the cold process only. An object that
    later runs with another configuration (a decision made late) gets a further record."""
    sub = _CTX["sub"]
    ls = getattr(tuner, "launchers", None)
    if sub is None or not ls:
        return
    prev = _SUBKEYS.get((id(tuner), sub))
    if prev is not None and prev[1] == id(ls[0]):
        return
    cfg = _cfg(ls[0])
    if prev is not None and prev[2] == cfg:  # a new launcher object with the same configuration
        _SUBKEYS[(id(tuner), sub)] = (prev[0], id(ls[0]), cfg)
        return
    name = getattr(getattr(tuner, "fn", None), "__name__", "?")
    i = _UINST.get((sub, name), 0)
    _UINST[(sub, name)] = i + 1
    k = f"use{sub}/{name}#{i}"
    _SUBKEYS[(id(tuner), sub)] = (k, id(ls[0]), cfg)
    _SUBOBJ[(id(tuner), sub)] = tuner  # _ALL misses objects built before install (warm processes load them early)
    _USE[k] ={"subgraph": sub, "kernel": name, "chosen": cfg, "srcn": _srcn(tuner)}
    if _EAGER:
        flush()
    for cb in _ON_USAGE:
        try:
            cb()
        except Exception as e:  # never let tracing break the engine; but the guard must know
            _fail("on-usage", e)
            if _EAGER:
                flush()


def retune_for_test(sub, name):
    """Test hook (LD experiment): give every kernel object `name` that ran in subgraph `sub` another launch
    configuration, its reduction block halved (doubled if 16 or less), as a decision made after start-up
    would. Returns the (old, new) configurations applied."""
    import copy
    done = []
    for (tid, s), (_, _, cfg) in list(_SUBKEYS.items()):
        if sub is not None and s != sub:  # sub None: every subgraph, as a flip in a cold process (shared objects)
            continue
        t = _SUBOBJ.get((tid, s))
        if t is None or getattr(getattr(t, "fn", None), "__name__", "?") != name or not t.launchers:
            continue
        old = t.launchers[0].config
        new = copy.deepcopy(old)
        r = new.kwargs.get("R0_BLOCK")
        if not r:
            continue
        new.kwargs["R0_BLOCK"] = r // 2 if r > 16 else r * 2
        if getattr(t.fn, "_hash_lock", 0) is None:  # a JITFunction unpickled from a cache (warm, flag on) has no lock
            import threading
            t.fn._hash_lock = threading.RLock()
        try:
            from torch._dynamo.device_interface import DeviceGuard
            with DeviceGuard(t.get_device_interface(), t.triton_meta["device"]):
                res = t._precompile_config(new)
                launcher, exc = t._make_launcher(res)
        except Exception as e:
            import traceback
            done.append({"error": repr(e)[:300], "where": traceback.format_exc()[-1500:]})
            continue
        if launcher is None:
            done.append({"error": repr(exc)[:300]})
            continue
        t.compile_results, t.launchers = [res], [launcher]
        if hasattr(t, "_cached_launcher"):  # torch >= 2.13: run() calls a cached launcher (fast path)
            t._cached_launcher = None
        _VER[0] += 1
        done.append([_cfg_of(old), _cfg_of(new)])
    return done


def adopt(remote_use):
    """Repair (RP experiment): take over another process's launch configurations. For every usage record of
    `remote_use` ({key: {"subgraph", "kernel", "chosen", "srcn"}}) whose key, kernel, and source digest match one
    of ours and whose configuration differs, rebuild our kernel object's launcher with the remote configuration
    and update our record in place, so that our identity then describes what we run. Returns
    {"adopted": n, "unmatched": keys whose kernel or source differs or that exist on one side only}."""
    import copy
    by_key = {v[0]: (tid, s) for (tid, s), v in _SUBKEYS.items()}
    adopted, unmatched = [], sorted(set(_USE) ^ set(remote_use))
    for k, rec in remote_use.items():
        mine = _USE.get(k)
        if mine is None or mine.get("chosen") == rec.get("chosen"):
            continue
        if mine.get("kernel") != rec.get("kernel") or mine.get("srcn") != rec.get("srcn") or k not in by_key:
            unmatched.append(k)
            continue
        tid, s = by_key[k]
        t = _SUBOBJ.get((tid, s))
        want = rec["chosen"]
        try:
            new = copy.deepcopy(t.launchers[0].config)
            new.kwargs.clear()
            new.kwargs.update(want["kwargs"])
            new.num_warps, new.num_stages = want["num_warps"], want["num_stages"]
            from torch._dynamo.device_interface import DeviceGuard
            with DeviceGuard(t.get_device_interface(), t.triton_meta["device"]):
                res = t._precompile_config(new)
                launcher, exc = t._make_launcher(res)
            if launcher is None:
                raise RuntimeError(repr(exc))
        except Exception as e:
            unmatched.append(k)
            continue
        t.compile_results, t.launchers = [res], [launcher]
        if hasattr(t, "_cached_launcher"):
            t._cached_launcher = None
        _SUBKEYS[(tid, s)] = (k, id(launcher), _cfg(launcher))
        _USE[k] = dict(mine, chosen=_cfg(launcher))
        _VER[0] += 1
        adopted.append(k)
    if _EAGER:
        flush()
    return {"adopted": len(adopted), "unmatched": unmatched}


def _cfg_of(c):
    return {"kwargs": dict(c.kwargs), "num_warps": c.num_warps, "num_stages": c.num_stages}


_UINST = {}
_USE = {}   # per-(subgraph, object) usage records (identity v6)
_STACK = {}  # hardware and software stack of this process (identity v6)


def stack():
    """GPU model, compute capability, driver, and CUDA/PyTorch/Triton/vLLM versions. Read lazily, once CUDA is
    already initialized by the engine, so that tracing never initializes CUDA itself."""
    if _STACK:
        return _STACK
    try:
        import torch
        if not torch.cuda.is_initialized():
            return {}
        dev = torch.cuda.current_device()
        p = torch.cuda.get_device_properties(dev)
        # SM count too: cuBLAS promises bitwise reproducibility only for the same architecture and SM count,
        # and MIG or MPS partitions change it while the GPU name stays the same
        out = {"gpu": p.name, "cc": f"{p.major}.{p.minor}", "sms": p.multi_processor_count, "torch": torch.__version__,
               "cuda": torch.version.cuda}
        out["driver"] = None
        try:
            import ctypes
            v = ctypes.c_int()
            if ctypes.CDLL("libcuda.so.1").cuDriverGetVersion(ctypes.byref(v)) == 0:
                out["driver"] = v.value
        except Exception:
            pass
        try:
            import triton
            out["triton"] = triton.__version__
        except Exception:
            out["triton"] = None
        try:
            import vllm
            out["vllm"] = vllm.__version__
        except Exception:
            out["vllm"] = None
        _STACK.update(out)
    except Exception:
        return {}
    return _STACK


def snapshot():
    """Final configuration of every autotuner object, independent of launch order or sharing.
    Key: kernel name + source digest + occurrence index among identical (name, source) objects."""
    out, seen = {}, {}
    for t in _ALL:
        try:
            name = getattr(getattr(t, "fn", None), "__name__", "?")
            src = getattr(t, "_lockstep_src", None)
            if src is None:
                src = t._lockstep_src = _src(t)
            ls = getattr(t, "launchers", None) or []
            i = seen.get((name, src), 0)
            seen[(name, src)] = i + 1
            out[f"{name}@{src}#{i}"] = {"n_launchers": len(ls), "chosen": _cfg(ls[0]) if len(ls) == 1 else None,
                                        "n_configs": len(getattr(t, "configs", []) or [])}
        except Exception:
            continue
    return out


def _structure(path):
    """Lowering decisions visible in a compiled module's source: which ops became extern library calls
    (e.g. cuBLAS GEMMs, which bypass dispatcher-level overrides such as batch-invariant matmuls) and
    which generated kernels exist."""
    import re
    try:
        _test_fail("structure")
        src = open(path).read()
    except Exception as e:
        _fail("structure", e)
        return None
    ext = {}
    for m in re.finditer(r"extern_kernels\.([A-Za-z_]+)\(", src):
        ext[m.group(1)] = ext.get(m.group(1), 0) + 1
    kernels = re.findall(r"async_compile\.triton\('([^']+)'", src)
    return {"extern": ext, "n_triton": len(kernels), "user_kernels": sorted({k for k in kernels if not k.startswith("triton_")})}


def flush():
    p = _path()
    if p and _STRUCT:
        q = p.replace("decisions_", "structure_")
        json.dump(_STRUCT, open(q + ".tmp", "w"), indent=0)
        os.replace(q + ".tmp", q)
    if p and _DEC:
        tmp = p + ".tmp"
        json.dump(_DEC, open(tmp, "w"), indent=1, default=str)
        os.replace(tmp, p)
    if p and stack():
        q = p.replace("decisions_", "stack_")
        json.dump(_STACK, open(q + ".tmp", "w"), indent=0, default=str)
        os.replace(q + ".tmp", q)
    if p:
        q = p.replace("decisions_", "health_")
        json.dump(_HEALTH, open(q + ".tmp", "w"))
        os.replace(q + ".tmp", q)
    if p and _USE:
        q = p.replace("decisions_", "usage_")
        json.dump(_USE, open(q + ".tmp", "w"), indent=0, default=str)
        os.replace(q + ".tmp", q)
    if p and _HIST:
        q = p.replace("decisions_", "history_")
        json.dump(_HIST, open(q + ".tmp", "w"), indent=0, default=str)
        os.replace(q + ".tmp", q)
    if p and _ALL:
        q = p.replace("decisions_", "snapshot_")
        json.dump(snapshot(), open(q + ".tmp", "w"), indent=0, default=str)
        os.replace(q + ".tmp", q)


def install():
    from torch._inductor.runtime import triton_heuristics as th

    CA = th.CachingAutotuner
    if getattr(CA, "_lockstep_decisions", False):
        return
    CA._lockstep_decisions = True
    orig_autotune = CA.autotune_to_one_config
    orig_run = CA.run
    orig_init = CA.__init__
    try:  # run-time engine choices outside compiled code: the attention backend actually selected
        from vllm.platforms import cuda as _cuda
        _orig_sel = _cuda.CudaPlatform.get_attn_backend_cls.__func__

        def get_attn_backend_cls(cls, *a, **k):
            r = _orig_sel(cls, *a, **k)
            _DEC["__runtime__/attention_backend#" + str(r)] = {
                "kernel": "attention_backend", "subgraph": None, "candidates": 1, "benchmarked": False,
                "chosen": {"backend": str(r)}}
            return r
        _cuda.CudaPlatform.get_attn_backend_cls = classmethod(get_attn_backend_cls)
    except Exception as e:
        print("[lockstep] attention-backend tracing unavailable:", e, flush=True)
        _fail("install-attention-backend", e)
    try:  # lowering structure of every compiled module loaded in this process (cache hits included)
        from torch._inductor.codecache import PyCodeCache
        orig_load = PyCodeCache.load_by_key_path.__func__

        def load_by_key_path(cls, key, path, *a, **k):
            mod = orig_load(cls, key, path, *a, **k)
            name = os.path.basename(path)
            if name not in _STRUCT:
                st = _structure(path)
                if st is not None:
                    _STRUCT[name] = st
            return mod
        PyCodeCache.load_by_key_path = classmethod(load_by_key_path)
    except Exception as e:
        print("[lockstep] structure tracing unavailable:", e, flush=True)
        _fail("install-structure", e)

    def __init__(self, *a, **k):
        orig_init(self, *a, **k)
        _ALL.append(self)  # strong refs: compiled kernels live for the whole process anyway
    CA.__init__ = __init__

    def autotune_to_one_config(self, *args, **kwargs):
        n_cand = len(getattr(self, "launchers", []) or [])
        timings = {}
        forced = {}
        orig_bench = self.benchmark_all_configs
        want = _force_map().get(_key(self)) if _force_map() else None
        if want is not None:
            forced["requested"] = want
            forced["applied"] = any(str(_cfg(l)) == want for l in (getattr(self, "launchers", None) or []))
        orig_one = self.bench
        had_own = "bench" in self.__dict__

        def bench_one(launcher, *a, **k):
            # Force inside the timing loop: Inductor closes each losing static launcher as soon as a
            # faster one is timed, so the requested candidate must win while the loop runs.
            t = orig_one(launcher, *a, **k)
            timings[str(_cfg(launcher))] = float(t)
            if forced.get("applied"):
                return 0.0 if str(_cfg(launcher)) == want else float(t) + 1.0
            return t

        def bench(*a, **k):
            t = orig_bench(*a, **k)
            if not forced.get("applied"):
                timings.update({str(_cfg(l)): float(v) for l, v in t.items()})
            return t
        self.benchmark_all_configs = bench
        self.bench = bench_one
        try:
            out = orig_autotune(self, *args, **kwargs)
        finally:
            self.benchmark_all_configs = orig_bench
            if had_own:
                self.bench = orig_one
            else:
                del self.bench
        k = _key(self)
        _HIST.append({"key": k, "tuner": id(self), "subgraph_now": _CTX["sub"], "candidates": n_cand,
                      "chosen": _cfg(self.launchers[0]) if self.launchers else None, "timings_ms": timings})
        _DEC[k] = {"kernel": k.split("/")[-1], "subgraph": _CTX["sub"], "candidates": n_cand,
                   "benchmarked": True, "chosen": _cfg(self.launchers[0]) if self.launchers else None,
                   "timings_ms": timings, "src": _src(self),
                   "bin": _bin(self, self.launchers[0]) if self.launchers else None}
        if forced:
            _DEC[k]["forced"] = forced
        if _EAGER:
            flush()
        return out

    dbg_runs = {} if os.environ.get("LOCKSTEP_DEBUG_RUNS") else None
    track_use = os.environ.get("LOCKSTEP_NO_USAGE") is None

    def run(self, *args, **kwargs):
        out = orig_run(self, *args, **kwargs)
        if track_use:
            try:
                _test_fail("usage")
                _usage(self)
            except Exception as e:  # never let tracing break the engine; but the identity is now partial
                _fail("usage", e)
                if _EAGER:
                    flush()
        if dbg_runs is not None:  # diagnostics: the first launches of every kernel object and their context
            c = dbg_runs.get(id(self), 0)
            if c < 3:
                dbg_runs[id(self)] = c + 1
                _HIST.append({"event": "run", "n": c, "tuner": id(self), "key": _KEYS.get(id(self)),
                              "name": getattr(getattr(self, "fn", None), "__name__", "?"), "sub": _CTX["sub"],
                              "n_launchers": len(getattr(self, "launchers", None) or [])})
        k0 = _KEYS.get(id(self))
        if k0 is not None and k0.startswith("subNone/") and _CTX["sub"] is not None:
            # First seen outside any piecewise subgraph (a cold process tunes some kernels while
            # compiling). Keys are provisional until the kernel runs inside a subgraph; otherwise a
            # cold and a warm process would label the same decisions differently (H800 g1).
            rec = _DEC.pop(k0, None)
            del _KEYS[id(self)]
            k1 = _key(self)
            if rec is not None:
                rec["subgraph"] = _CTX["sub"]
                rec["kernel"] = k1.split("/")[-1]
                _DEC[k1] = rec
            _SEEN.discard(id(self))  # fall through: record under the new key if nothing moved
        if id(self) in _SEEN:  # per object: copies of a tuner must not inherit "seen" (see _key)
            return out
        _SEEN.add(id(self))
        _KEPT.append(self)
        k = _key(self)
        if k not in _DEC and getattr(self, "launchers", None):
            _DEC[k] = {"kernel": k.split("/")[-1], "subgraph": _CTX["sub"], "candidates": len(self.launchers),
                       "benchmarked": False, "chosen": _cfg(self.launchers[0]), "src": _src(self),
                       "bin": _bin(self, self.launchers[0])}
            if _EAGER:  # engines whose workers are killed at shutdown (atexit never runs)
                flush()
        return out

    CA.autotune_to_one_config = autotune_to_one_config
    CA.run = run
    atexit.register(flush)


def diff(dec_a, dec_b):
    """Compare two decision traces; return kernels whose chosen configuration differs."""
    out = []
    for k in sorted(set(dec_a) | set(dec_b)):
        a, b = dec_a.get(k), dec_b.get(k)
        if a is None or b is None:
            out.append({"kernel": k, "only_in": "a" if b is None else "b"})
        elif a.get("chosen") != b.get("chosen"):
            out.append({"kernel": k, "subgraph": a.get("subgraph"), "a": a.get("chosen"), "b": b.get("chosen"),
                        "benchmarked": (a.get("benchmarked"), b.get("benchmarked")),
                        "candidates": (a.get("candidates"), b.get("candidates"))})
    return out


def bin_diff(dec_a, dec_b, fields=("n_regs", "n_spills", "shared", "n_compiled", "rblock_scaling")):
    """Kernels with the same chosen config whose binary-level facts differ (E25). The autotune-cache
    state is reported but not compared by default, since cold and warm processes differ in it."""
    out = []
    for k in sorted(set(dec_a) & set(dec_b)):
        x, y = dec_a[k].get("bin") or {}, dec_b[k].get("bin") or {}
        d = {f: (x.get(f), y.get(f)) for f in fields if x.get(f) != y.get(f)}
        if d:
            out.append({"kernel": k, "subgraph": dec_a[k].get("subgraph"), "diff": d,
                        "cache_state": (x.get("cache_state"), y.get("cache_state"))})
    return out


def load_rank(capture, rank):
    """One tensor-parallel rank's trace from a capture directory: (usage, decisions, stack, structure). Several
    processes can write files for rank 0 (the engine's own process, which runs no model under TP, and the rank-0
    worker); the worker is the process with the most usage records."""
    import glob
    best, pid = -1, None
    for f in glob.glob(os.path.join(capture, f"usage_rank{rank}_pid*.json")):
        try:
            n = len(json.load(open(f)))
        except Exception:
            continue
        if n > best:
            best, pid = n, f.rsplit("_pid", 1)[1][:-5]
    if pid is None:
        return {}, {}, {}, {}
    out = []
    for kind in ("usage", "decisions", "stack", "structure"):
        try:
            out.append(json.load(open(os.path.join(capture, f"{kind}_rank{rank}_pid{pid}.json"))))
        except Exception:
            out.append({})
    return tuple(out)


def load_rank_health(capture, rank):
    """The tracing failures that rank's worker recorded (its health file), or None if the file is missing."""
    import glob
    fs = sorted(glob.glob(os.path.join(capture, f"health_rank{rank}_pid*.json")))
    out = {}
    for f in fs:
        try:
            out.update(json.load(open(f)))
        except Exception as e:
            out["health-file"] = {"n": 1, "first": repr(e)[:200]}
    return out if fs else None


def load(run_dir):
    import glob
    dec = {}
    for f in glob.glob(os.path.join(run_dir, "capture", "decisions_rank0_pid*.json")):
        dec.update(json.load(open(f)))
    return dec


def load_stack(run_dir):
    import glob
    for f in glob.glob(os.path.join(run_dir, "capture", "stack_rank0_pid*.json")):
        return json.load(open(f))
    return {}


def load_usage(run_dir):
    import glob
    use = {}
    for f in glob.glob(os.path.join(run_dir, "capture", "usage_rank0_pid*.json")):
        use.update(json.load(open(f)))
    return use


def load_snapshot(run_dir):
    import glob
    snap = {}
    for f in glob.glob(os.path.join(run_dir, "capture", "snapshot_rank0_pid*.json")):
        snap.update(json.load(open(f)))
    return snap


def snapshot_diff(a, b):
    """Kernels whose final configuration differs, plus kernels present in only one process."""
    out = []
    for k in sorted(set(a) | set(b)):
        x, y = a.get(k), b.get(k)
        if x is None or y is None:
            out.append({"kernel": k, "only_in": "a" if y is None else "b"})
        elif x.get("chosen") != y.get("chosen"):
            out.append({"kernel": k, "a": x.get("chosen"), "b": y.get("chosen")})
    return out


def load_structure(run_dir):
    import glob
    st = {}
    for f in glob.glob(os.path.join(run_dir, "capture", "structure_rank0_pid*.json")):
        st.update(json.load(open(f)))
    return st


def lint(run_dir):
    """Static contract check: in batch-invariant mode, every GEMM must go through the batch-invariant
    implementation. Extern library GEMMs inside compiled code bypass it (F-004)."""
    cfg = {}
    try:
        cfg = json.load(open(os.path.join(run_dir, "cfg.json")))
    except Exception:
        pass
    bi = str(cfg.get("env", {}).get("VLLM_BATCH_INVARIANT", "")) == "1"
    st = load_structure(run_dir)
    issues = []
    for mod, v in st.items():
        g = {k: n for k, n in v.get("extern", {}).items() if k in _GEMM_EXTERN}
        if g and bi:
            issues.append({"module": mod, "extern_gemms": g})
    return {"batch_invariant_mode": bi, "modules": len(st), "issues": issues}


def identity(dec, with_src=False):
    """Numerical identity of a process: digest of every traced kernel instance and the configuration
    it actually ran with. Two processes of the same deployment configuration whose identities match
    executed the same generated kernels with the same launch configurations. Used by the guard: KV
    or prefix state produced by one instance is reused by another only if the identities match.
    Covers Inductor-generated kernels only (not extern library calls or per-batch graph routing)."""
    import hashlib
    items = sorted((k, json.dumps(v.get("chosen"), sort_keys=True, default=str), v.get("src") if with_src else None)
                   for k, v in dec.items())
    return hashlib.sha256(json.dumps(items).encode()).hexdigest()[:16]


def risk(dec, tau=0.05):
    """Single-run reproducibility risk: run-time benchmarked decisions whose best and second-best
    measured timings are within a relative margin tau. A near-tie can flip in another process (timing
    noise), changing the reduction order. An exact tie is risky too: it reflects timer resolution in
    this process, not equal speed, and another process may measure a strict order (E10 observed such
    a flip). Returns (risky records sorted by margin, number of benchmarked decisions)."""
    out, nb = [], 0
    for k, v in dec.items():
        t = sorted((v.get("timings_ms") or {}).values())
        if not v.get("benchmarked") or len(t) < 2:
            continue
        nb += 1
        m = (t[1] - t[0]) / t[0] if t[0] > 0 else 0.0
        if m < tau:
            out.append({"kernel": k, "margin": m, "chosen": v.get("chosen")})
    return sorted(out, key=lambda r: r["margin"]), nb


if __name__ == "__main__":
    import sys
    if sys.argv[1] == "--risk":
        r, nb = risk(load(sys.argv[2]), float(sys.argv[3]) if len(sys.argv) > 3 else 0.05)
        print(f"benchmarked decisions: {nb}; near-ties (could flip across processes): {len(r)}")
        for x in r[:20]:
            print(f"  {x['kernel']}  margin={x['margin']:.4f}")
        sys.exit(1 if r else 0)
    if sys.argv[1] == "--lint":
        for r in sys.argv[2:]:
            res = lint(r)
            print(f"{r}: BI={res['batch_invariant_mode']} modules={res['modules']} issues={len(res['issues'])}")
            for x in res["issues"][:10]:
                print("   ", json.dumps(x))
        sys.exit(0)
    if sys.argv[1] == "--snapdiff":
        a, b = load_snapshot(sys.argv[2]), load_snapshot(sys.argv[3])
        d = snapshot_diff(a, b)
        print(f"autotuner objects: a={len(a)} b={len(b)}; differing: {len(d)}")
        for x in d[:40]:
            print(" ", json.dumps(x, default=str))
        sys.exit(0)
    if sys.argv[1] == "--identity":
        for r in sys.argv[2:]:
            print(identity(load(r)), r)
        sys.exit(0)
    A, B = load(sys.argv[1]), load(sys.argv[2])
    d = diff(A, B)
    print(f"kernels traced: a={len(A)} b={len(B)}; differing decisions: {len(d)}")
    for x in d[:40]:
        print(" ", json.dumps(x, default=str))
