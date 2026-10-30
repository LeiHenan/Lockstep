"""In-connector guard: fold the instance's numerical identity into vLLM NIXL's compatibility hash.

Enabled by the vLLM plugin when LOCKSTEP_LIVE_GUARD=1 (requires LOCKSTEP_DIR, i.e. decision tracing).
vLLM's NIXL handshake already compares a compatibility hash of the engine version, model, KV layout,
and attention backend (computed once, when the KV cache is registered). That hash does not cover the
compiler decisions a process makes while it warms up, so two instances with equal configurations can
pass it and still compute different bits. This module recomputes the identity of `lockstep.guard`
(configuration under the default-deny rule + run-time decision trace) at handshake time, after
warm-up, and combines it with vLLM's hash:

  * prefill side (the handshake listener, which serves the payload): the served hash becomes
    H(vllm_hash | identity_now);
  * decode side (`_nixl_handshake`, which checks the payload): the local hash used in the comparison
    becomes H(vllm_hash | identity_now) for the duration of the handshake.

On a mismatch vLLM's own check raises, the handshake fails, and with
kv_load_failure_policy="recompute" the decode instance recomputes the prefill locally. No vLLM source
is modified. Every serve/check is logged to $LOCKSTEP_DIR/live_guard.jsonl.

With one GPU per instance (uniprocess executor) the scheduler and the model run in one process, and the identity
comes from the in-memory trace. Under tensor parallelism each rank's worker runs its share of the model in its own
process and handshakes with the producer's rank of the same index, while the producer's handshake listener runs in
the engine's process, which runs no model. There, every rank's identity is computed by `identity_rank` from the
files that the rank's worker writes (its eagerly flushed trace, and its environment and resolved configuration,
written when the connector worker is created), so that both sides of a rank's handshake compute it the same way.
"""
import hashlib
import json
import os
import time


def _log(rec):
    d = os.environ.get("LOCKSTEP_DIR")
    if not d:
        return
    os.makedirs(d, exist_ok=True)
    rec = dict(rec, pid=os.getpid(), t=time.time())
    with open(os.path.join(d, "live_guard.jsonl"), "a") as f:
        f.write(json.dumps(rec, default=str) + "\n")


def identity_now():
    """Same digest as lockstep.guard.identity (version 3), computed from the in-memory trace.
    Returns (identity hex, number of trace records, milliseconds spent)."""
    from lockstep import decisions, guard
    t0 = time.perf_counter()
    d = os.environ["LOCKSTEP_DIR"].rstrip("/")
    cfg = json.load(open(os.path.join(os.path.dirname(d), "cfg.json")))
    dec = dict(decisions._DEC)
    if os.environ.get("LOCKSTEP_GUARD_VERSION") == "6":  # canonical per-subgraph usage identity
        use = dict(decisions._USE)
        return guard.identity_v6(cfg, use, dec, decisions.stack()), len(use), 1000 * (time.perf_counter() - t0)
    items = sorted(("run:" + k, json.dumps(v.get("chosen"), sort_keys=True)) for k, v in dec.items())
    traced = any(not k.startswith("__runtime__") for k in dec)
    blob = json.dumps({"cfg": guard.config_part(cfg, traced), "kernels": items}, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16], len(dec), 1000 * (time.perf_counter() - t0)


# --- version 7 (review round 7): resolved configuration, coverage, on-demand canaries, invalidation --------
_VLLM_CFG = []      # the engine's VllmConfig, captured when the NIXL worker is created
_MODEL = {}         # model path -> digest of its files (computed once)
_SERVED = {}        # producer side: usage records and registry entries at the last serve
_INJECTED = []      # test hook state


def v7():
    return os.environ.get("LOCKSTEP_GUARD_VERSION", "7") == "7"  # default 7; set 6 or 3 for the earlier versions


def resolved_config():
    """Engine options as vLLM resolved them (not as they were passed), or None if not yet available."""
    if not _VLLM_CFG:
        return None
    c = _VLLM_CFG[0]
    m, cc, pc = c.model_config, c.cache_config, c.parallel_config
    out = {"model": m.model, "revision": getattr(m, "revision", None), "dtype": str(m.dtype),
           "quantization": getattr(m, "quantization", None), "max_model_len": m.max_model_len,
           "block_size": cc.block_size, "cache_dtype": str(cc.cache_dtype), "tp": pc.tensor_parallel_size,
           "pp": pc.pipeline_parallel_size}
    try:
        out["compilation"] = _compilation_hash(c.compilation_config)
    except Exception:
        out["compilation"] = None
    return out


# Fields of vLLM's compilation configuration that vLLM derives from the token budget and the sequence limit. They
# act only through what the trace records (the kernels compiled per compile range, and which batch sizes replay
# CUDA graphs of the same kernels), so the identity leaves them out unless LOCKSTEP_V7_BUDGET_DIGEST=1 (the
# digest of the first version-7 runs: L1, V7S, LD, ENV), which separated every pair with different budgets.
BUDGET_DERIVED = {"compile_ranges_endpoints": [], "cudagraph_capture_sizes": [], "max_cudagraph_capture_size": 0}


def _compilation_hash(cc):
    if os.environ.get("LOCKSTEP_V7_BUDGET_DIGEST") == "1":
        return cc.compute_hash()
    import copy
    c2 = copy.copy(cc)
    for k, v in BUDGET_DERIVED.items():
        if hasattr(c2, k):
            setattr(c2, k, copy.copy(v))
    return c2.compute_hash()


def model_files(path):
    """Digest of a local model's metadata files and the names and sizes of its weight files."""
    if path in _MODEL:
        return _MODEL[path]
    h, out = hashlib.sha256(), None
    try:
        names = sorted(os.listdir(path))
        for n in names:
            p = os.path.join(path, n)
            if n.endswith(".json"):
                h.update(n.encode() + open(p, "rb").read())
            elif n.endswith((".safetensors", ".bin", ".pt")):
                h.update(f"{n}:{os.path.getsize(p)}".encode())
        out = h.hexdigest()[:16]
    except Exception:
        out = None
    _MODEL[path] = out
    return out


def _tp_size():
    if _VLLM_CFG:
        return _VLLM_CFG[0].parallel_config.tensor_parallel_size
    try:
        d = os.environ["LOCKSTEP_DIR"].rstrip("/")
        return int(json.load(open(os.path.join(os.path.dirname(d), "cfg.json")))["engine_kwargs"].get(
            "tensor_parallel_size", 1))
    except Exception:
        return 1


def _rank():
    try:
        import torch.distributed as dist
        return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    except Exception:
        return 0


def write_rank_meta():
    """Worker side, tensor parallelism: this rank's environment and resolved configuration, for identity_rank."""
    d = os.environ.get("LOCKSTEP_DIR")
    if not d:
        return
    p = os.path.join(d, f"rankmeta_r{_rank()}.json")
    json.dump({"env": dict(os.environ), "resolved": resolved_config(), "pid": os.getpid()}, open(p + ".tmp", "w"),
              default=str)
    os.replace(p + ".tmp", p)


def identity_rank(rank):
    """(identity, coverage reasons, number of usage records, milliseconds) of one tensor-parallel rank, from the
    files its worker wrote. Missing files give an empty trace, hence coverage "no-trace"."""
    from lockstep import decisions, guard
    t0 = time.perf_counter()
    d = os.environ["LOCKSTEP_DIR"].rstrip("/")
    cfg = json.load(open(os.path.join(os.path.dirname(d), "cfg.json")))
    use, dec, st_, structure = decisions.load_rank(d, rank)
    try:
        meta = json.load(open(os.path.join(d, f"rankmeta_r{rank}.json")))
    except Exception:
        meta = {"env": None, "resolved": None}
    res = meta.get("resolved")
    mf = model_files(res["model"]) if res and os.path.isdir(str(res["model"])) else None
    health = decisions.load_rank_health(d, rank)
    ident, why = guard.identity_v7(cfg, use, dec, st_, meta.get("env") or {}, res, mf, structure, health or {})
    if meta.get("env") is None:
        why = list(why) + ["no-rank-meta"]
    if health is None and use:
        why = list(why) + ["no-health"]  # the rank's worker wrote no health file: its tracing state is unknown
    p = os.path.join(d, f"v7_parts_r{rank}_{ident}.json")
    if not os.path.exists(p):
        json.dump({"rank": rank, "cfg": guard.config_part(cfg, bool(use)), "env": guard.env_part(meta.get("env") or {}),
                   "resolved": res, "model": mf, "stack": st_, "coverage": why,
                   "lowering": guard.lowering_set(structure), "kernels6": guard.usage_kernels(use, dec)},
                  open(p, "w"), indent=1, sort_keys=True, default=str)
    return ident, why, len(use), 1000 * (time.perf_counter() - t0)


def identity_v7_now():
    """(identity, coverage reasons, number of usage records, milliseconds)."""
    from lockstep import decisions, guard
    t0 = time.perf_counter()
    d = os.environ["LOCKSTEP_DIR"].rstrip("/")
    cfg = json.load(open(os.path.join(os.path.dirname(d), "cfg.json")))
    use, dec, st = dict(decisions._USE), dict(decisions._DEC), dict(decisions._STRUCT)
    res = resolved_config()
    mf = model_files(res["model"]) if res and os.path.isdir(str(res["model"])) else None
    ident, why = guard.identity_v7(cfg, use, dec, decisions.stack(), dict(os.environ), res, mf, st,
                                   dict(decisions._HEALTH))
    p = os.path.join(d, f"v7_parts_{ident}.json")
    if not os.path.exists(p):  # what this identity covers, once per distinct value (for audits and debugging)
        json.dump({"cfg": guard.config_part(cfg, bool(use)), "env": guard.env_part(dict(os.environ)),
                   "resolved": res, "model": mf, "stack": decisions.stack(), "coverage": why,
                   "lowering": guard.lowering_set(st), "kernels6": guard.usage_kernels(use, dec)},
                  open(p, "w"), indent=1, sort_keys=True, default=str)
    return ident, why, len(use), 1000 * (time.perf_counter() - t0)


def decide(policy, equal, covered, own_canary, remote_canary):
    """The guard's decision for one handshake. "strict": admit only on equal identities with full coverage.
    "canary": also admit when the identities differ or coverage is partial, if both canaries exist and
    agree bit for bit (a tested relaxation, not a guarantee). Returns admit | admit_canary | block."""
    if equal and covered and policy != "canary_only":
        return "admit"
    if policy not in ("canary", "canary_only") or own_canary is None or remote_canary is None:
        return "block"
    return "admit_canary" if own_canary == remote_canary else "block"


def needs_invalidation(n_at_handshake, n_now, served_then, served_now):
    """A transfer path is stale if this instance recorded new kernel usage after its handshake, or the
    producer republished a different identity since."""
    return (n_at_handshake is not None and n_now != n_at_handshake) or \
        (served_then is not None and served_now is not None and served_now != served_then)


def canary_get(capture, wait_s=None):
    """An instance's canary; with LOCKSTEP_CANARY_ON_DEMAND=1, ask the instance's driver to compute it
    (file canary_request in its capture directory) and wait for it."""
    p = os.path.join(capture, "canary.json")
    try:
        return json.load(open(p))
    except Exception:
        pass
    if os.environ.get("LOCKSTEP_CANARY_ON_DEMAND") != "1":
        return None
    open(os.path.join(capture, "canary_request"), "a").close()
    t_end = time.time() + float(wait_s or os.environ.get("LOCKSTEP_CANARY_WAIT_S", "60"))
    while time.time() < t_end:
        try:
            return json.load(open(p))
        except Exception:
            time.sleep(0.2)
    return None


def _inject_for_test():
    """Test hook (LOCKSTEP_TEST_INJECT_AFTER_S): add one synthetic usage record this many seconds after the
    first handshake, as a decision that a process makes late would."""
    t = os.environ.get("LOCKSTEP_TEST_INJECT_AFTER_S")
    if not t or _INJECTED or "t_first" not in _SERVED:
        return
    if time.time() - _SERVED["t_first"] >= float(t):
        from lockstep import decisions
        decisions._USE["test/injected#0"] = {"subgraph": 0, "kernel": "injected", "chosen": {"test": 1},
                                            "srcn": "injected"}
        _INJECTED.append(time.time())
        _log({"event": "test_injected"})


def _retune_for_test():
    """Test hook (LOCKSTEP_TEST_RETUNE_AFTER_S, LD experiment): this many seconds after the first handshake,
    change the launch configuration of kernel LOCKSTEP_TEST_RETUNE_KERNEL in subgraph LOCKSTEP_TEST_RETUNE_SUB,
    a decision made after the handshake that really changes the numerics."""
    t = os.environ.get("LOCKSTEP_TEST_RETUNE_AFTER_S")
    if not t or _INJECTED or "t_first" not in _SERVED or time.time() - _SERVED["t_first"] < float(t):
        return
    from lockstep import decisions
    sub = os.environ.get("LOCKSTEP_TEST_RETUNE_SUB", "22")
    done = decisions.retune_for_test(None if sub == "all" else int(sub),
                                     os.environ.get("LOCKSTEP_TEST_RETUNE_KERNEL", "triton_red_fused_3"))
    _INJECTED.append(time.time())
    _log({"event": "test_retuned", "changed": done})


def _on_new_usage():
    """A new kernel-usage record (decisions._ON_USAGE). A canary holds only for the identity it was computed
    under, so drop this instance's cached canary (unless LOCKSTEP_CANARY_STALE_OK=1, the LD control, or the
    canary-only baseline, which never re-checks). A producer that has served its identity republishes at
    once, before consumers load KV computed with the new decision."""
    policy = os.environ.get("LOCKSTEP_GUARD_POLICY", "strict")
    d = os.environ.get("LOCKSTEP_DIR")
    if d and policy != "canary_only" and os.environ.get("LOCKSTEP_CANARY_STALE_OK") != "1":
        p = os.path.join(d, "canary.json")
        if os.path.exists(p):
            os.replace(p, f"{p}.stale.{time.time_ns()}")
            _log({"event": "canary_dropped"})
    if "addr" in _SERVED and _SERVE_FN:
        from lockstep import decisions
        old = _SERVED["served"]
        served, why, _, _ = _SERVE_FN[0](*_SERVED["addr"])
        _log({"event": "republished", "old": old, "identity": served, "coverage": why,
              "n_records": len(decisions._USE), "at": "usage"})


_SERVE_FN = []  # install_v7's _serve_identity


def _canary():
    """This instance's canary value (written by the harness after warm-up), or None."""
    d = os.environ.get("LOCKSTEP_DIR")
    try:
        return json.load(open(os.path.join(d, "canary.json")))
    except Exception:
        return None


def _registry(host, port, rank=0):
    """Hybrid mode (LOCKSTEP_HYBRID_DIR): where a prefill instance publishes its identity and canary,
    keyed by its handshake address (and, under tensor parallelism, the rank), so that a decode instance can
    compare canaries after a mismatch."""
    d = os.environ.get("LOCKSTEP_HYBRID_DIR")
    return os.path.join(d, f"{host}_{port}.json" if not rank else f"{host}_{port}_r{rank}.json") if d else None


def combine(vllm_hash, ident):
    return hashlib.sha256(f"{vllm_hash}|lockstep:{ident}".encode()).hexdigest()


def install():
    import msgspec
    import zmq
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl import base_scheduler as bs
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl import base_worker as bw
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import NixlHandshakePayload

    W, S = bw.NixlBaseConnectorWorker, bs.NixlBaseConnectorScheduler
    if getattr(W, "_lockstep_live_guard", False):
        return
    W._lockstep_live_guard = True
    if v7():
        return install_v7(bs, bw, W, S, NixlHandshakePayload, msgspec, zmq)

    # decode side: compare against H(local vllm hash | local identity) during the handshake
    orig_hs = W._nixl_handshake

    def _nixl_handshake(self, *a, **k):
        base = self.__dict__.setdefault("_lockstep_base_hash", self.compat_hash)
        ident, n, ms = identity_now()
        self.compat_hash = combine(base, ident)
        try:
            out = orig_hs(self, *a, **k)
            _log({"event": "check", "result": "admitted", "identity": ident, "n_records": n, "identity_ms": ms})
            return out
        except Exception as e:
            mismatch = "hash mismatch" in str(e)
            reg = _registry(*(list(a[:2]) or [k.get("host"), k.get("port")])) if mismatch else None
            if reg:  # hybrid: identities differ; admit only if both instances' canaries agree bit for bit
                try:
                    remote = json.load(open(reg))
                except Exception:
                    remote = None
                own = _canary()
                if remote and own is not None and remote.get("canary") is not None and remote["canary"] == own:
                    self.compat_hash = combine(base, remote["identity"])
                    out = orig_hs(self, *a, **k)
                    _log({"event": "check", "result": "admitted_by_canary", "identity": ident,
                          "remote_identity": remote["identity"], "n_records": n, "identity_ms": ms})
                    return out
                _log({"event": "check", "result": "blocked", "canary_checked": True,
                      "canary_equal": False if (remote and own is not None) else None,
                      "identity": ident, "n_records": n, "identity_ms": ms, "error": str(e)[:300]})
                raise
            _log({"event": "check", "result": "blocked" if mismatch else "error",
                  "identity": ident, "n_records": n, "identity_ms": ms, "error": str(e)[:300]})
            raise
        finally:
            self.compat_hash = base
    W._nixl_handshake = _nixl_handshake

    # prefill side: serve H(vllm hash | identity at the time of the request)
    def _nixl_handshake_listener(encoded_data, ready_event, stop_event, host, port):
        path = bs.make_zmq_path("tcp", host, port)
        with bs.zmq_ctx(zmq.ROUTER, path) as sock:
            sock.setsockopt(zmq.RCVTIMEO, 1000)
            ready_event.set()
            while True:
                try:
                    identity, _, msg = sock.recv_multipart()
                except zmq.Again:
                    if stop_event.is_set():
                        break
                    continue
                msg, pp, tp = msgspec.msgpack.decode(msg)
                if msg != bs.GET_META_MSG:
                    bs.logger.warning("Connection listener got unexpected message %s", msg)
                raw = encoded_data[(pp, tp)]
                try:
                    p = msgspec.msgpack.decode(raw, type=NixlHandshakePayload)
                    ident, n, ms = identity_now()
                    reg = _registry(host, port)
                    if reg:  # hybrid: publish identity and canary for decode instances that see a mismatch
                        os.makedirs(os.path.dirname(reg), exist_ok=True)
                        json.dump({"identity": ident, "canary": _canary()}, open(reg + ".tmp", "w"))
                        os.replace(reg + ".tmp", reg)
                    raw = msgspec.msgpack.encode(NixlHandshakePayload(
                        compatibility_hash=combine(p.compatibility_hash, ident),
                        agent_metadata_bytes=p.agent_metadata_bytes))
                    _log({"event": "serve", "identity": ident, "n_records": n, "identity_ms": ms})
                except Exception as e:  # never break the side channel; serve vLLM's own payload
                    _log({"event": "serve", "result": "error", "error": repr(e)[:300]})
                ts = msgspec.msgpack.encode(time.perf_counter())
                sock.send_multipart((identity, b"", raw, ts))
    S._nixl_handshake_listener = staticmethod(_nixl_handshake_listener)
    _log({"event": "installed"})


def _own_key():
    """What this process's in-memory identity depends on that can change while it serves."""
    from lockstep import decisions
    return len(decisions._USE), decisions._VER[0]


def _epoch_key():
    """_own_key of the process whose KV a request uses. Under TP the engine process runs no model: each rank's
    eagerly flushed usage files (a flush replaces the file, so its inode changes with every write)."""
    tp = _tp_size()
    if tp == 1:
        return _own_key()
    import glob
    d = os.environ["LOCKSTEP_DIR"]
    out = []
    for r in range(tp):
        st = []
        for f in sorted(glob.glob(os.path.join(d, f"usage_rank{r}_pid*.json"))):
            try:
                s = os.stat(f)
                st.append((f, s.st_ino, s.st_mtime_ns, s.st_size))
            except OSError:
                pass
        out.append(tuple(st))
    return tuple(out)


_TAG = {}  # producer: the served identities for the last epoch key


def served_tag():
    """The identity every rank serves now, as the handshake listener would serve it (list indexed by rank)."""
    e = _epoch_key()
    if _TAG.get("e") != e:
        tp = _tp_size()
        vals = [identity_rank(r) for r in range(tp)] if tp > 1 else [identity_v7_now()]
        _TAG.update(e=e, tag=[i if not w else f"partial:{i}" for i, w, _, _ in vals])
    return _TAG["tag"]


def tag_verdict(tag, expected, own_then, own_now, idx):
    """Consumer, at the moment it would read a request's KV: None if the read may go ahead, else the reason it
    may not. `tag` is what the producer attached when the request finished (its identity then, or "mixed" when
    the request's KV spans a change of its decisions); `expected` is the identity this path was admitted on."""
    if expected is None:
        return "no-admission"
    if tag is None:
        return "untagged"
    if tag == "mixed":
        return "mixed"
    if idx >= len(tag) or tag[idx] != expected:
        return "producer-changed"
    if own_now != own_then:
        return "consumer-changed"
    return None


def install_xfer_tag(S, W):
    """Close the window between a decision made after the handshake and the consumers' next check (the
    lifecycle poll runs at most every 0.5 s). The producer tags every request it hands over with the identity
    its KV was computed under; the consumer compares the tag with the identity it admitted the path on right
    before it issues the read, and with LOCKSTEP_XFER_TAG=enforce recomputes the prefill instead (vLLM's
    recompute-on-load-failure path) and invalidates the path. LOCKSTEP_XFER_TAG=observe only logs.

    A request's KV is consistent only if no decision changed between its first scheduling and its end, and
    every full block it uses (own or from the prefix cache) was cached in the current epoch; blocks are
    registered when vLLM caches them, at scheduling time, before they are computed, so a change during their
    computation makes them stale too."""
    mode = os.environ.get("LOCKSTEP_XFER_TAG")
    from vllm.v1.core import block_pool as bp
    epoch_of, start, broken = {}, {}, []  # broken: a block's epoch could not be registered; tag "mixed" from then on

    orig_cache = bp.BlockPool.cache_full_blocks

    def cache_full_blocks(self, request, blocks, num_cached_blocks, num_full_blocks, *a, **k):
        try:
            e = _epoch_key()
            for b in blocks[num_cached_blocks:num_full_blocks]:
                epoch_of[b.block_id] = e
        except Exception as ex:
            broken.append(ex)
            _log({"event": "xfer_tag_error", "where": "cache", "error": repr(ex)[:300]})
        return orig_cache(self, request, blocks, num_cached_blocks, num_full_blocks, *a, **k)
    bp.BlockPool.cache_full_blocks = cache_full_blocks

    from vllm.distributed.kv_transfer.kv_connector.v1.nixl import base_scheduler as bsm
    mods = [bsm]
    for mod in ("pull_scheduler", "push_scheduler", "pull_worker", "push_worker"):
        try:
            mods.append(__import__(f"vllm.distributed.kv_transfer.kv_connector.v1.nixl.{mod}", fromlist=["x"]))
        except Exception:
            pass
    classes = {v for m in mods for v in vars(m).values() if isinstance(v, type)}

    for cls in [c for c in classes if issubclass(c, S) and "update_state_after_alloc" in vars(c)]:
        def update_state_after_alloc(self, request, blocks, num_external_tokens, _orig=cls.update_state_after_alloc):
            start.setdefault(request.request_id, _epoch_key())
            return _orig(self, request, blocks, num_external_tokens)
        cls.update_state_after_alloc = update_state_after_alloc

    for cls in [c for c in classes if issubclass(c, S) and "request_finished" in vars(c)]:
        def request_finished(self, request, block_ids, _orig=cls.request_finished):
            delay, params = _orig(self, request, block_ids)
            s0 = start.pop(request.request_id, None)
            if params and params.get("do_remote_prefill"):  # producer handing the request over
                try:
                    end = _epoch_key()
                    stale = [b for g in (block_ids or ()) for b in g if epoch_of.get(b, end) != end]
                    ok = s0 == end and not stale and not broken
                    params["lockstep_tag"] = served_tag() if ok else "mixed"
                    if not ok:
                        _log({"event": "xfer_tag_mixed", "req": request.request_id, "start_changed": s0 != end,
                              "stale_blocks": len(stale)})
                except Exception as ex:
                    params["lockstep_tag"] = "mixed"
                    _log({"event": "xfer_tag_error", "where": "finish", "error": repr(ex)[:300]})
            return delay, params
        cls.request_finished = request_finished

    orig_bcm = S.build_connector_meta

    def build_connector_meta(self, scheduler_output):
        tags = {rid: (v[0].kv_transfer_params or {}).get("lockstep_tag")
                for rid, v in getattr(self, "_reqs_need_recv", {}).items()}
        meta = orig_bcm(self, scheduler_output)
        if tags:
            meta.lockstep_tags = tags  # travels with the metadata to every worker
        return meta
    S.build_connector_meta = build_connector_meta

    for cls in [c for c in classes if issubclass(c, W) and "start_load_kv" in vars(c)]:
        def start_load_kv(self, metadata, _orig=cls.start_load_kv):
            self.__dict__.setdefault("_lockstep_tags", {}).update(getattr(metadata, "lockstep_tags", None) or {})
            return _orig(self, metadata)
        cls.start_load_kv = start_load_kv

    for cls in [c for c in classes if issubclass(c, W) and "_read_blocks_for_req" in vars(c)]:
        def _read_blocks_for_req(self, req_id, meta, _orig=cls._read_blocks_for_req):
            eid = meta.remote.engine_id if meta.remote is not None else None
            if eid is None or eid not in self._remote_agents or not any(len(g) > 0 for g in meta.local_block_ids):
                # no path yet (vLLM handshakes, then calls again), or nothing to read (a full local hit)
                return _orig(self, req_id, meta)
            why = "error"
            try:
                addr = next((a for a, e in self._engine_by_address.items() if e == eid), None)
                expected, own_then = self.__dict__.get("_lockstep_admit", {}).get(addr, (None, None))
                tag = self._lockstep_tags.pop(req_id, None)
                why = tag_verdict(tag, expected, own_then, _own_key(), _rank() if _tp_size() > 1 else 0)
            except Exception as ex:
                _log({"event": "xfer_tag_error", "where": "read", "error": repr(ex)[:300]})
                tag = expected = None
            _log({"event": "xfer_check", "req": req_id, "verdict": why or "ok", "mode": mode,
                  "tag": tag, "expected": expected})
            if why and mode == "enforce":
                self.__dict__.setdefault("_lockstep_stale", set()).add(addr)  # _lifecycle invalidates the path
                self._handle_failed_transfer(req_id, None, self._recv_failures, record_failed_transfer=False)
                return
            return _orig(self, req_id, meta)
        cls._read_blocks_for_req = _read_blocks_for_req
    _log({"event": "xfer_tag_installed", "mode": mode, "file": __file__})


def install_v7(bs, bw, W, S, NixlHandshakePayload, msgspec, zmq):
    """Version 7: identity_v7 with coverage, policy LOCKSTEP_GUARD_POLICY (strict | canary, default strict),
    canaries on demand, and invalidation of transfer paths when either side records new kernel usage after
    the handshake. The decode side re-handshakes through vLLM's own path for released engines."""
    policy = os.environ.get("LOCKSTEP_GUARD_POLICY", "strict")
    own_cap = os.environ.get("LOCKSTEP_DIR")

    orig_init = W.__init__

    def __init__(self, vllm_config, *a, **k):
        _VLLM_CFG[:] = [vllm_config]
        orig_init(self, vllm_config, *a, **k)
        if vllm_config.parallel_config.tensor_parallel_size > 1:
            try:
                write_rank_meta()
            except Exception as e:
                _log({"event": "rank_meta_error", "error": repr(e)[:300]})
    W.__init__ = __init__

    def _ident_here():  # this process's identity: in-memory trace, or under TP this rank's files
        if _tp_size() > 1:
            return identity_rank(_rank())
        return identity_v7_now()

    def _remote(host, port):
        reg = _registry(host, port, _rank() if _tp_size() > 1 else 0)
        url = os.environ.get("LOCKSTEP_HYBRID_URL")  # producer on another host: its registry served over HTTP
        try:
            if reg and url:
                import urllib.request
                with urllib.request.urlopen(f"{url.rstrip('/')}/{os.path.basename(reg)}", timeout=3) as r:
                    return json.loads(r.read())
            return json.load(open(reg)) if reg else None
        except Exception:
            return None

    orig_hs = W._nixl_handshake

    def _nixl_handshake(self, *a, **k):
        if policy == "off":  # RW2 control: vLLM's own handshake, no identity; the test hooks still run
            _SERVED.setdefault("t_first", time.time())
            return orig_hs(self, *a, **k)
        base = self.__dict__.setdefault("_lockstep_base_hash", self.compat_hash)
        host, port = (list(a[:2]) or [k.get("host"), k.get("port")])
        key0 = _own_key()  # before the identity: a record made in between makes the path's reads stale
        admit = self.__dict__.setdefault("_lockstep_admit", {})
        ident, why, n, ms = _ident_here()
        rec = {"event": "check", "policy": policy, "identity": ident, "coverage": why, "n_records": n,
               "identity_ms": ms, "rank": _rank()}
        _SERVED.setdefault("t_first", time.time())
        try:
            equal_tried = False
            if not why and policy != "canary_only":  # the canary-only baseline never compares identities
                self.compat_hash = combine(base, ident)
                try:
                    out = orig_hs(self, *a, **k)
                    self.__dict__.setdefault("_lockstep_paths", {})[(host, port)] = (
                        n, (_remote(host, port) or {}).get("served"))
                    admit[(host, port)] = (ident, key0)
                    _log(dict(rec, result="admitted"))
                    return out
                except Exception as e:
                    if "hash mismatch" not in str(e):
                        _log(dict(rec, result="error", error=str(e)[:300]))
                        raise
                    equal_tried = True
            # the identities differ, or coverage is partial on this side
            if policy == "canary_only" and _remote(host, port) is None:
                # the baseline skips the identity comparison, but the producer publishes its registry entry (and
                # canary capture) only when it serves a handshake: request one, expect vLLM's hash check to fail
                try:
                    orig_hs(self, *a, **k)
                except Exception:
                    pass
            remote = _remote(host, port)
            if policy == "repair" and remote and remote.get("usage") and not why and not remote.get("coverage"):
                # repair instead of block: adopt the producer's launch configurations where only they differ,
                # then admit if our identity now equals the producer's. UNSOUND with CUDA graphs (RP addendum 1):
                # captured graphs keep the old configurations, so use only on consumers without CUDA graphs
                from lockstep import decisions
                t0 = time.perf_counter()
                plan = decisions.adopt(remote["usage"])
                ident2, why2, n2, _ = identity_v7_now()
                rec.update(repair=plan["adopted"], repair_unmatched=len(plan["unmatched"]), repair_ms=1000 * (time.perf_counter() - t0),
                           identity_after=ident2, remote_served=remote.get("served"))
                if not why2 and ident2 == remote.get("identity"):
                    self.compat_hash = combine(base, remote["served"])
                    out = orig_hs(self, *a, **k)
                    self.__dict__.setdefault("_lockstep_paths", {})[(host, port)] = (n2, remote.get("served"))
                    admit[(host, port)] = (remote.get("served"), _own_key())
                    _log(dict(rec, result="admitted_after_repair"))
                    return out
                _log(dict(rec, result="blocked", identity_compared=equal_tried))
                raise RuntimeError("lockstep: transfer blocked (repair did not make the identities equal)")
            t0 = time.perf_counter()
            canary = policy in ("canary", "canary_only")
            own_c = canary_get(own_cap) if canary else None
            rem_c = None
            if canary and remote:
                rem_c = remote.get("canary") or canary_get(remote.get("capture") or "")
            verdict = decide(policy, False, not why, own_c, rem_c)
            rec.update(remote_served=(remote or {}).get("served"), canary_ms=1000 * (time.perf_counter() - t0),
                       canary_checked=canary,
                       canary_equal=None if own_c is None or rem_c is None else own_c == rem_c,
                       remote_coverage=(remote or {}).get("coverage"))
            if verdict == "admit_canary":
                self.compat_hash = combine(base, remote["served"])
                out = orig_hs(self, *a, **k)
                if policy != "canary_only":  # the baseline checks once and never re-checks a connection
                    self.__dict__.setdefault("_lockstep_paths", {})[(host, port)] = (n, remote.get("served"))
                admit[(host, port)] = (remote.get("served"), key0)
                _log(dict(rec, result="admitted_by_canary"))
                return out
            _log(dict(rec, result="blocked", identity_compared=equal_tried))
            raise RuntimeError(f"lockstep: transfer blocked (policy {policy}, coverage {why or 'full'})")
        finally:
            self.compat_hash = base
    W._nixl_handshake = _nixl_handshake

    def _serve_identity(host, port, rank=0):
        tp = _tp_size()
        ident, why, n, ms = identity_rank(rank) if tp > 1 else identity_v7_now()
        served = ident if not why else f"partial:{ident}"  # never equal to a fully covered identity
        reg = _registry(host, port, rank if tp > 1 else 0)
        if reg:
            os.makedirs(os.path.dirname(reg), exist_ok=True)
            from lockstep import decisions
            json.dump({"identity": ident, "served": served, "coverage": why, "capture": own_cap,
                       "canary": None if os.environ.get("LOCKSTEP_CANARY_ON_DEMAND") == "1" else _canary(),
                       "n_records": n, "t": time.time(),
                       "usage": dict(decisions._USE)},  # for consumers that repair instead of block (RP)
                      open(reg + ".tmp", "w"), default=str)
            os.replace(reg + ".tmp", reg)
        if tp == 1:  # under TP, the serving process runs no model; producer-side republishing is not implemented
            _SERVED.update(n=n, served=served, addr=(host, port))
        _SERVED.setdefault("t_first", time.time())
        return served, why, n, ms

    def _nixl_handshake_listener(encoded_data, ready_event, stop_event, host, port):
        path = bs.make_zmq_path("tcp", host, port)
        with bs.zmq_ctx(zmq.ROUTER, path) as sock:
            sock.setsockopt(zmq.RCVTIMEO, 1000)
            ready_event.set()
            while True:
                try:
                    identity, _, msg = sock.recv_multipart()
                except zmq.Again:
                    if stop_event.is_set():
                        break
                    continue
                msg, pp, tp = msgspec.msgpack.decode(msg)
                if msg != bs.GET_META_MSG:
                    bs.logger.warning("Connection listener got unexpected message %s", msg)
                raw = encoded_data[(pp, tp)]
                if policy == "off":  # serve vLLM's own payload; start the test hooks' clock
                    _SERVED.setdefault("t_first", time.time())
                    sock.send_multipart((identity, b"", raw, msgspec.msgpack.encode(time.perf_counter())))
                    continue
                try:
                    p = msgspec.msgpack.decode(raw, type=NixlHandshakePayload)
                    served, why, n, ms = _serve_identity(host, port, tp)
                    raw = msgspec.msgpack.encode(NixlHandshakePayload(
                        compatibility_hash=combine(p.compatibility_hash, served),
                        agent_metadata_bytes=p.agent_metadata_bytes))
                    _log({"event": "serve", "identity": served, "coverage": why, "n_records": n,
                          "identity_ms": ms, "rank": tp})
                except Exception as e:  # never break the side channel; serve vLLM's own payload
                    _log({"event": "serve", "result": "error", "error": repr(e)[:300]})
                ts = msgspec.msgpack.encode(time.perf_counter())
                sock.send_multipart((identity, b"", raw, ts))
    S._nixl_handshake_listener = staticmethod(_nixl_handshake_listener)

    def _lifecycle(self):
        from lockstep import decisions
        now = time.monotonic()
        if now - self.__dict__.get("_lockstep_t_check", 0.0) < 0.5:
            return
        self._lockstep_t_check = now
        _inject_for_test()
        _retune_for_test()
        n = len(decisions._USE)
        if "n" in _SERVED and n != _SERVED["n"]:  # producer: republish the identity it now has
            old = _SERVED["served"]
            served, why, _, _ = _serve_identity(*_SERVED["addr"])
            _log({"event": "republished", "old": old, "identity": served, "coverage": why, "n_records": n})
        paths = self.__dict__.get("_lockstep_paths", {})
        if not paths:
            return
        busy = {m.remote.engine_id for m in self._recving_metadata.values() if getattr(m, "remote", None)}
        stale = self.__dict__.get("_lockstep_stale", set())  # paths whose reads the transfer tag refused
        for addr, (n_hs, served_then) in list(paths.items()):
            if addr not in stale and not needs_invalidation(n_hs, n, served_then, (_remote(*addr) or {}).get("served")):
                continue
            eid = self._engine_by_address.get(addr)
            if eid is None or eid not in self._remote_agents or eid in busy:
                continue  # retry at a later step, once its transfers have drained
            self._cleanup_remote_engine(eid, log_eviction=False)
            del paths[addr]
            stale.discard(addr)
            self.__dict__.get("_lockstep_admit", {}).pop(addr, None)
            _log({"event": "invalidated", "addr": list(addr), "n_at_handshake": n_hs, "n_records": n,
                  "served_then": served_then})

    _SERVE_FN[:] = [_serve_identity]
    from lockstep import decisions
    if _on_new_usage not in decisions._ON_USAGE:
        decisions._ON_USAGE.append(_on_new_usage)

    for mod in ("pull_worker", "push_worker"):
        try:
            m = __import__(f"vllm.distributed.kv_transfer.kv_connector.v1.nixl.{mod}", fromlist=["x"])
        except Exception:
            continue
        for cls in [v for v in vars(m).values()
                    if isinstance(v, type) and issubclass(v, W) and "start_load_kv" in vars(v)]:
            orig_sl = cls.start_load_kv

            def start_load_kv(self, metadata, _orig=orig_sl):
                try:
                    if _tp_size() > 1 and "t_first" not in _SERVED and getattr(metadata, "reqs_to_send", None):
                        # test hooks under TP: the engine process serves the handshakes, so a producer rank's
                        # clock starts with the first request it hands over (RW addendum 3)
                        _SERVED["t_first"] = time.time()
                    _lifecycle(self)
                except Exception as e:  # the guard must never break serving
                    _log({"event": "lifecycle_error", "error": repr(e)[:300]})
                return _orig(self, metadata)
            cls.start_load_kv = start_load_kv
    if os.environ.get("LOCKSTEP_XFER_TAG") in ("observe", "enforce"):
        install_xfer_tag(S, W)
    _log({"event": "installed", "version": 7, "policy": policy, "file": __file__})
