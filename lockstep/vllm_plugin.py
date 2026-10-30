"""vLLM general plugin: boundary capture at piecewise-compiled subgraph boundaries.

Enabled only when LOCKSTEP_DIR is set. It wraps (at runtime, without touching any
compiled code):
  * GPUModelRunner.execute_model  -> marks real forward steps (skips warmup/graph capture)
  * GPUModelRunner._prepare_inputs -> records row -> (prompt_key, position) layout
  * PiecewiseBackend.__call__      -> per-row fingerprints of subgraph inputs/outputs
Records are appended (pickle stream) to $LOCKSTEP_DIR/rank{R}.pkl.
"""
import os
import pickle
import zlib

_STATE = {"captured_inner": False, "active": False, "step": -1, "layout": None, "n_tokens": 0, "fh": None, "calls": []}


def _rank():
    try:
        import torch.distributed as dist
        return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    except Exception:
        return 0


def _fh():
    if _STATE["fh"] is None:
        d = os.environ["LOCKSTEP_DIR"]
        os.makedirs(d, exist_ok=True)
        _STATE["fh"] = open(os.path.join(d, f"rank{_rank()}_pid{os.getpid()}.pkl"), "ab")
    return _STATE["fh"]


def _flush_step():
    if _STATE["layout"] is not None and _STATE["calls"]:
        import torch
        calls = [(i, rng, io, slot, h.cpu().numpy(), w) for (i, rng, io, slot, h, w) in _STATE["calls"]]
        rec = {"step": _STATE["step"], "layout": _STATE["layout"], "n_tokens": _STATE["n_tokens"], "calls": calls}
        if _STATE.get("rows") is not None and _STATE["rows"][1] is not None:
            rec["rows_pk"], rec["rows_pos"] = _STATE["rows"]
        pickle.dump(rec, _fh())
        _STATE["fh"].flush()
    _STATE["calls"] = []
    try:
        from lockstep import decisions as _d
        _d.flush()
    except Exception:
        pass


def register():
    if not os.environ.get("LOCKSTEP_DIR"):
        return
    import numpy as np
    import torch
    from vllm.compilation import piecewise_backend as pb
    from vllm.v1.worker import gpu_model_runner as gmr
    from lockstep.hashing import row_hash

    if getattr(pb.PiecewiseBackend, "_lockstep_patched", False):
        return
    from lockstep import decisions as _dec
    _dec.install()
    if os.environ.get("LOCKSTEP_LIVE_GUARD") == "1":
        try:
            from lockstep.live_guard import install as _guard_install
            _guard_install()
        except Exception as e:
            print("[lockstep] live guard unavailable:", repr(e), flush=True)
    if os.environ.get("LOCKSTEP_AWQ_OPAQUE") == "1":
        from lockstep.awq_intervention import install as _awq_install
        _awq_install()
    if os.environ.get("LOCKSTEP_DEBUG"):
        print(f"[lockstep] register pid={os.getpid()} dir={os.environ.get('LOCKSTEP_DIR')}", flush=True)
    if os.environ.get("LOCKSTEP_MODE", "boundary") == "off":
        # Decision tracing only (LOCKSTEP_TRACE_ONLY): no boundary capture hooks. Without the per-step
        # flush of the capture path, write the trace whenever a new decision is recorded.
        import atexit
        _dec._EAGER = True
        atexit.register(_dec.flush)
        _orig = pb.PiecewiseBackend.__call__

        def _tag_only(self, *args):  # attribute decisions to the subgraph that launched them
            _dec.set_subgraph(self.piecewise_compile_index)
            return _orig(self, *args)
        pb.PiecewiseBackend.__call__ = _tag_only
        pb.PiecewiseBackend._lockstep_patched = True
        return
    max_slots = int(os.environ.get("LOCKSTEP_MAX_SLOTS", "64"))
    orig_call = pb.PiecewiseBackend.__call__

    pkeys = {}  # req_id -> stable prompt key (crc32 of prompt token ids)

    def _pkey(ptoks):
        return zlib.crc32(np.asarray(ptoks if ptoks is not None else [], dtype=np.int64).tobytes())

    def wrap_execute(cls):
        orig = cls.execute_model

        def execute_model(self, scheduler_output, *a, **k):
            for nr in getattr(scheduler_output, "scheduled_new_reqs", []) or []:
                pkeys[nr.req_id] = _pkey(nr.prompt_token_ids)
            _STATE["active"] = True
            _STATE["step"] += 1
            _STATE["layout"] = None
            _STATE["rows"] = None
            try:
                return orig(self, scheduler_output, *a, **k)
            finally:
                _STATE["active"] = False
                _flush_step()
        cls.execute_model = execute_model

    # --- Model Runner V1 ---
    wrap_execute(gmr.GPUModelRunner)
    orig_prep = gmr.GPUModelRunner._prepare_inputs

    def _prepare_inputs(self, scheduler_output, num_scheduled_tokens, *a, **k):
        if _STATE["active"]:
            nr = self.input_batch.num_reqs
            _STATE["layout"] = [(pkeys.get(rid, -1), int(self.input_batch.num_computed_tokens_cpu[i]),
                                 int(num_scheduled_tokens[i])) for i, rid in enumerate(self.input_batch.req_ids[:nr])]
            _STATE["n_tokens"] = int(np.sum(num_scheduled_tokens[:nr]))
        return orig_prep(self, scheduler_output, num_scheduled_tokens, *a, **k)
    gmr.GPUModelRunner._prepare_inputs = _prepare_inputs

    # --- Model Runner V2 ---
    try:
        from vllm.v1.worker.gpu import model_runner as gmr2
        wrap_execute(gmr2.GPUModelRunner)
        orig_prep2 = gmr2.GPUModelRunner.prepare_inputs

        def prepare_inputs(self, scheduler_output, batch_req_state, *a, **k):
            ret = orig_prep2(self, scheduler_output, batch_req_state, *a, **k)
            if _STATE["active"]:
                b = batch_req_state
                n = int(b.num_tokens)
                keys = [pkeys.get(rid, -1) for rid in b.req_ids]
                counts = np.asarray(b.num_scheduled_tokens[: len(keys)], dtype=np.int64)
                # exact per-row labels: request key repeated per scheduled token, positions from the
                # GPU buffer the model actually consumes (the CPU mirror can be optimistic)
                rows_pk = np.repeat(np.asarray(keys, dtype=np.int64), counts)[:n]
                pos_t = getattr(ret, "positions", None)
                rows_pos = pos_t[:n].detach().to("cpu").numpy().astype(np.int64) if pos_t is not None else None
                _STATE["layout"] = [(int(kk), -1, int(c)) for kk, c in zip(keys, counts)]
                _STATE["rows"] = (rows_pk, rows_pos)
                _STATE["n_tokens"] = n
            return ret
        gmr2.GPUModelRunner.prepare_inputs = prepare_inputs
    except Exception as e:  # pragma: no cover
        print("[lockstep] V2 runner patch skipped:", e)

    def _capture(backend, args, out):
        """Record per-row fingerprints of a piecewise subgraph's inputs and outputs."""
        if not _STATE["active"] or _STATE["layout"] is None or not getattr(backend, "sym_shape_indices", None):
            return
        if torch.cuda.is_current_stream_capturing():
            return
        rt = args[backend.sym_shape_indices[0]]
        rt = int(rt) if not isinstance(rt, int) else rt
        n = _STATE["n_tokens"]
        try:
            rng = backend._find_range_for_shape(rt).compile_range
            rng = (rng.start, rng.end) if hasattr(rng, "start") else str(rng)
        except Exception:
            rng = None
        idx = backend.piecewise_compile_index
        outs = out if isinstance(out, (tuple, list)) else (out,)
        for io, tensors in (("in", args), ("out", outs)):
            s = 0
            for slot, t in enumerate(tensors):
                if isinstance(t, torch.Tensor) and t.dim() >= 1 and t.shape[0] == rt and t.is_cuda and s < max_slots:
                    w = int(t[:1].numel())  # row width (elements); differs for TP-sharded tensors
                    _STATE["calls"].append((idx, rng, io, slot, row_hash(t, n), w))
                    s += 1
        _STATE["captured_inner"] = True

    # E4 fault injection (ground truth for localization accuracy):
    # LOCKSTEP_INJECT="<subgraph>:<out_slot>:<max_tokens>" flips the lowest bit of element 0 of the
    # first n rows of that output slot, only in steps with <= max_tokens tokens (a planted
    # batch-size-dependent bug).
    inj = os.environ.get("LOCKSTEP_INJECT")
    inj = tuple(int(x) for x in inj.split(":")) if inj else None

    def _inject(backend, out):
        if inj is None or not _STATE["active"] or backend.piecewise_compile_index != inj[0]:
            return
        n = _STATE["n_tokens"]
        if n <= 0 or n > inj[2]:
            return
        outs = out if isinstance(out, (tuple, list)) else (out,)
        if inj[1] < len(outs) and isinstance(outs[inj[1]], torch.Tensor):
            t = outs[inj[1]]
            v = t[:n].reshape(n, -1)
            bits = v[:, :1].view(torch.int16) if v.element_size() == 2 else v[:, :1].view(torch.int32)
            bits.bitwise_xor_(1)

    def __call__(self, *args):
        _dec.set_subgraph(self.piecewise_compile_index)
        out = orig_call(self, *args)
        _inject(self, out)
        _capture(self, args, out)
        return out

    # CUDA-graph replays of piecewise subgraphs bypass PiecewiseBackend.__call__; wrap the
    # graph wrapper too and record on replay (outputs are the graph's static buffers).
    try:
        from vllm.compilation import cuda_graph as cg
        orig_wrap_call = cg.CUDAGraphWrapper.__call__

        def wrapper_call(self, *args, **kwargs):
            _STATE["captured_inner"] = False
            out = orig_wrap_call(self, *args, **kwargs)
            inner = self.__dict__.get("runnable")
            if not _STATE["captured_inner"] and isinstance(inner, pb.PiecewiseBackend):
                _inject(inner, out)
                _capture(inner, args, out)
            return out
        cg.CUDAGraphWrapper.__call__ = wrapper_call
    except Exception as e:  # pragma: no cover
        print("[lockstep] CUDAGraphWrapper patch skipped:", e)

    mode = os.environ.get("LOCKSTEP_MODE", "boundary")
    if mode in ("naive", "naive_inline", "naive_break"):
        # Baseline for E3: naive instrumentation inside the compiled region. Wrap decoder-layer
        # forward methods so fingerprinting runs inside the model code that torch.compile traces.
        # Fingerprints are stashed on a side list; this changes what Dynamo/Inductor see.
        import importlib
        from lockstep.hashing import row_hash_traceable
        naive_buf = []
        # naive_inline: hashing is traced into the compiled graph;
        # naive_break / naive: hashing runs eagerly via a graph break at every decoder layer.
        hfn = row_hash_traceable if mode == "naive_inline" else torch.compiler.disable(lambda h: row_hash(h, h.shape[0]))
        for modname, clsname in (("vllm.model_executor.models.qwen3", "Qwen3DecoderLayer"),
                                 ("vllm.model_executor.models.llama", "LlamaDecoderLayer"),
                                 ("vllm.model_executor.models.qwen2", "Qwen2DecoderLayer")):
            try:
                cls = getattr(importlib.import_module(modname), clsname)
            except Exception:
                continue
            orig_fwd = cls.forward

            def fwd(self, *args, __orig=orig_fwd, **kw):
                out = __orig(self, *args, **kw)
                h = out[0] if isinstance(out, (tuple, list)) else out
                naive_buf.append(hfn(h))
                return out
            cls.forward = fwd
        pb.PiecewiseBackend.__call__ = __call__
    else:
        pb.PiecewiseBackend.__call__ = __call__
    pb.PiecewiseBackend._lockstep_patched = True
