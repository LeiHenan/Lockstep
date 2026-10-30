"""SGLang adapter: boundary capture at decoder-layer outputs (SGLang runs eagerly by default,
so module forward hooks do not perturb code generation; decode steps replayed from CUDA graphs
are not observed and are covered by token comparison instead).

Records use the same format as the vLLM plugin, so lockstep.align works unchanged:
  subgraph index := decoder layer index (final norm = num_layers), slots = (hidden, residual).
Loaded in every SGLang process through sitecustomize (see sitecustom/sitecustomize.py).
"""
import os
import pickle
import zlib

_S = {"active": False, "layout": None, "n": 0, "calls": [], "fh": None, "step": -1, "pkeys": {}}


def _rank():
    try:
        import torch.distributed as dist
        return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    except Exception:
        return 0


def _flush():
    if _S["layout"] is not None and _S["calls"]:
        if _S["fh"] is None:
            d = os.environ["LOCKSTEP_DIR"]
            os.makedirs(d, exist_ok=True)
            _S["fh"] = open(os.path.join(d, f"rank{_rank()}_pid{os.getpid()}.pkl"), "ab")
        calls = [(i, None, io, slot, h.cpu().numpy(), w) for (i, io, slot, h, w) in _S["calls"]]
        pickle.dump({"step": _S["step"], "layout": _S["layout"], "n_tokens": _S["n"], "calls": calls}, _S["fh"])
        _S["fh"].flush()
    _S["calls"] = []


def register():
    if not os.environ.get("LOCKSTEP_DIR"):
        return
    import numpy as np
    import torch
    from sglang.srt.model_executor import forward_batch_info as fbi
    from sglang.srt.model_executor import model_runner as mr
    from lockstep.hashing import row_hash

    if getattr(mr.ModelRunner, "_lockstep_patched", False):
        return
    mr.ModelRunner._lockstep_patched = True

    orig_init_new = fbi.ForwardBatch.init_new.__func__

    def init_new(cls, batch, model_runner, *a, **k):
        for req in getattr(batch, "reqs", []) or []:
            ids = getattr(req, "origin_input_ids", None)
            if ids is not None and req.rid not in _S["pkeys"]:
                _S["pkeys"][req.rid] = zlib.crc32(np.asarray(list(ids), dtype=np.int64).tobytes())
        return orig_init_new(cls, batch, model_runner, *a, **k)
    fbi.ForwardBatch.init_new = classmethod(init_new)

    def layout_of(fb):
        pos = fb.positions[: fb.positions.shape[0]].detach().to("cpu", non_blocking=False).numpy()
        rids = fb.rids or []
        if fb.forward_mode.is_extend() and fb.extend_seq_lens_cpu is not None:
            lens = list(fb.extend_seq_lens_cpu)
        else:
            lens = [1] * len(rids)
        out, off = [], 0
        for rid, n in zip(rids, lens):
            out.append((_S["pkeys"].get(rid, -1), int(pos[off]) if n else 0, int(n)))
            off += n
        return out, off

    import atexit
    atexit.register(_flush)

    def on_batch(fb):
        """Called from the first hooked layer that sees a new ForwardBatch."""
        if _S.get("fb_id") == id(fb):
            return _S["active"]
        _flush()
        _S["fb_id"] = id(fb)
        ok = fb is not None and fb.rids is not None and not fb.forward_mode.is_idle()
        _S["active"] = bool(ok)
        if ok:
            _S["step"] += 1
            _S["layout"], _S["n"] = layout_of(fb)
        return _S["active"]

    orig_load = mr.ModelRunner.load_model

    def load_model(self, *a, **k):
        r = orig_load(self, *a, **k)
        inner = getattr(self.model, "model", None)
        layers = getattr(inner, "layers", None)
        if layers is None:
            print("[lockstep-sglang] no model.model.layers; capture disabled", flush=True)
            return r

        def mk(idx):
            def hook(mod, args, kwargs, out):
                if torch.cuda.is_current_stream_capturing():
                    return
                fb = kwargs.get("forward_batch")
                if fb is None:
                    fb = next((x for x in args if isinstance(x, fbi.ForwardBatch)), None)
                if fb is not None and not on_batch(fb):
                    return
                if not _S["active"]:
                    return
                outs = out if isinstance(out, (tuple, list)) else (out,)
                for slot, t in enumerate(outs):
                    if isinstance(t, torch.Tensor) and t.dim() >= 1 and t.shape[0] >= _S["n"] > 0:
                        _S["calls"].append((idx, "out", slot, row_hash(t, _S["n"]), int(t[:1].numel())))
            return hook
        for i, layer in enumerate(layers):
            layer.register_forward_hook(mk(i), with_kwargs=True)
        if getattr(inner, "norm", None) is not None:
            inner.norm.register_forward_hook(mk(len(layers)), with_kwargs=True)
        if os.environ.get("LOCKSTEP_DEBUG"):
            print(f"[lockstep-sglang] hooked {len(layers)} layers pid={os.getpid()}", flush=True)
        return r
    mr.ModelRunner.load_model = load_model
    try:  # hidden-decision tracing (Inductor autotuning) also applies to SGLang's compiled decode graphs
        os.environ.setdefault("LOCKSTEP_DEC_EAGER_FLUSH", "1")
        from lockstep import decisions
        decisions._EAGER = os.environ["LOCKSTEP_DEC_EAGER_FLUSH"] == "1"
        decisions.install()
        atexit.register(decisions.flush)
    except Exception as e:  # tracing is best-effort; never break the engine
        print(f"[lockstep-sglang] decision tracer not installed: {e}", flush=True)
