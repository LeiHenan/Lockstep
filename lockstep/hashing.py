"""Order-independent, exact per-row fingerprints of GPU tensors.

h(row) = sum_i word_i * c_i  (mod 2**64), computed in int64 on the GPU. Integer
addition is associative, so the result does not depend on reduction order and the
fingerprint itself is deterministic. Two rows have equal fingerprints iff their raw
bytes are equal (up to negligible collision probability).
"""
import torch

_COEF = {}


def _coef(m: int, device) -> torch.Tensor:
    key = (m, str(device))
    c = _COEF.get(key)
    if c is None:
        g = torch.Generator().manual_seed(0x10C5)
        c = torch.randint(-(2**62), 2**62, (m,), generator=g, dtype=torch.int64) | 1
        c = c.to(device)
        _COEF[key] = c
    return c


def row_hash(t: torch.Tensor, n: int) -> torch.Tensor:
    """Fingerprint the first n rows of t (rows = dim 0). Returns int64 [n] on t.device."""
    x = t[:n].contiguous().reshape(n, -1)
    nbytes = x.shape[1] * x.element_size()
    if nbytes % 4 == 0:
        w = x.view(torch.int32) if x.dtype != torch.int32 else x
    else:
        w = x.view(torch.uint8)
    w = w.to(torch.int64)
    return (w * _coef(w.shape[1], w.device)).sum(dim=1)


def row_hash_traceable(t: torch.Tensor) -> torch.Tensor:
    """Variant used only by the E3 'naive' baselines: coefficients come from arithmetic on
    arange (no Generator), so torch.compile can trace it into the model graph."""
    x = t.contiguous().reshape(t.shape[0], -1)
    w = (x.view(torch.int32) if (x.shape[1] * x.element_size()) % 4 == 0 else x.view(torch.uint8)).to(torch.int64)
    c = torch.arange(w.shape[1], device=w.device, dtype=torch.int64) * 6364136223846793005 + 1442695040888963407
    return (w * (c | 1)).sum(dim=1)
