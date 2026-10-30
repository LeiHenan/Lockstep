"""Localizer step 4: diff Inductor-generated code between two configurations.

Input: two vLLM logs produced with TORCH_LOGS=output_code (one per configuration).
Each "Output code:" block is one compiled graph (piecewise subgraph x compile range),
in compilation order. For a chosen block we extract every generated Triton kernel and
its @triton_heuristics decorator (size_hints, reduction hints, block sizes) and report
kernels whose source or heuristics differ.

python -m lockstep.codediff LOG_A LOG_B --block 0 [--show-source]
"""
import argparse
import difflib
import re

_PREFIX = re.compile(r"^.*?\[__output_code\] ?")


def blocks(path):
    txt = open(path, errors="ignore").read()
    out = []
    for part in txt.split("Output code:")[1:]:
        lines = [_PREFIX.sub("", l, count=1) for l in part.splitlines() if "[__output_code]" in l or not l.startswith("[rank")]
        out.append("\n".join(lines))
    return out


_KERNEL = re.compile(r"(@triton_heuristics\.(\w+)\((.*?)\)\s*\n@triton\.jit\s*\ndef (\w+)\((.*?)(?=\n'''|\n\"\"\"))", re.S)


def kernels(code):
    ks = {}
    for m in _KERNEL.finditer(code):
        full, kind, heur, name, _ = m.groups()
        ks[name] = {"kind": kind, "heuristics": " ".join(heur.split()), "source": full}
    return ks


def normalize(src):
    # kernel names embed hashes; normalize them so pure renames do not count as diffs
    return re.sub(r"triton_\w+?_\d+", "triton_K", src)


def diff_block(ba, bb, show_source=False):
    ka, kb = kernels(ba), kernels(bb)
    rep = {"n_kernels_a": len(ka), "n_kernels_b": len(kb), "diffs": []}
    names = sorted(set(ka) | set(kb))
    for n in names:
        a, b = ka.get(n), kb.get(n)
        if a is None or b is None:
            rep["diffs"].append({"kernel": n, "only_in": "a" if b is None else "b"})
            continue
        if a["heuristics"] != b["heuristics"] or normalize(a["source"]) != normalize(b["source"]):
            d = {"kernel": n, "kind": a["kind"], "heuristics_a": a["heuristics"], "heuristics_b": b["heuristics"]}
            if show_source:
                d["source_diff"] = "\n".join(difflib.unified_diff(
                    a["source"].splitlines(), b["source"].splitlines(), "a", "b", lineterm="", n=1))
            rep["diffs"].append(d)
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log_a")
    ap.add_argument("log_b")
    ap.add_argument("--block", type=int, default=0)
    ap.add_argument("--show-source", action="store_true")
    a = ap.parse_args()
    A, B = blocks(a.log_a), blocks(a.log_b)
    print(f"blocks: a={len(A)} b={len(B)}")
    rep = diff_block(A[a.block], B[a.block], a.show_source)
    print(f"block {a.block}: kernels a={rep['n_kernels_a']} b={rep['n_kernels_b']} differing={len(rep['diffs'])}")
    for d in rep["diffs"]:
        print("--", d.get("kernel"), d.get("kind", ""), d.get("only_in", ""))
        if "heuristics_a" in d:
            print("   a:", d["heuristics_a"][:600])
            print("   b:", d["heuristics_b"][:600])
        if d.get("source_diff"):
            print(d["source_diff"][:3000])


if __name__ == "__main__":
    main()
