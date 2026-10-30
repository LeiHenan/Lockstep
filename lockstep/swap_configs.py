"""Localizer intervention for autotuning-induced divergence (F-002): swap individual autotune decisions.

Given two Inductor caches A and B produced by independent compilations of the same configuration,
build caches that are copies of A with exactly one kernel's `.best_config` taken from B. Running the
engine on each (with vLLM's compile cache disabled so Inductor reloads the decisions) tells which
kernel's autotune choice is sufficient to change the numerics.

python -m lockstep.swap_configs --a CACHE_A --b CACHE_B --out DIR --spec-out spec.json --base-spec base.json
"""
import argparse
import glob
import json
import os
import shutil


def best_configs(root):
    return {os.path.relpath(f, root): json.load(open(f)) for f in glob.glob(os.path.join(root, "**", "*.best_config"), recursive=True)}


def strip(cfg):
    return {k: v for k, v in cfg.items() if k in ("XBLOCK", "YBLOCK", "R0_BLOCK", "R1_BLOCK", "num_warps", "num_stages")}


def kernel_name(root, rel):
    for src in glob.glob(os.path.join(root, os.path.dirname(rel), "*.py")):
        for line in open(src, errors="ignore"):
            if line.startswith("def triton_"):
                return line.split("def ")[1].split("(")[0]
    return "?"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--out", required=True, help="directory for the prepared caches")
    ap.add_argument("--base-spec", required=True, help="spec whose base config is reused (env/kwargs)")
    ap.add_argument("--spec-out", required=True)
    a = ap.parse_args()
    ca, cb = best_configs(a.a), best_configs(a.b)
    diff = sorted(k for k in ca.keys() & cb.keys() if strip(ca[k]) != strip(cb[k]))
    os.makedirs(a.out, exist_ok=True)
    variants, manifest = [], {}

    def prep(name, src, swap=None):
        dst = os.path.join(a.out, name)
        if not os.path.exists(dst):
            shutil.copytree(src, dst)
            if swap:
                shutil.copyfile(os.path.join(a.b, swap), os.path.join(dst, swap))
        variants.append({"name": name, "env": {"TORCHINDUCTOR_CACHE_DIR": dst, "VLLM_DISABLE_COMPILE_CACHE": "1"}})

    prep("cacheA", a.a)
    prep("cacheB", a.b)
    for i, k in enumerate(diff):
        nm = f"swap{i}"
        prep(nm, a.a, swap=k)
        manifest[nm] = {"kernel_file": k, "kernel": kernel_name(a.a, k), "A": strip(ca[k]), "B": strip(cb[k])}
    base = json.load(open(a.base_spec))
    spec = {"prompts": base["prompts"], "base": base["base"], "variants": variants}
    json.dump(spec, open(a.spec_out, "w"), indent=1)
    json.dump(manifest, open(os.path.join(a.out, "manifest.json"), "w"), indent=1)
    print(f"{len(diff)} differing autotune decisions; wrote {len(variants)} variants")
    for nm, m in manifest.items():
        print(nm, m["kernel"], "A:", m["A"], "B:", m["B"])


if __name__ == "__main__":
    main()
