"""Offline analyses for review round 13 (O1-O3). Reads the mirrored live deployments only; changes nothing.

PYTHONPATH=lockstep python -m lockstep.offline_r13 o1 ROOT [ROOT ...] > offline_r13_o1_versions.json
PYTHONPATH=lockstep python -m lockstep.offline_r13 o2 ROOT [ROOT ...] > offline_r13_o2_static_hash.json
PYTHONPATH=lockstep python -m lockstep.offline_r13 o3 ROOT [ROOT ...] > offline_r13_o3_canary.json

Deployment set, KV-unsafe, planted, cell and family: exactly lockstep.cluster_stats.rows (the same 645 deployments
as cluster_report_r14.json for the same roots).

O1  per identity version of the live guard. Version: 7 if an "installed" event of either instance carries
    version 7 (only install_v7 logs a version). Versions 3 and 6 log a bare "installed" event and the
    environment variable LOCKSTEP_GUARD_VERSION is not recorded in meta.json or cfg.json, so they are told apart
    from the logged handshake identity and record count: version 6 if a logged identity equals the offline
    guard.identity(version=6) of that instance or a logged n_records equals the instance's number of usage
    records (version 6 counts usage records, version 3 decision records); version 3 if a logged identity equals
    guard.identity(version=3) or a logged n_records equals the number of decision records, or the instance has
    no usage records at all (version 6 cannot run without them). Each deployment reports the rule that fired.
O2  static-hash baselines vs identities (deployments whose two instances both have cfg.json).
O3  multi-prompt canary replay on local_a/local_b.
"""
import glob
import json
import math
import os
import random
import sys

from lockstep import ablate_identity, cluster_stats, decisions, guard


# ---------------------------------------------------------------- statistics
def upper95(k, n):
    """One-sided 95% exact (Clopper-Pearson) upper bound on a binomial rate with k events in n trials."""
    if not n:
        return None
    if k >= n:
        return 1.0
    if k == 0:
        return 1 - 0.05 ** (1 / n)

    def cdf(p):  # P(X <= k)
        return sum(math.exp(math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
                            + i * math.log(p) + (n - i) * math.log1p(-p)) for i in range(k + 1))
    lo, hi = k / n, 1.0
    for _ in range(100):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if cdf(mid) > 0.05 else (lo, mid)
    return hi


# ---------------------------------------------------------------- live-guard events
def events(d, side):
    f = os.path.join(d, f"inst_{side}", "capture", "live_guard.jsonl")
    if not os.path.exists(f):
        return []
    out = []
    for line in open(f):
        try:
            out.append(json.loads(line))
        except Exception:
            pass
    return out


def classify_version(d):
    ev = {s: events(d, s) for s in "ab"}
    if any(e.get("event") == "installed" and e.get("version") == 7 for s in "ab" for e in ev[s]):
        return 7, "installed.version=7"
    votes = []
    for s in "ab":
        inst = os.path.join(d, f"inst_{s}")
        logged = [e for e in ev[s] if e.get("event") in ("check", "serve") and "identity" in e]
        if not logged or not os.path.exists(os.path.join(inst, "cfg.json")):
            continue
        ids = {e["identity"] for e in logged}
        nrec = {e.get("n_records") for e in logged}
        n_use, n_dec = len(decisions.load_usage(inst)), len(decisions.load(inst))
        try:
            if guard.identity(inst, version=6)[0] in ids:
                votes.append((6, "identity=offline v6"))
                continue
        except Exception:
            pass
        if n_use and n_use in nrec and n_use != n_dec:
            votes.append((6, "n_records=usage records"))
            continue
        try:
            if guard.identity(inst, version=3)[0] in ids:
                votes.append((3, "identity=offline v3"))
                continue
        except Exception:
            pass
        if n_dec in nrec:
            votes.append((3, "n_records=decision records"))
        elif not n_use:
            votes.append((3, "no usage records"))
    vs = {v for v, _ in votes}
    if len(vs) == 1:
        return vs.pop(), "+".join(sorted({w for _, w in votes}))
    return None, f"ambiguous {votes}"


def live_info(d):
    """Per-deployment handshake facts from both instances' live_guard.jsonl."""
    checks = [e for s in "ab" for e in events(d, s) if e.get("event") == "check"]
    inst = [e for s in "ab" for e in events(d, s) if e.get("event") == "installed"]
    res = [e.get("result") for e in checks]
    pol = sorted({str(e.get("policy")) for e in checks + inst if e.get("policy")})
    canary_seen = any(e.get("result") == "admitted_by_canary" or e.get("canary_checked") for e in checks)
    if not pol:
        pol = ["hybrid(v3/v6 canary on mismatch)"] if canary_seen else ["identity-only or hybrid (not logged)"]
    not_on_identity = any(r in ("blocked", "admitted_by_canary", "admitted_after_repair") for r in res)
    partial = any(e.get("coverage") for e in checks)
    identity_admit = bool(res) and all(r in ("admitted", "error") for r in res) and "admitted" in res
    return {"results": sorted(set(map(str, res))), "policy": "/".join(pol), "not_admitted_on_identity": not_on_identity,
            "coverage_partial": partial, "canary_only": "canary_only" in pol, "identity_admit": identity_admit,
            "admitted_by_canary": "admitted_by_canary" in res}


def tagged_rows(roots):
    for r in cluster_stats.rows(roots):
        d = r["dir"]
        r = dict(r)
        r["version"], r["version_rule"] = classify_version(d)
        r.update(live_info(d))
        ts = [e.get("t") for s in "ab" for e in events(d, s) if e.get("event") == "installed" and e.get("t")]
        r["t_installed"] = min(ts) if ts else None
        parts = d.replace("\\", "/").split("/")
        base = parts[-1].strip().strip("")
        r["warmup"] = base.startswith(("prime_", "warm_", "a_warm_", "b_warm_"))
        r["smoke"] = any("smoke" in p for p in parts)
        yield r


# ---------------------------------------------------------------- O1
def summarize_o1(rs):
    nat = [r for r in rs if r["unsafe"] and not r["planted"]]
    ua = [r for r in rs if r["unsafe"] and r["admitted"]]
    ua_nat = [r for r in ua if not r["planted"]]
    out = {"deployments": len(rs), "kv_unsafe": sum(r["unsafe"] for r in rs), "natural_kv_unsafe": len(nat),
           "planted": sum(r["planted"] for r in rs), "planted_kv_unsafe": sum(r["unsafe"] and r["planted"] for r in rs),
           "identities_differed_or_not_admitted_on_identity": sum(r["not_admitted_on_identity"] for r in rs),
           "  of_which_blocked_final": sum(r["not_admitted_on_identity"] and not r["admitted"] for r in rs),
           "  of_which_admitted_by_canary": sum(r["not_admitted_on_identity"] and r["admitted"] for r in rs),
           "  of_which_coverage_partial": sum(r["not_admitted_on_identity"] and r["coverage_partial"] for r in rs),
           "  of_which_canary_only_policy(identity never compared)": sum(r["not_admitted_on_identity"] and r["canary_only"] for r in rs),
           "natural_kv_unsafe_with_identity_differed": sum(r["not_admitted_on_identity"] for r in nat),
           "natural_kv_unsafe_admitted_on_identity": sum(r["identity_admit"] for r in nat),
           "unsafe_admits": len(ua), "unsafe_admits_natural": len(ua_nat), "unsafe_admits_planted": len(ua) - len(ua_nat),
           "unsafe_admits_detail": [{"dir": r["dir"], "planted": r["planted"], "policy": r["policy"],
                                     "mechanism": "canary" if r["admitted_by_canary"] else "identity"} for r in ua],
           "per_deployment": {"n_natural_unsafe": len(nat), "natural_unsafe_admits": len(ua_nat),
                              "bound_zero_failure_formula": cluster_stats._bound(len(nat)),
                              "bound_exact_with_observed": upper95(len(ua_nat), len(nat))}}
    for g in ("cell", "family"):
        groups = sorted({r[g] for r in nat})
        bad = sorted({r[g] for r in ua_nat})
        out[g] = {"groups_with_natural_unsafe": len(groups), "groups_with_natural_unsafe_admit": len(bad),
                  "bound_zero_failure_formula": cluster_stats._bound(len(groups)),
                  "bound_exact_with_observed": upper95(len(bad), len(groups)),
                  "groups_with_admit": bad}
    return out


def o1(roots):
    rs = list(tagged_rows(roots))
    out = {"roots": roots, "deployments": len(rs),
           "classification": {}, "by_version": {}, "definitions": {
               "version": __doc__.split("O1  ")[1].split("O2  ")[0].strip(),
               "identities_differed": "any handshake check with result blocked, admitted_by_canary or admitted_after_repair "
                                      "(i.e. not admitted on identity); for v7 this includes partial coverage and the "
                                      "canary_only baseline, counted separately",
               "verdict": "cluster_stats.verdict: admitted iff some check admitted (incl. by canary) and none blocked",
               "bounds": "bound_zero_failure_formula = 1-0.05^(1/n) as in cluster_stats (n = natural KV-unsafe "
                         "deployments / groups); bound_exact_with_observed = one-sided 95% Clopper-Pearson upper bound "
                         "with the observed number of natural unsafe admits"}}
    import collections
    cl = collections.Counter((str(r["version"]), r["version_rule"]) for r in rs)
    out["classification"] = {f"v{v}: {w}": n for (v, w), n in sorted(cl.items())}
    import datetime
    for v in sorted({r["version"] for r in rs}, key=str):
        sub = [r for r in rs if r["version"] == v]
        s = summarize_o1(sub)
        ts = [r["t_installed"] for r in sub if r["t_installed"]]
        if ts:
            s["installed_time_range"] = [str(datetime.datetime.fromtimestamp(min(ts))),
                                         str(datetime.datetime.fromtimestamp(max(ts)))]
        s["roots"] = dict(collections.Counter(r["dir"].replace("\\", "/").split("/")[0] for r in sub))
        out["by_version"][f"v{v}"] = s
    v7 = [r for r in rs if r["version"] == 7]
    main_ = [r for r in v7 if not r["warmup"] and not r["smoke"]]
    out["v7_excluding_warmups_and_smoke"] = summarize_o1(main_)
    out["v7_excluded_dirs"] = {"warmup(prime_/warm_/a_warm_/b_warm_)": [r["dir"] for r in v7 if r["warmup"]],
                               "smoke(path contains 'smoke')": [r["dir"] for r in v7 if r["smoke"]]}
    out["v7_excluding_warmups_smoke_and_planted"] = summarize_o1([r for r in main_ if not r["planted"]])
    out["all"] = summarize_o1(rs)
    out["per_deployment_rows"] = [{k: r[k] for k in ("dir", "version", "version_rule", "unsafe", "planted", "admitted",
                                                     "policy", "results", "warmup", "smoke")} for r in rs]
    return out


# ---------------------------------------------------------------- O2
def _cfg(inst):
    return json.load(open(os.path.join(inst, "cfg.json")))


def static_part(inst, drop_budget=False):
    """ablate_identity variant "budget": guard.config_part(cfg, traced=False) - every engine option except
    guard.NEUTRAL_KEYS minus the token budget (so max_num_batched_tokens counts, max_num_seqs does not), incl. the
    ibi flag and any --compilation-config; cfg env minus LOCKSTEP_/CUDA_VISIBLE/NIXL-side-channel/host-IP and
    minus compilation-cache locations - plus decisions.load_stack (GPU model, CC, SMs, driver, torch, CUDA,
    Triton, vLLM) when recorded."""
    part = guard.config_part(_cfg(inst), False)
    part["env"] = {k: v for k, v in part["env"].items() if k not in guard.TRACE_DEPENDENT_ENV}
    if drop_budget:
        part["ek"] = {k: v for k, v in part["ek"].items() if k not in ("max_num_batched_tokens", "max_num_seqs")}
    return {"cfg": part, "stack": decisions.load_stack(inst)}


def flag_on(inst):
    """Inductor batch_invariant flag: set by hand (cfg engine_kwargs.ibi) or by vLLM itself, which vLLM main
    (a .dev build after 0.32.0, incl. PR #61118; the A8U runs) does under VLLM_BATCH_INVARIANT=1."""
    cfg = _cfg(inst)
    if cfg.get("engine_kwargs", {}).get("ibi"):
        return True
    v = str(decisions.load_stack(inst).get("vllm", ""))
    return "dev" in v and (cfg.get("env") or {}).get("VLLM_BATCH_INVARIANT") == "1"


O2_POLICIES = ("static", "static_flag_enforced", "static_budget_excluded", "static_budget_excluded_flag_enforced",
               "live_identity", "live_final", "v6", "v6p")


def o2_rows(roots):
    for r in tagged_rows(roots):
        d = r["dir"]
        ia, ib = os.path.join(d, "inst_a"), os.path.join(d, "inst_b")
        if not all(os.path.exists(os.path.join(x, "cfg.json")) for x in (ia, ib)):
            continue
        stack = bool(decisions.load_stack(ia)) and bool(decisions.load_stack(ib))
        h = lambda x, drop: ablate_identity._h(static_part(x, drop))
        st = h(ia, False) == h(ib, False)
        assert (ablate_identity.ident(ia, "budget") == ablate_identity.ident(ib, "budget")) == st
        stx = h(ia, True) == h(ib, True)
        fa, fb = flag_on(ia), flag_on(ib)
        usage = bool(decisions.load_usage(ia)) and bool(decisions.load_usage(ib))
        adm = {"static": st, "static_flag_enforced": st and fa and fb, "static_budget_excluded": stx,
               "static_budget_excluded_flag_enforced": stx and fa and fb,
               "live_identity": r["identity_admit"], "live_final": r["admitted"],
               "v6": guard.identity(ia, 6)[0] == guard.identity(ib, 6)[0],
               "v6p": guard.identity(ia, "6p")[0] == guard.identity(ib, "6p")[0]}
        vllm = {str(decisions.load_stack(x).get("vllm")) for x in (ia, ib)}
        yield {"dir": d, "unsafe": r["unsafe"], "planted": r["planted"], "version": r["version"], "stack": stack,
               "usage": usage, "flag": "both_on" if fa and fb else ("both_off" if not fa and not fb else "mixed"),
               "vllm": "/".join(sorted(vllm)), "admit": adm}


def summarize_o2(rs):
    out = {"deployments": len(rs), "kv_unsafe": sum(r["unsafe"] for r in rs),
           "natural_kv_unsafe": sum(r["unsafe"] and not r["planted"] for r in rs),
           "planted_kv_unsafe": sum(r["unsafe"] and r["planted"] for r in rs),
           "proxy_identical": sum(not r["unsafe"] for r in rs)}
    for p in O2_POLICIES:
        out[p] = {"admits": sum(r["admit"][p] for r in rs),
                  "unsafe_admits_natural": sum(r["admit"][p] and r["unsafe"] and not r["planted"] for r in rs),
                  "unsafe_admits_planted": sum(r["admit"][p] and r["unsafe"] and r["planted"] for r in rs),
                  "unneeded_blocks": sum(not r["admit"][p] and not r["unsafe"] for r in rs)}
    return out


def o2(roots):
    rs = list(o2_rows(roots))
    import collections
    out = {"roots": roots, "definitions": {
        "static": static_part.__doc__.strip(), "flag": flag_on.__doc__.strip(),
        "static_flag_enforced": "admit iff static hashes equal AND both instances have the flag on",
        "static_budget_excluded": "static without max_num_batched_tokens and max_num_seqs (max_num_seqs is already "
                                  "neutral in static); everything else as static",
        "static_budget_excluded_flag_enforced": "static_budget_excluded AND both flags on",
        "live_identity": "the live guard admitted on identity at every (non-error) handshake check (version 3/6/7 as run)",
        "live_final": "cluster_stats verdict incl. canary admits (v3/v6 hybrid, v7 canary policies)",
        "v6/v6p": "guard.identity(version=6 / '6p') recomputed offline; without usage records it degenerates to "
                  "config+stack, see with_usage_records",
        "unsafe": "cluster_stats: first generated token's log-prob differs on any prompt",
        "unneeded_block": "policy blocks although the proxy shows identical first-token log-probs on every prompt"},
        "excluded_no_cfg": None,
        "all": summarize_o2(rs),
        "by_flag": {f: summarize_o2([r for r in rs if r["flag"] == f]) for f in ("both_on", "mixed", "both_off")},
        "with_stack": summarize_o2([r for r in rs if r["stack"]]),
        "with_usage_records": summarize_o2([r for r in rs if r["usage"]]),
        "with_usage_records_by_flag": {f: summarize_o2([r for r in rs if r["usage"] and r["flag"] == f])
                                       for f in ("both_on", "mixed", "both_off")},
        "by_live_version": {f"v{v}": summarize_o2([r for r in rs if r["version"] == v]) for v in (3, 6, 7)},
        "vllm_versions": dict(collections.Counter(r["vllm"] for r in rs)),
        "unsafe_admit_dirs": {p: [r["dir"] for r in rs if r["admit"][p] and r["unsafe"]] for p in O2_POLICIES}}
    n_all = sum(1 for _ in cluster_stats.rows(roots))
    out["excluded_no_cfg"] = n_all - len(rs)
    return out


# ---------------------------------------------------------------- O3
def _subset_name(d):
    p = d.replace("\\", "/")
    if "/r0320/" in p:
        return "r0320"
    if "/v7s/" in p:
        return "v7s"
    if "/single/" in f"/{p}" and "a8t_v032" in p:
        return "single_a8t_v032"
    if "/tp14/" in p and "v032" in os.path.basename(p.rstrip("/")):
        return "tp14_v032"
    return None


K_VALUES = (1, 2, 4, 8)
N_RANDOM = 20


def o3(roots, seed=13):
    rows = []
    for r in cluster_stats.rows(roots):
        d = r["dir"]
        a, b = json.load(open(os.path.join(d, "local_a.json"))), json.load(open(os.path.join(d, "local_b.json")))
        n = min(len(a), len(b))
        ft = [(a[i]["logprobs"] or [None])[0] != (b[i]["logprobs"] or [None])[0] for i in range(n)]
        sq = [(a[i]["token_ids"], a[i]["logprobs"]) != (b[i]["token_ids"], b[i]["logprobs"]) for i in range(n)]
        stack = decisions.load_stack(os.path.join(d, "inst_b"))
        rows.append({"dir": d, "n": n, "ft": ft, "sq": sq, "planted": r["planted"], "subset": _subset_name(d),
                     "vllm": str(stack.get("vllm"))})
    rng = random.Random(seed)
    subsets = {}  # (dir, k) -> list of index tuples: first-k, then N_RANDOM random

    for r in rows:
        for k in K_VALUES:
            subsets[(r["dir"], k)] = [tuple(range(k))] + [tuple(sorted(rng.sample(range(r["n"]), k)))
                                                          for _ in range(N_RANDOM)]

    def evaluate(rs, signal):
        """signal: 'ft' (first-token log-prob) or 'sq' (full generated sequence) - what the canary compares.
        Truth is always the cluster_stats proxy (first-token log-prob on prompts)."""
        res = {}
        for k in K_VALUES:
            e = {}
            for sel, idx in (("first_k", [0]), ("random", list(range(1, N_RANDOM + 1)))):
                ins_miss, ins_n, ho_miss, ho_n, ho_fb, ho_safe, any_miss, ins_fb, ins_safe = [], 0, [], 0, [], 0, 0, [], 0
                for r in rs:
                    unsafe_all = any(r["ft"])
                    miss_r, hm_r, hfb_r, ifb_r = [], [], [], []
                    for j in idx:
                        S = subsets[(r["dir"], k)][j]
                        admit = not any(r[signal][i] for i in S)
                        rest = [i for i in range(r["n"]) if i not in S]
                        unsafe_ho = any(r["ft"][i] for i in rest)
                        if unsafe_all:
                            miss_r.append(admit)
                        else:
                            ifb_r.append(not admit)
                        if unsafe_ho:
                            hm_r.append(admit)
                        else:
                            hfb_r.append(not admit)
                    if unsafe_all:
                        ins_n += 1
                        ins_miss.append(sum(miss_r) / len(miss_r))
                        any_miss += any(miss_r)
                    else:
                        ins_safe += 1
                        ins_fb.append(sum(ifb_r) / len(ifb_r))
                    if hm_r:
                        ho_n += 1
                        ho_miss.append(sum(hm_r) / len(hm_r))
                    if hfb_r:
                        ho_safe += 1
                        ho_fb.append(sum(hfb_r) / len(hfb_r))
                e[sel] = {"kv_unsafe": ins_n,
                          "miss_rate_in_sample": sum(ins_miss) / ins_n if ins_n else None,
                          "missed_deployments_in_sample(mean)": sum(ins_miss),
                          "deployments_missed_by_any_subset": any_miss if sel == "random" else None,
                          "false_block_rate_in_sample": sum(ins_fb) / ins_safe if ins_safe else None,
                          "heldout_kv_unsafe(mean over subsets, counted per deployment)": ho_n,
                          "miss_rate_heldout": sum(ho_miss) / ho_n if ho_n else None,
                          "false_block_rate_heldout": sum(ho_fb) / ho_safe if ho_safe else None}
            res[f"k={k}"] = e
        return res

    def block(rs):
        return {"deployments": len(rs), "kv_unsafe": sum(any(r["ft"]) for r in rs),
                "natural_kv_unsafe": sum(any(r["ft"]) and not r["planted"] for r in rs),
                "prompts_per_deployment": sorted({r["n"] for r in rs}),
                "fraction_of_prompts_differing_in_kv_unsafe(mean)":
                    (lambda u: sum(sum(r["ft"]) / r["n"] for r in u) / len(u) if u else None)([r for r in rs if any(r["ft"])]),
                "canary_first_token": evaluate(rs, "ft"), "canary_full_sequence": evaluate(rs, "sq")}

    out = {"roots": roots, "seed": seed, "random_subsets": N_RANDOM, "definitions": {
        "truth": "KV-unsafe iff the first generated token's log-prob differs on any prompt (cluster_stats)",
        "canary": "admit iff the k canary prompts agree: first_token = first generated token's log-prob; "
                  "full_sequence = generated token ids and all log-probs",
        "selection": "first_k = prompts 0..k-1; random = 20 uniform k-subsets per deployment and k (random.Random(13))",
        "in_sample": "miss rate = mean over KV-unsafe deployments (truth on all prompts, canary prompts included) of the "
                     "fraction of subsets the canary admits; false_block = mean fraction of subsets blocking a "
                     "deployment that is identical on every prompt",
        "heldout": "truth computed on the prompts NOT used as canary; miss rate = among (deployment, subset) with a "
                   "held-out difference, the fraction admitted, averaged per deployment then over deployments; "
                   "false_block = blocked although the held-out prompts are all identical (the only blocks this can "
                   "count for the first-token canary are deployments whose only differences are on canary prompts)"},
        "all": block(rows), "natural_only": block([r for r in rows if not r["planted"]]),
        "vllm_0320_matrices": {s: block([r for r in rows if r["subset"] == s])
                               for s in ("r0320", "v7s", "a8x_single_a8t_v032", "tp14_v032")},
        "vllm_0320_matrices_pooled": block([r for r in rows if r["subset"]]),
        "vllm_0320_by_stack": block([r for r in rows if r["vllm"] == "0.32.0"]),
        "not_vllm_0320_by_stack": block([r for r in rows if r["vllm"] != "0.32.0"])}
    out["kv_unsafe_missed_by_first_prompt_canary"] = [r["dir"] for r in rows if any(r["ft"]) and not r["ft"][0]]
    return out


def main():
    cmd, roots = sys.argv[1], sys.argv[2:]
    out = {"o1": o1, "o2": o2, "o3": o3}[cmd](roots)
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
