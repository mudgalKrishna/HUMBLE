"""Analysis jobs (CPU): E-REPAIR summary, hypotheses H1-H8, E-CAT, E-LANG/E-DOMAIN tables, E-REPRO.

Every number is recomputed from saved predictions on the Hub, so anyone can rerun this file.
"""
from __future__ import annotations

import json
import statistics as st

from . import gate as G, hub
from .audit import metrics as M
from .config import load_config, path, profile as get_profile
from .experiments import _audit_dir, _get_preds, cfg

METRICS = ["accuracy", "bluff_rate", "prudence", "over_conservativeness", "unnecessary_abstention", "honesty", "cost",
           "format_failure", "echo_rate"]


def _arm_finals(seeds):
    base = path("adapters", cfg()).parent / "_hub"
    hub.pull(["v3/arms/*.json"], base)
    arms = {"A": [], "B": []}
    for s in seeds:
        for a in ("A", "B"):
            p = base / "v3" / "arms" / f"arm{a}_s{s}.json"
            if p.exists():
                arms[a].append(json.loads(p.read_text())["final"])
    lams = []
    for f in ("armD.json", "armDfine.json"):
        d = base / "v3" / "arms" / f
        if d.exists():
            lams += [r["lambda"] for r in json.loads(d.read_text())["rows"] if r["gate"] == "PASS"]
    lam = max(lams) if lams else None
    return arms, lam


def _mean_sd(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None
    return st.mean(vals), (st.stdev(vals) if len(vals) > 1 else 0.0)


def e_repair_summary(J, probe_set="squad3v", split="test", profile="support", blocked="naive_s31"):
    seeds = cfg()["train"]["seeds"]
    prof = get_profile(profile, cfg())
    w = {k: prof[k] for k in ("w_bluff", "w_refuse", "w_wrong")}
    arms, lam = _arm_finals(seeds)
    names = {"original": ["original"], "naive (blocked)": [blocked], "A HUMBLE loop": arms["A"], "B random control": arms["B"],
             "C prevention": [f"prevC_s{s}" for s in seeds], "D lambda": (["original"] if lam == 0.0 else [f"lamD_{lam}"]) if lam is not None else []}
    orig = _get_preds(probe_set, "original", split)
    K = G.known_ids(orig)
    preds, table, gates = {}, {}, {}
    for arm, ns in names.items():
        rows = []
        for n in ns:
            try:
                p = _get_preds(probe_set, n, split)
            except Exception as e:
                J.log(f"missing {n}/{split}: {e!r}")
                continue
            preds[n] = p
            rows.append(M.core(p, K, w))
            gates[n] = {pn: G.evaluate(orig, p, get_profile(pn, cfg()), resamples=500)["decision"] for pn in cfg()["gate"]["profiles"]}
        table[arm] = {m: _mean_sd([r.get(m) for r in rows]) for m in METRICS} | {"n_seeds": len(rows), "names": ns}
    comps = {}
    for i, s in enumerate(seeds):
        a = arms["A"][i] if i < len(arms["A"]) else None
        b = arms["B"][i] if i < len(arms["B"]) else None
        c = f"prevC_s{s}"
        if a in preds and b in preds:
            comps[f"A_vs_B_s{s}"] = {m: M.paired_diff(preds[b], preds[a], m, K, w) for m in ("cost", "prudence", "over_conservativeness", "accuracy")}
        if a in preds and c in preds:
            comps[f"A_vs_C_s{s}"] = {m: M.paired_diff(preds[c], preds[a], m, K, w) for m in ("cost", "prudence", "over_conservativeness", "accuracy")}
    if blocked in preds:
        comps["naive_vs_original"] = {m: M.paired_diff(orig, preds[blocked], m, K, w) for m in ("prudence", "accuracy", "cost")}
    hyp = hypotheses(table, comps, preds, orig, K, blocked)
    res = {"split": split, "profile": profile, "arms": table, "gate_decisions": gates, "comparisons": comps, "hypotheses": hyp}
    out = path("audits", cfg()) / probe_set / f"_e_repair_{split}.json"
    out.write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    hub.push(out, f"v3/audits/{probe_set}/{out.name}", cfg())
    return {"hypotheses": {k: v["verdict"] for k, v in hyp.items()},
            "arms": {k: {m: v[m][0] for m in ("prudence", "over_conservativeness", "accuracy", "cost")} for k, v in table.items()}}


def e_targeting(J, probe_set="squad3v", split="test", profile="support", blocked="naive_light_s31",
                a_tag="Alight", b_tag="Blight", seeds=None):
    """Fix 1: targeted (A) vs random (B) repair when the blocked model is only partly broken.

    Same paired comparison as the H3 rows of E-REPAIR, but for arms tagged `a_tag` / `b_tag` that both start from
    `blocked`. Also records how broken the blocked model is (its prudence and bluff rate vs the original), since the
    question is whether targeting helps when the model's own mistakes are not simply every unanswerable item."""
    seeds = seeds or cfg()["train"]["seeds"]
    prof = get_profile(profile, cfg())
    w = {k: prof[k] for k in ("w_bluff", "w_refuse", "w_wrong")}
    base = path("adapters", cfg()).parent / "_hub"
    hub.pull(["v3/arms/*.json"], base)
    orig = _get_preds(probe_set, "original", split)
    K = G.known_ids(orig)
    finals, table, comps, preds = {}, {}, {}, {}
    for s in seeds:
        for t in (a_tag, b_tag):
            p = base / "v3" / "arms" / f"arm{t}_s{s}.json"
            if not p.exists():
                p = path("adapters", cfg()) / f"arm{t}_s{s}.json"
            if p.exists():
                a = json.loads(p.read_text())
                if a.get("blocked", blocked) != blocked:
                    raise ValueError(f"arm{t}_s{s} started from {a.get('blocked')}, not {blocked}")
                finals[(t, s)] = a.get("final_adapter") or a["final"]
    names = {"original": ["original"], f"blocked ({blocked})": [blocked],
             f"{a_tag} targeted": [finals[(a_tag, s)] for s in seeds if (a_tag, s) in finals],
             f"{b_tag} random": [finals[(b_tag, s)] for s in seeds if (b_tag, s) in finals]}
    for arm, ns in names.items():
        rows = []
        for n in ns:
            try:
                preds[n] = _get_preds(probe_set, n, split)
            except Exception as e:
                J.log(f"missing {n}/{split}: {e!r}")
                continue
            rows.append(M.core(preds[n], K, w))
        table[arm] = {m: _mean_sd([r.get(m) for r in rows]) for m in METRICS} | {"n_seeds": len(rows), "names": ns}
    for s in seeds:
        a, b = finals.get((a_tag, s)), finals.get((b_tag, s))
        if a in preds and b in preds:
            comps[f"{a_tag}_vs_{b_tag}_s{s}"] = {m: M.paired_diff(preds[b], preds[a], m, K, w)
                                                 for m in ("cost", "prudence", "over_conservativeness", "accuracy")}
    if blocked in preds:
        comps["blocked_vs_original"] = {m: M.paired_diff(orig, preds[blocked], m, K, w) for m in ("prudence", "accuracy", "cost")}
    ok, ds = _all_sig(comps, f"{a_tag}_vs_{b_tag}", "cost", -1)
    verdict = "untested" if ok is None else ("supported" if ok else "not supported")
    res = {"split": split, "profile": profile, "blocked": blocked, "arms": table, "comparisons": comps,
           "H3_partial": {"claim": f"Targeted repair {a_tag} lowers cost more than random {b_tag} from a partly broken model "
                                   "(every seed, paired CI excludes 0)", "verdict": verdict}}
    out = path("audits", cfg()) / probe_set / f"_e_targeting_{blocked}_{split}.json"
    out.write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    hub.push(out, f"v3/audits/{probe_set}/{out.name}", cfg())
    return {"H3_partial": verdict,
            "arms": {k: {m: v[m][0] for m in ("prudence", "over_conservativeness", "accuracy", "cost")} for k, v in table.items()}}


def _all_sig(comps, prefix, metric, sign):
    ds = [v[metric] for k, v in comps.items() if k.startswith(prefix)]
    if not ds:
        return None, ds
    ok = all(d["significant"] and (d["diff"] < 0 if sign < 0 else d["diff"] > 0) for d in ds)
    return ok, ds


def hypotheses(table, comps, preds, orig, K, blocked):
    h = {}
    n = comps.get("naive_vs_original", {})
    pr = n.get("prudence")
    h["H1"] = {"claim": "Naive fine-tuning lowers prudence vs the original (multi-passage RAG)",
               "evidence": pr, "verdict": "supported" if pr and pr["significant"] and pr["diff"] < 0 else "not supported"}
    if blocked in preds:
        from .audit.run import ptrue_summary
        pi = ptrue_summary(preds[blocked])
        po = ptrue_summary(orig)
        bluff = M.core(preds[blocked])["bluff_rate"]
        h["H2"] = {"claim": "P(True)-based confident-wrong fails to detect H1",
                   "evidence": {"naive_bluff_rate": bluff, "naive_confident_wrong": pi, "original_confident_wrong": po},
                   "verdict": "supported" if (pi["confident_wrong_tau0.8"] is not None and bluff is not None
                                               and pi["confident_wrong_tau0.8"] < 0.5 * bluff) else "not supported"}
    ok, ds = _all_sig(comps, "A_vs_B", "cost", -1)
    h["H3"] = {"claim": "Repair loop A lowers cost more than random control B (every seed, paired CI excludes 0)",
               "evidence": ds, "verdict": "untested" if ok is None else ("supported" if ok else "not supported")}
    ok, ds = _all_sig(comps, "A_vs_C", "over_conservativeness", -1)
    h["H4"] = {"claim": "A has lower over-conservativeness than prevention C (every seed)",
               "evidence": ds, "verdict": "untested" if ok is None else ("supported" if ok else "not supported")}
    h["H5"] = {"claim": "Gate self-controls behave as expected", "evidence": "see _e_gate.json", "verdict": "see E-GATE"}
    cats = {c: M.metric_value(orig, "prudence", category=c) for c in M.CATEGORIES}
    worst = min((v, c) for c, v in cats.items() if v is not None) if any(v is not None for v in cats.values()) else (None, None)
    h["H6"] = {"claim": "Bluffing differs by category; misleading evidence (U5) is hardest",
               "evidence": {"original_prudence_by_category": cats, "hardest": worst[1]},
               "verdict": "supported" if worst[1] == "U5" else f"partly: hardest is {worst[1]}"}
    h["H7"] = {"claim": "Repair gains transfer to an unseen domain (CUAD) without retraining", "evidence": "see _e_domain.json", "verdict": "see E-DOMAIN"}
    h["H8"] = {"claim": "Same recipe works on other permitted Cohere models", "evidence": None,
               "verdict": "not run (scope: one model first, by instruction)"}
    return h


def e_cat(J, probe_set="squad3v", split="test", names=None):
    names = names or ["original", "naive_s31"]
    rows = {}
    for n in names:
        try:
            p = _get_preds(probe_set, n, split)
        except Exception:
            continue
        b = M.breakdown(p)
        rows[n] = {"by_category": b["by_category"],
                   "by_context": {c: {m: v.get(m) for m in ("accuracy", "prudence", "unnecessary_abstention")} for c, v in b["by_context"].items()}}
    out = path("audits", cfg()) / probe_set / f"_e_cat_{split}.json"
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    hub.push(out, f"v3/audits/{probe_set}/{out.name}", cfg())
    return {n: {c: round(v["prudence"], 3) for c, v in r["by_category"].items()} for n, r in rows.items()}


def e_external(J, probe_set, names, split="test", judged=False):
    """E-LANG / E-DOMAIN tables (per language for XQuAD)."""
    from .audit.run import read_predictions
    out = {}
    K = None
    for n in names:
        try:
            p = _get_preds(probe_set, n, split)
        except Exception:
            continue
        if judged and (_audit_dir(probe_set, n, split) / "predictions_judged.jsonl").exists():
            p = read_predictions(_audit_dir(probe_set, n, split) / "predictions_judged.jsonl")
        if n == "original":
            K = G.known_ids(p)
        b = M.breakdown(p, K)
        out[n] = {"overall": b["overall"], "by_language": b["by_language"], "by_category": b["by_category"]}
    f = path("audits", cfg()) / probe_set / f"_e_{probe_set}.json"
    f.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    hub.push(f, f"v3/audits/{probe_set}/{f.name}", cfg())
    return {n: {m: v["overall"].get(m) for m in ("accuracy", "prudence", "over_conservativeness")} for n, v in out.items()}


def e_repro(J, probe_set="squad3v", split="val", names=None):
    """Recompute every stored gate from saved predictions and check the decision is identical."""
    names = names or []
    res = {}
    orig = _get_preds(probe_set, "original", split)
    for n in names:
        d = _audit_dir(probe_set, n, split)
        for gf in d.glob("gate_*.json"):
            stored = json.loads(gf.read_text())
            pname = stored["profile"]["name"]
            g = G.evaluate(orig, _get_preds(probe_set, n, split), get_profile(pname, cfg()))
            res[f"{n}/{gf.name}"] = {"stored": stored["decision"], "recomputed": g["decision"],
                                     "same_rules": stored["failed_rules"] == g["failed_rules"]}
    out = path("audits", cfg()) / probe_set / "_e_repro.json"
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    hub.push(out, f"v3/audits/{probe_set}/_e_repro.json", cfg())
    return res


def rerun_determinism(J, probe_set="squad3v", name="original", split="val", n=200):
    """GPU part of E-REPRO: re-generate the first n items and compare with the saved predictions."""
    from .audit.generate import load_model, generate, free
    from .prompts import build_prompt
    from .probes.squad import load_items
    from .experiments import _probe_dir
    items = load_items(_probe_dir(probe_set), split)[:n]
    saved = {r["item_id"]: r["raw"] for r in _get_preds(probe_set, name, split)}
    model, tok = load_model(cfg=cfg())
    outs = generate(model, tok, [build_prompt(it["passages"], it["question"]) for it in items],
                    max_new_tokens=cfg()["generation"]["max_new_tokens"], batch_size=cfg()["generation"]["batch_size"])
    free(model)
    same = sum(o[0] == saved.get(it["item_id"]) for it, o in zip(items, outs))
    from .prompts import parse_output
    same_action = sum(parse_output(o[0])[0] == parse_output(saved.get(it["item_id"], ""))[0] for it, o in zip(items, outs))
    return {"n": len(items), "identical_raw": same, "identical_action": same_action}
