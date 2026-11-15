"""E-REPAIR arms (plan §9.3). All arms start from the same blocked naive adapter.

A  HUMBLE loop      repair data from the model's own mistakes (§9.1), 1-3 rounds
B  random control   same rounds, same sizes and target ratio, random probe-pool items, same replay and steps
C  prevention       fresh honest SFT from the original (experiments.train_prevention)
D  lambda scaling   training-free, AlignTune merge; largest lambda that passes the gate
"""
from __future__ import annotations

import json
import os
import random as _r
import shutil

from . import hub
from .config import path
from .experiments import (_audit_dir, _get_adapter, _get_preds, _probe_dir, _push_adapter, audit_candidate, cfg)


def _load_split(probe_set, split):
    from .probes.squad import load_items
    return load_items(_probe_dir(probe_set), split)


def _repair_context(probe_set):
    from .probes.squad import load_items, load_corpus
    from .probes.corpus import wrong_answer_pool
    pdir = _probe_dir(probe_set)
    probe = load_items(pdir, "probe")
    corpus = load_corpus(pdir, "probe")
    replay = [it for it in load_items(pdir, "train") if it["label"] == "answer"]
    wp = wrong_answer_pool([it["source"] for it in probe if it["source"].get("answers")])
    return pdir, probe, corpus, replay, wp


def _maybe_judge_filter(examples, J):
    """Verify generated ABSTAIN variants with the sufficient-context judge when it is available."""
    from . import judge as JG
    try:
        jd = JG.Judge(cfg()["judge"]["model_id"], token=os.environ.get("HF_TOKEN"))
    except Exception as e:
        J.log(f"judge unavailable for repair-variant verification: {e!r}")
        return examples, {"judge": "unavailable", "error": repr(e)[:200]}
    gen_roles = {"delete_fact", "misleading", "raft_no_gold_flipped"}
    cand = [e for e in examples if e["role"] in gen_roles]
    keep = JG.verify(cand, jd.sufficiency(cand)) if cand else []
    jd.free()
    drop = {id(e) for e, k in zip(cand, keep) if not k}
    return [e for e in examples if id(e) not in drop], {"judge": cfg()["judge"]["model_id"], "checked": len(cand),
                                                        "dropped": len(drop)}


def _sft_repair(examples, name, init_adapter, seed, J):
    from .curate import curate_sft
    from .repair import to_sft_examples
    from . import train as T
    msgs, cs = curate_sft(to_sft_examples(examples), path("curated", cfg()) / name, name)
    J.log(f"{name} curated {cs}")
    out = path("adapters", cfg()) / name
    adir, info = T.sft(msgs, out, init_adapter=init_adapter, seed=seed, log=J.log)
    hub.push(path("curated", cfg()) / name, f"v3/curated/{name}", cfg())
    _push_adapter(out, name)
    return adir, cs, info


def _weights(prof):
    return {x: prof[x] for x in ("w_bluff", "w_refuse", "w_wrong")}


def _val_audit(probe_set, name, adapter, K, w, J, seed=0, merged=None):
    from .audit.generate import load_model, free
    from .audit.run import audit
    model, tok = load_model(adapter=adapter, base=merged, cfg=cfg())
    vrecs, vm = audit(model, tok, _load_split(probe_set, "val"), _audit_dir(probe_set, name, "val"), name, "val",
                      adapter=adapter, known_ids=K, weights=w, cfg=cfg(), log=J.log, seed=seed, ptrue=True)
    free(model)
    hub.push(_audit_dir(probe_set, name, "val"), f"v3/audits/{probe_set}/{name}/val", cfg())
    return vrecs, vm


def arm_A(J, seed=31, probe_set="squad3v", blocked="naive_s31", profile="support", tag="A", twins=True, two_way_dpo=None, replay_ratio=None,
          cat_weights=None):
    """HUMBLE repair loop from the blocked adapter (§9.1).

    Fix 2: cat_weights={"U3": 2, "U4": 2, "U5": 2} with tag="Acat" upweights the weak categories.
    """
    from .audit.generate import load_model, free
    from .audit.run import audit
    from .audit import metrics as M
    from . import gate as G, repair as R
    from .config import profile as get_profile
    prof = get_profile(profile, cfg())
    rc = cfg()["repair"]
    k = cfg()["prompt"]["passages_default"]
    pdir, probe, corpus, replay, wp = _repair_context(probe_set)
    by_id = {it["item_id"]: it for it in probe}
    slices = R.probe_slices(probe, rc["max_rounds"], seed)
    orig_val = _get_preds(probe_set, "original", "val")
    K = G.known_ids(orig_val)
    w = _weights(prof)
    current, current_name = _get_adapter(blocked), blocked
    current_adapter = blocked
    history, prev_val = [], _get_preds(probe_set, blocked, "val")
    rng = _r.Random(seed)
    for r in range(rc["max_rounds"]):
        name = f"rep{tag}_s{seed}_r{r + 1}"
        model, tok = load_model(adapter=current, cfg=cfg())
        precs, _ = audit(model, tok, slices[r], _audit_dir(probe_set, name, "probe"), current_name, f"probe_r{r + 1}",
                         samples=cfg()["generation"]["diagnose_samples"], ptrue=False, cfg=cfg(), log=J.log, seed=seed)
        free(model)
        hub.push(_audit_dir(probe_set, name, "probe"), f"v3/audits/{probe_set}/{name}/probe", cfg())
        diag = R.diagnose(precs, cfg()["generation"]["bluff_majority"])
        J.log(f"round {r + 1} diagnosis: bluffs {len(diag['bluffs'])} {diag['bluffs_by_category']}, "
              f"over-refusals {len(diag['over_refusals'])}, wrong {len(diag['wrong'])}")
        dsum = {key: (len(v) if isinstance(v, list) else v) for key, v in diag.items()}
        if not diag["bluffs"] and not diag["over_refusals"]:
            history.append({"round": r + 1, "diagnosis": dsum, "stopped": "no mistakes found"})
            break
        ex, counts = R.build_repair_set(diag, by_id, corpus, rng, k, wp, replay, rc["replay_ratio"] if replay_ratio is None else replay_ratio, twins=twins,
                                       cat_weights=cat_weights)
        ex, jinfo = _maybe_judge_filter(ex, J)
        R.save_json({"diagnosis": diag, "counts": counts, "judge": jinfo}, path("curated", cfg()) / name / "repair_build.json")
        n_abs, n_ans, n_rep = R.target_counts(ex)
        adir, cs, info = _sft_repair(ex, name, current, seed, J)
        adapter_name = name
        if (rc.get("two_way_dpo") if two_way_dpo is None else two_way_dpo):
            from .curate import curate_dpo
            from . import train as T
            dfile, ds = curate_dpo(R.two_way_pairs(ex, rng), path("curated", cfg()) / f"{name}_dpo", f"{name}_dpo")
            adir, dinfo = T.dpo(dfile, path("adapters", cfg()) / f"{name}_dpo", init_adapter=adir, seed=seed, log=J.log)
            info["two_way_dpo"] = dinfo
            adapter_name = f"{name}_dpo"  # the val audit below uses this adapter; tests must use it too
            _push_adapter(adir, adapter_name)
        vrecs, vm = _val_audit(probe_set, name, adir, K, w, J, seed)
        g = G.evaluate(orig_val, vrecs, prof)
        G.write(g, _audit_dir(probe_set, name, "val") / f"gate_{prof['name']}.json")
        hub.push(_audit_dir(probe_set, name, "val") / f"gate_{prof['name']}.json",
                 f"v3/audits/{probe_set}/{name}/val/gate_{prof['name']}.json", cfg())
        imp = M.paired_diff(prev_val, vrecs, "cost", K, w)
        history.append({"round": r + 1, "adapter": name, "adapter_used": adapter_name, "diagnosis": dsum, "counts": counts,
                        "targets": {"abstain": n_abs, "answer": n_ans, "replay": n_rep}, "curation": cs, "train": info,
                        "judge": jinfo, "gate": g["decision"], "failed_rules": g["failed_rules"],
                        "val": {m: vm["overall"].get(m) for m in ("accuracy", "prudence", "over_conservativeness", "cost", "honesty")},
                        "cost_change_vs_previous": imp})
        J.log(f"round {r + 1}: gate {g['decision']} {g['failed_rules']} cost change {imp['diff']} CI {imp['ci95']}")
        current, current_name, current_adapter, prev_val = adir, name, adapter_name, vrecs
        if g["decision"] == "PASS":
            break
        if not (imp["diff"] is not None and imp["diff"] < 0 and imp["ci95"][1] is not None and imp["ci95"][1] < 0):
            history[-1]["stopped"] = "cost did not improve beyond its CI"
            break
    res = {"arm": tag, "seed": seed, "blocked": blocked, "final": current_name, "final_adapter": current_adapter, "twins": twins, "replay_ratio": rc["replay_ratio"] if replay_ratio is None else replay_ratio,
           "two_way_dpo": bool(rc.get("two_way_dpo") if two_way_dpo is None else two_way_dpo), "cat_weights": cat_weights,
           "rounds": history}
    R.save_json(res, path("adapters", cfg()) / f"arm{tag}_s{seed}.json")
    hub.push(path("adapters", cfg()) / f"arm{tag}_s{seed}.json", f"v3/arms/arm{tag}_s{seed}.json", cfg())
    return {"final": current_name, "rounds": [{key: h.get(key) for key in ("round", "gate", "failed_rules", "targets", "val", "stopped")}
                                              for h in history]}


def arm_B(J, seed=31, probe_set="squad3v", blocked="naive_s31", profile="support", tag="B", match="A"):
    """Random control matched to arm `match`'s rounds, sizes, target ratio, replay and steps.
    Defaults reproduce the published names (armB_s<seed>.json, ctlB_s<seed>_r<n>); a new experiment sets its own
    tag (e.g. tag="Blight", match="Alight") so it never overwrites those files."""
    from . import gate as G, repair as R
    from .config import profile as get_profile
    prof = get_profile(profile, cfg())
    hub.pull([f"v3/arms/arm{match}_s{seed}.json"], path("adapters", cfg()).parent / "_hub")
    a = json.loads((path("adapters", cfg()).parent / "_hub" / "v3" / "arms" / f"arm{match}_s{seed}.json").read_text())
    if a.get("blocked", blocked) != blocked:
        raise ValueError(f"arm{match}_s{seed} repaired {a.get('blocked')}, not {blocked}: the control must start from the same adapter")
    pdir, probe, corpus, replay, wp = _repair_context(probe_set)
    orig_val = _get_preds(probe_set, "original", "val")
    K = G.known_ids(orig_val)
    w = _weights(prof)
    rng = _r.Random(1000 + seed)
    current, hist, name = _get_adapter(blocked), [], blocked
    for h in [h for h in a["rounds"] if "targets" in h]:
        t = h["targets"]
        name = f"ctl{tag}_s{seed}_r{h['round']}"
        ex, counts = R.random_control(t["abstain"], t["answer"], probe, rng, replay, t["replay"])
        adir, cs, info = _sft_repair(ex, name, current, seed, J)
        current = adir
        hist.append({"round": h["round"], "matched_targets": t, "counts": counts, "curation": cs, "train": info})
    vrecs, vm = _val_audit(probe_set, name, current, K, w, J, seed)
    g = G.evaluate(orig_val, vrecs, prof)
    G.write(g, _audit_dir(probe_set, name, "val") / f"gate_{prof['name']}.json")
    res = {"arm": tag, "matched_to": match, "seed": seed, "blocked": blocked, "final": name, "rounds": hist, "gate": g["decision"], "failed_rules": g["failed_rules"]}
    R.save_json(res, path("adapters", cfg()) / f"arm{tag}_s{seed}.json")
    hub.push(path("adapters", cfg()) / f"arm{tag}_s{seed}.json", f"v3/arms/arm{tag}_s{seed}.json", cfg())
    return {"final": name, "gate": g["decision"], "failed_rules": g["failed_rules"],
            "val": {m: vm["overall"].get(m) for m in ("accuracy", "prudence", "over_conservativeness", "cost")}}


def arm_D(J, probe_set="squad3v", blocked="naive_s31", profile="support", lambdas=None, tag="D"):
    """Training-free fallback: lambda scaling of the blocked adapter via AlignTune merge; largest passing lambda."""
    from . import gate as G, train as T
    from .config import profile as get_profile
    prof = get_profile(profile, cfg())
    orig_val = _get_preds(probe_set, "original", "val")
    K = G.known_ids(orig_val)
    w = _weights(prof)
    adir = _get_adapter(blocked)
    rows = []
    for lam in sorted(lambdas or cfg()["repair"]["lambdas"], reverse=True):
        name = f"lamD_{lam}"
        mdir = path("adapters", cfg()) / f"merged_{blocked}_{lam}"
        if lam == 0.0:
            vrecs, vm = _val_audit(probe_set, name, None, K, w, J)
        elif lam == 1.0:
            vrecs, vm = _val_audit(probe_set, name, adir, K, w, J)
        else:
            T.scale(adir, lam, mdir)
            vrecs, vm = _val_audit(probe_set, name, None, K, w, J, merged=str(mdir))
        g = G.evaluate(orig_val, vrecs, prof)
        rows.append({"lambda": lam, "gate": g["decision"], "failed_rules": g["failed_rules"],
                     "val": {m: vm["overall"].get(m) for m in ("accuracy", "prudence", "over_conservativeness", "cost")}})
        J.log(f"lambda {lam}: {g['decision']} {g['failed_rules']}")
    passing = [r for r in rows if r["gate"] == "PASS"]
    chosen = max(passing, key=lambda r: r["lambda"])["lambda"] if passing else None
    for r in rows:
        d = path("adapters", cfg()) / f"merged_{blocked}_{r['lambda']}"
        if r["lambda"] != chosen and d.exists():
            shutil.rmtree(d, ignore_errors=True)
    res = {"arm": "D", "rows": rows, "chosen_lambda": chosen,
           "note": "lambda=0 is the original model; choosing 0 means scaling could not keep any of the fine-tune"}
    (path("adapters", cfg()) / f"arm{tag}.json").write_text(json.dumps(res, indent=2))
    hub.push(path("adapters", cfg()) / f"arm{tag}.json", f"v3/arms/arm{tag}.json", cfg())
    return res


def test_once(J, name, adapter=None, probe_set="squad3v", merged_lambda=None, blocked="naive_s31"):
    """Evaluate a candidate on the locked test exactly once."""
    merged = None
    if merged_lambda is not None and merged_lambda not in (0.0, 1.0):
        from . import train as T
        merged = str(T.scale(_get_adapter(blocked), merged_lambda, path("adapters", cfg()) / f"merged_{blocked}_{merged_lambda}"))
    elif merged_lambda == 1.0:
        adapter = blocked
    return audit_candidate(J, name, adapter=adapter, split="test", probe_set=probe_set, merged=merged)
