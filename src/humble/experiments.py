"""Experiment jobs (plan §12). Each takes a hub.Job `J` for logging/status and is resumable.

Run on a GPU notebook via tools/nbrun.sh, or locally with `python -m humble.experiments <job>`.
Outputs live under HUMBLE_HOME and are mirrored to the HF repo under v3/.
"""
from __future__ import annotations

import json
import os
import tarfile
from pathlib import Path

from . import hub
from .config import load_config, path

CFG = None


def cfg():
    global CFG
    if CFG is None:
        CFG = load_config()
        _check_model_repo(CFG)
    return CFG


def _check_model_repo(c):
    """Audit names (original, naive_s31, ...) are not namespaced by model, so every base model needs its own
    Hub repo; otherwise a run could silently gate against another model's `original` audit."""
    from .hub import read_repos
    tiny = lambda r: r in ("KM2007/humble-runs", "KM2007/humble-v3") or r.startswith("KM2007/humble-v3-")
    if c["model"]["base_id"] != "CohereLabs/tiny-aya-global" and any(tiny(r) for r in read_repos(c)):
        raise RuntimeError("base model is not Tiny Aya Global but the Hub repos are the Tiny Aya ones: set "
                           "HUMBLE_HF_REPO=<you>/humble-<model>, HUMBLE_LEGACY_REPO=none, HUMBLE_PER_NODE=false")


# ------------------------------------------------------------------ environment
def env_check(J):
    import torch, transformers, trl, peft
    import aligntune, curatorkit, auditkit
    from .audit.generate import load_model, stop_token_ids, generate
    from .prompts import build_prompt
    model, tok = load_model(cfg=cfg())
    stops = stop_token_ids(tok)
    out = generate(model, tok, [build_prompt(["Paris is the capital of France."], "What is the capital of France?"),
                                build_prompt(["Paris is the capital of France."], "What is the capital of Peru?")])
    info = {"torch": torch.__version__, "transformers": transformers.__version__, "trl": trl.__version__,
            "peft": peft.__version__, "aligntune": getattr(aligntune, "__version__", "?"),
            "curatorkit": curatorkit.__version__, "auditkit": auditkit.__version__,
            "stop_ids": stops, "stop_tokens": tok.convert_ids_to_tokens(stops), "outputs": out,
            "chat_template_sample": tok.apply_chat_template([{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}], tokenize=False)}
    J.log(json.dumps(info, ensure_ascii=False))
    return info


# ------------------------------------------------------------------ probes
def _used_paragraph_dirs():
    """Folders of earlier iterations whose paragraphs must not appear in the new val/test."""
    base = Path("/marimo/humble")
    from huggingface_hub import hf_hub_download
    for name, sub in (("real_bundle", "real"), ("real2_bundle", "real2")):
        if not (base / sub / "documents").exists():
            b = hf_hub_download("KM2007/humble-runs", f"bundles/{name}.tar.gz", repo_type="dataset",
                                token=os.environ.get("HF_TOKEN"), local_dir=str(base / "_bundle"))
            with tarfile.open(b, "r:gz") as t:
                t.extractall(base)
    if not (base / "external").exists():
        hub.pull(["external/**"], base)
    return [base / "real", base / "real2", base / "external"]


def build_squad_probes(J, probe_set="squad3v", reuse_descriptors_from="squad3", judge_model=None):
    """Build the verified SQuAD probe set: mechanical construction + sufficient-context judge (§6.4)."""
    from .audit.generate import load_model, free
    from .probes import squad, describe
    from . import judge as JG
    out = path("probes", cfg()) / probe_set
    prev = path("probes", cfg()) / reuse_descriptors_from / "descriptors.json"
    if reuse_descriptors_from and not prev.exists():
        try:
            _probe_dir(reuse_descriptors_from)
        except Exception:
            pass
    if prev.exists() and not (out / "descriptors.json").exists():
        out.mkdir(parents=True, exist_ok=True)
        (out / "descriptors.json").write_bytes(prev.read_bytes())
    if (out / "MANIFEST.json").exists():
        J.log("probe set exists; skipping build")
        return json.loads((out / "MANIFEST.json").read_text())
    used = squad.used_paragraph_texts(_used_paragraph_dirs())
    J.log(f"excluding {len(used)} paragraphs used by earlier iterations")
    df = squad.load_squad_frames()
    titles = sorted(set(df["title"]))
    desc_file = out / "descriptors.json"
    if desc_file.exists():
        desc = json.loads(desc_file.read_text(encoding="utf-8"))
    else:
        model, tok = load_model(cfg=cfg())
        desc = describe.describe(titles, model, tok, log=J.log)
        free(model)
        out.mkdir(parents=True, exist_ok=True)
        desc_file.write_text(json.dumps(desc, indent=1, ensure_ascii=False), encoding="utf-8")
    J.log(f"descriptors for {len(desc)}/{len(titles)} articles")
    jm = judge_model or cfg()["judge"]["model_id"]
    judge = JG.Judge(jm, token=os.environ.get("HF_TOKEN"))  # raises if the judge model is gated: no silent fallback
    J.log(f"judge loaded: {jm}")

    def verify_fn(items):
        return JG.verify(items, judge.sufficiency(items))

    m = squad.build(out, cfg(), used_texts=used, descriptors=desc, df=df, probe_set=probe_set, verify_fn=verify_fn)
    m["judge_model"] = jm
    (out / "MANIFEST.json").write_text(json.dumps(m, indent=1, ensure_ascii=False), encoding="utf-8")
    judge.free()
    hub.push(out, f"v3/probes/{probe_set}", cfg())
    return {"cells": m["cells"], "failures": m["failures"], "test_lock": m["test_lock_sha256"]}


# ------------------------------------------------------------------ smoke test of the training path
def build_u45_probes(J, probe_set="squad3u45", n=150, judge_model=None):
    """Fix 2: larger U4/U5 test set (>= 100 judged items each) from articles outside squad3v val/test.
    Never touches squad3v. Evaluate with audit_candidate(J, name, split="test", probe_set="squad3u45")."""
    from .probes import squad
    from . import judge as JG
    out = path("probes", cfg()) / probe_set
    if (out / "MANIFEST.json").exists():
        J.log("probe set exists; skipping build")
        return json.loads((out / "MANIFEST.json").read_text())
    excl = [_probe_dir("squad3"), _probe_dir("squad3v")]
    desc = json.loads((excl[1] / "descriptors.json").read_text(encoding="utf-8"))
    used = squad.used_paragraph_texts(_used_paragraph_dirs())
    jm = judge_model or cfg()["judge"]["model_id"]
    judge = JG.Judge(jm, token=os.environ.get("HF_TOKEN"))
    J.log(f"judge loaded: {jm}")
    m = squad.build_u45(out, cfg(), desc, excl, probe_set=probe_set, n=n, used_texts=used,
                        verify_fn=lambda items: JG.verify(items, judge.sufficiency(items)))
    m["judge_model"] = jm
    (out / "MANIFEST.json").write_text(json.dumps(m, indent=1, ensure_ascii=False), encoding="utf-8")
    judge.free()
    for cell, c in m["cells"].items():
        if c["built"] < 100:
            J.log(f"WARNING {cell}: only {c['built']} judged items (< 100); raise n or overgenerate")
    hub.push(out, f"v3/probes/{probe_set}", cfg())
    return {"cells": m["cells"], "judge": m.get("judge"), "test_lock": m["test_lock_sha256"]}


def smoke_train(J):
    """Tiny end-to-end check: CuratorKIT -> AlignTune SFT -> continue adapter -> DPO -> merge vs LoRA scale."""
    import torch
    from .curate import curate_sft, curate_dpo
    from .prompts import build_prompt
    from . import train as T
    from .audit.generate import load_model, generate, free
    w = path("adapters", cfg()) / "_smoke"
    ex = []
    for i in range(48):
        ex.append({"instruction": build_prompt([f"The code word number {i} is BANANA{i}."], f"What is code word number {i}?"),
                   "input": "", "output": f"ANSWER: BANANA{i}"})
        ex.append({"instruction": build_prompt([f"The code word number {i} is BANANA{i}."], f"What is the colour of item {i}?"),
                   "input": "", "output": "ABSTAIN"})
    msgs, s = curate_sft(ex, w / "data", "smoke")
    J.log(f"curated {s}")
    a1, i1 = T.sft(msgs, w / "sft", max_steps=6, log=J.log)
    a2, i2 = T.sft(msgs, w / "sft_cont", init_adapter=a1, max_steps=3, log=J.log)
    model, tok = load_model(adapter=a2, cfg=cfg())
    outs = generate(model, tok, [ex[0]["instruction"], ex[1]["instruction"]], max_new_tokens=20)
    raw = tok.decode(model.generate(**tok(tok.apply_chat_template([{"role": "user", "content": ex[0]["instruction"]}], tokenize=False, add_generation_prompt=True), add_special_tokens=False, return_tensors="pt").to("cuda"), max_new_tokens=20, do_sample=False)[0][-24:])
    free(model)
    dfile, ds = curate_dpo([{"prompt": ex[1]["instruction"], "chosen": "ABSTAIN", "rejected": "ANSWER: BANANA0"},
                            {"prompt": ex[0]["instruction"], "chosen": "ANSWER: BANANA0", "rejected": "ABSTAIN"}] * 8, w / "dpo_data", "smoke_dpo")
    try:
        a3, i3 = T.dpo(dfile, w / "dpo", init_adapter=a2, log=J.log)
        dpo_ok = True
    except Exception as e:
        dpo_ok = repr(e)[:800]
    try:
        merged = T.scale(a2, 0.5, w / "merged_0.5")
        lora_half = T.scale_lora(a2, 0.5, w / "lora_0.5")
        m1, tok1 = load_model(base=str(merged), cfg=cfg())
        p = tok1(ex[0]["instruction"], return_tensors="pt").to("cuda")
        with torch.no_grad():
            l1 = m1(**p).logits[0, -1].float()
        free(m1)
        m2, _ = load_model(adapter=lora_half, cfg=cfg())
        with torch.no_grad():
            l2 = m2(**p).logits[0, -1].float()
        free(m2)
        merge_ok = {"max_abs_logit_diff": float((l1 - l2).abs().max()), "argmax_equal": bool(l1.argmax() == l2.argmax())}
    except Exception as e:
        merge_ok = repr(e)[:800]
    return {"curation": s, "sft": i1, "sft_cont": i2, "outputs_after_training": outs, "raw_tail": raw,
            "dpo": dpo_ok, "merge_vs_lora_scale": merge_ok}


# ------------------------------------------------------------------ candidates
def _train_items(probe_set):
    from .probes.squad import load_items
    pdir = _probe_dir(probe_set)
    return load_items(pdir, "train")


def _probe_dir(probe_set):
    d = path("probes", cfg()) / probe_set
    if not (d / "MANIFEST.json").exists():
        hub.pull([f"v3/probes/{probe_set}/*"], path("probes", cfg()).parent / "_hub")
        src = path("probes", cfg()).parent / "_hub" / "v3" / "probes" / probe_set
        d.mkdir(parents=True, exist_ok=True)
        for f in src.iterdir():
            (d / f.name).write_bytes(f.read_bytes())
    return d


def _push_adapter(adir, name):
    hub.push(adir, f"v3/adapters/{name}", cfg())


def _get_adapter(name):
    """Local adapter dir for `name`, pulled from the Hub if needed."""
    local = path("adapters", cfg()) / name / "adapter"
    if (local / "adapter_config.json").exists():
        return local
    hub.pull([f"v3/adapters/{name}/**"], path("adapters", cfg()).parent / "_hub")
    src = path("adapters", cfg()).parent / "_hub" / "v3" / "adapters" / name / "adapter"
    import shutil
    if not (src / "adapter_config.json").exists():
        raise FileNotFoundError(f"adapter {name} not on the Hub yet")
    shutil.rmtree(local, ignore_errors=True)
    shutil.copytree(src, local)
    return local


def train_naive(J, probe_set="squad3", seed=31, name=None, max_steps=None, lr=None):
    """The candidate a company would ship: answer-only SFT on the train pool (no ABSTAIN targets).

    `max_steps` / `lr` train a lighter, only partly broken naive model (Fix 1: targeted vs random repair when the
    damage is partial); give it its own `name` (e.g. naive_light_s31) so it never collides with naive_s<seed>.
    The default call is unchanged and still trains naive_s<seed>."""
    from .curate import curate_sft, examples_from_items
    from .prompts import target_text
    from . import train as T
    if name is None:
        if max_steps or lr is not None:
            raise ValueError("a lighter naive model needs its own name, e.g. name='naive_light_s31'")
        name = f"naive_s{seed}"
    out = path("adapters", cfg()) / name
    if (out / "adapter" / "adapter_config.json").exists():
        return {"skipped": name}
    items = [it for it in _train_items(probe_set) if it["label"] == "answer"]
    msgs, cs = curate_sft(examples_from_items(items, target_text), path("curated", cfg()) / name, name)
    J.log(f"curated {cs}")
    adir, info = T.sft(msgs, out, seed=seed, max_steps=max_steps, lr=lr, log=J.log)
    info.update({"max_steps": max_steps, "lr_override": lr})
    (out / "train_info.json").write_text(json.dumps(info, indent=2, default=str), encoding="utf-8")
    hub.push(path("curated", cfg()) / name / "curated", f"v3/curated/{name}", cfg())
    _push_adapter(out, name)
    return {"adapter": name, "curation": cs, "train": info}


def train_prevention(J, probe_set="squad3", seed=31):
    """Arm C: the v2 recipe in RAG format, fresh honest SFT from the original (answerable + 30% unanswerable)."""
    from .curate import curate_sft, examples_from_items
    from .prompts import target_text
    from . import train as T
    name = f"prevC_s{seed}"
    out = path("adapters", cfg()) / name
    if (out / "adapter" / "adapter_config.json").exists():
        return {"skipped": name}
    items = _train_items(probe_set)
    msgs, cs = curate_sft(examples_from_items(items, target_text), path("curated", cfg()) / name, name)
    J.log(f"curated {cs}")
    adir, info = T.sft(msgs, out, seed=seed, log=J.log)
    hub.push(path("curated", cfg()) / name / "curated", f"v3/curated/{name}", cfg())
    _push_adapter(out, name)
    return {"adapter": name, "curation": cs, "train": info,
            "unanswerable_share": sum(it["label"] == "abstain" for it in items) / len(items)}


# ------------------------------------------------------------------ audits and gates
def _audit_dir(probe_set, name, split):
    return path("audits", cfg()) / probe_set / name / split


def _get_preds(probe_set, name, split):
    from .audit.run import read_predictions
    d = _audit_dir(probe_set, name, split)
    if not (d / "predictions.jsonl").exists():
        hub.pull([f"v3/audits/{probe_set}/{name}/{split}/*"], path("audits", cfg()).parent / "_hub")
        src = path("audits", cfg()).parent / "_hub" / "v3" / "audits" / probe_set / name / split
        d.mkdir(parents=True, exist_ok=True)
        for f in src.iterdir():
            (d / f.name).write_bytes(f.read_bytes())
    return read_predictions(d / "predictions.jsonl")


def audit_candidate(J, name, adapter=None, split="val", probe_set="squad3v", samples=0, ptrue=True, merged=None,
                    items=None, seed=0):
    """Audit one candidate (adapter name on the Hub, a merged model dir, or None = original) on one split."""
    from .audit.generate import load_model, free
    from .audit.run import audit
    from .gate import known_ids
    from .probes.squad import load_items
    out = _audit_dir(probe_set, name, split)
    if (out / "metrics.json").exists() and (split != "test" or (out / "DONE").exists()):
        if not hub.exists_remote(f"v3/audits/{probe_set}/{name}/{split}/metrics.json", cfg()):
            hub.push(out, f"v3/audits/{probe_set}/{name}/{split}", cfg())  # resumed job: finish the upload
        return {"skipped": str(out)}
    if split == "test" and hub.exists_remote(f"v3/audits/{probe_set}/{name}/test/DONE", cfg()):
        return {"skipped": "locked test already evaluated for this candidate (DONE marker on the Hub)"}
    pdir = _probe_dir(probe_set)
    items = items if items is not None else load_items(pdir, split)
    K, w = None, None
    if name != "original":
        try:
            K = known_ids(_get_preds(probe_set, "original", split))
        except Exception:
            K = None
    prof = cfg()["gate"]["profiles"][cfg()["gate"]["default_profile"]]
    w = {k: prof[k] for k in ("w_bluff", "w_refuse", "w_wrong")}
    adir = _get_adapter(adapter) if adapter else None
    model, tok = load_model(adapter=adir, base=merged, cfg=cfg())
    lock = (pdir / "TEST_LOCK.txt").read_text().strip() if (pdir / "TEST_LOCK.txt").exists() else None
    recs, mets = audit(model, tok, items, out, name, split, adapter=adir, probe_file=pdir / f"items_{split}.jsonl",
                       test_lock=lock if split == "test" else None, samples=samples, ptrue=ptrue, known_ids=K,
                       weights=w, cfg=cfg(), log=J.log, seed=seed)
    if name == "original":
        K = known_ids(recs)
        from .audit import metrics as M
        mets = M.with_cis(recs, K, w)
        (out / "metrics.json").write_text(json.dumps(mets, indent=2), encoding="utf-8")
    free(model)
    hub.push(out, f"v3/audits/{probe_set}/{name}/{split}", cfg())
    o = mets["overall"]
    return {k: o.get(k) for k in ("accuracy", "bluff_rate", "prudence", "over_conservativeness", "unnecessary_abstention",
                                  "honesty", "cost", "format_failure", "echo_rate")} | {"by_category": {c: v["prudence"] for c, v in mets["by_category"].items()}}


def gate_job(J, name, split="val", probe_set="squad3v", profile=None):
    from . import gate as G
    from .config import profile as get_profile
    prof = get_profile(profile, cfg())
    g = G.evaluate(_get_preds(probe_set, "original", split), _get_preds(probe_set, name, split), prof,
                   cfg()["gate"]["format_failure_max"], cfg()["metrics"]["bootstrap_resamples"])
    # AuditKIT paired significance as a cross-check
    try:
        from .audit.score import load_run, compare
        ra = load_run(_audit_dir(probe_set, "original", split) / "auditkit_run.json")
        rb = load_run(_audit_dir(probe_set, name, split) / "auditkit_run.json")
        g["auditkit_crosscheck"] = {m: compare(ra, rb, m) for m in ("honest", "abstain", "correct", "parse_error")}
    except Exception as e:
        g["auditkit_crosscheck"] = {"error": repr(e)}
    out = _audit_dir(probe_set, name, split) / f"gate_{prof['name']}.json"
    G.write(g, out)
    hub.push(out, f"v3/audits/{probe_set}/{name}/{split}/{out.name}", cfg())
    return {"decision": g["decision"], "failed_rules": g["failed_rules"], "reasons": g["reasons"]}


def e_gate(J, probe_set="squad3v", split="val", naive="naive_s31"):
    """E-GATE: gate v2 self-controls on the new probes + gate v2 rerun on all saved v2 predictions."""
    from . import gate as G
    from .config import profile as get_profile
    res = {"self_controls": {}, "v2_saved_predictions": {}}
    orig = _get_preds(probe_set, "original", split)
    nv = _get_preds(probe_set, naive, split)
    for pname in cfg()["gate"]["profiles"]:
        res["self_controls"][pname] = G.self_controls(orig, get_profile(pname, cfg()), naive_recs=nv)
    res["v2_saved_predictions"] = gate_v2_on_saved(J)
    out = path("audits", cfg()) / probe_set / "_e_gate.json"
    out.write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    hub.push(out, f"v3/audits/{probe_set}/_e_gate.json", cfg())
    return {p: {k: v.get("got") for k, v in d.items() if isinstance(v, dict)} | {"all_ok": d["all_ok"]}
            for p, d in res["self_controls"].items()} | {"v2": {k: v.get("decision") for k, v in res["v2_saved_predictions"].items()}}


def _from_v2(rec):
    """Map a v2 prediction record (single passage, question_type) onto the v3 schema."""
    lab = "answer" if rec["question_type"] == "answerable" else "abstain"
    return {"item_id": rec["item_id"], "doc_id": rec.get("org_id", rec["item_id"]), "label": lab,
            "category": "A" if lab == "answer" else "U3", "context_type": "C1", "language": "en",
            "action": rec["action"], "correct": bool(rec.get("correct")), "echo": False}


def gate_v2_on_saved(J):
    """Rerun gate v2 on every saved v2 real-text prediction file (original vs each candidate on the same split)."""
    from . import gate as G
    from .config import profile as get_profile
    base = path("audits", cfg()).parent / "_v2runs"
    hub.pull(["runs/*/real_test/predictions.jsonl", "runs/*/test2/predictions.jsonl", "runs/*/val2/predictions.jsonl",
              "runs/*/real_val/predictions.jsonl"], base)
    prof = get_profile("support", cfg())
    out = {}
    for split, orig_name in (("real_test", "real_original"), ("test2", "r2_original"), ("val2", "r2_original"), ("real_val", "real_original")):
        op = base / "runs" / orig_name / split / "predictions.jsonl"
        if not op.exists():
            continue
        orig = [_from_v2(json.loads(l)) for l in op.read_text(encoding="utf-8").splitlines() if l.strip()]
        for cand_dir in sorted((base / "runs").iterdir()):
            cp = cand_dir / split / "predictions.jsonl"
            if cand_dir.name == orig_name or not cp.exists():
                continue
            cand = [_from_v2(json.loads(l)) for l in cp.read_text(encoding="utf-8").splitlines() if l.strip()]
            try:
                g = G.evaluate(orig, cand, prof, resamples=500)
                out[f"{split}/{cand_dir.name}"] = {"decision": g["decision"], "failed_rules": g["failed_rules"],
                                                   "prudence": g["metrics"]["candidate"]["prudence"],
                                                   "over_conservativeness": g["metrics"]["candidate"]["over_conservativeness"],
                                                   "accuracy": g["metrics"]["candidate"]["accuracy"]}
            except Exception as e:
                out[f"{split}/{cand_dir.name}"] = {"error": repr(e)}
    return out


# ------------------------------------------------------------------ arms (implemented in arms.py)
def arm_A(J, **kw):
    from .arms import arm_A as f
    return f(J, **kw)


def arm_B(J, **kw):
    from .arms import arm_B as f
    return f(J, **kw)


def arm_D(J, **kw):
    from .arms import arm_D as f
    return f(J, **kw)


def test_once(J, name, **kw):
    from .arms import test_once as f
    return f(J, name, **kw)


# ------------------------------------------------------------------ external probe sets (E-LANG, E-DOMAIN)
def _judge_verify_fn(J):
    from . import judge as JG
    jm = cfg()["judge"]["model_id"]
    jd = JG.Judge(jm, token=os.environ.get("HF_TOKEN"))
    J.log(f"judge loaded: {jm}")
    return jd, (lambda items: JG.verify(items, jd.sufficiency(items)))


def build_external_probes(J, which="xquad"):
    from .probes import external as X
    out = path("probes", cfg()) / which
    if (out / "MANIFEST.json").exists():
        return json.loads((out / "MANIFEST.json").read_text(encoding="utf-8"))["cells"]
    jd, vf = _judge_verify_fn(J)
    m = X.build_xquad(out, cfg(), verify_fn=vf) if which == "xquad" else X.build_cuad(out, cfg(), verify_fn=vf)
    m["judge_model"] = jd.model_id
    (out / "MANIFEST.json").write_text(json.dumps(m, indent=1, ensure_ascii=False), encoding="utf-8")
    jd.free()
    hub.push(out, f"v3/probes/{which}", cfg())
    return {"cells": m["cells"], "judge": m.get("judge")}


def judge_rescore(J, probe_set, names, split="test", min_gold_words=4):
    """M9: re-grade answers on long-answer probes with the judge (equivalence) where token F1 says wrong."""
    from . import judge as JG
    from .audit import metrics as M
    from .gate import known_ids
    jd = JG.Judge(cfg()["judge"]["model_id"], token=os.environ.get("HF_TOKEN"))
    out = {}
    orig_K = None
    for name in names:
        recs = _get_preds(probe_set, name, split)
        todo = [r for r in recs if r["label"] == "answer" and r["action"] == "answer" and not r["correct"]
                and r["gold_answers"] and len(r["gold_answers"][0].split()) >= min_gold_words]
        p = jd.equivalence([r["question"] for r in todo], [r["gold_answers"][0] for r in todo], [r["answer"] for r in todo]) if todo else []
        flips = 0
        for r, v in zip(todo, p):
            r["judge_equiv"] = v
            if v >= 0.5:
                r["correct_f1"] = False
                r["correct"] = True
                flips += 1
        d = _audit_dir(probe_set, name, split)
        with open(d / "predictions_judged.jsonl", "w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        if name == "original":
            orig_K = known_ids(recs)
        mets = M.with_cis(recs, orig_K)
        (d / "metrics_judged.json").write_text(json.dumps(mets, indent=2), encoding="utf-8")
        hub.push(d, f"v3/audits/{probe_set}/{name}/{split}", cfg())
        out[name] = {"regraded": len(todo), "flipped_to_correct": flips, "accuracy_judged": mets["overall"]["accuracy"]}
    jd.free()
    return out


def regate_judged(J, probe_set, names, splits=("val", "test"), profile="support", min_gold_words=1):
    """For models that answer in full sentences: judge-grade every answer token F1 marked wrong (M9), then rebuild
    the original's known set and recompute each gate from the saved predictions. No new generation, so the
    test-once rule is untouched. Writes predictions_judged.jsonl, metrics_judged.json, gate_<profile>_judged.json."""
    from . import gate as G
    from .audit.run import read_predictions
    from .config import profile as get_profile
    prof = get_profile(profile, cfg())
    names = ["original"] + [n for n in names if n != "original"]
    out = {}
    for split in splits:
        have = [n for n in names if _has_split(probe_set, n, split)]
        out[f"{split}/rescore"] = judge_rescore(J, probe_set, have, split=split, min_gold_words=min_gold_words)
        orig = read_predictions(_audit_dir(probe_set, "original", split) / "predictions_judged.jsonl")
        for n in have:
            if n == "original":
                continue
            d = _audit_dir(probe_set, n, split)
            g = G.evaluate(orig, read_predictions(d / "predictions_judged.jsonl"), prof)
            g["grading"] = f"judge-regraded (min_gold_words={min_gold_words})"
            G.write(g, d / f"gate_{prof['name']}_judged.json")
            hub.push(d / f"gate_{prof['name']}_judged.json", f"v3/audits/{probe_set}/{n}/{split}/gate_{prof['name']}_judged.json", cfg())
            out[f"{split}/{n}"] = {"decision": g["decision"], "failed_rules": g["failed_rules"], "n_known": g.get("n_known")}
    return out


def _has_split(probe_set, name, split):
    try:
        _get_preds(probe_set, name, split)
        return True
    except Exception:
        return False


# ------------------------------------------------------------------ product mode: probes from company documents
def build_docs_probes(J, src="/marimo/humble/data/documents", n_docs=44, probe_set="docs_demo2"):
    """Demo company: full policy documents of the synthetic corpus (fictional organisations)."""
    import shutil
    from .audit.generate import load_model, free
    from .probes import docs as D
    from . import judge as JG
    out = path("probes", cfg()) / probe_set
    if (out / "MANIFEST.json").exists():
        return json.loads((out / "MANIFEST.json").read_text())["cells"]
    folder = path("probes", cfg()) / f"_{probe_set}_input"
    folder.mkdir(parents=True, exist_ok=True)
    full = sorted(p for p in Path(src).glob("*__full.md"))[:n_docs]
    for p in full:
        shutil.copy(p, folder / p.name)
    model, tok = load_model(cfg=cfg())
    jd = JG.Judge(cfg()["judge"]["model_id"], token=os.environ.get("HF_TOKEN"))
    judge_llm = D.LocalLLM(jd.model, jd.tok, cfg()["judge"]["model_id"] + "(local)")
    m = D.build(folder, out, cfg(), model, tok, verify_fn=lambda items: JG.verify(items, jd.sufficiency(items)),
                judge_llm=judge_llm, log=J.log)
    m["generator"] = cfg()["model"]["base_id"]
    m["judge_model"] = cfg()["judge"]["model_id"]
    (out / "MANIFEST.json").write_text(json.dumps(m, indent=1, ensure_ascii=False), encoding="utf-8")
    free(model)
    jd.free()
    hub.push(out, f"v3/probes/{probe_set}", cfg())
    return {"qa": m["qa"], "cells": m["cells"]}


def wait_hub(J, remote_file, every=60, max_hours=6):
    """Block a job chain until `remote_file` exists on the Hub (cross-notebook dependency)."""
    import time
    t0 = time.time()
    while not hub.exists_remote(remote_file, cfg()):
        if time.time() - t0 > max_hours * 3600:
            raise TimeoutError(remote_file)
        time.sleep(every)
    J.log(f"dependency ready: {remote_file}")
    return remote_file


def analysis(J, fn, **kw):
    from . import analysis as A
    return getattr(A, fn)(J, **kw)


# ------------------------------------------------------------------ E-JUDGE (needs human labels)
def export_judge_sheet(J, probe_set="squad3v", split="val", n=100):
    from . import judge as JG
    from .probes.squad import load_items
    items = load_items(_probe_dir(probe_set), split)
    out = path("probes", cfg()) / probe_set / f"judge_label_sheet_{split}.csv"
    JG.export_label_sheet(items, out, n=n)
    hub.push(out, f"v3/judge/{out.name}", cfg())
    return {"sheet": f"v3/judge/{out.name}", "n": n,
            "instructions": "fill human_sufficient with yes/no, upload as v3/judge/labels_filled_<split>.csv"}


def judge_agreement(J, probe_set="squad3v", split="val"):
    from . import judge as JG
    from .probes.squad import load_items
    hub.pull([f"v3/judge/labels_filled_{split}.csv"], path("probes", cfg()).parent / "_hub")
    f = path("probes", cfg()).parent / "_hub" / "v3" / "judge" / f"labels_filled_{split}.csv"
    items = {it["item_id"]: it for it in load_items(_probe_dir(probe_set), split)}
    return JG.agreement(f, items)


def certify_job(J, names, split="test", probe_set="squad3v", profile="support"):
    """Gate each candidate against the original on `split` and write certificate.json / certificate.md."""
    from . import report
    out = {}
    for n in names:
        try:
            g = gate_job(J, n, split=split, probe_set=probe_set, profile=profile)
            d = _audit_dir(probe_set, n, split)
            cert = report.certificate(d, d / f"gate_{profile}.json", _audit_dir(probe_set, "original", split))
            cdir = path("certificates", cfg()) / probe_set / n
            report.write(cert, cdir)
            hub.push(cdir, f"v3/certificates/{probe_set}/{n}", cfg())
            out[n] = g["decision"]
        except Exception as e:
            out[n] = f"error: {e!r}"[:200]
    return out


def test_arm(J, tag, seed=31):
    """Wait for arm `tag` to finish (its json on the Hub), then evaluate its final adapter on the locked test once."""
    wait_hub(J, f"v3/arms/arm{tag}_s{seed}.json")
    hub.pull([f"v3/arms/arm{tag}_s{seed}.json"], path("adapters", cfg()).parent / "_hub")
    a = json.loads((path("adapters", cfg()).parent / "_hub" / "v3" / "arms" / f"arm{tag}_s{seed}.json").read_text())
    # With two-way DPO the val-audited adapter is "<final>_dpo"; test exactly that adapter.
    final = a.get("final_adapter") or a["final"]
    return test_once(J, final, adapter=final)


def retest_dpo_arm(J, tag, seed=31):
    """Arms run before `final_adapter` existed recorded only the SFT name, so their test audited the SFT step,
    not the DPO adapter that the val audit used. Backfill `final_adapter`, push the DPO adapter, test it once."""
    p = path("adapters", cfg()) / f"arm{tag}_s{seed}.json"
    a = json.loads(p.read_text())
    dpo_name = f"{a['final']}_dpo"
    if a.get("two_way_dpo") is not False and (path("adapters", cfg()) / dpo_name / "adapter" / "adapter_config.json").exists():
        a["final_adapter"] = dpo_name
        a.setdefault("notes", []).append("final_adapter backfilled: earlier test of the SFT-only step is not the DPO model")
        p.write_text(json.dumps(a, indent=2, default=str), encoding="utf-8")
        _push_adapter(path("adapters", cfg()) / dpo_name / "adapter", dpo_name)
        hub.push(p, f"v3/arms/{p.name}", cfg())
    else:
        raise FileNotFoundError(f"no DPO adapter {dpo_name} on this node")
    return test_once(J, dpo_name, adapter=dpo_name)


DATA_CARD = """---
license: cc-by-sa-4.0
tags: [humble, rag, abstention, honesty, lexsi-hackathon]
---
# HUMBLE v3 probes and audits

Probe sets (SQuAD-derived multi-passage RAG probes, XQuAD, CUAD, a fictional-company documents demo), every audit
folder (`predictions.jsonl`, `metrics.json`, gate files, `audit.toml`), certificates and arm histories behind
https://github.com/mudgalKrishna/HUMBLE (results/v3/RESULTS.md). SQuAD 2.0 / XQuAD derived data: CC BY-SA 4.0;
CUAD derived data: CC BY 4.0. Model outputs come from adapters on CohereLabs/tiny-aya-global (CC BY-NC 4.0).
"""


def publish_private(J, owner="KM2007"):
    """Create the PRIVATE model repos, the dataset repo and a private collection (plan section 16).
    Making anything public is a separate, explicit step (publish.make_public)."""
    from . import publish as P
    picks = {"humble-tinyaya-naive-blocked": ("naive_s31", "naive answer-only fine-tune (blocked by the gate)"),
             "humble-tinyaya-repaired-A": ("repA_s31_r1", "arm A: HUMBLE repair loop from the blocked adapter"),
             "humble-tinyaya-control-B": ("ctlB_s31_r1", "arm B: matched random-data repair control"),
             "humble-tinyaya-prevention-C": ("prevC_s31", "arm C: fresh honest fine-tune (prevention)")}
    stage = path("adapters", cfg()).parent / "_publish"
    hub.pull(["v3/probes/**", "v3/audits/**", "v3/certificates/**", "v3/arms/**", "v3/judge/**"], stage)
    models = {}
    for repo, (name, role) in picks.items():
        adir = _get_adapter(name)
        t = stage / "v3" / "audits" / "squad3v" / name / "test"
        m = json.loads((t / "metrics.json").read_text())["overall"]
        g = json.loads((t / "gate_support.json").read_text())
        if m.get("over_conservativeness") is None:
            g2 = [r for r in g["rules"] if r["rule"] == "G2"]
            m["over_conservativeness"] = g2[0]["candidate"] if g2 else None
        keep = {k: m.get(k) for k in ("accuracy", "prudence", "over_conservativeness", "honesty", "cost", "echo_rate")}
        ti = path("adapters", cfg()) / name / "train_info.json"
        training = json.loads(ti.read_text()) if ti.exists() else {"adapter": name}
        models[repo] = {"adapter_dir": adir, "card": P.model_card(f"HUMBLE: {role}", role, g, keep, training)}
    (stage / "README.md").write_text(DATA_CARD, encoding="utf-8")
    ids = P.create_private_repos(owner, models, dataset_folder=stage)
    slug = P.make_collection(owner, "HUMBLE: catch and repair bluffing in fine-tuned RAG models", ids,
                             "Tiny Aya Global adapters (blocked, repaired, control, prevention) and all probes/audits.")
    return {"repos": ids, "collection": slug, "private": True}


def model_recipe(J, naive_seeds=(31,), a_seeds=(31, 32, 33), b_seeds=(31, 32, 33), c_seeds=(31,), probe_set="squad3v"):
    """The whole Tiny Aya recipe for another base model, in order, on one GPU. Resumable: finished audits, tests,
    adapters and arms are skipped, so after a crash or restart the same call continues where it stopped."""
    out = {}

    def step(key, fn):
        J.log(f"recipe step {key}")
        out[key] = fn()
        J.status("running", progress=key)
        return out[key]

    def arm(tag, s, fn):
        f = path("adapters", cfg()) / f"arm{tag}_s{s}.json"
        if f.exists():
            out[f"arm{tag}_s{s}"] = {"skipped": str(f)}
        else:
            step(f"arm{tag}_s{s}", fn)
        step(f"test{tag}_s{s}", lambda: test_arm(J, tag, s))

    step("env", lambda: env_check(J))
    step("orig_val", lambda: audit_candidate(J, "original", split="val", probe_set=probe_set))
    step("orig_test", lambda: test_once(J, "original", probe_set=probe_set))
    for s in naive_seeds:
        n = f"naive_s{s}"
        step(f"train_{n}", lambda: train_naive(J, seed=s))
        step(f"{n}_val", lambda: audit_candidate(J, n, adapter=n, split="val", probe_set=probe_set))
        g = step(f"{n}_gate", lambda: gate_job(J, n, split="val", probe_set=probe_set))
        step(f"{n}_test", lambda: test_once(J, n, adapter=n, probe_set=probe_set))
        if s == naive_seeds[0] and g.get("decision") != "BLOCK":
            out["stopped"] = f"{n} was not blocked on val ({g.get('decision')}): repair arms skipped, check before going on"
            return out
    for s in a_seeds:
        arm("A", s, lambda: arm_A(J, seed=s, probe_set=probe_set))
    for s in b_seeds:
        arm("B", s, lambda: arm_B(J, seed=s, probe_set=probe_set))
    for s in c_seeds:
        c = f"prevC_s{s}"
        step(f"train_{c}", lambda: train_prevention(J, seed=s))
        step(f"{c}_val", lambda: audit_candidate(J, c, adapter=c, split="val", probe_set=probe_set))
        step(f"{c}_test", lambda: test_once(J, c, adapter=c, probe_set=probe_set))
    finals = ["naive_s31"] + [f"repA_s{s}_r1" for s in a_seeds] + [f"ctlB_s{s}_r1" for s in b_seeds] + [f"prevC_s{s}" for s in c_seeds]
    step("certify", lambda: certify_job(J, finals, probe_set=probe_set))
    step("summary", lambda: analysis(J, "e_repair_summary", split="test"))
    step("e_cat", lambda: analysis(J, "e_cat", split="test"))
    return out


def flush(J):
    """Re-upload everything that failed to reach the Hub (rate limits) into this node's repo, plus all local audits."""
    n = hub.flush_pending(cfg())
    base = path("audits", cfg())
    pushed = 0
    for ps in [d for d in base.iterdir() if d.is_dir() and not d.name.startswith("_")]:
        pushed += bool(hub.push(ps, f"v3/audits/{ps.name}", cfg(), message=f"flush audits {ps.name}"))
    arms = [p for p in path("adapters", cfg()).glob("arm*.json")]
    for p in arms:
        hub.push(p, f"v3/arms/{p.name}", cfg())
    return {"pending_reuploaded": n, "audit_sets_pushed": pushed, "arm_files": len(arms)}
