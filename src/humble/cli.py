"""HUMBLE command line (plan §14). Exit codes for `gate`: PASS 0, BLOCK 1, error 2.

  humble probe   --docs ./company_docs --out ./probes                 (GPU: local generator + judge)
  humble audit   --adapter ./my_finetune --probes ./probes --split val --out ./audits/my_finetune   (GPU)
  humble gate    ./audits/original ./audits/my_finetune --profile finance                      (CPU)
  humble repair  --blocked ./my_finetune --probes ./probes --original-audit ./audits/original --out ./adapters/repaired (GPU)
  humble scale   ./my_finetune --lambdas 0,0.25,0.5,0.75,1 --out ./adapters/scaled              (GPU)
  humble certify ./audits/my_finetune_repaired --gate ./audits/my_finetune_repaired/gate.json --out ./certificates
  humble selfcheck ./audits/original --profile finance                                          (CPU)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _preds(d):
    from .audit.run import read_predictions
    return read_predictions(Path(d) / "predictions.jsonl")


def cmd_probe(a):
    from .config import load_config
    from .audit.generate import load_model, free
    from .probes import docs as D
    from . import judge as JG
    cfg = load_config(a.config)
    model, tok = load_model(cfg=cfg)
    jd = None if a.no_judge else JG.Judge(a.judge or cfg["judge"]["model_id"])
    m = D.build(a.docs, a.out, cfg, model, tok,
                verify_fn=(lambda items: JG.verify(items, jd.sufficiency(items))) if jd else None,
                judge_llm=D.LocalLLM(jd.model, jd.tok, jd.model_id) if jd else None, n_per_cell=a.n, log=print)
    free(model)
    print(json.dumps(m["cells"], indent=1))
    return 0


def cmd_audit(a):
    from .config import load_config
    from .audit.generate import load_model, free
    from .audit.run import audit
    from .gate import known_ids
    from .probes.squad import load_items
    cfg = load_config(a.config)
    items = load_items(a.probes, a.split)
    K = known_ids(_preds(a.original_audit)) if a.original_audit else None
    model, tok = load_model(adapter=a.adapter, base=a.base, cfg=cfg)
    lock = Path(a.probes, "TEST_LOCK.txt")
    recs, mets = audit(model, tok, items, a.out, a.name or Path(a.out).name, a.split, adapter=a.adapter,
                       probe_file=Path(a.probes, f"items_{a.split}.jsonl"),
                       test_lock=lock.read_text().strip() if lock.exists() else None,
                       samples=a.samples, ptrue=not a.no_ptrue, known_ids=K, cfg=cfg, log=print)
    free(model)
    print(json.dumps(mets["overall"], indent=1))
    return 0


def cmd_gate(a):
    from . import gate as G
    from .config import load_config, profile
    cfg = load_config(a.config)
    prof = profile(a.profile, cfg)
    g = G.evaluate(_preds(a.original), _preds(a.candidate), prof, cfg["gate"]["format_failure_max"],
                   cfg["metrics"]["bootstrap_resamples"])
    out = Path(a.out) if a.out else Path(a.candidate) / "gate.json"
    G.write(g, out)
    print(f"{g['decision']}  (profile {prof['name']})")
    for r in g["reasons"]:
        print("  -", r)
    print(f"gate written to {out}")
    return g["exit_code"]


def cmd_selfcheck(a):
    from . import gate as G
    from .config import load_config, profile
    res = G.self_controls(_preds(a.original), profile(a.profile, load_config(a.config)),
                          naive_recs=_preds(a.naive) if a.naive else None)
    print(json.dumps(res, indent=1))
    return 0 if res["all_ok"] else 1


def cmd_repair(a):
    """Generic repair loop on any probe folder with a probe pool (items_probe.jsonl + paragraphs_probe.jsonl)."""
    import random
    from .config import load_config, profile
    from .audit.generate import load_model, free
    from .audit.run import audit
    from .probes.squad import load_items, load_corpus
    from .probes.corpus import wrong_answer_pool
    from . import gate as G, repair as R, train as T
    from .curate import curate_sft
    cfg = load_config(a.config)
    prof = profile(a.profile, cfg)
    probe = load_items(a.probes, "probe")
    corpus = load_corpus(a.probes, "probe")
    val = load_items(a.probes, "val")
    replay = [it for it in (load_items(a.probes, "train") if Path(a.probes, "items_train.jsonl").exists() else probe)
              if it["label"] == "answer"]
    wp = wrong_answer_pool([it["source"] for it in probe if it["source"].get("answers")])
    orig = _preds(a.original_audit)
    K = G.known_ids(orig)
    by_id = {it["item_id"]: it for it in probe}
    slices = R.probe_slices(probe, a.rounds, a.seed)
    current = a.blocked
    rng = random.Random(a.seed)
    out = Path(a.out)
    for r in range(a.rounds):
        model, tok = load_model(adapter=current, cfg=cfg)
        precs, _ = audit(model, tok, slices[r], out / f"round{r + 1}" / "probe_audit", "candidate", f"probe_r{r + 1}",
                         samples=cfg["generation"]["diagnose_samples"], ptrue=False, cfg=cfg, log=print)
        free(model)
        diag = R.diagnose(precs, cfg["generation"]["bluff_majority"])
        ex, counts = R.build_repair_set(diag, by_id, corpus, rng, cfg["prompt"]["passages_default"], wp, replay,
                                        cfg["repair"]["replay_ratio"])
        msgs, cs = curate_sft(R.to_sft_examples(ex), out / f"round{r + 1}" / "data", f"repair_r{r + 1}")
        current, info = T.sft(msgs, out / f"round{r + 1}", init_adapter=current, seed=a.seed, cfg=cfg, log=print)
        model, tok = load_model(adapter=current, cfg=cfg)
        vrecs, _ = audit(model, tok, val, out / f"round{r + 1}" / "val_audit", "repaired", "val", adapter=current,
                         known_ids=K, cfg=cfg, log=print)
        free(model)
        g = G.evaluate(orig, vrecs, prof)
        G.write(g, out / f"round{r + 1}" / "gate.json")
        print(f"round {r + 1}: {counts} -> {g['decision']} {g['failed_rules']}")
        if g["decision"] == "PASS":
            break
    print(f"final adapter: {current}")
    return 0


def cmd_scale(a):
    from . import train as T
    for lam in [float(x) for x in a.lambdas.split(",")]:
        d = T.scale(a.adapter, lam, Path(a.out) / f"lambda_{lam}")
        print(lam, d)
    return 0


def cmd_certify(a):
    from . import report
    cert = report.certificate(a.audit, a.gate, a.original_audit)
    d = report.write(cert, a.out)
    print(f"{cert['decision']}: certificate written to {d}")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(prog="humble", description="Catch and repair bluffing in fine-tuned RAG models.")
    p.add_argument("--config", default=None)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("probe"); s.add_argument("--docs", required=True); s.add_argument("--out", required=True)
    s.add_argument("--n", type=int, default=60); s.add_argument("--judge", default=None); s.add_argument("--no-judge", action="store_true")
    s.add_argument("--profile", default=None); s.set_defaults(fn=cmd_probe)
    s = sub.add_parser("audit"); s.add_argument("--probes", required=True); s.add_argument("--out", required=True)
    s.add_argument("--adapter", default=None); s.add_argument("--base", default=None); s.add_argument("--model", default=None)
    s.add_argument("--split", default="val"); s.add_argument("--name", default=None); s.add_argument("--samples", type=int, default=0)
    s.add_argument("--original-audit", default=None); s.add_argument("--no-ptrue", action="store_true"); s.set_defaults(fn=cmd_audit)
    s = sub.add_parser("gate"); s.add_argument("original"); s.add_argument("candidate"); s.add_argument("--profile", default=None)
    s.add_argument("--out", default=None); s.set_defaults(fn=cmd_gate)
    s = sub.add_parser("selfcheck"); s.add_argument("original"); s.add_argument("--naive", default=None)
    s.add_argument("--profile", default=None); s.set_defaults(fn=cmd_selfcheck)
    s = sub.add_parser("repair"); s.add_argument("--blocked", required=True); s.add_argument("--probes", required=True)
    s.add_argument("--original-audit", required=True); s.add_argument("--out", required=True)
    s.add_argument("--rounds", type=int, default=3); s.add_argument("--seed", type=int, default=31)
    s.add_argument("--profile", default=None); s.set_defaults(fn=cmd_repair)
    s = sub.add_parser("scale"); s.add_argument("adapter"); s.add_argument("--lambdas", default="0,0.25,0.5,0.75,1")
    s.add_argument("--out", required=True); s.set_defaults(fn=cmd_scale)
    s = sub.add_parser("certify"); s.add_argument("audit"); s.add_argument("--gate", required=True)
    s.add_argument("--original-audit", default=None); s.add_argument("--out", required=True); s.set_defaults(fn=cmd_certify)
    a = p.parse_args(argv)
    try:
        return a.fn(a)
    except Exception as e:
        print(f"error: {e!r}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
