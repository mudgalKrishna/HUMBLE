"""Job definitions. Each function is run inside a GPU notebook by the runner.

Phases:
  A  original_val, train_audit(sft_naive|sft_honest|dpo_naive|dpo_honest) -> val metrics
  B  repair(targeted|control, target), lambda_val(target)                -> adapters, lambda choice
  C  test audits (each notebook audits its own adapter, once)            -> final numbers
"""
import gc
import random
import json
from pathlib import Path

import torch

from . import runner as R
from .audit import audit_split, read_predictions
from .gate import release_gate
from .metrics import compute_metrics
from .prompts import target_for
from .train import (SFT_LR, load_adapter, load_base, new_lora_model, scale_adapter, sft_examples,
                    dpo_pairs, train_dpo, train_sft)

CANDIDATES = {
    "sft_naive": ("sft", False),
    "sft_honest": ("sft", True),
    "dpo_naive": ("dpo", False),
    "dpo_honest": ("dpo", True),
}
REPAIR_LR, REPAIR_EPOCHS = 1e-4, 1


def _free(*objs):
    for o in objs:
        del o
    gc.collect()
    torch.cuda.empty_cache()


def audit_val(name, adapter_dir=None, log=None):
    tok, model = load_base(R.MODEL)
    if adapter_dir:
        model = load_adapter(model, adapter_dir)
    model.eval()
    items = R.load_items("val")
    recs, metrics = audit_split(model, tok, items, R.DATA / "documents", run_id=name, candidate=name,
                                out_dir=R.RUNS / name / "val", log=log)
    _free(model)
    R.sync_up(R.RUNS / name / "val", f"runs/{name}/val")
    return metrics


def audit_test(name, adapter_dir=None, log=None):
    """Final, single test evaluation. Refuses to run twice for the same name."""
    marker = R.RUNS / name / "test" / "DONE"
    if marker.exists():
        raise RuntimeError(f"test already evaluated for {name}; test is locked")
    tok, model = load_base(R.MODEL)
    if adapter_dir:
        model = load_adapter(model, adapter_dir)
    model.eval()
    items = R.load_items("test")
    recs, metrics = audit_split(model, tok, items, R.DATA / "documents", run_id=name, candidate=name,
                                out_dir=R.RUNS / name / "test", log=log)
    marker.write_text("locked-after-single-evaluation\n", encoding="utf-8")
    _free(model)
    R.sync_up(R.RUNS / name / "test", f"runs/{name}/test")
    return metrics


def audit_external(name, adapter_dir=None, log=None):
    """External real-data check (SQuAD 2.0 sample). Runs once per model, like the test set."""
    marker = R.RUNS / name / "external" / "DONE"
    if marker.exists():
        raise RuntimeError(f"external check already run for {name}; it is locked")
    base_dir = R.BASE / "external"
    tok, model = load_base(R.MODEL)
    if adapter_dir:
        model = load_adapter(model, adapter_dir)
    model.eval()
    items = [json.loads(l) for l in (base_dir / "items_external.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    recs, metrics = audit_split(model, tok, items, base_dir / "documents", run_id=name, candidate=name,
                                out_dir=R.RUNS / name / "external", log=log)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("locked-after-single-evaluation\n", encoding="utf-8")
    _free(model)
    R.sync_up(R.RUNS / name / "external", f"runs/{name}/external")
    return metrics


def train_and_audit(candidate, seed=1, log=None):
    """Train one candidate with one seed, then audit it on val. Seed 1 keeps the plain name."""
    method, honest = CANDIDATES[candidate]
    name = candidate if seed == 1 else f"{candidate}_s{seed}"
    adapter_dir = R.ADAPTERS / name
    if not (adapter_dir / "adapter_config.json").exists():
        tok, base = load_base(R.MODEL)
        model = new_lora_model(base)
        items = R.load_items("train")
        if method == "sft":
            examples = sft_examples(items, R.DATA / "documents", honest)
            train_sft(model, tok, examples, adapter_dir, seed=seed, log=log)
        else:
            pairs = dpo_pairs(items, R.DATA / "documents", honest, seed=seed)
            train_dpo(model, tok, pairs, adapter_dir, seed=seed, log=log)
        R.sync_up(adapter_dir, f"adapters/{name}")
        _free(model, base)
    return audit_val(name, adapter_dir, log=log)


def _ensure_run(name, sub):
    """Return a local folder with runs/<name>/<sub>, downloading it from the HF repo if this notebook lacks it."""
    local = R.RUNS / name / sub
    if (local / "metrics.json").exists() or (local / "predictions.jsonl").exists():
        return local
    R.sync_down(f"runs/{name}/{sub}", R.BASE / "_dl")
    return R.BASE / "_dl" / "runs" / name / sub


def _targeted_predictions(target):
    return read_predictions(_ensure_run(target, "val") / "predictions.jsonl")


def repair(mode, target, log=None):
    """mode: targeted (confident-wrong items from val + equal train mix) or control (2x random train)."""
    name = f"repaired_{mode}" if mode == "targeted" else "repaired_control"
    src = R.ADAPTERS / target
    if not (src / "adapter_config.json").exists():
        R.sync_down(f"adapters/{target}", R.ADAPTERS.parent / "_dl")
        src = R.ADAPTERS.parent / "_dl" / "adapters" / target
    train = R.load_items("train")
    val = R.load_items("val")
    rng = random.Random(7)
    if mode == "targeted":
        recs = _targeted_predictions(target)
        bad = {(r["org_id"], r["fact_type"]) for r in recs
               if r["action"] == "parse_error" or (r["action"] == "answer" and not r["correct"]
                                                   and (r["conf"] or 0) >= 0.8)
               or (r["question_type"] == "unanswerable" and r["action"] == "answer")}
        repair_items = [it for it in val if (it["org_id"], it["fact_type"]) in bad]
        mix = [it for it in train if it["question_type"] in ("answerable", "unanswerable")]
        extra = rng.sample(mix, min(len(repair_items), len(mix)))
        chosen = repair_items + extra
    else:
        budget = 2 * len(_targeted_size(target))
        chosen = rng.sample(train, min(budget, len(train)))
    examples = sft_examples(chosen, R.DATA / "documents", honest=True)
    tok, base = load_base(R.MODEL)
    model = load_adapter(base, src, trainable=True)
    out = R.ADAPTERS / name
    train_sft(model, tok, examples, out, epochs=REPAIR_EPOCHS, lr=REPAIR_LR, log=log)
    R.sync_up(out, f"adapters/{name}")
    _free(model, base)
    return {"name": name, "items": len(chosen), "targeted_items": len(repair_items) if mode == "targeted" else None}


def _targeted_size(target):
    """Control is budget-matched to targeted: 2x the number of targeted-repair items."""
    recs = _targeted_predictions(target)
    bad = {(r["org_id"], r["fact_type"]) for r in recs
           if r["action"] == "parse_error" or (r["action"] == "answer" and not r["correct"]
                                               and (r["conf"] or 0) >= 0.8)
           or (r["question_type"] == "unanswerable" and r["action"] == "answer")}
    return [it for it in R.load_items("val") if (it["org_id"], it["fact_type"]) in bad]


def lambda_val(target, log=None):
    """Scale the target adapter by each lambda, audit on val, pick the largest lambda that passes release gate."""
    orig = json.loads((_ensure_run("original", "val") / "metrics.json").read_text(encoding="utf-8"))
    results = {}
    for lam in R.LAMBDAS:
        name = f"lambda_{lam}"
        tok, base = load_base(R.MODEL)
        src = R.ADAPTERS / target
        if not (src / "adapter_config.json").exists():
            R.sync_down(f"adapters/{target}", R.ADAPTERS.parent / "_dl")
            src = R.ADAPTERS.parent / "_dl" / "adapters" / target
        out = R.ADAPTERS / f"lambda_{lam}"
        scale_adapter(base, src, lam, out)
        _free(base)
        m = audit_val(name, out, log=log)
        results[str(lam)] = {"metrics": m, "gate": release_gate(orig, m)["decision"]}
    passing = [float(k) for k, v in results.items() if v["gate"] == "PASS"]
    chosen = max(passing) if passing else None
    summary = {"target": target, "chosen_lambda": chosen, "results": {k: v["gate"] for k, v in results.items()}}
    (R.RUNS / "lambda_scaled").mkdir(parents=True, exist_ok=True)
    (R.RUNS / "lambda_scaled" / "choice.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    R.sync_up(R.RUNS / "lambda_scaled" / "choice.json", "runs/lambda_scaled/choice.json")
    return summary
