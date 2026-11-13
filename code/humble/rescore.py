"""Re-score saved predictions with the current parser. No GPU needed.

Each predictions.jsonl keeps the raw model text (prediction_raw), so the action,
answer, and correctness can be recomputed. P(True) confidence is kept only when the
parsed answer text is unchanged; otherwise it is cleared (the answer was never shown).
"""
import json
from pathlib import Path

from .metrics import bootstrap_ci, compute_metrics
from .prompts import is_correct, parse_output


def rescore_run(run_dir, resamples=1000):
    run_dir = Path(run_dir)
    pred_path = run_dir / "predictions.jsonl"
    old = [json.loads(l) for l in pred_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    new, changed = [], 0
    for r in old:
        action, answer, conf01 = parse_output(r["prediction_raw"])
        r2 = dict(r)
        if action != r["action"] or answer != r.get("answer"):
            changed += 1
        r2["action"] = action
        r2["answer"] = answer
        if action == "answer":
            r2["correct"] = is_correct(answer, r["gold_answer"])
            if answer != r.get("answer") or r.get("conf") is None:
                r2["conf"] = None  # confidence was computed for a different answer text
        else:
            r2["correct"] = False
            r2["conf"] = None
        new.append(r2)
    pred_path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in new) + "\n", encoding="utf-8")
    metrics = compute_metrics(new)
    lo, hi = bootstrap_ci(new, "cw", resamples=resamples)
    metrics["cw_ci95"] = [lo, hi]
    metrics["rescored_changed_items"] = changed
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics, changed
