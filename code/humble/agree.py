"""Agreement-abstention rule, applied after the audit (no retraining).

support = share of the model's sampled answers that match its greedy answer
(normalized exact match or token F1 >= 0.8). If support < t, the answer becomes ABSTAIN.
Needs audits run with sample_all=True so unanswerable questions also have samples.
"""
from .metrics import compute_metrics
from .prompts import is_correct, normalize, token_f1


def _support(rec):
    ans, samples = rec.get("answer"), rec.get("samples") or []
    if rec["action"] != "answer" or not samples or ans is None:
        return None
    hits = sum(1 for s in samples if s not in ("ABSTAIN", "PARSE_ERROR")
               and (normalize(s) == normalize(ans) or token_f1(s, ans) >= 0.8))
    return hits / len(samples)


def apply_rule(recs, t):
    out = []
    for r in recs:
        r2 = dict(r)
        sup = _support(r)
        if sup is not None and sup < t:
            r2["action"], r2["correct"], r2["conf"] = "abstain", False, None
        out.append(r2)
    return out


def sweep(recs, thresholds=(0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)):
    res = {}
    for t in thresholds:
        m = compute_metrics(apply_rule(recs, t))
        res[t] = {k: m[k] for k in ("accuracy", "unanswerable_bluff", "unnecessary_abstention")}
        res[t]["score"] = m["unanswerable_bluff"] + m["unnecessary_abstention"]
    return res
