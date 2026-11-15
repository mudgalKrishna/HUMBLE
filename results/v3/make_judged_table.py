"""Judge-graded summary table for a run done with `regate_judged` (e.g. Aya Expanse 8B).

Usage: python results/v3/make_judged_table.py <HUMBLE_HOME>/audits/squad3v
"""
import json
import statistics as st
import sys
from pathlib import Path

B = Path(sys.argv[1])
ROWS = [("Original", ["original"]), ("Naive SFT s31", ["naive_s31"]),
        ("A HUMBLE repair", [f"repA_s{s}_r1" for s in (31, 32, 33)]),
        ("B random control", [f"ctlB_s{s}_r1" for s in (31, 32, 33)]), ("C prevention", ["prevC_s31"])]


def metrics(n, s):
    p = B / n / s / "metrics_judged.json"
    return json.loads(p.read_text())["overall"] if p.exists() else None


def gate(n, s):
    p = B / n / s / "gate_support_judged.json"
    if not p.exists():
        return "reference" if n == "original" else "-"
    g = json.loads(p.read_text())
    return g["decision"] + (f" ({', '.join(g['failed_rules'])})" if g["failed_rules"] else "")


L = []
for s in ("test", "val"):
    L += [f"### {s}", "", "| Model | Accuracy (judge) | Prudence | Over-cons. | Gate (judge-graded) |", "|---|---|---|---|---|"]
    for label, names in ROWS:
        ms = [(n, metrics(n, s)) for n in names]
        ms = [(n, m) for n, m in ms if m]
        if not ms:
            continue

        def cell(k):
            v = [100 * m[k] for _, m in ms if m.get(k) is not None]
            return "-" if not v else f"{st.mean(v):.1f}" + (f" ± {st.stdev(v):.1f}" if len(v) > 1 else "")

        gates = ", ".join(sorted({gate(n, s) for n, _ in ms}))
        L.append(f"| {label} (n={len(ms)}) | " + " | ".join(cell(k) for k in ("accuracy", "prudence", "over_conservativeness")) + f" | {gates} |")
    L.append("")
print("\n".join(L))
