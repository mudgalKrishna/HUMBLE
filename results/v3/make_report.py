"""Build results/v3/RESULTS.md from the saved audit files on the Hub (no hand-typed numbers).

Usage: HF_TOKEN=... python results/v3/make_report.py [download_dir]
"""
import json
import os
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

DL = Path(sys.argv[1] if len(sys.argv) > 1 else "humble_v3_results")
OUT = Path(__file__).resolve().parent / "RESULTS.md"
PS = "squad3v"


def fetch():
    for rid in ["KM2007/humble-runs", "KM2007/humble-v3"] + [f"KM2007/humble-v3-nb{i}" for i in range(1, 9)]:
        try:
            snapshot_download(rid, repo_type="dataset", local_dir=str(DL), token=os.environ.get("HF_TOKEN"),
                              allow_patterns=["v3/audits/**/metrics.json", "v3/audits/**/metrics_judged.json",
                                              "v3/audits/**/gate_*.json", "v3/audits/**/audit.json", "v3/audits/*/_*.json",
                                              "v3/arms/*.json", "v3/probes/*/MANIFEST.json"])
        except Exception as e:
            print("fetch", rid, e)


def load(p):
    q = DL / p
    m = json.loads(q.read_text(encoding="utf-8")) if q.exists() else None
    if m and q.name == "metrics.json" and m.get("overall", {}).get("over_conservativeness") is None:
        # audits run without the original's reference (C) have no over-conservativeness in metrics.json;
        # the gate computed it against the original on the same split, so take it from there.
        g = q.parent / "gate_support.json"
        if g.exists():
            r = [x for x in json.loads(g.read_text())["rules"] if x["rule"] == "G2"]
            if r and r[0].get("candidate") is not None:
                o = m["overall"]
                o["over_conservativeness"] = r[0]["candidate"]
                if o.get("prudence") is not None:
                    o["honesty"] = (o["prudence"] + 1 - o["over_conservativeness"]) / 2
    return m


def f(x, pct=True):
    if x is None:
        return "–"
    return f"{100 * x:.1f}" if pct else f"{x:.2f}"


def ci(m, key):
    c = (m or {}).get("ci95", {}).get(key)
    return f" ({f(c[0])}–{f(c[1])})" if c and c[0] is not None else ""


def row(name, m, with_ci=False):
    o = m["overall"]
    keys = ["accuracy", "prudence", "over_conservativeness", "unnecessary_abstention", "honesty"]
    cells = [f(o.get(k)) + (ci(m, k) if with_ci and k in ("accuracy", "prudence") else "") for k in keys]
    cells.append(f(o.get("cost"), pct=False))
    cells.append(f(o.get("echo_rate")))
    return f"| {name} | " + " | ".join(cells) + " |"


HEAD = ("| Model | Accuracy | Prudence | Over-cons. | Unnec. abstention | Honesty | Cost | U5 echo |\n"
        "|---|---|---|---|---|---|---|---|")


def mean_row(name, ms):
    ms = [m for m in ms if m]
    if not ms:
        return f"| {name} | (missing) |"
    keys = ["accuracy", "prudence", "over_conservativeness", "unnecessary_abstention", "honesty", "cost", "echo_rate"]
    vals = []
    for k in keys:
        xs = [m["overall"].get(k) for m in ms if m["overall"].get(k) is not None]
        if not xs:
            vals.append("–")
            continue
        mu = sum(xs) / len(xs)
        sd = (sum((x - mu) ** 2 for x in xs) / max(1, len(xs) - 1)) ** 0.5
        vals.append((f"{mu:.2f}" if k == "cost" else f"{100 * mu:.1f}") + (f" ± {sd:.2f}" if k == "cost" else f" ± {100 * sd:.1f}") if len(xs) > 1 else (f"{mu:.2f}" if k == "cost" else f"{100 * mu:.1f}"))
    return f"| {name} (n={len(ms)}) | " + " | ".join(vals) + " |"


def main():
    if "--no-fetch" not in sys.argv:
        fetch()
    arms = {}
    for p in (DL / "v3" / "arms").glob("*.json"):
        arms[p.stem] = json.loads(p.read_text())
    A = [arms.get(f"armA_s{s}", {}).get("final") for s in (31, 32, 33)]
    B = [arms.get(f"armB_s{s}", {}).get("final") for s in (31, 32, 33)]
    L = []
    L += ["# HUMBLE v3 results (Tiny Aya Global)", "",
          "All numbers are percentages unless marked; computed from saved `predictions.jsonl` files. "
          "Prudence = ABSTAIN on unanswerable probes; over-conservativeness = ABSTAIN on probes the original answered "
          "correctly; cost uses the `support` profile weights (2/2/2). Test = locked test, evaluated once per model.", ""]
    for split in ("test", "val"):
        L += [f"## E-REPAIR on the SQuAD-derived RAG probes ({split})", "", HEAD]
        L.append(row("Original", load(f"v3/audits/{PS}/original/{split}/metrics.json") or {"overall": {}}, True))
        for s in (31, 32, 33):
            m = load(f"v3/audits/{PS}/naive_s{s}/{split}/metrics.json")
            if m:
                L.append(row(f"Naive SFT s{s} (blocked)", m, True))
        L.append(mean_row("A HUMBLE repair loop", [load(f"v3/audits/{PS}/{a}/{split}/metrics.json") for a in A if a]))
        L.append(mean_row("B random control", [load(f"v3/audits/{PS}/{b}/{split}/metrics.json") for b in B if b]))
        L.append(mean_row("C prevention", [load(f"v3/audits/{PS}/prevC_s{s}/{split}/metrics.json") for s in (31, 32, 33)]))
        labels = {"AnoTw": "Ablation: A without twins", "AnoRep": "Ablation: A without replay",
                  "ADPO2": "A + two-way DPO", "Ab32": "A repairing naive s32", "Ab33": "A repairing naive s33"}
        for tag, label in labels.items():
            # test audits are stored under the adapter actually evaluated ("<final>_dpo" for two-way DPO)
            finals = [(arms.get(f"arm{tag}_s{s}", {}).get("final_adapter") if split == "test" else None)
                      or arms.get(f"arm{tag}_s{s}", {}).get("final") for s in (31, 32, 33)]
            ms = [load(f"v3/audits/{PS}/{a}/{split}/metrics.json") for a in finals if a]
            if any(ms):
                L.append(mean_row(label, ms))
        L.append("")
    L += ["## Gate decisions (val, profile `support`)", "", "| Candidate | Decision | Failed rules |", "|---|---|---|"]
    for p in sorted((DL / "v3" / "audits" / PS).glob("*/val/gate_support.json")):
        g = json.loads(p.read_text())
        L.append(f"| {p.parent.parent.name} | {g['decision']} | {', '.join(g['failed_rules']) or '–'} |")
    L.append("")
    L += ["## Gate decisions on the locked test (certificates, profile `support`)", "",
          "| Candidate | Decision | Failed rules |", "|---|---|---|"]
    for p in sorted((DL / "v3" / "audits" / PS).glob("*/test/gate_support.json")):
        g = json.loads(p.read_text())
        L.append(f"| {p.parent.parent.name} | {g['decision']} | {', '.join(g['failed_rules']) or '–'} |")
    L.append("")
    D = [arms.get("armD"), arms.get("armDfine")]
    L += ["## Arm D: lambda scaling of the blocked adapter (val)", "", "| lambda | Gate | Accuracy | Prudence |", "|---|---|---|---|"]
    rows = sorted([r for d in D if d for r in d["rows"]], key=lambda r: -r["lambda"])
    for r in rows:
        L.append(f"| {r['lambda']} | {r['gate']} | {f(r['val']['accuracy'])} | {f(r['val']['prudence'])} |")
    L.append("")
    cat = load(f"v3/audits/{PS}/original/test/metrics.json")
    L += ["## E-CAT: prudence by unanswerable category (test)", "", "| Model | U1 missing | U2 out of DB | U3 false premise | U4 underspecified | U5 misleading |", "|---|---|---|---|---|---|"]
    for name, path in [("Original", "original"), ("Naive s31", "naive_s31")] + [(f"A s{s}", A[i]) for i, s in enumerate((31, 32, 33)) if A[i]] + [(f"C s{s}", f"prevC_s{s}") for s in (31, 32, 33)]:
        m = load(f"v3/audits/{PS}/{path}/test/metrics.json")
        if m:
            bc = m["by_category"]
            L.append(f"| {name} | " + " | ".join(f(bc.get(c, {}).get("prudence")) for c in ("U1", "U2", "U3", "U4", "U5")) + " |")
    L.append("")
    for ps, title in (("xquad", "E-LANG: XQuAD (es, hi, ar), evaluation only"), ("cuad", "E-DOMAIN: CUAD contracts, evaluation only"),
                      ("docs_demo2", "Product mode: probes generated from a fictional company's 44 policy documents")):
        L += [f"## {title}", "", HEAD]
        for name, path in [("Original", "original"), ("Naive s31", "naive_s31")] + [(f"A s{s}", A[i]) for i, s in enumerate((31, 32, 33)) if A[i]] + [(f"C s{s}", f"prevC_s{s}") for s in (31, 32, 33)]:
            m = load(f"v3/audits/{ps}/{path}/test/metrics.json")
            if m:
                L.append(row(name, m))
        if ps == "cuad":
            L += ["", "With judge grading of long answers (M9):", "", "| Model | Accuracy (F1 only) | Accuracy (judge) |", "|---|---|---|"]
            for name, path in [("Original", "original"), ("Naive s31", "naive_s31")] + [(f"A s{s}", A[i]) for i, s in enumerate((31, 32, 33)) if A[i]] + [(f"C s{s}", f"prevC_s{s}") for s in (31, 32, 33)]:
                m, mj = load(f"v3/audits/cuad/{path}/test/metrics.json"), load(f"v3/audits/cuad/{path}/test/metrics_judged.json")
                if m and mj:
                    L.append(f"| {name} | {f(m['overall']['accuracy'])} | {f(mj['overall']['accuracy'])} |")
        L.append("")
    eg = load(f"v3/audits/{PS}/_e_gate.json")
    if eg:
        L += ["## E-GATE: self-controls (val)", "", "| Profile | Original vs itself | Always abstain | Always answer | Naive SFT | All as expected |", "|---|---|---|---|---|---|"]
        for p, d in eg["self_controls"].items():
            L.append(f"| {p} | {d['original_vs_itself']['got']} | {d['always_abstain']['got']} {d['always_abstain']['failed_rules']} | "
                     f"{d['always_answer']['got']} {d['always_answer']['failed_rules']} | {d.get('naive_sft', {}).get('got', '–')} | {d['all_ok']} |")
        L += ["", "Gate v2 rerun on saved v2 predictions (single passage, SQuAD 2.0):", "", "| Run | Decision | Failed rules | Prudence | Over-cons. |", "|---|---|---|---|---|"]
        for k, v in sorted(eg["v2_saved_predictions"].items()):
            if "decision" in v:
                L.append(f"| {k} | {v['decision']} | {', '.join(v['failed_rules']) or '–'} | {f(v['prudence'])} | {f(v['over_conservativeness'])} |")
        L.append("")
    er = load(f"v3/audits/{PS}/_e_repair_test.json")
    if er:
        L += ["## Hypotheses (pre-registered)", "", "| ID | Claim | Verdict |", "|---|---|---|"]
        for k, v in er["hypotheses"].items():
            L.append(f"| {k} | {v['claim']} | {v['verdict']} |")
        L += ["", "Paired differences on the locked test (document-clustered bootstrap, 1,000 resamples):", "",
              "| Comparison | Metric | Diff | 95% CI | Significant |", "|---|---|---|---|---|"]
        for comp, d in er["comparisons"].items():
            for m, v in d.items():
                if v.get("diff") is not None:
                    L.append(f"| {comp} | {m} | {v['diff']:+.3f} | {v['ci95'][0]:+.3f} … {v['ci95'][1]:+.3f} | {v['significant']} |")
        L.append("")
    man = load(f"v3/probes/{PS}/MANIFEST.json")
    if man:
        L += ["## Probe set", "", f"Locked test sha256 `{man.get('test_lock_sha256')}`; judge `{man.get('judge_model')}`.", "",
              "| Cell | Built | Judge pass rate |", "|---|---|---|"]
        for k, v in man["cells"].items():
            j = man.get("judge", {}).get(k)
            L.append(f"| {k} | {v['built']} | {f(j['pass_rate']) if j else '–'} |")
    OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
