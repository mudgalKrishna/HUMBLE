"""Certificate (plan §5 step 5, §14): metrics with CIs, the gate decision with reasons, and every hash
and version needed to reproduce the decision from saved predictions.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from .. import __version__


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest() if p and Path(p).exists() else None


def certificate(audit_dir, gate_file, original_audit_dir=None, extra=None):
    a = Path(audit_dir)
    meta = json.loads((a / "audit.json").read_text(encoding="utf-8"))
    mets = json.loads((a / "metrics.json").read_text(encoding="utf-8"))
    gate = json.loads(Path(gate_file).read_text(encoding="utf-8"))
    cert = {
        "humble_version": __version__,
        "issued_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "candidate": meta["candidate"], "split": meta["split"], "model_id": meta["model_id"],
        "adapter_hash": meta.get("adapter_hash"), "probe_sha256": meta.get("probe_sha256"),
        "test_lock": meta.get("test_lock"), "config_hash": meta.get("config_hash"),
        "auditkit_fingerprint": meta.get("auditkit_fingerprint"), "versions": meta.get("versions"),
        "predictions_sha256": _sha(a / "predictions.jsonl"),
        "original_predictions_sha256": _sha(Path(original_audit_dir) / "predictions.jsonl") if original_audit_dir else None,
        "decision": gate["decision"], "exit_code": gate["exit_code"], "profile": gate["profile"],
        "failed_rules": gate["failed_rules"], "reasons": gate["reasons"], "rules": gate["rules"],
        "metrics": mets["overall"], "ci95": mets.get("ci95"), "by_category": mets.get("by_category"),
        "by_context": {k: {m: v.get(m) for m in ("accuracy", "prudence", "unnecessary_abstention")} for k, v in mets.get("by_context", {}).items()},
        "reproduce": "humble gate <original audit dir> <candidate audit dir> --profile " + gate["profile"]["name"],
    }
    if extra:
        cert.update(extra)
    return cert


def _f(x, pct=True):
    if x is None:
        return "n/a"
    return f"{100 * x:.1f}%" if pct else f"{x:.3f}"


def to_markdown(cert):
    m, ci = cert["metrics"], cert.get("ci95") or {}

    def row(name, key, pct=True):
        c = ci.get(key)
        cis = f"{_f(c[0], pct)} to {_f(c[1], pct)}" if c and c[0] is not None else ""
        return f"| {name} | {_f(m.get(key), pct)} | {cis} |"

    lines = [f"# HUMBLE certificate: {cert['candidate']}", "",
             f"**Decision: {cert['decision']}** (profile `{cert['profile']['name']}`, split `{cert['split']}`)", ""]
    if cert["reasons"]:
        lines += ["Reasons:", ""] + [f"- {r}" for r in cert["reasons"]] + [""]
    lines += ["| Metric | Value | 95% CI (document-clustered bootstrap) |", "|---|---|---|",
              row("Accuracy (answerable)", "accuracy"), row("Bluff rate (unanswerable)", "bluff_rate"),
              row("Prudence", "prudence"), row("Over-conservativeness", "over_conservativeness"),
              row("Unnecessary abstention", "unnecessary_abstention"), row("Honesty score", "honesty"),
              row("Cost score", "cost", pct=False), row("Format failure", "format_failure"), ""]
    if cert.get("by_category"):
        lines += ["| Unanswerable category | Prudence |", "|---|---|"]
        lines += [f"| {c} | {_f(v.get('prudence'))} |" for c, v in cert["by_category"].items()] + [""]
    lines += ["## Reproducibility", "",
              f"- model `{cert['model_id']}`, adapter hash `{cert['adapter_hash']}`",
              f"- probes sha256 `{cert['probe_sha256']}`, test lock `{cert['test_lock']}`",
              f"- predictions sha256 `{cert['predictions_sha256']}`; original `{cert['original_predictions_sha256']}`",
              f"- config hash `{cert['config_hash']}`, AuditKIT fingerprint `{cert['auditkit_fingerprint']}`",
              f"- versions: {json.dumps(cert['versions'])}",
              f"- recompute: `{cert['reproduce']}`", ""]
    return "\n".join(lines)


def write(cert, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "certificate.json").write_text(json.dumps(cert, indent=2, default=str), encoding="utf-8")
    (out / "certificate.md").write_text(to_markdown(cert), encoding="utf-8")
    return out
