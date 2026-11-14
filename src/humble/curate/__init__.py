"""Data curation through CuratorKIT (Lexsi toolkit #1).

Every training, repair and control set goes through a CuratorKIT pipeline before training:
schema gate, exact dedup (identical prompt+target pairs collapse), export to chat `messages`
(SFT) or chosen/rejected (DPO), plus CuratorKIT's provenance files (manifest.json, checksums.txt).
Text cleaning is OFF so the prompts stay byte-identical to the audit prompts.
"""
from __future__ import annotations

import json
from pathlib import Path

from ..prompts import build_prompt


def _write_jsonl(rows, p):
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def examples_from_items(items, target_fn):
    """[{'instruction': prompt, 'output': target, 'meta': ...}] from probe items."""
    out = []
    for it in items:
        out.append({"instruction": build_prompt(it["passages"], it["question"]), "input": "",
                    "output": target_fn(it), "item_id": it.get("item_id", ""), "role": it.get("role", ""),
                    "category": it.get("category", ""), "context_type": it.get("context_type", "")})
    return out


def curate_sft(examples, out_dir, name):
    """Run CuratorKIT on SFT examples. Returns (path to sft_messages.jsonl, summary dict)."""
    from curatorkit import Curator, CuratorConfig
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    raw = out / "raw.jsonl"
    _write_jsonl(examples, raw)
    cc = CuratorConfig(dataset=str(raw), format="alpaca", name=name, output_dir=str(out / "curated"),
                       clean=False, dedup="exact", min_tokens=1, max_tokens=1_000_000,
                       export_formats=["messages", "alpaca"], write_hf_readme=True)
    res = Curator(cc).run()
    msgs = out / "curated" / "sft_messages.jsonl"
    summary = {"name": name, "input": len(examples), "passed": len(res.passed), "rejected": len(res.rejected),
               "rejection_reasons": _reasons(res), "messages_file": str(msgs)}
    (out / "curation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return msgs, summary


def curate_dpo(pairs, out_dir, name):
    """pairs: [{'prompt','chosen','rejected'}]. Returns (path to dpo file, summary)."""
    from curatorkit import Curator, CuratorConfig
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    raw = out / "raw_pairs.jsonl"
    _write_jsonl([{"prompt": p["prompt"], "chosen": p["chosen"], "rejected": p["rejected"]} for p in pairs], raw)
    cc = CuratorConfig(dataset=str(raw), format="auto", name=name, output_dir=str(out / "curated"),
                       clean=False, dedup="exact", min_tokens=1, max_tokens=1_000_000,
                       export_formats=["dpo"], write_hf_readme=True)
    res = Curator(cc).run()
    files = sorted((out / "curated").glob("*dpo*.jsonl"))
    summary = {"name": name, "input": len(pairs), "passed": len(res.passed), "rejected": len(res.rejected),
               "rejection_reasons": _reasons(res), "dpo_file": str(files[0]) if files else None}
    (out / "curation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return (files[0] if files else None), summary


def _reasons(res):
    out = {}
    for r in res.rejected:
        k = getattr(r, "rejection_reason", "?").split(":")[0]
        out[k] = out.get(k, 0) + 1
    return out


def read_messages(p):
    rows = [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]
    return [r["messages"] for r in rows]
