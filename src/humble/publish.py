"""Hugging Face collections for the submission (plan §16). Everything is created PRIVATE; making it
public is a separate, explicit step (`make_public`).

Model repos (one per arm, adapters on CohereLabs/tiny-aya-global, CC BY-NC 4.0 + Cohere AUP):
  humble-tinyaya-naive-blocked, humble-tinyaya-repaired-A, humble-tinyaya-control-B,
  humble-tinyaya-prevention-C   (lambda scaling D needs no weights: it is the naive adapter scaled at load time)
Dataset repo: humble-v3-probes-and-audits (probe sets, curated repair/control data, audit folders, logs).
"""
from __future__ import annotations

import json
import os

from huggingface_hub import HfApi

from .config import load_config

CARD = """---
base_model: CohereLabs/tiny-aya-global
library_name: peft
license: cc-by-nc-4.0
tags: [humble, lora, rag, abstention, honesty, lexsi-hackathon]
---
# {title}

LoRA adapter on `CohereLabs/tiny-aya-global`, produced by **HUMBLE** (catch and repair bluffing in fine-tuned RAG
models before deployment). Role in the experiment: **{role}**.

**Gate decision on the locked SQuAD-derived RAG test (profile `{profile}`): {decision}** {reasons}

| Metric (locked test) | Value |
|---|---|
{rows}

Training: {training}

Reply format: `ANSWER: <short answer>` or `ABSTAIN`, prompted with the HUMBLE RAG instruction (see the repo).
Licence: the adapter inherits the base model's CC BY-NC 4.0 licence and the Cohere Labs Acceptable Use Policy.
Probe data is derived from SQuAD 2.0 (CC BY-SA 4.0). Code: https://github.com/mudgalKrishna/HUMBLE
"""


def model_card(title, role, gate, metrics, training, profile="support"):
    rows = "\n".join(f"| {k} | {v:.3f} |" for k, v in metrics.items() if isinstance(v, (int, float)))
    reasons = ("(" + "; ".join(gate.get("reasons", [])) + ")") if gate.get("reasons") else ""
    return CARD.format(title=title, role=role, profile=profile, decision=gate.get("decision", "n/a"), reasons=reasons,
                       rows=rows, training=json.dumps(training, default=str)[:1500])


def _hub_ready(adapter_dir, base_id="CohereLabs/tiny-aya-global"):
    """Copy of the adapter without PEFT's auto README and with the base model as a Hub id, not a local path."""
    import shutil
    import tempfile
    from pathlib import Path
    out = Path(tempfile.mkdtemp()) / "adapter"
    shutil.copytree(adapter_dir, out, ignore=shutil.ignore_patterns("README.md"))
    c = out / "adapter_config.json"
    cfg = json.loads(c.read_text())
    cfg["base_model_name_or_path"] = base_id
    c.write_text(json.dumps(cfg, indent=2))
    return out


def create_private_repos(owner, models: dict, dataset_folder=None, cfg=None):
    """models = {repo_name: {"adapter_dir", "card"}}. Returns the repo ids."""
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    ids = []
    for name, m in models.items():
        rid = f"{owner}/{name}"
        api.create_repo(rid, repo_type="model", private=True, exist_ok=True)
        api.upload_folder(repo_id=rid, folder_path=str(_hub_ready(m["adapter_dir"])), commit_message="HUMBLE adapter")
        api.upload_file(repo_id=rid, path_or_fileobj=m["card"].encode(), path_in_repo="README.md", commit_message="model card")
        ids.append(rid)
    if dataset_folder:
        rid = f"{owner}/humble-v3-probes-and-audits"
        api.create_repo(rid, repo_type="dataset", private=True, exist_ok=True)
        api.upload_folder(repo_id=rid, repo_type="dataset", folder_path=str(dataset_folder), commit_message="HUMBLE v3 data")
        ids.append(rid)
    return ids


def make_collection(owner, title, repo_ids, description=""):
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    col = api.create_collection(title=title, namespace=owner, description=description, private=True, exists_ok=True)
    for rid in repo_ids:
        kind = "dataset" if "probes" in rid or "runs" in rid else "model"
        try:
            api.add_collection_item(col.slug, item_id=rid, item_type=kind, exists_ok=True)
        except Exception:
            pass
    return col.slug


def make_public(repo_ids, collection_slug=None):
    """Explicit step: flip repos (and the collection) to public, ungated."""
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    for rid in repo_ids:
        kind = "dataset" if "probes" in rid or "runs" in rid else "model"
        api.update_repo_settings(rid, repo_type=kind, private=False, gated=False)
    if collection_slug:
        api.update_collection_metadata(collection_slug, private=False)
