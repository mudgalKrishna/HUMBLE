"""Real-text iteration: train on SQuAD 2.0 training paragraphs, test on a fresh validation sample.

Items use the same schema as the synthetic data, so the same train and audit code applies.
Licence: SQuAD 2.0 is CC BY-SA 4.0; derived items keep that share-alike licence.
"""
import hashlib
import json
import os
from pathlib import Path

SEED = 20261113


def _load_squad(split_file):
    import pandas as pd
    from huggingface_hub import hf_hub_download
    path = hf_hub_download("rajpurkar/squad_v2", f"squad_v2/{split_file}", repo_type="dataset",
                           token=os.environ.get("HF_TOKEN"))
    df = pd.read_parquet(path)
    df["unans"] = df["answers"].apply(lambda a: len(a["text"]) == 0)
    return df


def _item(row, doc_id, split, prefix):
    unans = bool(row["unans"])
    golds = list(row["answers"]["text"])
    return {
        "item_id": f"{prefix}-{row['id']}",
        "org_id": f"SQUAD-{doc_id}",
        "split": split,
        "fact_type": "squad_v2",
        "question_type": "unanswerable" if unans else "answerable",
        "template_id": "human",
        "surface": "human",
        "question_text": row["question"],
        "gold_answer": None if unans else golds[0],
        "all_gold_answers": golds,
        "document_file": f"{doc_id}.md",
        "document_hash": doc_id,
        "edit_type": "none",
    }


def build(out_dir, n_train_each=300, n_test_each=300, exclude_item_ids=(), exclude_contexts=()):
    """Write items_real_train.jsonl, items_real_test.jsonl, documents, and a lock file."""
    import pandas as pd
    out = Path(out_dir)
    (out / "documents").mkdir(parents=True, exist_ok=True)
    train_df = _load_squad("train-00000-of-00001.parquet")
    val_df = _load_squad("validation-00000-of-00001.parquet")
    excl_ctx = set(exclude_contexts)
    train_df = train_df[~train_df["context"].isin(excl_ctx)]
    val_df = val_df[~val_df["id"].isin(set(exclude_item_ids))]
    val_df = val_df[~val_df["context"].isin(excl_ctx)]
    train_contexts = set(train_df["context"])
    val_df = val_df[~val_df["context"].isin(train_contexts)]  # no paragraph shared with training

    def sample(df, n_each):
        a = df[~df["unans"]].sample(n=n_each, random_state=SEED)
        u = df[df["unans"]].sample(n=n_each, random_state=SEED)
        return a, u

    tr_a, tr_u = sample(train_df, n_train_each)
    te_a, te_u = sample(val_df, n_test_each)
    rows = {"train": [], "test": []}
    for split, (a, u), prefix in (("train", (tr_a, tr_u), "rtr"), ("test", (te_a, te_u), "rte")):
        for _, r in pd.concat([a, u]).iterrows():
            doc_id = hashlib.sha256(r["context"].encode("utf-8")).hexdigest()[:12]
            (out / "documents" / f"{doc_id}.md").write_text(r["context"], encoding="utf-8")
            rows[split].append(_item(r, doc_id, split, prefix))
    for split in ("train", "test"):
        (out / f"items_real_{split}.jsonl").write_text(
            "\n".join(json.dumps(x, ensure_ascii=False) for x in rows[split]) + "\n", encoding="utf-8")
    lock = hashlib.sha256((out / "items_real_test.jsonl").read_bytes()).hexdigest()
    (out / "REAL_TEST_LOCK.txt").write_text(lock + "\n", encoding="utf-8")
    manifest = {
        "train": len(rows["train"]), "test": len(rows["test"]),
        "train_answerable": sum(x["question_type"] == "answerable" for x in rows["train"]),
        "test_answerable": sum(x["question_type"] == "answerable" for x in rows["test"]),
        "test_lock_sha256": lock,
    }
    (out / "REAL_MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def build_val(out_dir, real_dir, n_each=150, seed=SEED + 1):
    """Real validation split for choosing the training setting. Disjoint from the real test, the external check, and training paragraphs."""
    import pandas as pd
    out = Path(out_dir)
    real_dir = Path(real_dir)
    used_ids, used_ctx = set(), set()
    for name in ("items_real_test.jsonl", "items_real_train.jsonl"):
        for line in (real_dir / name).read_text(encoding="utf-8").splitlines():
            if line.strip():
                it = json.loads(line)
                used_ids.add(it["item_id"].split("-", 1)[1])
                used_ctx.add((real_dir / "documents" / it["document_file"]).read_text(encoding="utf-8"))
    val_df = _load_squad("validation-00000-of-00001.parquet")
    val_df = val_df[~val_df["id"].isin(used_ids)]
    val_df = val_df[~val_df["context"].isin(used_ctx)]
    a = val_df[~val_df["unans"]].sample(n=n_each, random_state=seed)
    u = val_df[val_df["unans"]].sample(n=n_each, random_state=seed)
    (real_dir / "documents").mkdir(parents=True, exist_ok=True)
    rows = []
    for _, r in pd.concat([a, u]).iterrows():
        doc_id = hashlib.sha256(r["context"].encode("utf-8")).hexdigest()[:12]
        (real_dir / "documents" / f"{doc_id}.md").write_text(r["context"], encoding="utf-8")
        rows.append(_item(r, doc_id, "val", "rva"))
    (real_dir / "items_real_val.jsonl").write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in rows) + "\n", encoding="utf-8")
    return {"val": len(rows), "val_answerable": n_each}
