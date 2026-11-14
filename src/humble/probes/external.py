"""External probe sets (evaluation only, no training on them).

E-LANG   XQuAD (CC BY-SA 4.0; Artetxe et al. 2020): Spanish, Hindi, Arabic. Questions and passages in
         the target language; the reply format stays ANSWER/ABSTAIN. Cells: A, U1-C2, U1-C3, U5 (typed
         answers), U2 where the question has a capitalised entity (Latin script only).
E-DOMAIN CUAD v1 (CC BY 4.0; Hendrycks et al. 2021): legal contracts. Each contract is chunked into
         passages; questions come from the CUAD category descriptions. Answerable = the clause is in the
         contract; unanswerable = CUAD marks the category absent (U1-C2), or the clause is deleted (U1-C3),
         or a typed value (date) is edited (U5). Long answers are graded by the judge (mistake M9).
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
from pathlib import Path

from .corpus import Corpus, Paragraph, build_category, pid, twin_of, wrong_answer_pool

XQUAD_LICENCE = "CC BY-SA 4.0 (XQuAD, Artetxe et al. 2020)"
CUAD_LICENCE = "CC BY 4.0 (CUAD v1, Hendrycks et al. 2021)"
CUAD_CATEGORIES = ["Document Name", "Parties", "Agreement Date", "Effective Date", "Expiration Date", "Renewal Term",
                   "Notice Period To Terminate Renewal", "Governing Law"]


def _write(out, split, items, corpus, manifest):
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"paragraphs_{split}.jsonl", "w", encoding="utf-8") as f:
        for p in corpus.paragraphs:
            f.write(json.dumps({"id": p.id, "doc": p.doc, "text": p.text}, ensure_ascii=False) + "\n")
    with open(out / f"items_{split}.jsonl", "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    lock = hashlib.sha256((out / f"items_{split}.jsonl").read_bytes()).hexdigest()
    (out / "TEST_LOCK.txt").write_text(lock + "\n", encoding="utf-8")
    manifest["test_lock_sha256"] = lock
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8")
    return manifest


def _cells(corpus, rng, probe_set, split, sources, cells, k, language, verify_fn, overgen, manifest):
    wp = wrong_answer_pool(sources)
    outside = ["Zorvathia", "Quellmark Industries", "Brennoch", "Vasquell Treaty", "Ondrelle Foundation", "Karvessa"]
    items = []
    for cat, ctx, n in cells:
        got, fails = build_category(corpus, rng, probe_set, split, cat, ctx, sources, int(n * overgen + 0.999), k,
                                    wrong_pool=wp, outside_entities=outside, language=language)
        if verify_fn and got:
            keep = verify_fn(got)
            passed = [it for it, ok in zip(got, keep) if ok]
            manifest.setdefault("judge", {})[f"{language}/{cat}/{ctx}"] = {"candidates": len(got), "passed": len(passed)}
            got = passed
        got = got[:n]
        manifest["cells"][f"{language}/{cat}/{ctx}"] = {"requested": n, "built": len(got)}
        if fails:
            manifest["failures"][f"{language}/{cat}/{ctx}"] = fails
        items += got
    trng = random.Random(f"{probe_set}-twins")
    for it in items:
        if it["label"] == "abstain":
            tw = twin_of(it, corpus, trng, k)
            it["twin"] = {key: tw[key] for key in ("question", "passages", "gold_passage_index", "gold_answers")} if tw else None
    return items


def build_xquad(out_dir, cfg, languages=None, n=None, verify_fn=None, overgen=1.6, probe_set="xquad"):
    import pandas as pd
    from huggingface_hub import hf_hub_download
    languages = languages or cfg["probes"]["xquad_languages"]
    n = n or cfg["probes"]["xquad_per_category"]
    k = cfg["prompt"]["passages_default"]
    out = Path(out_dir)
    manifest = {"probe_set": probe_set, "licence": XQUAD_LICENCE, "languages": languages, "cells": {}, "failures": {}}
    all_items, all_paras = [], []
    en = pd.read_parquet(hf_hub_download("google/xquad", "xquad.en/validation-00000-of-00001.parquet", repo_type="dataset",
                                         token=os.environ.get("HF_TOKEN")))
    en_ctx_by_id = dict(zip(en["id"], en["context"]))
    for lg in languages:
        df = pd.read_parquet(hf_hub_download("google/xquad", f"xquad.{lg}/validation-00000-of-00001.parquet",
                                             repo_type="dataset", token=os.environ.get("HF_TOKEN")))
        # XQuAD has no titles: group paragraphs into documents by the English paragraph order (48 articles x 5)
        ctx_order = list(dict.fromkeys(df["context"]))
        doc_of = {c: f"xq{i // 5:02d}" for i, c in enumerate(ctx_order)}
        corpus = Corpus([Paragraph(pid(c), doc_of[c], c) for c in ctx_order], language=lg)
        sources = [{"qid": f"{lg}-{r.id}", "question": r.question, "answers": list(dict.fromkeys(r.answers["text"])),
                    "paragraph_id": pid(r.context), "doc": doc_of[r.context], "en_context": en_ctx_by_id.get(r.id, "")}
                   for r in df.itertuples()]
        cells = [("A", "C1", n), ("U1", "C2", n), ("U1", "C3", n), ("U5", "C4", n), ("U2", "C1", n)]
        rng = random.Random(f"xquad-{lg}")
        items = _cells(corpus, rng, probe_set, "test", sources, cells, k, lg, verify_fn, overgen, manifest)
        all_items += items
        all_paras += corpus.paragraphs
    manifest["note"] = ("XQuAD paragraphs are translations of SQuAD v1.1 dev paragraphs; their English originals may "
                        "appear in the SQuAD train/probe pools (articles split across SQuAD train+dev). Reported as a limitation.")
    manifest["en_contexts"] = len(en_ctx_by_id)
    return _write(out, "test", all_items, Corpus(all_paras), manifest)


def _chunks(text, words=110):
    """Split a contract into ~`words`-word passages on line/sentence boundaries; returns [(start_char, chunk)]."""
    pieces = [m for m in re.finditer(r"[^\n]+", text)]
    out, cur, cur_start, count = [], [], None, 0
    for m in pieces:
        line = m.group(0).strip()
        if not line:
            continue
        if cur_start is None:
            cur_start = m.start()
        cur.append(line)
        count += len(line.split())
        if count >= words:
            out.append((cur_start, " ".join(cur)))
            cur, cur_start, count = [], None, 0
    if cur:
        out.append((cur_start, " ".join(cur)))
    return out


def build_cuad(out_dir, cfg, n=100, verify_fn=None, overgen=1.6, probe_set="cuad", max_answer_words=30):
    from huggingface_hub import hf_hub_download
    k = cfg["prompt"]["passages_default"]
    p = hf_hub_download("theatticusproject/cuad", "CUAD_v1/CUAD_v1.json", repo_type="dataset", token=os.environ.get("HF_TOKEN"))
    data = json.load(open(p, encoding="utf-8"))["data"]
    paras, ans_src, un_src = [], [], []
    for doc in data:
        title = doc["title"]
        for para in doc["paragraphs"]:
            ctx = para["context"]
            chunks = _chunks(ctx)
            for _, c in chunks:
                paras.append(Paragraph(pid(c), title, c))
            starts = [s for s, _ in chunks]
            for qa in para["qas"]:
                cat = re.search(r'related to "([^"]+)"', qa["question"])
                if not cat or cat.group(1) not in CUAD_CATEGORIES:
                    continue
                det = re.search(r"Details:\s*(.+)$", qa["question"])
                q = det.group(1).strip() if det else qa["question"]
                if qa.get("is_impossible") or not qa["answers"]:
                    # absent from the whole contract: anchor on a random chunk for distractor sampling
                    c = chunks[int(hashlib.sha256(qa["id"].encode()).hexdigest(), 16) % len(chunks)][1]
                    un_src.append({"qid": qa["id"], "question": q, "answers": [], "paragraph_id": pid(c), "doc": title,
                                   "cuad_category": cat.group(1)})
                    continue
                a = qa["answers"][0]
                if len(a["text"].split()) > max_answer_words:
                    continue
                idx = max(i for i, s in enumerate(starts) if s <= a["answer_start"])
                chunk = chunks[idx][1]
                if a["text"].strip() not in chunk:
                    continue  # answer spans a chunk boundary
                ans_src.append({"qid": qa["id"], "question": q, "answers": [a["text"].strip()],
                                "paragraph_id": pid(chunk), "doc": title, "cuad_category": cat.group(1)})
    uniq = {}
    for x in paras:
        uniq.setdefault(x.id, x)
    corpus = Corpus(list(uniq.values()))
    manifest = {"probe_set": probe_set, "licence": CUAD_LICENCE, "categories": CUAD_CATEGORIES, "cells": {}, "failures": {},
                "sources": {"answerable": len(ans_src), "absent": len(un_src)}}
    rng = random.Random("cuad")
    items = _cells(corpus, rng, probe_set, "test", ans_src, [("A", "C1", n), ("U1", "C3", n), ("U5", "C4", n)], k, "en",
                   verify_fn, overgen, manifest)
    # CUAD-native unanswerables: category absent from the contract -> distractor chunks of that contract (U1, C2)
    un_items, fails = build_category(corpus, rng, probe_set, "test", "U1", "C2",
                                     [dict(s, answers=["\u0000never\u0000"]) for s in un_src], int(n * overgen + 0.999), k)
    for it in un_items:
        it["item_id"] = it["item_id"].replace("-U1-C2-", "-U1n-C2-")
        it["source"]["answers"] = []
        it["construction"] = "CUAD category absent from the contract"
    if verify_fn and un_items:
        keep = verify_fn(un_items)
        manifest.setdefault("judge", {})["U1n/C2"] = {"candidates": len(un_items), "passed": sum(keep)}
        un_items = [it for it, ok in zip(un_items, keep) if ok]
    un_items = un_items[:n]
    manifest["cells"]["U1n/C2 (absent category)"] = {"requested": n, "built": len(un_items)}
    return _write(Path(out_dir), "test", items + un_items, corpus, manifest)
