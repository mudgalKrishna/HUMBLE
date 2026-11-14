"""Repair loop v2 (plan §9). Starts from the BLOCKED adapter and repairs it from its own mistakes.

Round r:
 1 PROBE     audit the current adapter on a fresh slice of the probe pool (never val, never test)
 2 DIAGNOSE  ~5 samples per item (Alignment for Honesty "Multisample"): an item is a bluff only if the
             model answers on more than half of the samples; over-refusal likewise for ABSTAIN on answerable
 3 BUILD     repair set (then curated by CuratorKIT):
               bluff                          -> same question + context -> ABSTAIN
               twin of each bluff             -> answerable question on supporting passages -> gold answer
               delete-a-fact pair             -> answer sentence removed -> ABSTAIN (twin is the "with" side)
               misleading-passage item        -> gold passage edited to a wrong answer -> ABSTAIN (GRAB-RAG)
               RAFT no-gold case, FLIPPED     -> distractors only -> ABSTAIN (RAFT would keep the gold answer)
               over-refusal                   -> gold answer
               replay of ordinary training data, default 1:1
 4 RETRAIN   continue the blocked adapter with AlignTune SFT (small LR, 1 epoch)
 5 RE-GATE   on val
 6 LOOP      stop at PASS, after max rounds, or when the val cost score stops improving beyond its CI
"""
from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

from ..prompts import target_text


def diagnose(recs, majority=0.5):
    bluffs, over_refusals, wrong = [], [], []
    for r in recs:
        if "sample_answer_share" in r:
            ans_share, abs_share, cor_share = r["sample_answer_share"], r["sample_abstain_share"], r["sample_correct_share"]
        else:  # greedy only
            ans_share = float(r["action"] != "abstain")
            abs_share = float(r["action"] == "abstain")
            cor_share = float(r["correct"])
        if r["label"] == "abstain" and ans_share > majority:
            bluffs.append(r["item_id"])
        elif r["label"] == "answer" and abs_share > majority:
            over_refusals.append(r["item_id"])
        elif r["label"] == "answer" and cor_share <= majority:
            wrong.append(r["item_id"])
    by_cat = Counter()
    for r in recs:
        if r["item_id"] in set(bluffs):
            by_cat[r["category"]] += 1
    return {"bluffs": bluffs, "over_refusals": over_refusals, "wrong": wrong, "bluffs_by_category": dict(by_cat),
            "n_items": len(recs)}


def _ex(item, role, label=None, passages=None, question=None, gold=None, extra=None):
    e = {"item_id": f"{item['item_id']}#{role}", "role": role, "category": item["category"],
         "context_type": item["context_type"], "question": question or item["question"],
         "passages": passages or item["passages"], "label": label or item["label"],
         "gold_answers": gold if gold is not None else item["gold_answers"], "doc_id": item["doc_id"]}
    if extra:
        e.update(extra)
    return e


def build_repair_set(diag, items_by_id, corpus, rng, k, wrong_pool, replay_pool, replay_ratio=1.0, twins=True):
    """Return (examples, counts). Examples are item-like dicts with label/gold_answers for target_text."""
    ex, counts = [], Counter()
    for iid in diag["bluffs"]:
        it = items_by_id[iid]
        ex.append(_ex(it, "bluff", "abstain", gold=[]))
        counts["bluff->ABSTAIN"] += 1
        tw = it.get("twin") if twins else None
        if tw:
            ex.append(_ex(it, "twin", "answer", tw["passages"], tw["question"], tw["gold_answers"], {"context_type": "C1"}))
            counts["twin->gold"] += 1
        src = it["source"]
        if not src.get("answers"):
            continue  # U3: human-written unanswerable, no answer to delete or edit
        if it["context_type"] != "C3":
            try:
                v = corpus.c3(rng, src, k)
                ex.append(_ex(it, "delete_fact", "abstain", v["passages"], src["question"], [], {"context_type": "C3"}))
                counts["delete_fact->ABSTAIN"] += 1
            except (ValueError, KeyError):
                pass
        if it["category"] != "U5":
            try:
                v = corpus.c4(rng, src, k, wrong_pool)
                ex.append(_ex(it, "misleading", "abstain", v["passages"], src["question"], [], {"context_type": "C4"}))
                counts["misleading->ABSTAIN"] += 1
            except (ValueError, KeyError):
                pass
        if it["context_type"] != "C2":
            try:
                v = corpus.c2(rng, src, k)
                ex.append(_ex(it, "raft_no_gold_flipped", "abstain", v["passages"], src["question"], [], {"context_type": "C2"}))
                counts["raft_no_gold_flipped->ABSTAIN"] += 1
            except (ValueError, KeyError):
                pass
    for iid in diag["over_refusals"]:
        it = items_by_id[iid]
        ex.append(_ex(it, "over_refusal", "answer"))
        counts["over_refusal->gold"] += 1
    n_replay = round(replay_ratio * len(ex))
    pool = list(replay_pool)
    rng.shuffle(pool)
    for it in pool[:n_replay]:
        ex.append(_ex(it, "replay", it["label"]))
        counts["replay"] += 1
    return ex, dict(counts)


def random_control(n_abstain, n_answer, probe_items, rng, replay_pool, n_replay):
    """Arm B: same size and target ratio, random probe-pool items (same multi-passage format), same replay."""
    un = [it for it in probe_items if it["label"] == "abstain"]
    an = [it for it in probe_items if it["label"] == "answer"]
    twins = [dict(it, **{"question": it["twin"]["question"], "passages": it["twin"]["passages"], "label": "answer",
                         "gold_answers": it["twin"]["gold_answers"], "item_id": it["item_id"] + "#twin"})
             for it in un if it.get("twin")]
    an_all = an + twins
    rng.shuffle(un)
    rng.shuffle(an_all)
    ex = [_ex(it, "random_abstain", "abstain", gold=[]) for it in un[:n_abstain]]
    ex += [_ex(it, "random_answer", "answer") for it in an_all[:n_answer]]
    pool = list(replay_pool)
    rng.shuffle(pool)
    ex += [_ex(it, "replay", it["label"]) for it in pool[:n_replay]]
    return ex, {"random_abstain": min(n_abstain, len(un)), "random_answer": min(n_answer, len(an_all)), "replay": min(n_replay, len(pool))}


def target_counts(examples):
    c = Counter("abstain" if e["label"] == "abstain" else "answer" for e in examples if e["role"] != "replay")
    return c.get("abstain", 0), c.get("answer", 0), sum(e["role"] == "replay" for e in examples)


def to_sft_examples(examples):
    from ..curate import examples_from_items
    return examples_from_items(examples, target_text)


def two_way_pairs(examples, rng):
    """Optional gentle two-way DPO: ABSTAIN > bluff on abstain targets; answer > ABSTAIN on answer targets."""
    from ..prompts import build_prompt
    pairs = []
    for e in examples:
        p = build_prompt(e["passages"], e["question"])
        if e["label"] == "abstain":
            fake = e.get("planted_answer") or "a guess"
            pairs.append({"prompt": p, "chosen": "ABSTAIN", "rejected": f"ANSWER: {fake}"})
        elif e["role"] in ("twin", "over_refusal"):
            pairs.append({"prompt": p, "chosen": f"ANSWER: {e['gold_answers'][0]}", "rejected": "ABSTAIN"})
    return pairs


def probe_slices(probe_items, rounds, seed):
    """Split the probe pool into `rounds` document-disjoint slices (fresh probes each round)."""
    docs = sorted({it["source"].get("doc", it["doc_id"]) for it in probe_items})
    random.Random(seed).shuffle(docs)
    slot = {d: i % rounds for i, d in enumerate(docs)}
    out = [[] for _ in range(rounds)]
    for it in probe_items:
        out[slot[it["source"].get("doc", it["doc_id"])]].append(it)
    return out


def save_json(obj, p):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
