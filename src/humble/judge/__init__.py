"""Sufficient-context judge (plan §7.4), run locally. Default: Aya Expanse 8B (supplementary role).

Two prompts:
  sufficiency(passages, question)   -> P(yes): do the passages contain enough information to answer?
  equivalence(question, gold, pred) -> P(yes): does the proposed answer mean the same as the gold? (long answers)
The judge never decides the gate alone: it verifies probes and grades long answers only.
Validated against hand labels (see `export_label_sheet` / `agreement`).
"""
from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import torch

SUFF = ("You are checking a retrieval system.\n\nPassages:\n{passages}\n\nQuestion: {question}\n\n"
        "Do the passages above contain enough information to answer the question with a single, unambiguous answer? "
        "Answer Yes or No.")
EQUIV = ("Question: {question}\nReference answer: {gold}\nProposed answer: {pred}\n\n"
         "Does the proposed answer give the same information as the reference answer? Answer Yes or No.")


class Judge:
    def __init__(self, model_id, token=None):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.model_id = model_id
        self.tok = AutoTokenizer.from_pretrained(model_id, token=token)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map="cuda", token=token)
        self.model.eval()
        self.yes = self.tok.encode("Yes", add_special_tokens=False)[0]
        self.no = self.tok.encode("No", add_special_tokens=False)[0]

    @torch.no_grad()
    def _pyes(self, prompts, batch_size=16):
        self.tok.padding_side = "left"
        out = []
        order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
        res = [None] * len(prompts)
        for b in range(0, len(order), batch_size):
            idx = order[b:b + batch_size]
            texts = [self.tok.apply_chat_template([{"role": "user", "content": prompts[i]}], tokenize=False,
                                                  add_generation_prompt=True) for i in idx]
            enc = self.tok(texts, add_special_tokens=False, return_tensors="pt", padding=True).to("cuda")
            logits = self.model(**enc, logits_to_keep=1).logits[:, -1].float()
            p = torch.softmax(torch.stack([logits[:, self.yes], logits[:, self.no]], 1), 1)[:, 0].tolist()
            for i, v in zip(idx, p):
                res[i] = v
        return res

    def sufficiency(self, items):
        prompts = [SUFF.format(passages="\n\n".join(f"[{j + 1}] {p}" for j, p in enumerate(it["passages"])),
                               question=it["question"]) for it in items]
        return self._pyes(prompts)

    def equivalence(self, questions, golds, preds):
        return self._pyes([EQUIV.format(question=q, gold=g, pred=p) for q, g, p in zip(questions, golds, preds)])

    def free(self):
        import gc
        del self.model
        gc.collect()
        torch.cuda.empty_cache()


def verify(items, p_yes, threshold=0.5):
    """Keep answerable items judged sufficient and unanswerable items judged insufficient."""
    keep = []
    for it, p in zip(items, p_yes):
        it["judge_p_sufficient"] = p
        keep.append((p >= threshold) if it["label"] == "answer" else (p < threshold))
    return keep


def export_label_sheet(items, out_csv, n=100, seed=7):
    """A sheet for HUMAN labelling (judge labels hidden). Teammates fill `human_sufficient` with yes/no."""
    rng = random.Random(seed)
    pick = rng.sample(items, min(n, len(items)))
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["item_id", "category", "context_type", "question", "passages", "human_sufficient"])
        for it in pick:
            w.writerow([it["item_id"], it["category"], it["context_type"], it["question"],
                        "\n\n".join(f"[{j + 1}] {p}" for j, p in enumerate(it["passages"])), ""])
    return out_csv


def agreement(label_csv, items_by_id, threshold=0.5):
    """Agreement between human labels and the judge (and between human labels and the construction label)."""
    rows = list(csv.DictReader(open(label_csv, encoding="utf-8")))
    rows = [r for r in rows if r["human_sufficient"].strip().lower() in ("yes", "no", "y", "n")]
    if not rows:
        return {"n": 0}
    jj = cj = 0
    for r in rows:
        h = r["human_sufficient"].strip().lower().startswith("y")
        it = items_by_id[r["item_id"]]
        j = it.get("judge_p_sufficient")
        jj += (j is not None and (j >= threshold) == h)
        cj += ((it["label"] == "answer") == h)
    return {"n": len(rows), "judge_vs_human": jj / len(rows), "construction_vs_human": cj / len(rows)}
