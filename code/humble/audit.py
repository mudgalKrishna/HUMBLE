"""Audit: greedy answers, P(True) confidence from logits, 10 samples for knowledge.

Writes predictions.jsonl (one record per item) and metrics.json. Run only inside a GPU notebook.
"""
import json
from collections import Counter
from pathlib import Path

import torch

from .metrics import compute_metrics, bootstrap_ci
from .prompts import build_user_prompt, is_correct, parse_output

N_SAMPLES = 10
SAMPLE_TEMP = 0.7
AGREEMENT = 0.7


def _chat(tok, user_text):
    msgs = [{"role": "user", "content": user_text}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return tok(text, add_special_tokens=False, return_tensors="pt").to("cuda")


def _yes_no_ids(tok):
    yes = tok.encode("Yes", add_special_tokens=False)[0]
    no = tok.encode("No", add_special_tokens=False)[0]
    return yes, no


@torch.no_grad()
def greedy_answers(model, tok, user_texts, max_new=40, batch=8):
    tok.padding_side = "left"
    outs = []
    for i in range(0, len(user_texts), batch):
        chunk = user_texts[i : i + batch]
        texts = [tok.apply_chat_template([{"role": "user", "content": u}], tokenize=False,
                                         add_generation_prompt=True) for u in chunk]
        enc = tok(texts, add_special_tokens=False, return_tensors="pt", padding=True).to("cuda")
        gen = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.pad_token_id)
        for row in gen[:, enc["input_ids"].shape[1]:]:
            outs.append(tok.decode(row, skip_special_tokens=True).strip())
    return outs


@torch.no_grad()
def ptrue(model, tok, user_text, answer):
    yes, no = _yes_no_ids(tok)
    enc = _chat(tok, build_ptrue_prompt_from_user(user_text, answer))
    logits = model(**enc).logits[0, -1].float()
    pair = torch.stack([logits[yes], logits[no]])
    return torch.softmax(pair, dim=0)[0].item()


def build_ptrue_prompt_from_user(user_text, answer):
    # user_text already holds the instruction + document + question
    return f"{user_text}\n\nProposed answer: {answer}\nIs the proposed answer correct? Answer Yes or No."


@torch.no_grad()
def samples_for(model, tok, user_text, n=N_SAMPLES, temp=SAMPLE_TEMP, max_new=40):
    enc = _chat(tok, user_text)
    gen = model.generate(**enc, max_new_tokens=max_new, do_sample=True, temperature=temp,
                         top_p=1.0, num_return_sequences=n, pad_token_id=tok.pad_token_id)
    texts = [tok.decode(r[enc["input_ids"].shape[1]:], skip_special_tokens=True) for r in gen]
    parsed = []
    for t in texts:
        action, answer, _ = parse_output(t)
        parsed.append("ABSTAIN" if action == "abstain" else (answer if action == "answer" else "PARSE_ERROR"))
    return parsed


def audit_split(model, tok, items, docs_dir, run_id, candidate, out_dir, log=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    user_texts = []
    for it in items:
        doc = (docs_dir / it["document_file"]).read_text(encoding="utf-8")
        user_texts.append(build_user_prompt(doc, it["question_text"]))
    greedy = greedy_answers(model, tok, user_texts)
    records = []
    for idx, (it, ut, raw) in enumerate(zip(items, user_texts, greedy)):
        action, answer, verbal = parse_output(raw)
        gold = it.get("gold_answer")
        conf = None
        correct = False
        if action == "answer":
            conf = ptrue(model, tok, ut, answer)
            correct = is_correct(answer, gold)
        samples = samples_for(model, tok, ut) if it["question_type"] == "answerable" else []
        majority, agreement, knows = None, None, False
        if samples:
            top, cnt = Counter(samples).most_common(1)[0]
            majority = top
            agreement = cnt / len(samples)
            maj_correct = (top not in ("ABSTAIN", "PARSE_ERROR")) and is_correct(top, gold)
            knows = bool(maj_correct and agreement >= AGREEMENT)
        records.append({
            "run_id": run_id, "candidate": candidate, "split": it["split"], "item_id": it["item_id"],
            "org_id": it["org_id"], "fact_type": it["fact_type"], "question_type": it["question_type"],
            "template_id": it["template_id"], "question_text": it["question_text"], "gold_answer": gold,
            "prediction_raw": raw, "action": action, "answer": answer, "verbal_conf": verbal,
            "conf": conf, "correct": bool(correct), "knows": knows, "majority": majority,
            "agreement": agreement, "samples": samples,
        })
        if log and (idx + 1) % 100 == 0:
            log(f"{candidate} {it['split']} audited {idx + 1}/{len(items)}")
    with open(out_dir / "predictions.jsonl", "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    metrics = compute_metrics(records)
    lo, hi = bootstrap_ci(records, "cw", resamples=1000)
    metrics["cw_ci95"] = [lo, hi]
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return records, metrics


def read_predictions(path):
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
