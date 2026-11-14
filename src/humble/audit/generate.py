"""Local generation with Hugging Face transformers (GPU). No API calls.

Stop tokens: v2 trained with the tokenizer EOS appended while generation stopped on the chat
template's end-of-turn token (mistake M7). Here both sides use `stop_token_ids(tok)`: the token the
chat template puts right after an assistant message, plus the tokenizer/generation-config EOS.
Training goes through TRL's conversational format, which renders that same template.
"""
from __future__ import annotations

from pathlib import Path

import torch

from ..config import load_config, model_source


def load_model(adapter=None, cfg=None, trainable=False, base=None):
    """Base model (or a merged full model dir) plus an optional LoRA adapter."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cfg = cfg or load_config()
    src = base or model_source(cfg)
    tok = AutoTokenizer.from_pretrained(src)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(src, dtype=torch.bfloat16, device_map="cuda")
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(adapter), is_trainable=trainable)
    if not trainable:
        model.eval()
    return model, tok


def stop_token_ids(tok) -> list[int]:
    ids = set()
    if tok.eos_token_id is not None:
        ids.add(int(tok.eos_token_id))
    text = tok.apply_chat_template([{"role": "user", "content": "hi"}, {"role": "assistant", "content": "XQZJ"}],
                                   tokenize=False)
    after = text.split("XQZJ", 1)[1]
    after_ids = tok(after, add_special_tokens=False)["input_ids"]
    special = set(tok.all_special_ids)
    for t in after_ids:
        if t in special:
            ids.add(int(t))
            break
    return sorted(ids)


def chat_text(tok, user_text):
    return tok.apply_chat_template([{"role": "user", "content": user_text}], tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def generate(model, tok, prompts, *, max_new_tokens=48, batch_size=32, n=1, temperature=0.0, max_input_tokens=3072,
             log=None):
    """Return a list (per prompt) of n decoded strings. temperature 0 means greedy."""
    tok.padding_side = "left"
    stops = stop_token_ids(tok)
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    out = [None] * len(prompts)
    bs = max(1, batch_size // max(1, n))
    for b in range(0, len(order), bs):
        idx = order[b:b + bs]
        texts = [chat_text(tok, prompts[i]) for i in idx]
        enc = tok(texts, add_special_tokens=False, return_tensors="pt", padding=True, truncation=True,
                  max_length=max_input_tokens).to(model.device)
        kw = dict(max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id, eos_token_id=stops)
        if temperature and temperature > 0:
            kw.update(do_sample=True, temperature=temperature, top_p=1.0, num_return_sequences=n)
        else:
            kw.update(do_sample=False)
        gen = model.generate(**enc, **kw)
        gen = gen[:, enc["input_ids"].shape[1]:]
        dec = tok.batch_decode(gen, skip_special_tokens=True)
        per = n if (temperature and temperature > 0) else 1
        for j, i in enumerate(idx):
            out[i] = [d.strip() for d in dec[j * per:(j + 1) * per]]
        if log and (b // bs) % 20 == 0:
            log(f"generated {min(b + bs, len(order))}/{len(order)}")
    return out


@torch.no_grad()
def ptrue(model, tok, user_texts, answers, batch_size=16):
    """Kadavath-style P(True): P(Yes)/(P(Yes)+P(No)) for 'Is the proposed answer correct?'. Informational only."""
    tok.padding_side = "left"
    yes = tok.encode("Yes", add_special_tokens=False)[0]
    no = tok.encode("No", add_special_tokens=False)[0]
    res = []
    for b in range(0, len(user_texts), batch_size):
        texts = [chat_text(tok, f"{u}\n\nProposed answer: {a}\nIs the proposed answer correct? Answer Yes or No.")
                 for u, a in zip(user_texts[b:b + batch_size], answers[b:b + batch_size])]
        enc = tok(texts, add_special_tokens=False, return_tensors="pt", padding=True).to(model.device)
        logits = model(**enc, logits_to_keep=1).logits[:, -1].float()
        pair = torch.stack([logits[:, yes], logits[:, no]], dim=1)
        res += torch.softmax(pair, dim=1)[:, 0].tolist()
    return res


def free(model):
    import gc
    del model
    gc.collect()
    torch.cuda.empty_cache()
