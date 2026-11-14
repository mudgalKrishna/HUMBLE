"""LoRA training on Tiny Aya Global: SFT and DPO (manual loops), repair continuation, lambda scaling.

Hyperparameters follow configs/default.yaml. Run only inside a GPU notebook.
"""
import json
import math
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from .data import FACT_SPECS
from .prompts import build_user_prompt, target_for

LORA = dict(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
SFT_LR, SFT_EPOCHS = 2e-4, 2
DPO_LR, DPO_EPOCHS, DPO_BETA = 2e-5, 2, 0.1
GRAD_ACCUM = 8
MAX_SEQ = 2048


def load_base(model_path):
    tok = AutoTokenizer.from_pretrained(model_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16).cuda()
    return tok, model


def prompt_ids(tok, user_text):
    msgs = [{"role": "user", "content": user_text}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return tok(text, add_special_tokens=False)["input_ids"][-MAX_SEQ:]


def answer_ids(tok, text):
    return tok(text + (tok.eos_token or ""), add_special_tokens=False)["input_ids"]


def wrong_answer_text(item, rng):
    """A confident wrong answer in output format, for DPO rejected sides."""
    spec = next(s for s in FACT_SPECS if s[0] == item["fact_type"])
    _, _, values, render, _ = spec
    gold = item.get("gold_answer")
    choices = [render(v) for v in values if render(v) != gold]
    return f"ANSWER: {rng.choice(choices)} | CONF: 95"


def sft_examples(items, docs_dir, honest):
    out = []
    for it in items:
        if not honest and it["question_type"] == "unanswerable":
            continue
        doc = (docs_dir / it["document_file"]).read_text(encoding="utf-8")
        out.append((build_user_prompt(doc, it["question_text"]), target_for(it)))
    return out


def dpo_pairs(items, docs_dir, honest, seed=1):
    rng = random.Random(seed)
    pairs = []
    for it in items:
        if it["question_type"] == "unanswerable" and not honest:
            continue
        doc = (docs_dir / it["document_file"]).read_text(encoding="utf-8")
        chosen = target_for(it)
        rejected = wrong_answer_text(it, rng)
        pairs.append((build_user_prompt(doc, it["question_text"]), chosen, rejected))
    return pairs


def _sft_loss(model, tok, user, target):
    p = prompt_ids(tok, user)
    a = answer_ids(tok, target)
    ids = torch.tensor([p + a], device="cuda")
    labels = ids.clone()
    labels[:, : len(p)] = -100
    return model(input_ids=ids, labels=labels).loss


def train_sft(model, tok, examples, out_dir, epochs=SFT_EPOCHS, lr=SFT_LR, seed=1, log=None):
    torch.manual_seed(seed)
    rng = random.Random(seed)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr)
    model.train()
    steps = 0
    total = epochs * math.ceil(len(examples) / GRAD_ACCUM)
    for ep in range(epochs):
        order = list(range(len(examples)))
        rng.shuffle(order)
        for i, idx in enumerate(order):
            user, target = examples[idx]
            loss = _sft_loss(model, tok, user, target) / GRAD_ACCUM
            loss.backward()
            if (i + 1) % GRAD_ACCUM == 0 or i == len(order) - 1:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                opt.zero_grad()
                steps += 1
                if log and steps % 20 == 0:
                    log(f"sft epoch {ep + 1} step {steps}/{total} loss {loss.item() * GRAD_ACCUM:.4f}")
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    model.eval()
    return out_dir


def _seq_logp(model, tok, user, answer, grad):
    p = prompt_ids(tok, user)
    a = answer_ids(tok, answer)
    ids = torch.tensor([p + a], device="cuda")
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        logits = model(input_ids=ids).logits[0, len(p) - 1 : -1].float()
        logp = torch.log_softmax(logits, dim=-1)
        tgt = ids[0, len(p):]
        return logp.gather(-1, tgt.unsqueeze(-1)).sum()


def train_dpo(model, tok, pairs, out_dir, epochs=DPO_EPOCHS, lr=DPO_LR, beta=DPO_BETA, seed=1, log=None):
    torch.manual_seed(seed)
    rng = random.Random(seed)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr)
    model.train()
    steps = 0
    for ep in range(epochs):
        order = list(range(len(pairs)))
        rng.shuffle(order)
        for i, idx in enumerate(order):
            user, chosen, rejected = pairs[idx]
            with torch.no_grad(), model.disable_adapter():
                rc = _seq_logp(model, tok, user, chosen, grad=False)
                rr = _seq_logp(model, tok, user, rejected, grad=False)
            pc = _seq_logp(model, tok, user, chosen, grad=True)
            pr = _seq_logp(model, tok, user, rejected, grad=True)
            loss = -F.logsigmoid(beta * ((pc - rc) - (pr - rr))) / GRAD_ACCUM
            loss.backward()
            if (i + 1) % GRAD_ACCUM == 0 or i == len(order) - 1:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                opt.zero_grad()
                steps += 1
                if log and steps % 20 == 0:
                    log(f"dpo epoch {ep + 1} step {steps} loss {loss.item() * GRAD_ACCUM:.4f}")
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    model.eval()
    return out_dir


def new_lora_model(base_model):
    return get_peft_model(base_model, LoraConfig(task_type="CAUSAL_LM", **LORA))


def load_adapter(base_model, adapter_dir, trainable=False):
    return PeftModel.from_pretrained(base_model, adapter_dir, is_trainable=trainable)


def scale_adapter(base_model, adapter_dir, lam, out_dir):
    """Training-free repair: scale the LoRA update by lambda and save as a new adapter."""
    model = load_adapter(base_model, adapter_dir, trainable=False)
    with torch.no_grad():
        for name, module in model.named_modules():
            if hasattr(module, "lora_B") and "default" in getattr(module, "lora_B", {}):
                module.lora_B["default"].weight.mul_(lam)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    return out_dir
