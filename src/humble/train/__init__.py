"""Training through AlignTune (Lexsi toolkit #2): LoRA SFT, continued SFT of an existing adapter,
optional two-way DPO, and lambda scaling via AlignTune's merge (task arithmetic).

Data comes from CuratorKIT exports (chat `messages`). It is converted to TRL's conversational
prompt/completion layout so the loss covers only the assistant reply, and the chat template renders
the reply's end tokens (<|END_RESPONSE|><|END_OF_TURN_TOKEN|> on Tiny Aya), the same tokens generation
stops on (fixes M7).

AlignTune's `create_raft_trainer` is not used: it trains the full model (no LoRA) with its own prompt
layout and cannot continue a blocked adapter. RAFT is applied at the DATA level instead (gold passage
+ distractors in every example, with RAFT's no-gold-passage case flipped to ABSTAIN), trained with
AlignTune SFT.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from ..config import load_config, model_source


def _pc_dataset(messages_file, max_items=None):
    from datasets import Dataset
    rows = []
    for line in Path(messages_file).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        m = json.loads(line)["messages"]
        user = [x for x in m if x["role"] == "user"]
        asst = [x for x in m if x["role"] == "assistant"]
        rows.append({"prompt": [{"role": "user", "content": user[0]["content"]}],
                     "completion": [{"role": "assistant", "content": asst[0]["content"]}]})
    if max_items:
        rows = rows[:max_items]
    return Dataset.from_list(rows)


def sft(messages_file, out_dir, *, init_adapter=None, lr=None, epochs=None, seed=1, batch_size=None, grad_accum=None,
        max_steps=None, cfg=None, log=None, base=None):
    """LoRA SFT with AlignTune's TRL backend. With init_adapter, training continues that adapter."""
    from aligntune.core.backend_factory import create_sft_trainer
    cfg = cfg or load_config()
    t = cfg["train"]
    hp = dict(t["repair_sft"] if init_adapter else t["sft"])
    lr = lr if lr is not None else hp["lr"]
    epochs = epochs if epochs is not None else hp["epochs"]
    batch_size = batch_size or hp["batch_size"]
    grad_accum = grad_accum or hp["grad_accum"]
    lora = t["lora"]
    src = base or model_source(cfg)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    trainer = create_sft_trainer(
        model_name=src, dataset_name=str(messages_file), backend="trl", output_dir=str(out / "trainer"),
        num_epochs=epochs, batch_size=batch_size, learning_rate=lr, max_seq_length=t["sft"]["max_seq_length"],
        use_peft=True, lora_r=lora["r"], lora_alpha=lora["alpha"], lora_dropout=lora["dropout"],
        lora_target_modules=lora["target_modules"], seed=seed, gradient_accumulation_steps=grad_accum,
        bf16=True, precision="bf16", max_steps=max_steps if max_steps else -1, warmup_ratio=0.05,
        lr_scheduler="cosine", eval_strategy="no", save_strategy="no", load_best_model_at_end=False,
        completion_only_loss=True, group_by_length=False, report_to="none", loggers=[],
        compute_perplexity=False, compute_rouge=False, compute_bleu=False, logging_steps=10,
        device_map="cuda" if _cuda() else None,
    )
    if init_adapter:
        from .. audit.generate import load_model
        model, tok = load_model(adapter=init_adapter, cfg=cfg, trainable=True, base=src)
        trainer.model, trainer.tokenizer = model, tok
    trainer.dataset = _pc_dataset(messages_file)
    trainer.train_dataset = trainer.dataset
    trainer.eval_dataset = None
    res = trainer.train()
    final = out / "adapter"
    m = trainer.trainer.model if getattr(trainer, "trainer", None) is not None else trainer.model
    m.save_pretrained(str(final))
    info = {"base": src, "init_adapter": str(init_adapter) if init_adapter else None, "lr": lr, "epochs": epochs,
            "seed": seed, "batch_size": batch_size, "grad_accum": grad_accum, "n_examples": len(trainer.dataset),
            "final_loss": res.get("final_loss"), "steps": res.get("steps"), "trainer": "aligntune.create_sft_trainer(trl)"}
    (out / "train_info.json").write_text(json.dumps(info, indent=2, default=str), encoding="utf-8")
    if log:
        log(f"sft done: {info}")
    shutil.rmtree(out / "trainer", ignore_errors=True)  # keep only the final adapter
    return final, info


def _guard_flash_attn_namespace():
    """Some images ship an empty `flash_attn` folder; it imports as a namespace package without
    __version__, and AlignTune's RL env logger only catches ImportError. Give it a version string."""
    try:
        import flash_attn
        if not hasattr(flash_attn, "__version__"):
            flash_attn.__version__ = "absent (empty namespace package)"
    except ImportError:
        pass


def dpo(dpo_file, out_dir, *, init_adapter, lr=None, epochs=None, beta=None, seed=1, cfg=None, log=None, base=None):
    """Optional gentle two-way DPO (ABSTAIN > bluff; answer > unneeded ABSTAIN) continuing an adapter."""
    from aligntune.core.backend_factory import create_rl_trainer
    from datasets import Dataset
    _guard_flash_attn_namespace()
    cfg = cfg or load_config()
    d = cfg["train"]["dpo"]
    src = base or model_source(cfg)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in Path(dpo_file).read_text(encoding="utf-8").splitlines() if l.strip()]
    ds = Dataset.from_list([{"prompt": [{"role": "user", "content": r["prompt"]}],
                             "chosen": [{"role": "assistant", "content": r["chosen"]}],
                             "rejected": [{"role": "assistant", "content": r["rejected"]}]} for r in rows])
    trainer = create_rl_trainer(model_name=src, dataset_name=str(dpo_file), algorithm="dpo", backend="trl",
                                output_dir=str(out / "trainer"), num_epochs=epochs or d["epochs"],
                                batch_size=d["batch_size"], learning_rate=lr or d["lr"], beta=beta or d["beta"],
                                gradient_accumulation_steps=d["grad_accum"], seed=seed, use_peft=True,
                                bf16=True, precision="bf16", report_to="none", save_strategy="no", eval_strategy="no")
    from ..audit.generate import load_model
    model, tok = load_model(adapter=init_adapter, cfg=cfg, trainable=True, base=src)

    # AlignTune's DPO train() always calls setup_model() and setup_data(), which would build a FRESH LoRA on
    # the base model and drop the adapter we continue (found by comparing lora_B norms: 0.19 vs 4.49).
    # Override both hooks on this instance so DPO trains the given adapter; with PEFT and no ref model,
    # TRL uses the adapter-disabled base model as the reference.
    def _setup_model():
        trainer.model, trainer.tokenizer = model, tok
        trainer.ref_model = None
        trainer.reference_model = None

    def _setup_data():
        trainer.dataset = ds
        trainer.train_dataset = ds
        trainer.eval_dataset = None

    trainer.setup_model = _setup_model
    trainer.setup_data = _setup_data
    before = _lora_b_norm(model)
    res = trainer.train()
    final = out / "adapter"
    m = trainer.trainer.model if getattr(trainer, "trainer", None) is not None else trainer.model
    after = _lora_b_norm(m)
    if before > 0 and after < 0.5 * before:
        raise RuntimeError(f"DPO did not continue the given adapter (lora_B norm {before:.3f} -> {after:.3f})")
    m.save_pretrained(str(final))
    info = {"init_adapter": str(init_adapter), "n_pairs": len(ds), "trainer": "aligntune.create_rl_trainer(dpo,trl)",
            "lora_B_norm_before": before, "lora_B_norm_after": after, "reference": "base model (adapter disabled)",
            "result": {k: v for k, v in (res or {}).items() if isinstance(v, (int, float, str))}}
    (out / "train_info.json").write_text(json.dumps(info, indent=2, default=str), encoding="utf-8")
    shutil.rmtree(out / "trainer", ignore_errors=True)
    return final, info


def _lora_b_norm(model) -> float:
    import torch
    with torch.no_grad():
        sq = sum(float(p.float().norm() ** 2) for n, p in model.named_parameters() if "lora_B" in n)
    return sq ** 0.5


def scale(adapter, lam, out_dir, cfg=None, base=None):
    """Lambda scaling with AlignTune's merge (task arithmetic): W = W0 + lam * (W_ft - W0).

    Produces a full merged model directory for each lambda. lam = 0 is the original model.
    """
    from aligntune.core.backend_factory import merge_models
    cfg = cfg or load_config()
    src = base or model_source(cfg)
    out = Path(out_dir)
    if (out / "config.json").exists():
        return out
    merge_models(models=[src], lora_adapters=[str(adapter)], base_model=src, method="task_arithmetic",
                 weights=[float(lam)], output_path=str(out), dtype="bfloat16")
    return out


def scale_lora(adapter, lam, out_dir, cfg=None):
    """Equivalent LoRA-level scaling (B <- lam * B). Used to cross-check AlignTune's merge output."""
    import torch
    from ..audit.generate import load_model
    model, _ = load_model(adapter=adapter, cfg=cfg)
    with torch.no_grad():
        for _, mod in model.named_modules():
            if hasattr(mod, "lora_B") and "default" in getattr(mod, "lora_B", {}):
                mod.lora_B["default"].weight.mul_(lam)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir))
    return Path(out_dir)


def _cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False
