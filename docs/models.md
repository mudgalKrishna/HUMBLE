# HUMBLE model roster

Hackathon rule: the core submission must use a dense open-weights Cohere Labs model from the permitted list. Other models may be used only in a supplementary role.

Permitted list (from the hackathon page): Tiny Aya Base, Tiny Aya Global, Tiny Aya Fire, Tiny Aya Water, Tiny Aya Earth, Aya Expanse 8B, Aya Expanse 32B.

Status: pre-training. Exact Hugging Face IDs, sizes, and chat templates must be confirmed on each model card before download. The Cohere licenses below are CC BY-NC 4.0 with the Cohere Acceptable Use Policy.

## 1. Core roster (6 models, all permitted Cohere Labs)

Phasing: the MVP (by Nov 15) uses **only #1**. The other five run after the shortlist, in the Full phase. Each owner takes two models.

| # | Model (Hugging Face ID) | Size | Role | Phase | Owner |
|---|---|---|---|---|---|
| 1 | CohereLabs/tiny-aya-global | 3.35B (confirmed on card summary) | Primary and demo model; all MVP results | MVP | A + B |
| 2 | CohereLabs/tiny-aya-fire | Verify on card | Same family, South Asian languages | Full | A |
| 3 | CohereLabs/tiny-aya-water | Verify on card | Same family, European and Asia Pacific languages | Full | A |
| 4 | CohereLabs/tiny-aya-earth | Verify on card | Same family, West Asian and African languages | Full | B |
| 5 | CohereLabs/tiny-aya-base | Verify on card | Pretrained (non-instruct) base; checks whether the instruct tuning drives the effect | Full | B |
| 6 | CohereLabs/aya-expanse-8b | 8B | Scale check (naive SFT, honest SFT, and repair only) | Full (MVP only if GPU is free on Nov 14) | C |

Ownership of the Full roster:
- **A:** Tiny Aya Global (MVP), Tiny Aya Fire, Tiny Aya Water
- **B:** Tiny Aya Earth, Tiny Aya Base
- **C:** Aya Expanse 8B (plus integration)

Removed from earlier versions: CohereLabs/c4ai-command-r7b-12-2024 (not on the permitted list).

Aya Expanse 32B is on the permitted list but is not in the roster. It is an option only if compute allows, and it would need its own budget line.

## 2. Supplementary (model-agnostic check only)

| Model | License | Role |
|---|---|---|
| Qwen/Qwen3-4B or HuggingFaceTB/SmolLM3-3B (choose one) | Apache 2.0 | E12: shows the pipeline and gate work on a non-Cohere family. Never used for headline results. Labelled as supplementary in every table. |

## 3. Pilot checks (E0): every model must pass before training

1. Logit access for the "Yes" and "No" tokens (needed for P(True)). Record the exact token IDs.
2. At least 95% format compliance with `ANSWER: ... | CONF: ...` or `ABSTAIN` on 50 items.
3. At least 95% parser agreement with 50 hand-labelled items.
4. Chat template recorded in the run config.
5. License accepted on Hugging Face and model card text saved in the repo.

A model that fails a check is swapped for another permitted Cohere model from section 1, and the failure is logged in `runs/RUN_LOG.md`.

## 4. Publishing

- Adapters built on a Cohere base carry CC BY-NC 4.0 and the Cohere Acceptable Use Policy on their model cards.
- Collections are public and ungated (submission rule). Confirm with the organizers that public adapters on gated CC BY-NC bases are allowed. If not, publish code, configs, and predictions only.

## 5. Verification status (fill in on the day)

| Model | Card read | License confirmed | Size confirmed | Logits checked | Pilot E0 |
|---|---|---|---|---|---|
| tiny-aya-global | | | yes (3.35B) | | |
| tiny-aya-fire | | | | | |
| tiny-aya-water | | | | | |
| tiny-aya-earth | | | | | |
| tiny-aya-base | | | | | |
| aya-expanse-8b | | | | | |
