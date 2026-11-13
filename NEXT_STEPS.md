# HUMBLE: next steps for the team

## Where things stand

HUMBLE checks a fine-tuned model before deployment for two failures on real text: **bluffing** (answering a question the document can't answer) and **over-refusal** (declining a question it can answer). It then retrains the model to fix them.

**Current best recipe (Tiny Aya Global):** SFT-honest LoRA on 2,000 real SQuAD 2.0 answerable questions plus a 30% share of unanswerable ones (ABSTAIN target). Chosen on a validation split, then tested once on a locked test.

| On the locked real test (test2) | Original | Recipe, mean of 3 seeds |
|---|---|---|
| Bluff on unanswerable questions | 0.653 | **0.069** |
| Unnecessary abstention | 0.010 | 0.230 |
| Accuracy on answerable questions | 0.500 | 0.561 |
| Score (bluff + abstention, lower is better) | 0.663 | **0.299** |

Full numbers: `results/README.md`, `results/TABLES.md`. Raw predictions: `results/runs/`.

**Rules we follow (please keep them):**
- Choose settings on validation (`val2`), never on the test.
- Each model/seed is evaluated **once** on `test2`. The code refuses a second run for the same name.
- Report negative results (for example, the agreement rule below).
- Core results must use permitted Cohere Labs models (hackathon rule).

## Part A: fixes on Tiny Aya Global (priority order)

1. **Cut over-refusal (biggest gap).** Unnecessary abstention is 23% on test2 vs 11% on val2. Try:
   - a lower unanswerable share at 2,000+ questions (15%, 20%, 25%), 3 seeds each, chosen on val2;
   - more answerable data (5,000-10,000 SQuAD train questions);
   - a small preference step after SFT: chosen = correct answer, rejected = ABSTAIN on answerable questions; chosen = ABSTAIN, rejected = a bluff on unanswerable ones. Use a low learning rate: DPO-honest on synthetic data collapsed into always abstaining.
2. **Update the release gate to v2.** `code/humble/gate.py` still uses confident-wrong and ECE, which break on real text (P(True) confidence collapses after fine-tuning). New rules: block if unanswerable bluff, unnecessary abstention, or accuracy is worse than the original by a set margin. Re-run the gate self-controls (original vs itself = PASS, always-abstain = BLOCK, bluffer = BLOCK).
3. **Confidence intervals for the real-text metrics.** Add bootstrap CIs for bluff and unnecessary abstention, resampling by paragraph (`org_id`), like `metrics.bootstrap_ci` does for confident-wrong.
4. **Stop-token fix in training.** Generation now stops on both end tokens (IDs 3 and 6). Training still appends the tokenizer's end token; append the token generation stops on, so outputs end cleanly without relying on the generation fix.
5. **A second real dataset** with unanswerable questions (not SQuAD), as an external check. Verify the licence before using it.
6. **Already tried, did not help:** abstaining when the model's 10 samples disagree. It always raised over-refusal more than it cut bluffing (see README, iteration 2).

## Part B: other models

**Permitted Cohere models (core results):** Tiny Aya Base, Global, Fire, Water, Earth; Aya Expanse 8B and 32B.

| Model | Size | Why test it | Notes |
|---|---|---|---|
| CohereLabs/tiny-aya-fire | 3.35B | Same size, different regional training | Easiest first check |
| CohereLabs/tiny-aya-water | 3.35B | Same family | |
| CohereLabs/tiny-aya-earth | 3.35B | Same family | |
| CohereLabs/tiny-aya-base | 3.35B | Pretrained, not instruction-tuned | May have no chat template; check the prompt format first |
| CohereLabs/aya-expanse-8b | 8B | Larger model | ~16 GB weights; fits one RTX 6000 with LoRA |
| CohereLabs/aya-expanse-32b | 32B | Scale check | ~64 GB in bf16; very tight on 96 GB, likely needs 4-bit loading |
| Qwen/Qwen3-4B or HuggingFaceTB/SmolLM3-3B | 3-4B | Non-Cohere check | Supplementary only (Apache 2.0); label it as such |

**Before any download:** each gated Cohere model needs its terms accepted on Hugging Face by the account whose token you use (it returns 403 until then).

**Protocol per model:**
1. Run the original model once on `test2` (baseline).
2. Run the recipe **unchanged** (2,000 questions, 30% share), 3 seeds, each once on `test2`. This tests whether the recipe transfers.
3. Only if it doesn't, tune on `val2` and test 3 fresh seeds once.
4. Report per-model tables like the one above.

## How to run

- Code: `code/humble/`. Jobs are plain functions in `code/humble/jobs.py`, for example `jobs.real2_val(2000, 1)` (train + val2 audit) and `jobs.real2_final(2000, 21)` (train + one test2 evaluation).
- Paths assume a GPU notebook with the code at `/marimo/humble/code`, data at `/marimo/humble/real2`, and the model at `/marimo/models/tiny-aya-global` (`runner.MODEL`). **For other models, make `runner.MODEL` configurable** (environment variable or argument) and include the model name in run names so results don't collide.
- LoRA targets `q_proj`, `k_proj`, `v_proj`, `o_proj`. Check the module names for each new architecture.
- Data, adapters, and run logs are in the Hugging Face dataset `KM2007/humble-runs` (currently private). The real-text splits are in `bundles/real2_bundle.tar.gz`. Ask for access, or the repo needs to be made public for the submission anyway.
- Re-scoring saved predictions needs no GPU: `humble.rescore.rescore_run(<run folder>)`; tables: `python results/make_tables.py`.

## Licences

- Adapters trained on Cohere models: CC BY-NC 4.0 plus the Cohere Acceptable Use Policy (noncommercial).
- Data derived from SQuAD 2.0: CC BY-SA 4.0 (share-alike).

## Deadline

Submission: Nov 15, 11:59 PM AoE. Still needed: public Hugging Face model and dataset collections, the abstract (under 500 words), and the write-up.
