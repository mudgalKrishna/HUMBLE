# HUMBLE
Fine-tuning can quietly teach a model to bluff. HUMBLE is a one-command check that catches it before deployment, fixes it with curated 'I don't know' data, and proves the fix with a reproducible certificate. It runs fully local on open Cohere models.

## Status: pre-training plan (specs only, no data, no training)

HUMBLE is a gate that blocks a fine-tuned model if it became more confidently wrong, and repairs it with targeted "I don't know" data. This repository holds the shared specs that the team builds against. The full project document is in [PROJECT.md](PROJECT.md).

## Frozen before any work starts

| File | What it fixes |
|---|---|
| docs/models.md | The 6-model roster, licenses, and the pilot checks each model must pass |
| docs/corpus_and_question_spec.md | Fact tables, corpus size, question types, splits, holdouts, acceptance checks |
| docs/predictions_schema.json | The per-item record every evaluation writes |
| docs/run_log_template.md | The run log format and rules |
| configs/default.yaml | Every default value: splits, thresholds, LoRA, metrics, gate |

Changing any frozen file requires all three owners to agree, and a dated note in `runs/RUN_LOG.md`.

## Work split (2 models per person)

| Person | Models | Owns (functional) |
|---|---|---|
| A | tiny-aya-global, tiny-aya-fire | Corpus, questions, splits, test lock, CuratorKIT repair data, dataset collection |
| B | aya-expanse-8b, c4ai-command-r7b-12-2024 | AlignTune training runs for the Cohere 8B-class models, adapter uploads, GPU queue |
| C | Qwen3-4B, SmolLM3-3B | AuditKIT adapter, quadrants, metrics, bootstrap, gate, UI, certificate, write-up |

Everyone runs their two models through the same configs and the same grid (E0 → E6). Person C's gate code is shared by all six models.

## Order of work

1. **Now (pre-training, no GPU):**
   - Everyone reads these files and raises objections in writing.
   - A: write the generator code against corpus_and_question_spec.md.
   - B: confirm model cards, gates, and chat templates for the two assigned models.
   - C: write the metric and gate code against the predictions schema, with unit tests on hand-made records.
2. **Sign-off:** all three confirm the frozen specs. Only then start generation and runs.
3. **Pilot E0 per model:** logits, format compliance, parser agreement. Drop a model that fails and use a reserve.
4. **Parallel grid:** each person runs their two models. Everything logged in `runs/RUN_LOG.md`.
5. **Gate and repair:** Person C computes gate decisions. Person A builds repair data from validation results only.
6. **Final:** test set touched once. Abstract, article, and README written from the saved predictions.

## Hard rules

- The test set is locked. Touching it before the final evaluation invalidates the run.
- Every number in the write-up traces to a `predictions.jsonl` and a config hash.
- Negative results are reported.
- Unfinished or failed runs stay in the log.

## Open items (must be closed before sign-off)

- Confirm the Tiny Aya Fire parameter count and chat template.
- Confirm Command R7B's current revision (search only found the 12-2024 card).
- Confirm each model exposes logits for "Yes"/"No" (P(True)).
- Confirm with organizers that public adapters derived from CC BY-NC bases are allowed in the submission.
- Confirm the submission deadline: sources say Nov 15, 11:59 PM AoE.

