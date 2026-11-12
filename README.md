# HUMBLE
Fine-tuning can quietly teach a model to bluff. HUMBLE is a one-command check that catches it before deployment, fixes it with curated 'I don't know' data, and proves the fix with a reproducible certificate. It runs fully local on open Cohere models.

## Status: pre-training (specs only, no data, no training)

HUMBLE is a release gate for fine-tuned language models. It blocks a fine-tune that became more confidently wrong than its original, repairs it with targeted "I don't know" data (or training-free weight scaling), and issues a reproducible calibration certificate.

The full project document is [PROJECT.md](PROJECT.md). This folder holds the shared specs that the team builds against.

**Track:** Track 1, Fine-tuning and Data Curation (Lexsi AI Safety and Alignment Hackathon).
**Problem statements:** #3 primary (grounded answers, fewer unsupported claims); #1 and #8 secondary.
**Deadline:** Nov 15, 2026, 11:59 PM AoE (aim to submit before noon local time).

## Core roster (all permitted Cohere Labs models)

Tiny Aya Global (MVP model), Tiny Aya Fire, Tiny Aya Water, Tiny Aya Earth, Tiny Aya Base, and Aya Expanse 8B. Phasing and owners are in [docs/models.md](docs/models.md). Qwen3 or SmolLM3 appear only as a labelled supplementary check.

## Frozen before any work starts

| File | What it fixes |
|---|---|
| [PROJECT.md](PROJECT.md) | Full project document: hypotheses, scope, methods, metrics, gate, experiments, compute, timeline |
| [docs/models.md](docs/models.md) | Model roster, phasing, pilot checks, publishing constraints |
| [docs/corpus_and_question_spec.md](docs/corpus_and_question_spec.md) | Fact tables, corpus, question types, shortcut guard, splits, holdouts |
| [docs/predictions_schema.json](docs/predictions_schema.json) | Per-item prediction record that every evaluation writes |
| [docs/run_log_template.md](docs/run_log_template.md) | Run log format and rules |
| [configs/default.yaml](configs/default.yaml) | Every default value: splits, thresholds, LoRA, metrics, gate |

Changing any frozen file needs agreement from all owners and a dated entry in `runs/RUN_LOG.md`.

## Work split (by function)

| Owner | Responsibilities |
|---|---|
| A | Corpus, questions, shortcut guard, splits, test lock, CuratorKIT adapter, repair and control data, dataset collection |
| B | AlignTune adapter, training runs, λ scaling, GPU queue, adapter uploads, model collection, ROI |
| C | Audit (parser, sampling, P(True), metrics, bootstrap, quadrants), gate and self-controls, certificate, CLI packaging, README, write-up, abstract; integration owner |

## Order of work

1. **Pre-training (now):** read the specs, raise objections in writing, and confirm the Lexsi toolkit interfaces (CuratorKIT, AlignTune, AuditKIT).
2. **Sign-off:** all owners confirm the frozen files.
3. **MVP:** corpus and shortcut guard → E0 pilot on Tiny Aya Global → E1 baseline → E2 (sft_naive first, H1 check by Nov 13 midday) → E6a gate self-controls → E5 repair vs control → E5c λ scaling (if time) → one final test evaluation.
4. **Full (after the shortlist):** the remaining five Cohere models, 3 seeds, KTO/ORPO/GRPO, planted faults, sensitivity, robustness, and the supplementary non-Cohere check.

## Hard rules

- The test set is locked. Touching it before the final evaluation invalidates the run.
- Every number in the write-up traces to a `predictions.jsonl` and a config hash.
- Negative results are reported.
- Failed runs stay in the run log.
- Core results come only from permitted Cohere models.
