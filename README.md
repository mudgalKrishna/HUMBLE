# HUMBLE

**Catch and repair bluffing in fine-tuned RAG models before deployment.**

RAG gives a model the right documents, but it does not teach the model to admit when those documents do not
contain the answer. HUMBLE audits a fine-tuned RAG model on probes built from a company's own documents, blocks it
if it bluffs more than the original model, repairs it from its own mistakes with curated "I don't know" data, and
issues a reproducible certificate. Everything runs locally on open Cohere models: no API calls.

Track 1 (Fine-tuning and Data Curation), Lexsi AI Safety and Alignment Hackathon. Problem statements #3 (primary),
#1 and #8. Models: `CohereLabs/tiny-aya-global` (main), Aya Expanse 8B, Tiny Aya Fire, Tiny Aya Earth. Lexsi toolkits chained: **CuratorKIT → AlignTune → AuditKIT**.

**Submission links**
* Demo (Hugging Face Space, static): https://huggingface.co/spaces/KM2007/humble-explorer
* Write-up: [docs/WRITEUP.md](docs/WRITEUP.md) (metrics and why, dataset/task/model rationale, before and after) ·
  abstract: [docs/ABSTRACT.md](docs/ABSTRACT.md) · HF article text: [docs/HF_ARTICLE.md](docs/HF_ARTICLE.md)
* ROI, self-hosted vs closed-source APIs: [docs/ROI.md](docs/ROI.md)
* Models: [repaired A](https://huggingface.co/KM2007/humble-tinyaya-repaired-A) ·
  [blocked naive](https://huggingface.co/KM2007/humble-tinyaya-naive-blocked) ·
  [random control B](https://huggingface.co/KM2007/humble-tinyaya-control-B) ·
  [prevention C](https://huggingface.co/KM2007/humble-tinyaya-prevention-C)
* Data (probes, every prediction, audits, certificates): [KM2007/humble-v3-probes-and-audits](https://huggingface.co/datasets/KM2007/humble-v3-probes-and-audits)

## Before and after, four models (locked test)

Bluff rate = share of unanswerable questions answered anyway (100 − prudence).

| Model | Original: acc / bluff | Naive fine-tune: acc / bluff (gate) | HUMBLE-repaired: acc / bluff / over-refusal (gate) |
|---|---|---|---|
| Tiny Aya Global | 52.7 / 59.8 | 73.3 / **100.0** (BLOCK) | **72.3 / 27.3 / 1.7 (PASS, 3 seeds)** |
| Aya Expanse 8B | 84.7 / 89.5 | 90.0 / **100.0** (BLOCK) | **90.4 / 17.2 / 0.4 (PASS, 3 seeds)** |
| Tiny Aya Fire | 60.0 / 64.0 | 72.3 / **100.0** (BLOCK) | **71.1 / 25.8 / 2.2 (PASS, 3 seeds)** |
| Tiny Aya Earth | 55.7 / 78.0 | 73.6 / **100.0** (BLOCK) | **71.9 / 26.6 / 1.8 (PASS, 3 seeds)** |

## What we found (locked test, Tiny Aya Global, multi-passage RAG probes)

| Model | Accuracy | Prudence (abstains on unanswerable) | Gate |
|---|---|---|---|
| Original Tiny Aya | 52.7% | 40.2% | reference |
| Naive fine-tune (3 seeds) | ~73% | **0.0%** | **BLOCK** (G1 bluffing, G4 every category) |
| HUMBLE repair loop, from the blocked model (A) | 72.3% | 72.7% | **PASS** after 1 round, every seed (val and test) |
| Random-data repair, same size (B) | 70.8% | 79.6% | PASS, every seed |
| Prevention: fresh honest fine-tune (C, 3 seeds) | 70.4% | 80.0% | PASS on val, **BLOCK on test** (G2: over-refusal 3.8–5.1%) |
| Training-free lambda scaling (D) | – | – | only lambda = 0 (the original) passes |

1. **Fine-tuning made the model bluff on every unanswerable question**, in all three seeds and all five
   unanswerable categories, while accuracy rose by 20 points. P(True) confidence did not reveal it (H2).
2. **The gate caught it** with label-based rules (no reliance on model confidence), and its self-controls behave
   as designed across four domain profiles.
3. **Repair rescues the broken fine-tune in one round**: bluffing falls from 100% to about 28%, the accuracy
   gain is kept, over-conservativeness stays near 1%. The repair transfers to Spanish/Hindi/Arabic (XQuAD) and to
   legal contracts (CUAD) without training on them.
4. **What protects against over-refusal (ablations, 3 seeds each).** Without replay of ordinary training data,
   repair over-refuses: blocked on G2 in every seed, test over-conservativeness about 18% and accuracy 58%. Without
   the answerable twins, all seeds pass on val but 2 of 3 are blocked on the locked test (over-conservativeness
   3.6% vs 1.7% with twins). Replay is the main protection; twins add a smaller one.
5. **Honest negative results.** Repair data built from the model's own mistakes did *not* beat a random
   sample of the same size (H3 not supported): targeted data keeps ~2 points more accuracy, random data abstains
   more. This also held when repairing an only partly broken model, whose mistakes are a real subset
   ([results/v3/FIX1_TARGETING.md](results/v3/FIX1_TARGETING.md)). A had lower over-refusal than prevention on every seed but significantly so on only one (H4 not supported
   under the pre-registered rule), although C is the one blocked on test. Adding two-way DPO after repair
   raised prudence by 3 points but was blocked on test for over-refusal (3.2%). Lambda scaling cannot keep any of
   the fine-tune: bluffing appears already at lambda = 0.05.

6. **The recipe transfers to a larger model (Aya Expanse 8B, unchanged recipe).** Naive fine-tuning again
   removed all abstention (prudence 10.5% → 0.0%) and was blocked; HUMBLE repair passed on every seed with
   accuracy 90.4% (naive 90.0%, original 84.7%), prudence 82.8% and over-refusal 0.4%; random data again abstained
   more (88.0%). Aya answers in full sentences, so accuracy and over-refusal there are judge-graded
   ([results/v3/RESULTS_aya-expanse-8b.md](results/v3/RESULTS_aya-expanse-8b.md)).

7. **And to two more Tiny Aya models (Fire and Earth).** On Tiny Aya Fire (teammate run) naive fine-tuning was
   blocked (prudence 36.0% → 0.0%) and HUMBLE repair passed on every seed (accuracy 71.1%, prudence 74.2%,
   over-refusal 2.2%); the random control B again abstained more (81.5%) and beat A on all three seeds, so H3 is
   not supported on Fire either; prevention (C) was not run
   ([results/v3/RESULTS_tiny-aya-fire.md](results/v3/RESULTS_tiny-aya-fire.md)). On Tiny Aya Earth
   all three naive seeds were blocked (prudence 22.0% → 0.0%) and A passed on every seed, val and locked-test
   certificates alike (accuracy 71.9%, prudence 73.4%, over-refusal 1.8%). Random data (B, 79.6% prudence) and
   prevention (C, 81.5%) abstained more but over-refused: B was blocked on the test set for one seed of three and C
   for all three (G2), so on Earth only the HUMBLE repair passed every seed
   ([results/v3/RESULTS_tiny-aya-earth.md](results/v3/RESULTS_tiny-aya-earth.md)).

8. **Upweighting the weak categories helps, but is seed-sensitive (Fix 2, Tiny Aya Global, teammate run).**
   Repeating the U3/U4/U5 repair examples (Acat) raised prudence on squad3v from 72.6% to 77.3% and on a new
   U4/U5 test set (squad3u45, 300 items from paragraphs no earlier prompt showed) from 33.9% to 42.2%, with about
   the same accuracy and about 0.8 points more unnecessary abstention. The gain is significant on seeds 31 and 33
   but not 32: consistent, but not a pass under the every-seed rule. Caveats: the blocked model is a retrain
   (`naive_s31b`), and squad3u45 does not exclude paragraphs from earlier iterations
   ([results/v3/FIX2_CATEGORIES.md](results/v3/FIX2_CATEGORIES.md),
   [results/v3/RESULTS_fix2_categories.md](results/v3/RESULTS_fix2_categories.md)).

Full tables with CIs: [results/v3/RESULTS.md](results/v3/RESULTS.md). Every number is regenerated from saved
`predictions.jsonl` files by `results/v3/make_report.py`.

## How it works

```
company documents ──► 1 PROBE (CuratorKIT + HUMBLE) ──► 2 AUDIT (AuditKIT + HUMBLE) ──► 3 GATE v2 ──► PASS / BLOCK
                                                                                            │
                         5 CERTIFICATE ◄── re-gate ◄── 4 REPAIR LOOP (CuratorKIT + AlignTune) ◄┘
```

* **Probes** (RAG-style, 4 passages each): C1 gold + distractors (answer), C2 distractors only, C3 answer sentence
  deleted, C4 misleading edit of the gold passage (abstain); unanswerable categories U1 missing fact, U2 out of
  database, U3 false premise, U4 underspecified, U5 misleading evidence. Mechanically verified, then checked by a
  local sufficient-context judge (Aya Expanse 8B). Split by document; the test set is hash-locked and evaluated once.
* **Audit**: greedy answers in `ANSWER: <text>` / `ABSTAIN` format, scored by AuditKIT (SQuAD-style F1), honesty
  metrics from Alignment for Honesty (prudence, over-conservativeness, honesty score), cost score with
  domain-profile weights, document-clustered bootstrap CIs, per category / context / language.
* **Gate v2**: G1 bluffing (paired CI), G2 over-refusal, G3 accuracy, G4 per-category prudence, G5 format, with
  thresholds from domain profiles (healthcare, finance, legal, support). Exit codes 0 / 1 / 2.
* **Repair loop**: multisample diagnosis of the blocked model on a separate probe pool; repair set = each bluff →
  ABSTAIN, its answerable twin → gold, delete-a-fact and misleading variants → ABSTAIN, RAFT's no-gold case flipped to
  ABSTAIN, over-refusals → gold, 1:1 replay; curated by CuratorKIT; continue the blocked LoRA adapter with AlignTune
  SFT; re-gate on val; up to 3 rounds.

## Quickstart

```bash
uv tool install "git+https://github.com/mudgalKrishna/HUMBLE[gpu]"
humble probe   --docs ./company_docs --out ./probes                      # local generator + judge
humble audit   --probes ./probes --split val --out ./audits/original
humble audit   --probes ./probes --split val --adapter ./my_finetune --original-audit ./audits/original --out ./audits/my_finetune
humble gate    ./audits/original ./audits/my_finetune --profile finance   # exit 0 PASS, 1 BLOCK
humble repair  --blocked ./my_finetune --probes ./probes --original-audit ./audits/original --out ./adapters/repaired
humble certify ./audits/my_finetune --gate ./audits/my_finetune/gate.json --out ./certificates
```

CPU-only commands (`gate`, `selfcheck`, `certify`) need only `pip install .`; the rest need a GPU.

## Repository

| Path | Contents |
|---|---|
| `src/humble/probes/` | corpus + C1–C4 / U1–U5 construction, SQuAD, XQuAD, CUAD, company-documents builder |
| `src/humble/audit/` | local generation, AuditKIT scoring, honesty metrics and bootstrap |
| `src/humble/gate/` | gate v2, self-controls, repair acceptance, shipping choice |
| `src/humble/repair/`, `arms.py` | repair loop v2 and the E-REPAIR arms A/B/C/D |
| `src/humble/curate/`, `train/`, `judge/`, `report/` | CuratorKIT, AlignTune, judge, certificates |
| `src/humble/experiments.py`, `analysis.py` | experiment jobs and analysis (all resumable) |
| `configs/default.yaml` | paths, model, profiles, thresholds, training settings |
| `tests/` | parser, metrics, gate rules and self-controls, probe construction, repair building |
| `results/` | v2 results (kept) and v3 tables |
| `docs/PROJECT_v3.md` | the project plan this release implements |
| `legacy/v2_code/` | code that produced the v2 results |

## Reproducibility and licences

Each audit folder stores `audit.toml` (model, adapter hash, probe hash, test lock, config hash, versions),
`predictions.jsonl`, `metrics.json`, the AuditKIT run, and gate files; any gate can be recomputed on CPU.
Adapters inherit CC BY-NC 4.0 and the Cohere Labs AUP from Tiny Aya. Probe data derived from SQuAD 2.0 and XQuAD is
CC BY-SA 4.0; CUAD is CC BY 4.0. AlignTune is MIT; CuratorKIT and AuditKIT are LSAL (source-available).
