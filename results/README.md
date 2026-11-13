# HUMBLE results (parser-corrected, with real-text iteration)

Status: **partial.** Test results so far: original, SFT-naive, SFT-honest, DPO-naive. Still running or pending: the repaired models (targeted and control), the λ-scaled model, DPO-honest test, the external SQuAD 2.0 check, and extra seeds.

Every number is recomputed from `runs/<model>/<split>/predictions.jsonl`, which keeps the raw model text. Nothing is typed by hand. Each test set was evaluated once (see `DONE` markers).

## Parser correction (read this first)

The first version of the parser checked for `ANSWER: ... | CONF: ...` anywhere in the output, before checking for a leading `ABSTAIN`. Models trained with ABSTAIN targets kept generating after `ABSTAIN` (the stop token used in training does not match the generation stop token), so their abstentions were scored as confident answers. Fixed parser: the first word of the output decides the action. All saved outputs were re-scored from the raw text (`rescore.py`). Changed items are reported in each `metrics.json` as `rescored_changed_items`.

Effect: SFT-honest's unanswerable bluff rate drops from 0.906 to 0.000 on test, and its seed-to-seed disagreement disappears (val bluff 0.922 and 0.094 before, both 0.000 after). No other model's numbers change.

## Test split (locked, evaluated once)

| Model | n | Accuracy | Confident wrong (95% CI) | SelAcc@80 | Unnecessary abstention | Unanswerable bluff | ECE | Format failure |
|---|---|---|---|---|---|---|---|---|
| dpo_naive | 560 | 0.830 | 0.180 [0.139, 0.225] | 0.663 | 0.021 | 0.188 | 0.270 | 0.164 |
| original | 560 | 0.658 | 0.009 [0.004, 0.013] | 0.484 | 0.000 | 0.634 | 0.197 | 0.009 |
| sft_honest | 560 | 1.000 | 0.000 [0.000, 0.000] | 1.000 | 0.000 | 0.000 | 0.047 | 0.000 |
| sft_naive | 560 | 1.000 | 0.204 [0.168, 0.237] | 0.741 | 0.000 | 1.000 | 0.212 | 0.000 |

## Validation split (used for choices)

| Model | n | Accuracy | Confident wrong (95% CI) | SelAcc@80 | Unnecessary abstention | Unanswerable bluff | ECE | Format failure |
|---|---|---|---|---|---|---|---|---|
| dpo_honest | 480 | 0.000 | 0.000 [0.000, 0.000] | 0.396 | 1.000 | 0.000 | 0.000 | 0.000 |
| dpo_naive | 480 | 0.670 | 0.252 [0.202, 0.312] | 0.521 | 0.014 | 0.229 | 0.195 | 0.244 |
| dpo_naive_s2 | 480 | 0.694 | 0.242 [0.183, 0.312] | 0.521 | 0.003 | 0.333 | 0.138 | 0.223 |
| lambda_0.0 | 480 | 0.649 | 0.002 [0.000, 0.006] | 0.487 | 0.000 | 0.750 | 0.222 | 0.002 |
| original | 480 | 0.649 | 0.002 [0.000, 0.006] | 0.487 | 0.000 | 0.750 | 0.222 | 0.002 |
| sft_honest | 480 | 1.000 | 0.000 [0.000, 0.000] | 1.000 | 0.000 | 0.000 | 0.046 | 0.000 |
| sft_honest_s2 | 480 | 1.000 | 0.000 [0.000, 0.000] | 1.000 | 0.000 | 0.000 | 0.055 | 0.000 |
| sft_naive | 480 | 1.000 | 0.183 [0.160, 0.212] | 0.750 | 0.000 | 1.000 | 0.211 | 0.000 |
| sft_naive_s2 | 480 | 1.000 | 0.179 [0.163, 0.202] | 0.750 | 0.000 | 1.000 | 0.199 | 0.000 |
| sft_naive_s3 | 480 | 1.000 | 0.181 [0.146, 0.215] | 0.750 | 0.000 | 1.000 | 0.208 | 0.000 |

## What the results show

1. **Naive fine-tuning raises confident mistakes.** SFT-naive has confident-wrong 0.204 on test (CI 0.168–0.237), against 0.009 for the original. It answers every unanswerable question with high confidence. This supports the main hypothesis.
2. **Honest fine-tuning fixes it without losing accuracy.** SFT-honest abstains on all 224 unanswerable test questions, answers all answerable ones correctly, and has no confident-wrong answers. This is the clearest positive result so far.
3. **DPO-naive's confident-wrong rate is mostly formatting.** 16% of its test outputs can't be parsed, and unparseable outputs count as wrong with full confidence.
4. **DPO-honest collapsed to always abstaining** (val accuracy 0, unnecessary abstention 1.0). It is a failed method, and the release gate should block it.
5. **Sanity check passes.** λ = 0 reproduces the original exactly (val accuracy 0.649, confident wrong 0.002).

## Known limitations

- **Stop token mismatch.** Training ends answers with the chat end-of-turn token (ID 6), but generation stops on ID 3. Outputs keep going after the answer. Actions and scores are unaffected because the parser reads the first word, but outputs are untidy. A future iteration should set the generation stop to include ID 6 before any new evaluation.
- **Synthetic corpus.** Documents come from templates; the external SQuAD 2.0 check is the real-data test.
- **Single seed for most results.** SFT-naive and SFT-honest have 2–3 seeds on val, but only seed 1 is on test.
- **Test set reuse.** Each test split was used once. Any new model needs a new test split.


## Real-data check (SQuAD 2.0, partial)

The synthetic results above come from template documents, so they are easy: the repaired model scores perfectly there. The real-data check uses 600 human-written SQuAD 2.0 questions on real Wikipedia paragraphs (300 answerable, 300 unanswerable), run once per model.

Results so far:
- **Original:** accuracy 0.507, confident wrong 0.092, unanswerable bluff 0.620.
- **DPO-naive:** accuracy 0.450, confident wrong 0.303, unanswerable bluff 0.320.

Pending: SFT-naive, SFT-honest, DPO-honest, and the repaired models. Until those are in, the synthetic results should not be read as evidence that the method works on real documents.

Full tables, including the external section, are in `TABLES.md` (generated by `make_tables.py`).


## Real-text iteration (the result that matters)

Training on template documents did not transfer: on real SQuAD 2.0 questions, the template-trained honest model was more confidently wrong than the original. So the candidates were retrained on real text.

**Setup.** SFT-honest trained on real SQuAD 2.0 training paragraphs (300 answerable questions plus a share of unanswerable ones). The unanswerable share was chosen on a separate real validation split (300 questions), then three fresh seeds were evaluated once on a locked real test (600 questions). None of the three splits share paragraphs, and none overlap the external check.

**Detector change.** On real text, the P(True) confidence collapses after fine-tuning: naive models answer every unanswerable question yet show almost no confident-wrong answers. The gate therefore uses two ground-truth signals that don't depend on confidence: the unanswerable bluff rate and unnecessary abstention.

**Selection rule (fixed before results).** Score = bluff + unnecessary abstention, averaged over 3 seeds; eligible only if accuracy >= original - 0.02. Validation scores: 10% 0.396, 20% 0.436, **30% 0.369 (chosen)**, 50% 0.429 but ineligible (accuracy 0.482; one seed refused 62% of answerable questions). Original: 0.607.

**Locked real test (evaluated once per seed).**

| Run | Accuracy | Unanswerable bluff | Unnecessary abstention | Score |
|---|---|---|---|---|
| Original | 0.520 | 0.690 | 0.020 | 0.710 |
| 30% recipe, seed 11 | 0.493 | 0.113 | 0.287 | 0.400 |
| 30% recipe, seed 12 | 0.620 | 0.267 | 0.083 | 0.350 |
| 30% recipe, seed 13 | 0.577 | 0.210 | 0.137 | 0.347 |
| **30% recipe, mean** | **0.563** | **0.197** | **0.169** | **0.366** |
| Naive SFT, 3 seeds | ~0.63 | ~1.00 | 0.000 | ~1.00 |

**Verdict (rule fixed in advance): the fix holds.** Mean score 0.366 against 0.710 for the original, mean accuracy 0.563 (floor 0.500), and every seed beats the original. Bluffing on real unanswerable questions falls from 69% to 20% while accuracy rises.

**Limits.** One model (Tiny Aya Global), 300 real training questions, three seeds. Over-refusal still varies by seed (8%-29%), and about 1 in 5 real unanswerable questions is still answered. Generalization to other models is untested.


## Iteration 2 (Tiny Aya Global): more real data

**What changed.** A fresh real-text split was built from SQuAD 2.0 questions never used before: a 4,000-question training pool (2,000 answerable), a validation split (val2, 600) and a locked test (test2, 600). val2 and test2 share no paragraph with each other, with the training pool, or with any earlier split. Audits now sample every question and stop generation on both end tokens (the stop-token fix).

**What was compared (on val2, 3 seeds each).** Training size 300 / 1,000 / 2,000 answerable questions (30% unanswerable share) x an agreement-abstention threshold (off, or 0.3-0.9: abstain when the model's samples don't back its greedy answer).

**Rule (fixed before results).** Lowest mean score = bluff + unnecessary abstention; eligible only if accuracy >= original - 0.02; ties go to the simpler setting. Chosen: **2,000 questions, agreement rule off** (val2 score 0.293, seed range 0.277-0.303; original 0.687).

**Negative result.** The agreement rule never helped: at every size, raising the threshold cut bluffing but raised over-refusal by more, so the score got worse.

**Locked test2 (evaluated once per run).**

| Run | Accuracy | Unanswerable bluff | Unnecessary abstention | Score |
|---|---|---|---|---|
| Original | 0.500 | 0.653 | 0.010 | 0.663 |
| 2,000 questions, seed 21 | 0.513 | 0.053 | 0.273 | 0.327 |
| 2,000 questions, seed 22 | 0.613 | 0.083 | 0.163 | 0.247 |
| 2,000 questions, seed 23 | 0.557 | 0.070 | 0.253 | 0.323 |
| **Mean** | **0.561** | **0.069** | **0.230** | **0.299** |

**Verdict (rule fixed in advance): iteration 2 holds.** Mean score 0.299 against 0.663; mean accuracy 0.561 against 0.500; every seed beats the original. Bluffing on real unanswerable questions falls from 65% to 7%.

**Compared with iteration 1** (different test sets, so only roughly comparable): iteration 1 cut the score from 0.710 to 0.366 (-48%); iteration 2 cuts it from 0.663 to 0.299 (-55%), with far less bluffing (7% vs 20%).

**Limits.** Over-refusal on test2 (23%) is higher than on val2 (11%), so some answerable questions are still refused. One model only (Tiny Aya Global). Three seeds per setting.

## Deviations from the spec

- 2,720 items instead of about 3,000; train split has 1,440 items instead of 1,800.
- Documents are about 300 words, not 800–1,200.
- DPO learning rate 2e-5 (inside the spec's range).
- The repair target follows the pre-registered rule (highest val confident-wrong rate, DPO-naive). Its high rate is mostly formatting, so that repair result is confounded. A repair of SFT-naive is planned as an exploratory analysis, labelled as such.

## Reproduce

Code: `code/humble/`. Configs: `configs/default.yaml`. The dataset is generated deterministically by `humble.data.write_dataset`; `MANIFEST.json` and `TEST_LOCK.txt` record counts and the locked test-set hash. Training and audits need a GPU (RTX PRO 6000, 98 GB). Re-scoring needs no GPU: `humble.rescore.rescore_run(<run folder>)`.
