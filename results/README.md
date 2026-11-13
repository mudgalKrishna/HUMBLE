# HUMBLE results (snapshot, partial)

Status: **partial**. Test results so far cover the original, SFT-naive, SFT-honest, and DPO-naive. Pending: repaired models (targeted and control), the λ-scaled model, DPO-honest test, the external SQuAD 2.0 check, and extra seeds.

Every number comes from `runs/<model>/<split>/predictions.jsonl` and is recomputed in `metrics.json`. Nothing is edited by hand. Test runs were evaluated once each (see `DONE` markers).

## Test split (locked, evaluated once)

| Model | n | Accuracy | Confident wrong (95% CI) | SelAcc@80 | Unnecessary abstention | Unanswerable bluff | ECE | Format failure |
|---|---|---|---|---|---|---|---|---|
| dpo_naive | 560 | 0.830 | 0.180 [0.139, 0.225] | 0.663 | 0.021 | 0.188 | 0.270 | 0.164 |
| original | 560 | 0.658 | 0.009 [0.004, 0.013] | 0.484 | 0.000 | 0.634 | 0.197 | 0.009 |
| sft_honest | 560 | 1.000 | 0.000 [0.000, 0.000] | 0.750 | 0.000 | 0.906 | 0.036 | 0.000 |
| sft_naive | 560 | 1.000 | 0.204 [0.168, 0.237] | 0.741 | 0.000 | 1.000 | 0.212 | 0.000 |

## Validation split (used for choices)

| Model | n | Accuracy | Confident wrong (95% CI) | SelAcc@80 | Unnecessary abstention | Unanswerable bluff | ECE | Format failure |
|---|---|---|---|---|---|---|---|---|
| dpo_honest | 480 | 0.000 | 0.000 [0.000, 0.000] | 0.396 | 1.000 | 0.000 | 0.000 | 0.000 |
| dpo_naive | 480 | 0.670 | 0.252 [0.202, 0.312] | 0.521 | 0.014 | 0.229 | 0.195 | 0.244 |
| dpo_naive_s2 | 480 | 0.694 | 0.242 [0.183, 0.312] | 0.521 | 0.003 | 0.333 | 0.138 | 0.223 |
| lambda_0.0 | 480 | 0.649 | 0.002 [0.000, 0.006] | 0.487 | 0.000 | 0.750 | 0.222 | 0.002 |
| original | 480 | 0.649 | 0.002 [0.000, 0.006] | 0.487 | 0.000 | 0.750 | 0.222 | 0.002 |
| sft_honest | 480 | 1.000 | 0.000 [0.000, 0.000] | 0.750 | 0.000 | 0.922 | 0.033 | 0.000 |
| sft_honest_s2 | 480 | 1.000 | 0.000 [0.000, 0.000] | 0.953 | 0.000 | 0.094 | 0.052 | 0.000 |
| sft_naive | 480 | 1.000 | 0.183 [0.160, 0.212] | 0.750 | 0.000 | 1.000 | 0.211 | 0.000 |
| sft_naive_s2 | 480 | 1.000 | 0.179 [0.163, 0.202] | 0.750 | 0.000 | 1.000 | 0.199 | 0.000 |
| sft_naive_s3 | 480 | 1.000 | 0.181 [0.146, 0.215] | 0.750 | 0.000 | 1.000 | 0.208 | 0.000 |

## What these results show

1. **Naive fine-tuning raises confident mistakes.** SFT-naive confident-wrong on test is 0.204 (CI 0.168–0.237), against 0.009 for the original. It answers all unanswerable questions with high confidence (bluff rate 1.000). This supports the main hypothesis.
2. **SFT-honest's zero confident-wrong rate is not the same as abstention.** It answers 91% of unanswerable test questions (bluff 0.906), but at low confidence, so they don't count as confident wrong. Its honest training does not teach it to abstain. The confidence measure (P(True)) is doing the work here. This needs a follow-up.
3. **Seed instability.** SFT-honest seed 1 bluffs on 92% of unanswerable val questions; seed 2 bluffs on 9%. Confidence and abstention behavior vary a lot between seeds, so single-seed claims are weak.
4. **DPO-naive's confident-wrong rate is mostly formatting.** 16% of its test outputs can't be parsed, and parse failures count as wrong with full confidence.
5. **DPO-honest collapsed to always abstaining** (val accuracy 0, unnecessary abstention 1.0). It is a failed method, and the release gate should block it.
6. **Sanity check passes.** λ = 0 reproduces the original exactly (val accuracy 0.649, confident wrong 0.002).

## Deviations from the spec

- 2,720 items instead of about 3,000; train split has 1,440 items instead of 1,800.
- Documents are about 300 words, not 800–1,200.
- DPO learning rate 2e-5 (inside the spec's range).
- Repair ran on the target chosen by the pre-registered rule (highest val confident-wrong rate, which was DPO-naive). Its high rate is mostly formatting, so that repair result is confounded. A repair on SFT-naive is planned as an exploratory analysis, labelled as such.

## Reproduce

Code: `code/humble/`. Configs: `configs/default.yaml`. Dataset is generated deterministically from `humble.data.write_dataset`; `MANIFEST.json` and `TEST_LOCK.txt` record the counts and the locked test-set hash. Training and audits need a GPU; they ran on RTX PRO 6000 (98 GB).
