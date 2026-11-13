# HUMBLE results (snapshot, partial, parser-corrected)

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

## Deviations from the spec

- 2,720 items instead of about 3,000; train split has 1,440 items instead of 1,800.
- Documents are about 300 words, not 800–1,200.
- DPO learning rate 2e-5 (inside the spec's range).
- The repair target follows the pre-registered rule (highest val confident-wrong rate, DPO-naive). Its high rate is mostly formatting, so that repair result is confounded. A repair of SFT-naive is planned as an exploratory analysis, labelled as such.

## Reproduce

Code: `code/humble/`. Configs: `configs/default.yaml`. The dataset is generated deterministically by `humble.data.write_dataset`; `MANIFEST.json` and `TEST_LOCK.txt` record counts and the locked test-set hash. Training and audits need a GPU (RTX PRO 6000, 98 GB). Re-scoring needs no GPU: `humble.rescore.rescore_run(<run folder>)`.
