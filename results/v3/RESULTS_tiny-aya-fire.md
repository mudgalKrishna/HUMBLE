# HUMBLE v3 results (CohereLabs/tiny-aya-fire)

All numbers are percentages unless marked; computed from saved `predictions.jsonl` files. Prudence = ABSTAIN on unanswerable probes; over-conservativeness = ABSTAIN on probes the original answered correctly; cost uses the `support` profile weights (2/2/2). Test = locked test, evaluated once per model.

## E-REPAIR on the SQuAD-derived RAG probes (test)

| Model | Accuracy | Prudence | Over-cons. | Unnec. abstention | Honesty | Cost | U5 echo |
|---|---|---|---|---|---|---|---|
| Original | 60.0 (54.5–65.3) | 36.0 (31.7–40.5) | 0.0 | 0.3 | 68.0 | 2.07 | 38.0 |
| Naive SFT s31 (blocked) | 72.3 (67.4–77.2) | 0.0 (0.0–0.0) | 0.0 | 0.0 | 50.0 | 2.55 | 48.0 |
| A HUMBLE repair loop (n=3) | 71.1 ± 0.8 | 74.2 ± 2.5 | 2.2 ± 1.0 | 1.7 ± 0.6 | 86.0 ± 0.8 | 1.10 ± 0.03 | 34.0 ± 2.0 |
| B random control (n=3) | 70.3 ± 0.7 | 81.5 ± 0.4 | 2.6 ± 0.3 | 2.9 ± 0.2 | 89.4 ± 0.3 | 0.96 ± 0.02 | 26.0 ± 0.0 |
| C prevention | (missing) |

## E-REPAIR on the SQuAD-derived RAG probes (val)

| Model | Accuracy | Prudence | Over-cons. | Unnec. abstention | Honesty | Cost | U5 echo |
|---|---|---|---|---|---|---|---|
| Original | 60.7 (54.9–66.0) | 41.7 (37.3–46.6) | 0.0 | 0.7 | 70.8 | 1.94 | 47.6 |
| Naive SFT s31 (blocked) | 81.0 (76.4–85.2) | 0.0 (0.0–0.0) | 0.0 | 0.0 | 50.0 | 2.38 | 61.9 |
| A HUMBLE repair loop (n=3) | 79.2 ± 0.5 | 76.5 ± 2.8 | 0.5 ± 0.0 | 1.1 ± 0.2 | 88.0 ± 1.4 | 0.88 ± 0.05 | 35.7 ± 2.4 |
| B random control (n=3) | 79.0 ± 0.7 | 81.1 ± 1.2 | 0.9 ± 0.3 | 1.3 ± 0.3 | 90.1 ± 0.6 | 0.79 ± 0.01 | 30.2 ± 1.4 |
| C prevention | (missing) |

## Gate decisions (val, profile `support`)

| Candidate | Decision | Failed rules |
|---|---|---|
| naive_s31 | BLOCK | G1, G4 |
| repA_s31_r1 | PASS | – |
| repA_s32_r1 | PASS | – |
| repA_s33_r1 | PASS | – |

## Gate decisions on the locked test (certificates, profile `support`)

| Candidate | Decision | Failed rules |
|---|---|---|
| ctlB_s31_r1 | PASS | – |
| naive_s31 | BLOCK | G1, G4 |
| repA_s31_r1 | PASS | – |

## Arm D: lambda scaling of the blocked adapter (val)

| lambda | Gate | Accuracy | Prudence |
|---|---|---|---|

## E-CAT: prudence by unanswerable category (test)

| Model | U1 missing | U2 out of DB | U3 false premise | U4 underspecified | U5 misleading |
|---|---|---|---|---|---|
| Original | 44.0 | 48.0 | 27.0 | 10.7 | 12.0 |
| Naive s31 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| A s31 | 86.0 | 93.0 | 52.0 | 25.0 | 36.0 |
| A s32 | 91.0 | 94.0 | 62.0 | 32.1 | 38.0 |
| A s33 | 90.0 | 93.0 | 55.0 | 32.1 | 38.0 |

## E-LANG: XQuAD (es, hi, ar), evaluation only

| Model | Accuracy | Prudence | Over-cons. | Unnec. abstention | Honesty | Cost | U5 echo |
|---|---|---|---|---|---|---|---|

## E-DOMAIN: CUAD contracts, evaluation only

| Model | Accuracy | Prudence | Over-cons. | Unnec. abstention | Honesty | Cost | U5 echo |
|---|---|---|---|---|---|---|---|

With judge grading of long answers (M9):

| Model | Accuracy (F1 only) | Accuracy (judge) |
|---|---|---|

## Product mode: probes generated from a fictional company's 44 policy documents

| Model | Accuracy | Prudence | Over-cons. | Unnec. abstention | Honesty | Cost | U5 echo |
|---|---|---|---|---|---|---|---|

## Hypotheses (pre-registered)

| ID | Claim | Verdict |
|---|---|---|
| H1 | Naive fine-tuning lowers prudence vs the original (multi-passage RAG) | supported |
| H2 | P(True)-based confident-wrong fails to detect H1 | supported |
| H3 | Repair loop A lowers cost more than random control B (every seed, paired CI excludes 0) | not supported |
| H4 | A has lower over-conservativeness than prevention C (every seed) | untested |
| H5 | Gate self-controls behave as expected | see E-GATE |
| H6 | Bluffing differs by category; misleading evidence (U5) is hardest | partly: hardest is U4 |
| H7 | Repair gains transfer to an unseen domain (CUAD) without retraining | see E-DOMAIN |
| H8 | Same recipe works on other permitted Cohere models | not run (scope: one model first, by instruction) |

Paired differences on the locked test (document-clustered bootstrap, 1,000 resamples):

| Comparison | Metric | Diff | 95% CI | Significant |
|---|---|---|---|---|
| A_vs_B_s31 | cost | +0.178 | +0.098 … +0.259 | True |
| A_vs_B_s31 | prudence | -0.103 | -0.130 … -0.075 | True |
| A_vs_B_s31 | over_conservativeness | -0.017 | -0.037 … +0.000 | False |
| A_vs_B_s31 | accuracy | +0.017 | -0.007 … +0.040 | False |
| A_vs_B_s32 | cost | +0.088 | +0.032 … +0.149 | True |
| A_vs_B_s32 | prudence | -0.044 | -0.064 … -0.025 | True |
| A_vs_B_s32 | over_conservativeness | +0.000 | -0.016 … +0.016 | False |
| A_vs_B_s32 | accuracy | +0.010 | -0.011 … +0.030 | False |
| A_vs_B_s33 | cost | +0.173 | +0.114 … +0.241 | True |
| A_vs_B_s33 | prudence | -0.071 | -0.096 … -0.047 | True |
| A_vs_B_s33 | over_conservativeness | +0.006 | +0.000 … +0.018 | False |
| A_vs_B_s33 | accuracy | -0.003 | -0.024 … +0.016 | False |
| naive_vs_original | prudence | -0.360 | -0.404 … -0.316 | True |
| naive_vs_original | accuracy | +0.123 | +0.068 … +0.181 | True |
| naive_vs_original | cost | +0.480 | +0.337 … +0.627 | True |

