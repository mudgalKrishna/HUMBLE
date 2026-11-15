# HUMBLE v3 results (CohereLabs/tiny-aya-fire)

All numbers are percentages unless marked; computed from saved `predictions.jsonl` files. Prudence = ABSTAIN on unanswerable probes; over-conservativeness = ABSTAIN on probes the original answered correctly; cost uses the `support` profile weights (2/2/2). Test = locked test, evaluated once per model.

## E-REPAIR on the SQuAD-derived RAG probes (test)

| Model | Accuracy | Prudence | Over-cons. | Unnec. abstention | Honesty | Cost | U5 echo |
|---|---|---|---|---|---|---|---|
| Original | 60.0 (54.5–65.3) | 36.0 (31.7–40.5) | 0.0 | 0.3 | 68.0 | 2.07 | 38.0 |
| Naive SFT s31 (blocked) | 72.3 (67.4–77.2) | 0.0 (0.0–0.0) | 0.0 | 0.0 | 50.0 | 2.55 | 48.0 |
| A HUMBLE repair loop (n=1) | 72.0 | 71.5 | 1.1 | 1.0 | 85.2 | 1.13 | 34.0 |
| B random control (n=1) | 70.3 | 81.8 | 2.8 | 3.0 | 89.5 | 0.95 | 26.0 |
| C prevention | (missing) |

## E-REPAIR on the SQuAD-derived RAG probes (val)

| Model | Accuracy | Prudence | Over-cons. | Unnec. abstention | Honesty | Cost | U5 echo |
|---|---|---|---|---|---|---|---|
| Original | 60.7 (54.9–66.0) | 41.7 (37.3–46.6) | 0.0 | 0.7 | 70.8 | 1.94 | 47.6 |
| Naive SFT s31 (blocked) | 81.0 (76.4–85.2) | 0.0 (0.0–0.0) | 0.0 | 0.0 | 50.0 | 2.38 | 61.9 |
| A HUMBLE repair loop (n=1) | 79.7 | 73.2 | 0.5 | 1.0 | 86.3 | 0.93 | 38.1 |
| B random control (n=1) | 78.3 | 82.3 | 1.1 | 1.3 | 90.6 | 0.78 | 28.6 |
| C prevention | (missing) |

## Gate decisions (val, profile `support`)

| Candidate | Decision | Failed rules |
|---|---|---|
| naive_s31 | BLOCK | G1, G4 |
| repA_s31_r1 | PASS | – |

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
| naive_vs_original | prudence | -0.360 | -0.404 … -0.316 | True |
| naive_vs_original | accuracy | +0.123 | +0.068 … +0.181 | True |
| naive_vs_original | cost | +0.480 | +0.337 … +0.627 | True |

