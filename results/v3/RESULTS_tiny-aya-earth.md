# HUMBLE v3 results (CohereLabs/tiny-aya-earth)

All numbers are percentages unless marked; computed from saved `predictions.jsonl` files. Prudence = ABSTAIN on unanswerable probes; over-conservativeness = ABSTAIN on probes the original answered correctly; cost uses the `support` profile weights (2/2/2). Test = locked test, evaluated once per model.

## E-REPAIR on the SQuAD-derived RAG probes (test)

| Model | Accuracy | Prudence | Over-cons. | Unnec. abstention | Honesty | Cost | U5 echo |
|---|---|---|---|---|---|---|---|
| Original | 55.7 (50.0–61.0) | 22.0 (18.3–25.8) | 0.0 | 0.0 | 61.0 | 2.45 | 34.0 |
| Naive SFT s31 (blocked) | 73.7 (68.5–78.4) | 0.0 (0.0–0.0) | 0.0 | 0.0 | 50.0 | 2.53 | 46.0 |
| Naive SFT s32 (blocked) | 73.0 (68.0–78.0) | 0.0 (0.0–0.0) | 0.0 | 0.0 | 50.0 | 2.54 | 56.0 |
| Naive SFT s33 (blocked) | 74.0 (69.1–78.9) | 0.0 (0.0–0.0) | 0.0 | 0.0 | 50.0 | 2.52 | 54.0 |
| A HUMBLE repair loop (n=3) | 71.9 ± 0.5 | 73.4 ± 1.3 | 1.8 ± 0.6 | 1.1 ± 0.5 | 85.8 ± 0.6 | 1.11 ± 0.02 | 36.0 ± 2.0 |
| B random control (n=3) | 71.0 ± 0.3 | 79.6 ± 0.5 | 3.2 ± 0.3 | 3.2 ± 0.7 | 88.2 ± 0.4 | 0.99 ± 0.01 | 28.0 ± 2.0 |
| C prevention (n=3) | 71.1 ± 0.7 | 81.5 ± 1.1 | 4.8 ± 1.2 | 4.3 ± 0.7 | 88.3 ± 0.9 | 0.96 ± 0.04 | 37.3 ± 6.4 |
| Ablation: A without twins (n=3) | 70.6 ± 1.2 | 77.1 ± 1.3 | 4.0 ± 0.9 | 3.8 ± 0.7 | 86.6 ± 0.7 | 1.05 ± 0.04 | 32.0 ± 5.3 |
| Ablation: A without replay (n=3) | 60.4 ± 10.2 | 86.8 ± 8.8 | 16.6 ± 14.2 | 18.8 ± 13.8 | 85.1 ± 3.1 | 1.01 ± 0.06 | 16.0 ± 11.1 |

## E-REPAIR on the SQuAD-derived RAG probes (val)

| Model | Accuracy | Prudence | Over-cons. | Unnec. abstention | Honesty | Cost | U5 echo |
|---|---|---|---|---|---|---|---|
| Original | 57.7 (51.5–63.2) | 30.2 (26.2–34.5) | 0.0 | 0.3 | 65.1 | 2.24 | 47.6 |
| Naive SFT s31 (blocked) | 78.3 (73.8–82.8) | 0.0 (0.0–0.0) | 0.0 | 0.0 | 50.0 | 2.43 | 66.7 |
| Naive SFT s32 (blocked) | 80.0 (75.4–84.2) | 0.0 (0.0–0.0) | 0.0 | 0.0 | 50.0 | 2.40 | 71.4 |
| Naive SFT s33 (blocked) | 81.0 (76.6–85.3) | 0.0 (0.0–0.0) | 0.0 | 0.0 | 50.0 | 2.38 | 64.3 |
| A HUMBLE repair loop (n=3) | 78.3 ± 0.0 | 76.0 ± 1.7 | 0.0 ± 0.0 | 0.7 ± 0.0 | 88.0 ± 0.9 | 0.90 ± 0.03 | 34.9 ± 3.6 |
| B random control (n=3) | 77.6 ± 0.7 | 81.3 ± 0.5 | 0.4 ± 0.3 | 0.9 ± 0.2 | 90.4 ± 0.2 | 0.81 ± 0.01 | 29.4 ± 1.4 |
| C prevention (n=3) | 77.1 ± 0.8 | 83.4 ± 1.4 | 3.5 ± 0.0 | 3.7 ± 0.6 | 90.0 ± 0.7 | 0.79 ± 0.02 | 36.5 ± 1.4 |
| Ablation: A without twins (n=3) | 78.2 ± 0.2 | 79.7 ± 1.5 | 0.0 ± 0.0 | 0.7 ± 0.0 | 89.8 ± 0.7 | 0.83 ± 0.03 | 28.6 ± 2.4 |
| Ablation: A without replay (n=3) | 68.8 ± 12.2 | 87.5 ± 8.0 | 13.1 ± 13.7 | 14.2 ± 15.7 | 87.2 ± 3.6 | 0.85 ± 0.08 | 23.0 ± 10.7 |

## Gate decisions (val, profile `support`)

| Candidate | Decision | Failed rules |
|---|---|---|
| naive_s31 | BLOCK | G1, G4 |
| naive_s32 | BLOCK | G1, G4 |
| naive_s33 | BLOCK | G1, G4 |
| repA_s31_r1 | PASS | – |
| repA_s32_r1 | PASS | – |
| repA_s33_r1 | PASS | – |
| repAnoRep_s31_r1 | BLOCK | G2 |
| repAnoRep_s31_r2 | BLOCK | G2, G3 |
| repAnoRep_s32_r1 | BLOCK | G2 |
| repAnoRep_s32_r2 | BLOCK | G2 |
| repAnoRep_s33_r1 | BLOCK | G2 |
| repAnoRep_s33_r2 | BLOCK | G2 |
| repAnoTw_s31_r1 | PASS | – |
| repAnoTw_s32_r1 | PASS | – |
| repAnoTw_s33_r1 | PASS | – |

## Gate decisions on the locked test (certificates, profile `support`)

| Candidate | Decision | Failed rules |
|---|---|---|
| ctlB_s31_r1 | BLOCK | G2 |
| ctlB_s32_r1 | PASS | – |
| ctlB_s33_r1 | PASS | – |
| naive_s31 | BLOCK | G1, G4 |
| prevC_s31 | BLOCK | G2 |
| prevC_s32 | BLOCK | G2 |
| prevC_s33 | BLOCK | G2 |
| repA_s31_r1 | PASS | – |
| repA_s32_r1 | PASS | – |
| repA_s33_r1 | PASS | – |

## Arm D: lambda scaling of the blocked adapter (val)

| lambda | Gate | Accuracy | Prudence |
|---|---|---|---|

## E-CAT: prudence by unanswerable category (test)

| Model | U1 missing | U2 out of DB | U3 false premise | U4 underspecified | U5 misleading |
|---|---|---|---|---|---|
| Original | 30.5 | 27.0 | 13.0 | 3.6 | 6.0 |
| Naive s31 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| A s31 | 88.5 | 90.0 | 53.0 | 28.6 | 32.0 |
| A s32 | 90.5 | 91.0 | 54.0 | 35.7 | 40.0 |
| A s33 | 90.0 | 92.0 | 54.0 | 32.1 | 34.0 |
| C s31 | 87.5 | 98.0 | 91.0 | 39.3 | 38.0 |
| C s32 | 86.5 | 99.0 | 85.0 | 39.3 | 32.0 |
| C s33 | 86.5 | 98.0 | 89.0 | 42.9 | 36.0 |

## E-LANG: XQuAD (es, hi, ar), evaluation only

| Model | Accuracy | Prudence | Over-cons. | Unnec. abstention | Honesty | Cost | U5 echo |
|---|---|---|---|---|---|---|---|
| Original | 46.2 | 39.3 | 0.0 | 1.2 | 69.6 | 2.26 | 20.6 |
| Naive s31 | 67.5 | 0.0 | 0.0 | 0.0 | 50.0 | 2.65 | 50.0 |
| A s31 | 66.2 | 81.4 | 1.8 | 2.9 | 89.8 | 1.02 | 33.8 |
| C s31 | 62.9 | 83.2 | 4.5 | 7.1 | 89.4 | 1.03 | 38.2 |

## E-DOMAIN: CUAD contracts, evaluation only

| Model | Accuracy | Prudence | Over-cons. | Unnec. abstention | Honesty | Cost | U5 echo |
|---|---|---|---|---|---|---|---|
| Original | 24.0 | 14.7 | 0.0 | 0.0 | 57.3 | 3.23 | 19.2 |
| Naive s31 | 35.0 | 0.0 | 0.0 | 0.0 | 50.0 | 3.30 | 45.2 |
| A s31 | 34.0 | 59.7 | 4.2 | 7.0 | 77.8 | 2.07 | 41.1 |
| C s31 | 39.0 | 46.2 | 0.0 | 1.0 | 73.1 | 2.28 | 39.7 |

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
| H4 | A has lower over-conservativeness than prevention C (every seed) | not supported |
| H5 | Gate self-controls behave as expected | see E-GATE |
| H6 | Bluffing differs by category; misleading evidence (U5) is hardest | partly: hardest is U4 |
| H7 | Repair gains transfer to an unseen domain (CUAD) without retraining | see E-DOMAIN |
| H8 | Same recipe works on other permitted Cohere models | this model is part of the H8 evidence (naive blocked, A passes every seed); see RESULTS.md |

Paired differences on the locked test (document-clustered bootstrap, 1,000 resamples):

| Comparison | Metric | Diff | 95% CI | Significant |
|---|---|---|---|---|
| A_vs_B_s31 | cost | +0.146 | +0.069 … +0.223 | True |
| A_vs_B_s31 | prudence | -0.071 | -0.096 … -0.048 | True |
| A_vs_B_s31 | over_conservativeness | -0.018 | -0.041 … +0.000 | False |
| A_vs_B_s31 | accuracy | +0.010 | -0.010 … +0.029 | False |
| A_vs_C_s31 | cost | +0.220 | +0.111 … +0.324 | True |
| A_vs_C_s31 | prudence | -0.105 | -0.139 … -0.067 | True |
| A_vs_C_s31 | over_conservativeness | -0.018 | -0.046 … +0.006 | False |
| A_vs_C_s31 | accuracy | +0.003 | -0.021 … +0.029 | False |
| A_vs_B_s32 | cost | +0.119 | +0.058 … +0.181 | True |
| A_vs_B_s32 | prudence | -0.052 | -0.076 … -0.028 | True |
| A_vs_B_s32 | over_conservativeness | -0.006 | -0.020 … +0.000 | False |
| A_vs_B_s32 | accuracy | +0.000 | -0.013 … +0.014 | False |
| A_vs_C_s32 | cost | +0.123 | +0.018 … +0.228 | True |
| A_vs_C_s32 | prudence | -0.059 | -0.093 … -0.020 | True |
| A_vs_C_s32 | over_conservativeness | -0.024 | -0.053 … +0.000 | False |
| A_vs_C_s32 | accuracy | +0.000 | -0.028 … +0.027 | False |
| A_vs_B_s33 | cost | +0.100 | +0.024 … +0.172 | True |
| A_vs_B_s33 | prudence | -0.065 | -0.090 … -0.042 | True |
| A_vs_B_s33 | over_conservativeness | -0.018 | -0.040 … +0.000 | False |
| A_vs_B_s33 | accuracy | +0.017 | -0.006 … +0.040 | False |
| A_vs_C_s33 | cost | +0.110 | +0.001 … +0.220 | True |
| A_vs_C_s33 | prudence | -0.079 | -0.112 … -0.044 | True |
| A_vs_C_s33 | over_conservativeness | -0.048 | -0.082 … -0.017 | True |
| A_vs_C_s33 | accuracy | +0.020 | -0.010 … +0.048 | False |
| naive_vs_original | prudence | -0.220 | -0.256 … -0.182 | True |
| naive_vs_original | accuracy | +0.180 | +0.121 … +0.238 | True |
| naive_vs_original | cost | +0.079 | -0.061 … +0.216 | False |

## Probe set

Locked test sha256 `78c6540b174a4626b84ad2935f716754bb39ae1a4e1ceb916078abba4f46694b`; judge `CohereLabs/aya-expanse-8b`.

| Cell | Built | Judge pass rate |
|---|---|---|
| val/A/C1 | 300 | 90.6 |
| val/U1/C2 | 100 | 93.3 |
| val/U1/C3 | 100 | 86.1 |
| val/U2/C1 | 100 | 96.7 |
| val/U3/C1 | 100 | 73.9 |
| val/U4/C1 | 21 | 11.7 |
| val/U5/C4 | 42 | 23.3 |
| test/A/C1 | 300 | 89.1 |
| test/U1/C2 | 100 | 96.1 |
| test/U1/C3 | 100 | 75.0 |
| test/U2/C1 | 100 | 90.0 |
| test/U3/C1 | 100 | 71.7 |
| test/U4/C1 | 28 | 21.1 |
| test/U5/C4 | 50 | 27.8 |
| probe/A/C1 | 900 | 91.8 |
| probe/U1/C2 | 300 | 94.6 |
| probe/U1/C3 | 300 | 81.3 |
| probe/U2/C1 | 300 | 92.0 |
| probe/U3/C1 | 300 | 78.5 |
| probe/U4/C1 | 68 | 12.6 |
| probe/U5/C4 | 100 | 18.5 |
| train/A/C1 | 2000 | – |
| train/U3/C1 | 860 | – |
