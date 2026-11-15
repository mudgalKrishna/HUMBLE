# HUMBLE v3 results (CohereLabs/tiny-aya-global)

All numbers are percentages unless marked; computed from saved `predictions.jsonl` files. Prudence = ABSTAIN on unanswerable probes; over-conservativeness = ABSTAIN on probes the original answered correctly; cost uses the `support` profile weights (2/2/2). Test = locked test, evaluated once per model.

## E-REPAIR on the SQuAD-derived RAG probes (test)

| Model | Accuracy | Prudence | Over-cons. | Unnec. abstention | Honesty | Cost | U5 echo |
|---|---|---|---|---|---|---|---|
| Original | 52.7 (46.8–58.1) | 40.2 (35.8–44.4) | 0.0 | 1.3 | 70.1 | 2.12 | 30.0 |
| Naive SFT s31 (blocked) | 73.7 (68.5–78.5) | 0.0 (0.0–0.0) | 0.0 | 0.0 | 50.0 | 2.53 | 50.0 |
| Naive SFT s32 (blocked) | 73.3 (68.4–78.0) | 0.0 (0.0–0.0) | 0.0 | 0.0 | 50.0 | 2.53 | 50.0 |
| Naive SFT s33 (blocked) | 73.0 (68.1–78.0) | 0.0 (0.0–0.0) | 0.0 | 0.0 | 50.0 | 2.54 | 54.0 |
| A HUMBLE repair loop (n=3) | 72.3 ± 0.3 | 72.7 ± 0.8 | 1.7 ± 0.4 | 0.9 ± 0.2 | 85.5 ± 0.2 | 1.12 ± 0.02 | 38.7 ± 2.3 |
| B random control (n=3) | 70.8 ± 0.5 | 79.6 ± 0.8 | 2.5 ± 0.0 | 2.0 ± 0.3 | 88.5 ± 0.4 | 1.00 ± 0.03 | 31.3 ± 1.2 |
| C prevention (n=3) | 70.4 ± 0.5 | 80.0 ± 1.1 | 4.6 ± 0.7 | 4.1 ± 0.7 | 87.7 ± 0.2 | 0.99 ± 0.02 | 38.7 ± 5.0 |
| Ablation: A without twins (n=3) | 71.2 ± 1.1 | 75.7 ± 2.5 | 3.6 ± 1.3 | 3.2 ± 1.0 | 86.0 ± 0.6 | 1.07 ± 0.03 | 29.3 ± 3.1 |
| Ablation: A without replay (n=3) | 58.4 ± 4.1 | 88.1 ± 6.8 | 17.9 ± 6.5 | 21.7 ± 6.1 | 85.1 ± 0.2 | 0.99 ± 0.05 | 17.3 ± 11.0 |
| A + two-way DPO (n=1) | 71.0 | 75.1 | 3.2 | 3.0 | 86.0 | 1.08 | 34.0 |
| A repairing naive s32 (n=1) | 71.3 | 70.9 | 1.3 | 0.7 | 84.8 | 1.17 | 40.0 |
| A repairing naive s33 (n=1) | 71.3 | 71.8 | 1.9 | 1.0 | 84.9 | 1.16 | 42.0 |

## E-REPAIR on the SQuAD-derived RAG probes (val)

| Model | Accuracy | Prudence | Over-cons. | Unnec. abstention | Honesty | Cost | U5 echo |
|---|---|---|---|---|---|---|---|
| Original | 54.3 (48.3–59.9) | 47.7 (43.3–52.2) | 0.0 | 1.3 | 73.9 | 1.93 | 42.9 |
| Naive SFT s31 (blocked) | 78.7 (74.1–83.0) | 0.0 (0.0–0.0) | 0.0 | 0.0 | 50.0 | 2.43 | 66.7 |
| Naive SFT s32 (blocked) | 78.3 (73.7–82.7) | 0.0 (0.0–0.0) | 0.0 | 0.0 | 50.0 | 2.43 | 73.8 |
| Naive SFT s33 (blocked) | 80.0 (75.4–84.4) | 0.0 (0.0–0.0) | 0.0 | 0.0 | 50.0 | 2.40 | 66.7 |
| A HUMBLE repair loop (n=3) | 77.9 ± 0.4 | 74.0 ± 0.7 | 0.4 ± 0.4 | 0.6 ± 0.2 | 86.8 ± 0.2 | 0.96 ± 0.02 | 38.9 ± 1.4 |
| B random control (n=3) | 77.3 ± 0.3 | 80.6 ± 0.6 | 1.0 ± 0.4 | 1.4 ± 0.5 | 89.8 ± 0.4 | 0.83 ± 0.02 | 36.5 ± 2.7 |
| C prevention (n=3) | 76.7 ± 0.6 | 81.5 ± 1.4 | 1.5 ± 0.4 | 3.4 ± 0.2 | 90.0 ± 1.2 | 0.81 ± 0.05 | 42.1 ± 5.0 |
| Ablation: A without twins (n=3) | 77.7 ± 0.9 | 78.7 ± 2.2 | 0.6 ± 0.6 | 1.8 ± 0.5 | 89.0 ± 0.8 | 0.85 ± 0.02 | 34.9 ± 1.4 |
| Ablation: A without replay (n=3) | 64.2 ± 4.5 | 89.1 ± 6.2 | 17.4 ± 3.9 | 19.6 ± 6.3 | 85.9 ± 1.1 | 0.89 ± 0.08 | 22.2 ± 8.4 |
| A + two-way DPO (n=1) | 77.3 | 78.8 | 0.6 | 2.0 | 89.1 | 0.85 | 35.7 |
| A repairing naive s32 (n=1) | 79.3 | 73.7 | 0.0 | 0.7 | 86.8 | 0.93 | 42.9 |
| A repairing naive s33 (n=1) | 80.0 | 74.7 | 0.6 | 1.0 | 87.1 | 0.90 | 40.5 |

## Gate decisions (val, profile `support`)

| Candidate | Decision | Failed rules |
|---|---|---|
| ctlB_s31_r1 | PASS | – |
| ctlB_s33_r1 | PASS | – |
| naive_s31 | BLOCK | G1, G4 |
| repA_s31_r1 | PASS | – |
| repA_s32_r1 | PASS | – |
| repA_s33_r1 | PASS | – |
| repAb32_s31_r1 | PASS | – |
| repAb33_s31_r1 | PASS | – |
| repADPO2_s31_r1 | PASS | – |
| repADPO_s31_r1 | BLOCK | G5 |
| repADPO_s31_r2 | PASS | – |
| repAnoRep_s31_r1 | BLOCK | G2 |
| repAnoRep_s31_r2 | BLOCK | G2 |
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
| ctlB_s31_r1 | PASS | – |
| ctlB_s32_r1 | PASS | – |
| ctlB_s33_r1 | PASS | – |
| naive_s31 | BLOCK | G1, G4 |
| naive_s32 | BLOCK | G1, G4 |
| naive_s33 | BLOCK | G1, G4 |
| prevC_s31 | BLOCK | G2 |
| prevC_s32 | BLOCK | G2 |
| prevC_s33 | BLOCK | G2 |
| repA_s31_r1 | PASS | – |
| repA_s32_r1 | PASS | – |
| repA_s33_r1 | PASS | – |
| repAb32_s31_r1 | PASS | – |
| repAb33_s31_r1 | PASS | – |
| repADPO2_s31_r1_dpo | BLOCK | G2 |
| repAnoRep_s31_r2 | BLOCK | G2 |
| repAnoRep_s32_r2 | BLOCK | G2 |
| repAnoRep_s33_r2 | BLOCK | G2 |
| repAnoTw_s31_r1 | PASS | – |
| repAnoTw_s32_r1 | BLOCK | G2 |
| repAnoTw_s33_r1 | BLOCK | G2 |

## Arm D: lambda scaling of the blocked adapter (val)

| lambda | Gate | Accuracy | Prudence |
|---|---|---|---|
| 1.0 | BLOCK | 78.7 | 0.0 |
| 0.75 | BLOCK | 79.3 | 0.0 |
| 0.5 | BLOCK | 77.7 | 0.0 |
| 0.25 | BLOCK | 73.3 | 0.2 |
| 0.2 | BLOCK | 72.3 | 4.3 |
| 0.15 | BLOCK | 69.7 | 19.0 |
| 0.1 | BLOCK | 64.7 | 31.5 |
| 0.05 | BLOCK | 58.0 | 44.1 |
| 0.0 | PASS | 54.3 | 47.7 |

## E-CAT: prudence by unanswerable category (test)

| Model | U1 missing | U2 out of DB | U3 false premise | U4 underspecified | U5 misleading |
|---|---|---|---|---|---|
| Original | 53.0 | 50.0 | 27.0 | 10.7 | 12.0 |
| Naive s31 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| A s31 | 85.0 | 93.0 | 53.0 | 32.1 | 36.0 |
| A s32 | 90.5 | 89.0 | 54.0 | 32.1 | 34.0 |
| A s33 | 89.0 | 92.0 | 53.0 | 32.1 | 34.0 |
| C s31 | 85.0 | 97.0 | 89.0 | 39.3 | 40.0 |
| C s32 | 86.5 | 97.0 | 84.0 | 28.6 | 30.0 |
| C s33 | 84.5 | 97.0 | 88.0 | 39.3 | 36.0 |

## E-LANG: XQuAD (es, hi, ar), evaluation only

| Model | Accuracy | Prudence | Over-cons. | Unnec. abstention | Honesty | Cost | U5 echo |
|---|---|---|---|---|---|---|---|
| Original | 41.7 | 55.0 | 0.0 | 4.2 | 77.5 | 1.98 | 16.2 |
| Naive s31 | 67.9 | 0.0 | – | 0.0 | – | 2.64 | 50.0 |
| A s31 | 64.2 | 84.7 | 1.0 | 5.4 | 91.9 | 0.93 | 35.3 |
| A s32 | 65.4 | 84.1 | 0.0 | 3.8 | 92.1 | 0.93 | 33.8 |
| A s33 | 65.4 | 84.7 | 2.0 | 4.2 | 91.4 | 0.95 | 36.8 |
| C s31 | 63.7 | 83.1 | 5.0 | 7.5 | 89.0 | 1.01 | 33.8 |
| C s32 | 64.2 | 82.6 | 2.0 | 4.6 | 90.3 | 1.01 | 35.3 |
| C s33 | 63.7 | 82.8 | 3.0 | 7.1 | 89.9 | 0.99 | 35.3 |

## E-DOMAIN: CUAD contracts, evaluation only

| Model | Accuracy | Prudence | Over-cons. | Unnec. abstention | Honesty | Cost | U5 echo |
|---|---|---|---|---|---|---|---|
| Original | 20.0 | 34.4 | 0.0 | 0.0 | 67.2 | 2.91 | 19.2 |
| Naive s31 | 36.0 | 0.0 | 0.0 | 0.0 | 50.0 | 3.28 | 45.2 |
| A s31 | 36.0 | 52.7 | 5.0 | 3.0 | 73.9 | 2.27 | 43.8 |
| A s32 | 34.0 | 60.4 | 10.0 | 5.0 | 75.2 | 2.21 | 38.4 |
| A s33 | 36.0 | 57.1 | 5.0 | 2.0 | 76.1 | 2.20 | 41.1 |
| C s31 | 38.0 | 41.8 | 0.0 | 1.0 | 70.9 | 2.38 | 41.1 |
| C s32 | 38.0 | 47.3 | 0.0 | 1.0 | 73.6 | 2.27 | 34.2 |
| C s33 | 38.0 | 45.8 | 0.0 | 1.0 | 72.9 | 2.30 | 42.5 |

With judge grading of long answers (M9):

| Model | Accuracy (F1 only) | Accuracy (judge) |
|---|---|---|
| Original | 20.0 | 54.0 |
| Naive s31 | 36.0 | 63.0 |
| A s31 | 36.0 | 63.0 |
| A s32 | 34.0 | 61.0 |
| A s33 | 36.0 | 63.0 |
| C s31 | 38.0 | 67.0 |
| C s32 | 38.0 | 66.0 |
| C s33 | 38.0 | 67.0 |

## Product mode: probes generated from a fictional company's 44 policy documents

| Model | Accuracy | Prudence | Over-cons. | Unnec. abstention | Honesty | Cost | U5 echo |
|---|---|---|---|---|---|---|---|
| Original | 22.0 | 7.7 | 0.0 | 2.4 | 53.9 | 3.36 | 12.5 |
| Naive s31 | 39.0 | 0.0 | 0.0 | 0.0 | 50.0 | 3.22 | 18.8 |
| A s31 | 41.5 | 31.0 | 0.0 | 0.0 | 65.5 | 2.55 | 18.8 |
| A s32 | 39.0 | 29.6 | 0.0 | 0.0 | 64.8 | 2.63 | 25.0 |
| A s33 | 39.0 | 28.9 | 0.0 | 0.0 | 64.4 | 2.64 | 25.0 |
| C s31 | 39.0 | 28.9 | 0.0 | 0.0 | 64.4 | 2.64 | 18.8 |
| C s32 | 41.5 | 29.6 | 0.0 | 0.0 | 64.8 | 2.58 | 18.8 |
| C s33 | 39.0 | 26.8 | 0.0 | 2.4 | 63.4 | 2.64 | 25.0 |

## E-GATE: self-controls (val)

| Profile | Original vs itself | Always abstain | Always answer | Naive SFT | All as expected |
|---|---|---|---|---|---|
| healthcare | PASS | BLOCK ['G2', 'G3'] | BLOCK ['G1', 'G4'] | BLOCK | True |
| finance | PASS | BLOCK ['G2', 'G3'] | BLOCK ['G1', 'G4'] | BLOCK | True |
| legal | PASS | BLOCK ['G2', 'G3'] | BLOCK ['G1', 'G4'] | BLOCK | True |
| support | PASS | BLOCK ['G2', 'G3'] | BLOCK ['G1', 'G4'] | BLOCK | True |

Gate v2 rerun on saved v2 predictions (single passage, SQuAD 2.0):

| Run | Decision | Failed rules | Prudence | Over-cons. |
|---|---|---|---|---|
| real_test/real_sft_honest | BLOCK | G2 | 72.3 | 4.5 |
| real_test/real_sft_honest_s2 | BLOCK | G2, G3 | 93.7 | 33.3 |
| real_test/real_sft_honest_s3 | BLOCK | G2, G3 | 98.0 | 53.8 |
| real_test/real_sft_honest_s4 | BLOCK | G2 | 88.3 | 18.6 |
| real_test/real_sft_naive | BLOCK | G1, G4 | 0.0 | 0.0 |
| real_test/real_sft_naive_s2 | BLOCK | G1, G4 | 0.0 | 0.0 |
| real_test/real_sft_naive_s3 | BLOCK | G1, G4 | 0.0 | 0.0 |
| real_test/realfinal_u30_s11 | BLOCK | G2 | 88.7 | 17.9 |
| real_test/realfinal_u30_s12 | PASS | – | 73.0 | 1.9 |
| real_test/realfinal_u30_s13 | BLOCK | G2 | 79.0 | 4.5 |
| real_val/realsw_u10_s1 | BLOCK | G2 | 68.7 | 10.3 |
| real_val/realsw_u10_s2 | BLOCK | G2 | 63.3 | 5.1 |
| real_val/realsw_u10_s3 | BLOCK | G2 | 78.7 | 17.9 |
| real_val/realsw_u20_s1 | PASS | – | 60.0 | 1.3 |
| real_val/realsw_u20_s2 | PASS | – | 48.7 | 1.3 |
| real_val/realsw_u20_s3 | BLOCK | G2 | 77.3 | 10.3 |
| real_val/realsw_u30_s1 | BLOCK | G2 | 84.7 | 25.6 |
| real_val/realsw_u30_s2 | BLOCK | G2 | 73.3 | 9.0 |
| real_val/realsw_u30_s3 | BLOCK | G2 | 76.7 | 14.1 |
| real_val/realsw_u50_s1 | BLOCK | G2 | 77.3 | 5.1 |
| real_val/realsw_u50_s2 | BLOCK | G2 | 84.0 | 19.2 |
| real_val/realsw_u50_s3 | BLOCK | G2, G3 | 96.7 | 60.3 |
| test2/r2final_n2000_s21 | BLOCK | G2 | 94.7 | 28.7 |
| test2/r2final_n2000_s22 | BLOCK | G2 | 91.7 | 18.0 |
| test2/r2final_n2000_s23 | BLOCK | G2 | 93.0 | 21.3 |
| val2/r2_n1000_s1 | PASS | – | 61.3 | 2.0 |
| val2/r2_n1000_s2 | BLOCK | G2 | 76.7 | 6.5 |
| val2/r2_n1000_s3 | PASS | – | 71.0 | 2.6 |
| val2/r2_n2000_s1 | BLOCK | G2 | 82.3 | 10.5 |
| val2/r2_n2000_s2 | BLOCK | G2 | 82.7 | 6.5 |
| val2/r2_n2000_s3 | BLOCK | G2 | 80.7 | 7.8 |
| val2/r2_n300_s1 | BLOCK | G2 | 73.0 | 9.2 |
| val2/r2_n300_s2 | BLOCK | G2 | 81.0 | 20.3 |
| val2/r2_n300_s3 | BLOCK | G2 | 72.3 | 6.5 |

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
| H8 | Same recipe works on other permitted Cohere models | supported on 3 more models: Aya Expanse 8B, Tiny Aya Fire, Tiny Aya Earth (partial run) - naive blocked, A passes every seed; see RESULTS_<model>.md |

Paired differences on the locked test (document-clustered bootstrap, 1,000 resamples):

| Comparison | Metric | Diff | 95% CI | Significant |
|---|---|---|---|---|
| A_vs_B_s31 | cost | +0.093 | +0.022 … +0.160 | True |
| A_vs_B_s31 | prudence | -0.069 | -0.095 … -0.045 | True |
| A_vs_B_s31 | over_conservativeness | -0.013 | -0.032 … +0.000 | False |
| A_vs_B_s31 | accuracy | +0.020 | -0.003 … +0.044 | False |
| A_vs_C_s31 | cost | +0.141 | +0.034 … +0.249 | True |
| A_vs_C_s31 | prudence | -0.092 | -0.128 … -0.054 | True |
| A_vs_C_s31 | over_conservativeness | -0.038 | -0.067 … -0.012 | True |
| A_vs_C_s31 | accuracy | +0.020 | -0.010 … +0.049 | False |
| A_vs_B_s32 | cost | +0.116 | +0.057 … +0.180 | True |
| A_vs_B_s32 | prudence | -0.071 | -0.096 … -0.049 | True |
| A_vs_B_s32 | over_conservativeness | -0.006 | -0.021 … +0.000 | False |
| A_vs_B_s32 | accuracy | +0.020 | +0.003 … +0.039 | True |
| A_vs_C_s32 | cost | +0.088 | -0.004 … +0.180 | False |
| A_vs_C_s32 | prudence | -0.056 | -0.090 … -0.023 | True |
| A_vs_C_s32 | over_conservativeness | -0.019 | -0.048 … +0.006 | False |
| A_vs_C_s32 | accuracy | +0.017 | -0.007 … +0.039 | False |
| A_vs_B_s33 | cost | +0.128 | +0.063 … +0.194 | True |
| A_vs_B_s33 | prudence | -0.067 | -0.091 … -0.045 | True |
| A_vs_B_s33 | over_conservativeness | -0.006 | -0.021 … +0.000 | False |
| A_vs_B_s33 | accuracy | +0.007 | -0.013 … +0.026 | False |
| A_vs_C_s33 | cost | +0.112 | +0.001 … +0.222 | True |
| A_vs_C_s33 | prudence | -0.071 | -0.107 … -0.036 | True |
| A_vs_C_s33 | over_conservativeness | -0.032 | -0.064 … +0.000 | False |
| A_vs_C_s33 | accuracy | +0.020 | -0.009 … +0.049 | False |
| naive_vs_original | prudence | -0.402 | -0.443 … -0.358 | True |
| naive_vs_original | accuracy | +0.210 | +0.155 … +0.267 | True |
| naive_vs_original | cost | +0.410 | +0.268 … +0.568 | True |

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
