# Fix 2 results: upweighting weak categories (Acat) and the larger U4/U5 test (squad3u45)

Tiny Aya Global, default config, two molab RTX PRO 6000 notebooks (2026-11-15). Every adapter is evaluated
once on each locked test. Cost uses the `support` weights (2/2/2). Paired differences: document-clustered
bootstrap, 1,000 resamples (`metrics.paired_diff`), Acat minus Ab on the same seed.

**Setup notes**
- Blocked model: `naive_s31b`, a retrain of the published `naive_s31` recipe (seed 31). Krishna's `naive_s31`
  was not readable with this token. Because of that, plain repair A was re-run on it as `Ab`, so Ab and Acat
  start from the same model. Published names (`naive_s31`, `repA_*`) were not touched.
  `naive_s31b` val gate: BLOCK. Its squad3v test was not run (not needed for the comparison).
- `Acat` = `arm_A(..., cat_weights={"U3": 2, "U4": 2, "U5": 2}, tag="Acat")`. Every arm (Ab and Acat, all
  seeds) passed its gate in round 1.
- squad3u45: 150 U4 + 150 U5 judged items (judge pass rate 22.6% / 20.5%) from train-split articles. It uses only
  paragraphs that no squad3/squad3v prompt shows. **Earlier-iteration paragraphs (KM2007/humble-runs) were not
  excluded** (no access), as recorded in its MANIFEST. squad3v was not modified (test lock verified).

## squad3v locked test

| Model | Accuracy | Prudence | Unnec. abst. | Cost | U1 | U2 | U3 | U4 | U5 |
|---|---|---|---|---|---|---|---|---|---|
| Original | 52.7 | 40.2 | 1.3 | 2.12 | 53.0 | 50.0 | 27.0 | 10.7 | 12.0 |
| Ab s31 | 73.0 | 71.8 | 0.7 | 1.12 | 85.0 | 93.0 | 53.0 | 28.6 | 38.0 |
| Ab s32 | 72.3 | 73.2 | 1.0 | 1.11 | 90.5 | 89.0 | 54.0 | 32.1 | 34.0 |
| Ab s33 | 72.3 | 72.8 | 1.0 | 1.12 | 89.0 | 92.0 | 52.0 | 32.1 | 34.0 |
| **Acat s31** | 72.0 | 77.6 | 2.0 | 1.01 | 91.5 | 98.0 | 58.0 | 35.7 | 44.0 |
| **Acat s32** | 72.3 | 74.7 | 1.0 | 1.08 | 87.0 | 96.0 | 57.0 | 39.3 | 38.0 |
| **Acat s33** | 72.0 | 79.5 | 2.0 | 0.98 | 93.5 | 96.0 | 62.0 | 42.9 | 46.0 |
| *mean Ab* | 72.5 | 72.6 | 0.9 | 1.11 | 88.2 | 91.3 | 53.0 | 31.0 | 35.3 |
| *mean Acat* | 72.1 | **77.3** | 1.7 | **1.02** | 90.7 | 96.7 | **59.0** | **39.3** | **42.7** |

Cell sizes: U1 200, U2 100, U3 100, U4 28, U5 50 (U4/U5 are small, hence squad3u45).

Paired Acat − Ab (squad3v test):

| Seed | Prudence | Cost | Accuracy | Unnec. abst. | U3 prud. | U4 prud. | U5 prud. |
|---|---|---|---|---|---|---|---|
| 31 | **+5.9** [+3.6, +8.0] | **−0.111** [−0.171, −0.048] | −1.0 [−3.0, +1.0] | **+1.3** [+0.3, +2.6] | **+5.0** [+1.0, +9.5] | +7.1 [0.0, +16.7] | +6.0 [−2.5, +15.6] |
| 32 | +1.5 [−1.0, +3.7] | −0.029 [−0.092, +0.033] | 0.0 [−1.7, +1.6] | 0.0 [−1.0, +1.0] | +3.0 [−2.9, +8.9] | +7.1 [0.0, +17.9] | +4.0 [0.0, +10.6] |
| 33 | **+6.7** [+4.5, +9.0] | **−0.135** [−0.198, −0.070] | −0.3 [−2.3, +1.6] | +1.0 [0.0, +2.3] | **+10.0** [+4.6, +16.0] | +10.7 [0.0, +23.8] | **+12.0** [+3.8, +21.7] |

## squad3u45 locked test (U4/U5 only, 150 each)

| Model | Prudence | U4 | U5 |
|---|---|---|---|
| Original | 16.7 | 14.0 | 19.3 |
| naive_s31b | 0.0 | 0.0 | 0.0 |
| Ab s31 / s32 / s33 | 32.7 / 35.0 / 34.0 | 31.3 / 35.3 / 32.0 | 34.0 / 34.7 / 36.0 |
| **Acat s31 / s32 / s33** | 42.7 / 37.0 / 47.0 | 44.0 / 36.7 / 47.3 | 41.3 / 37.3 / 46.7 |
| *mean Ab → mean Acat* | 33.9 → **42.2** | 32.9 → **42.7** | 34.9 → **41.8** |

Paired Acat − Ab (squad3u45): s31 **+10.0** [+6.4, +13.6] (U4 **+12.7**, U5 **+7.3**); s32 +2.0 [−1.0, +4.8]
(U4 +1.3, U5 +2.7, n.s.); s33 **+13.0** [+9.3, +17.0] (U4 **+15.3**, U5 **+10.7**).

## Reading

- Upweighting U3/U4/U5 2× raises prudence on exactly those categories without losing accuracy (−0.4 points on
  average, not significant). Cost drops on every seed, significantly on 31 and 33. Effects on all three
  seeds point the same way; seed 32 is smaller and its CI includes 0.
- The price: unnecessary abstention rises ~+0.8 points (significant on s31).
- On the larger squad3u45 set the gain is clearer: U4 +10 and U5 +7 points on average. The weak categories are
  still weak, though (~42% prudence vs ~91–97% on U1/U2). 2× is a first step, not a fix. Trying 3–4× or a
  U4-specific repair target would be the next experiment.
- Statistically, this is "2 of 3 seeds significant". Under the H3-style rule (every seed, CI excludes 0) it does
  **not** pass. Report it as a consistent but seed-sensitive improvement.

Raw data: `icyBear02/humble-v3-mohit` (v3/audits/{squad3v,squad3u45}/..., v3/arms/arm{Ab,Acat}_s*.json,
v3/probes/squad3u45).
