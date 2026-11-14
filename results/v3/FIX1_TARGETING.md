# Fix 1: is targeted repair better than random when the damage is only partial?

Model: `CohereLabs/tiny-aya-global`. Probes: locked `squad3v` (test lock `78c6540b174a4626`). Profile: support.
Hub repo for this run: `TASMAYU/humble-tinyaya-fixes` (private). Raw numbers: `_e_targeting_naive_light_s31_test.json`.

## Why

On Tiny Aya Global, repair data built from the model's own mistakes (A) did not beat a random sample of the same
size (B) (H3 not supported). One explanation: the full naive model bluffed on every unanswerable item, so "its own
mistakes" were just a random sample. Fix 1 tests this with a naive model that is only partly broken.

## 1. A partly broken naive model

`train_naive` now takes `name`, `max_steps` and `lr` (the default call is unchanged). Same seed (31) and data as
`naive_s31`; only the amount of training differs. Val metrics:

| Training | Accuracy | Prudence | Note |
|---|---|---|---|
| original | 54.3% | 47.7% | reference |
| 5 steps, lr 2e-4 | 66.0% | 39.7% | |
| 8 steps, lr 2e-4 | 68.0% | 29.8% | |
| **10 steps, lr 2e-4** | **72.0%** | **21.8%** | **chosen → `naive_light_s31`** |
| 12 steps, lr 2e-4 | 72.0% | 13.6% | |
| 15 steps, lr 2e-4 | 70.7% | 2.2% | |
| 20 / 40 / 80 steps, lr 2e-4 | 73–77% | 0.0% | |
| 10–40 steps, lr 2e-5 … 3e-6 | 54–56% | 47.3–47.7% | barely moves |
| full naive (`naive_s31`, ~250 steps) | ~73% | 0.0% | published |

Abstention collapses within 10–20 optimizer steps at the training learning rate, while accuracy has already risen
by about 18 points: a company would ship this model after a few minutes of fine-tuning.

`naive_light_s31` is blocked by the gate (val: G1 bluffing, G4 categories); locked test: accuracy 69.3%,
prudence 12.6% (original 52.7% / 40.2%).

## 2. Targeted (Alight) vs random (Blight) repair, 3 seeds, locked test

Both arms start from `naive_light_s31`; B is matched to A's rounds, target counts, replay and steps
(`arm_B(..., tag='Blight', match='Alight')`). Every arm passed the val gate after one round.
A's diagnosis found 362–423 consistent bluffs per seed, so the targets are now a real subset, not every
unanswerable item. They are skewed: U1 (missing fact) is 38–43% of A's bluffs (U1 is 2/7 ≈ 29% of the probe design),
while U4 (underspecified) is only 6–9% and U5 (misleading) 8–11% (each 1/7 ≈ 14% of the design).
A's repair set per seed: each bluff → ABSTAIN, an answerable twin per bluff, plus delete-a-fact, misleading and
RAFT-flipped variants and replay. B's set has the same numbers of ABSTAIN targets, answer targets and replay
examples, drawn at random from the probe pool (no twins or variants).

| Arm (test) | Accuracy | Prudence | Over-conservativeness | Cost (support) |
|---|---|---|---|---|
| original | 52.7% | 40.2% | – | 2.117 |
| blocked `naive_light_s31` | 69.3% | 12.6% | 0.0% | 2.362 |
| **Alight targeted** (mean ± sd) | **71.8% ± 0.5** | 69.7% ± 3.0 | 0.8% ± 0.4 | 1.155 ± 0.064 |
| **Blight random** (mean ± sd) | 70.6% ± 1.2 | **74.8% ± 3.3** | 1.7% ± 1.0 | **1.072 ± 0.050** |

Paired differences A − B (document-clustered bootstrap, 1,000 resamples; * = 95% CI excludes 0):

| Seed | Cost | Prudence | Over-conservativeness | Accuracy |
|---|---|---|---|---|
| 31 | +0.099 [+0.039, +0.159] * | −4.6 pts [−6.7, −2.5] * | +0.0 pts | +0.7 pts [−1.5, +2.9] |
| 32 | +0.077 [+0.002, +0.154] * | −6.1 pts [−8.5, −3.9] * | −1.9 pts [−4.3, +0.0] | +2.3 pts [+0.0, +4.7] |
| 33 | +0.073 [+0.001, +0.144] * | −4.6 pts [−7.0, −2.3] * | −0.6 pts [−2.8, +1.4] | +0.7 pts [−2.0, +3.2] |

## Result

**H3 with partial damage: not supported.** Even when the blocked model's mistakes are a real subset, random
repair data is significantly better on cost and prudence in all three seeds (about 5 points more abstention on
unanswerable questions). Targeted data again keeps slightly more accuracy (+1.2 points on average) and refuses a
little less (−0.9 points), but neither difference is significant per seed. The pattern matches the full-naive
result (A − B cost on seed 31: +0.093 there, +0.099 here), so the "every item was a mistake" explanation does not
account for H3.

Possible reasons (not tested here): (1) coverage: B's ABSTAIN targets follow the probe design across all five
categories, while A's follow the model's bluffs, which over-weight U1 and under-weight U4/U5, the categories where
repaired models still bluff most; (2) diversity: B draws ~840 distinct unanswerable items, while about half of A's
ABSTAIN targets are variants (delete-a-fact, misleading, RAFT-flipped) of the same ~390 bluffed questions; (3) the
answerable twins, which share passages with the bluffed questions, may make A slightly more willing to answer, which
fits A's higher accuracy and lower over-refusal. A natural next test is a category-balanced targeted set (A's
mistakes resampled to the probe design's category mix) and an A-without-variants ablation.

## Reproduce

```bash
export HUMBLE_HOME=$PWD/work_fix1 HUMBLE_HF_REPO=<you>/humble-tinyaya-fixes HUMBLE_LEGACY_REPO=none HUMBLE_PER_NODE=false
unset HUMBLE_MODEL_ID
run(){ python -c "from humble import hub, experiments as E; hub.run_job('$1', lambda J: $2)"; }
run train_light "E.train_naive(J,seed=31,name='naive_light_s31',max_steps=10)"
run light_val   "E.audit_candidate(J,'naive_light_s31',adapter='naive_light_s31',split='val')"
run light_gate  "E.gate_job(J,'naive_light_s31')"
run light_test  "E.test_once(J,'naive_light_s31',adapter='naive_light_s31')"
for s in 31 32 33; do
  run armAlight$s  "E.arm_A(J,seed=$s,blocked='naive_light_s31',tag='Alight')"
  run armBlight$s  "E.arm_B(J,seed=$s,blocked='naive_light_s31',tag='Blight',match='Alight')"
  run testAlight$s "E.test_arm(J,'Alight',$s)"
  run testBlight$s "E.test_arm(J,'Blight',$s)"
done
run e_targeting "E.analysis(J,'e_targeting',split='test')"
```

Hardware: one NVIDIA RTX PRO 6000 (96 GB). Each repair arm took about 10–20 minutes.
