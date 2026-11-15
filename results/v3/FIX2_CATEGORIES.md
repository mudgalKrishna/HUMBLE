# Fix 2: weak categories (U3/U4/U5) and a bigger U4/U5 test set

Branch `fix-categories`, Tiny Aya Global, default config (unset the four `HUMBLE_*` model env vars).

## 2a. Upweighted repair (`Acat`)

`arm_A(..., cat_weights={"U3": 2, "U4": 2, "U5": 2}, tag="Acat")` repeats every repair example of those
categories (bluff → ABSTAIN, its twin, delete-a-fact, misleading, RAFT-flipped) 2× in total.
CuratorKIT's exact dedup would collapse plain copies, so each copy shows the same passages in another order.
Replay is sized after upweighting (repair:replay stays 1:1). Default `cat_weights=None` leaves arm A unchanged.

```bash
run(){ python -c "from humble import hub, experiments as E; hub.run_job('$1', lambda J: $2)"; }
W='{"U3": 2, "U4": 2, "U5": 2}'
for s in 31 32 33; do
  run armAcat$s "E.arm_A(J, seed=$s, tag='Acat', cat_weights=$W)"
  run testAcat$s "E.test_arm(J, 'Acat', $s)"
done
run ecat "E.analysis(J, 'e_cat', split='test', names=['original','naive_s31','repA_s31_r1','repAcat_s31_r1'])"
```
(Use each arm's actual final adapter name from `v3/arms/arm{tag}_s{seed}.json` in `names`.)

## 2b. `squad3u45`: larger U4/U5 test set

squad3v's article split covers all 477 SQuAD 2.0 articles (train 286 / probe 95 / val 48 / test 48), so no
article is entirely unseen. `build_u45_probes` uses **train-split articles** (never squad3v val/test) and
only paragraphs that **no prompt of squad3 or squad3v ever showed** (items, distractors, twins), plus the
earlier-iteration exclusions. Distractors come from the same fresh paragraphs. squad3v is read only.

Dry run without the judge (local, no GPU): 282 articles, 4,141 fresh paragraphs, mechanical
candidates U4 = 963 (all that exist), U5 ≥ 2,000. At squad3v's judge pass rates (U4 12–21%, U5 19–28%)
that gives roughly 115–200 U4 and 150 U5 judged items (n=150, overgenerate 8×). The job logs a warning if a
cell ends under 100. Caveat: the naive and prevention models were trained on other paragraphs of these
articles, so this set is article-level seen, paragraph-level fresh.

```bash
run u45build "E.build_u45_probes(J)"            # needs GPU for the judge (aya-expanse-8b)
run u45orig  "E.audit_candidate(J, 'original', split='test', probe_set='squad3u45')"
run u45naive "E.audit_candidate(J, 'naive_s31', adapter='naive_s31', split='test', probe_set='squad3u45')"
# then the final A / Acat / B adapters the same way; per-category table:
run u45cat   "E.analysis(J, 'e_cat', probe_set='squad3u45', split='test', names=[...])"
```
