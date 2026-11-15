#!/usr/bin/env bash
# Fix 2 on Tiny Aya Global in ONE notebook (RTX PRO 6000): upweighted repair Acat (seeds 31-33) and the larger
# U4/U5 test set squad3u45, steps in parallel on one GPU, all dependencies on local disk.
#
#   export HF_TOKEN=...                       # never commit it
#   bash scripts/global_fix2.sh               # PAR=1 runs the seed chains one after another (if OOM)
#
# Blocked model: Krishna's naive_s31, read from HUMBLE_LEGACY_REPO if this token can see it. Otherwise the
# script retrains it as naive_s31b and re-runs plain A (tag Ab) on it, so A vs Acat start from the same model.
# Published Tiny Aya names (naive_s31, repA_*) are never overwritten; results go to your own repo.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${HF_TOKEN:?set HF_TOKEN}"
unset HUMBLE_MODEL_ID                                   # default config = Tiny Aya Global
export HUMBLE_HOME=${HUMBLE_HOME:-$PWD/work}
export HUMBLE_HF_REPO=${HUMBLE_HF_REPO:-icyBear02/humble-v3-mohit}
export HUMBLE_LEGACY_REPO=${HUMBLE_LEGACY_REPO:-KM2007/humble-v3}   # read-only; where Krishna's naive_s31 lives
export HUMBLE_PER_NODE=false
export HF_XET_HIGH_PERFORMANCE=1 HF_HUB_DISABLE_PROGRESS_BARS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PAR=${PAR:-3}
W="{'U3': 2, 'U4': 2, 'U5': 2}"
mkdir -p logs

# ---- setup: install while the base model and the judge download
(pip install -q -U "huggingface_hub[hf_xet]" && hf download CohereLabs/tiny-aya-global >/dev/null \
  && hf download CohereLabs/aya-expanse-8b >/dev/null) &
pip install -q uv && uv pip install --python "$(command -v python)" -q -e ".[gpu]"
wait
[ -d "$HUMBLE_HOME/probes/squad3v" ] || { mkdir -p "$HUMBLE_HOME" && tar xzf data/humble_v3_shared.tar.gz -C "$HUMBLE_HOME"; }
# keep audits/squad3v/original: those ARE Tiny Aya Global's (its locked test is already evaluated; never rerun)
(cd "$HUMBLE_HOME/probes/squad3v" && echo "$(cat TEST_LOCK.txt)  items_test.jsonl" | sha256sum -c -)

run() {  # run <job> <python expr>; stdout+stderr in logs/<job>.log, non-zero exit on error
  echo "$(date +%T) start $1"
  python -c "from humble import hub, experiments as E; hub.run_job('$1', lambda J: $2)" >"logs/$1.log" 2>&1 \
    || { echo "FAILED $1 (logs/$1.log)"; tail -20 "logs/$1.log"; return 1; }
  echo "$(date +%T) done  $1"
}
FIN() { echo "(lambda a: a.get('final_adapter') or a['final'])(E.arm_json(J,'$1',$2))"; }

run env_check "E.env_check(J)"

# ---- stage 1: get the blocked model || build squad3u45 (judge on the same GPU)
run u45build "E.build_u45_probes(J)" > logs/_u45.out 2>&1 &
if python -c "from humble import experiments as E; E._get_adapter('naive_s31'); E._get_preds('squad3v','naive_s31','val')" 2>/dev/null; then
  BLOCKED=naive_s31; ATAG=A; echo "using Krishna's naive_s31 and his published A arms"
else
  BLOCKED=naive_s31b; ATAG=Ab; echo "naive_s31 not readable: retraining as naive_s31b and re-running A as Ab"
  run naive31b "E.train_naive(J,seed=31,name='naive_s31b')"
  run naive31b_val "E.audit_candidate(J,'naive_s31b',adapter='naive_s31b',split='val')"
  run naive31b_gate "E.gate_job(J,'naive_s31b')"
  DEC=$(python -c "import json,glob;print(json.load(open(glob.glob('$HUMBLE_HOME/audits/squad3v/naive_s31b/val/gate_*.json')[0]))['decision'])")
  echo "naive_s31b gate: $DEC"
  [ "$DEC" = BLOCK ] || { echo "naive_s31b PASSED the gate: tell Krishna before going on"; wait; exit 2; }
fi

# ---- stage 2: per-seed chains (Ab -> test, if retrained) then Acat -> test
seed_chain() {
  if [ "$ATAG" = Ab ]; then
    run armAb$1 "E.arm_A(J,seed=$1,blocked='naive_s31b',tag='Ab')" && run testAb$1 "E.test_arm(J,'Ab',$1)" || return 1
  fi
  run armAcat$1 "E.arm_A(J,seed=$1,blocked='$BLOCKED',tag='Acat',cat_weights=$W)" && run testAcat$1 "E.test_arm(J,'Acat',$1)"
}
for s in 31 32 33; do
  if [ "$PAR" -gt 1 ]; then seed_chain $s > "logs/_seed$s.out" 2>&1 & sleep 60; else seed_chain $s; fi
done
wait
cat logs/_u45.out logs/_seed*.out 2>/dev/null | grep -E "FAILED|done" || true

# ---- stage 3: per-category tables on squad3v test and on squad3u45
NAMES="['original','$BLOCKED']+[$(FIN $ATAG 31),$(FIN $ATAG 32),$(FIN $ATAG 33),$(FIN Acat 31),$(FIN Acat 32),$(FIN Acat 33)]"
run e_cat_v "E.analysis(J,'e_cat',split='test',names=$NAMES)"
run u45_orig "E.audit_candidate(J,'original',split='test',probe_set='squad3u45')"
for n in $(python -c "from humble import experiments as E; J=None; print(' '.join(($NAMES)[1:]))"); do
  run "u45_$n" "E.audit_candidate(J,'$n',adapter='$n',split='test',probe_set='squad3u45')"
done
run e_cat_u45 "E.analysis(J,'e_cat',probe_set='squad3u45',split='test',names=$NAMES)"
python -c "from humble import hub; print('pending pushes flushed:', hub.flush_pending())"
python - <<EOF
import json
for f in ("$HUMBLE_HOME/audits/squad3v/_e_cat_test.json", "$HUMBLE_HOME/audits/squad3u45/_e_cat_test.json"):
    rows = json.load(open(f))
    print("\n" + f)
    for n, r in rows.items():
        print(f"{n:22s}", {c: round(v['prudence'], 3) for c, v in r['by_category'].items()})
EOF
python -c "import json;m=json.load(open('$HUMBLE_HOME/probes/squad3u45/MANIFEST.json'));print('squad3u45 cells',m['cells'],'earlier iterations excluded:',m.get('earlier_iterations_excluded'))"
echo "send Krishna: the two tables above, logs/, and a zip of work/"
