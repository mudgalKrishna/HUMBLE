#!/usr/bin/env bash
# global: after the pre-lock processes ($@) exit, re-run seed 33, then stage 3 of global_fix2.sh once all
# Acat tests are done (seed 32 runs from global_seed32.sh, seed 31 from global_fix2.sh).
cd "$(dirname "$0")/.."
unset HUMBLE_MODEL_ID
export HUMBLE_HOME=${HUMBLE_HOME:-$PWD/work} HUMBLE_HF_REPO=${HUMBLE_HF_REPO:-icyBear02/humble-v3-mohit}
export HUMBLE_LEGACY_REPO=${HUMBLE_LEGACY_REPO:-KM2007/humble-v3} HUMBLE_PER_NODE=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for p in "$@"; do while kill -0 "$p" 2>/dev/null; do sleep 20; done; done
run() { echo "$(date +%T) start $1"; python -c "from humble import hub, experiments as E; hub.run_job('$1', lambda J: $2)" >"logs/$1.log" 2>&1 || { echo "FAILED $1"; return 1; }; echo "$(date +%T) done  $1"; }
FIN() { echo "(lambda a: a.get('final_adapter') or a['final'])(E.arm_json(J,'$1',$2))"; }
W="{'U3': 2, 'U4': 2, 'U5': 2}"
run armAb33 "E.arm_A(J,seed=33,blocked='naive_s31b',tag='Ab')" && run testAb33 "E.test_arm(J,'Ab',33)" \
 && run armAcat33 "E.arm_A(J,seed=33,blocked='naive_s31b',tag='Acat',cat_weights=$W)" && run testAcat33 "E.test_arm(J,'Acat',33)"
for s in 31 32; do until grep -qh "done  testAcat$s" logs/_*.out run.log 2>/dev/null; do sleep 30; done; done
NAMES="['original','naive_s31b']+[$(FIN Ab 31),$(FIN Ab 32),$(FIN Ab 33),$(FIN Acat 31),$(FIN Acat 32),$(FIN Acat 33)]"
run e_cat_v "E.analysis(J,'e_cat',split='test',names=$NAMES)"
run u45_orig "E.audit_candidate(J,'original',split='test',probe_set='squad3u45')"
for n in $(python -c "from humble import experiments as E; J=None; print(' '.join(($NAMES)[1:]))"); do
  run "u45_$n" "E.audit_candidate(J,'$n',adapter='$n',split='test',probe_set='squad3u45')"
done
run e_cat_u45 "E.analysis(J,'e_cat',probe_set='squad3u45',split='test',names=$NAMES)"
python -c "from humble import hub; print('pending pushes flushed:', hub.flush_pending())"
echo "$(date +%T) done  ALL"
