#!/usr/bin/env bash
# fire: after the pre-lock processes ($@ = their pids) exit, re-run seeds 32/33 (GPU work now serialised by
# config.one_gpu_job_at_a_time), then redo stage 4 of fire_parallel.sh once every seed's B test is done.
cd "$(dirname "$0")/.."
export HUMBLE_HOME=${HUMBLE_HOME:-$PWD/work} HUMBLE_MODEL_ID=CohereLabs/tiny-aya-fire HUMBLE_HF_REPO=${HUMBLE_HF_REPO:-icyBear02/humble-tiny-aya-fire}
export HUMBLE_LEGACY_REPO=none HUMBLE_PER_NODE=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for p in "$@"; do while kill -0 "$p" 2>/dev/null; do sleep 20; done; done
run() { echo "$(date +%T) start $1"; python -c "from humble import hub, experiments as E; hub.run_job('$1', lambda J: $2)" >"logs/$1.log" 2>&1 || { echo "FAILED $1"; return 1; }; echo "$(date +%T) done  $1"; }
chain() { run armA$1 "E.arm_A(J,seed=$1)" && run testA$1 "E.test_arm(J,'A',$1)" && run armB$1 "E.arm_B(J,seed=$1)" && run testB$1 "E.test_arm(J,'B',$1)"; }
chain 32 & chain 33 & wait
until grep -qh "done  testB31" logs/_*.out run.log 2>/dev/null; do sleep 30; done
F="[(lambda a: a.get('final_adapter') or a['final'])(E.arm_json(J,t,31)) for t in ('A','B')]"
run certify2 "E.certify_job(J,['naive_s31']+$F)"
run summary2 "E.analysis(J,'e_repair_summary',split='test')"
run e_cat2 "E.analysis(J,'e_cat',split='test',names=['original','naive_s31']+$F)"
python -c "from humble import hub; print('pending pushes flushed:', hub.flush_pending())"
python results/v3/make_report.py && echo "$(date +%T) done  report"
