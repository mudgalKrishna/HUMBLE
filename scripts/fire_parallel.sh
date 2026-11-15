#!/usr/bin/env bash
# Full HUMBLE recipe for one base model in ONE notebook, independent steps run as parallel processes on one GPU
# (a 3.35B model leaves most of a 96 GB RTX PRO 6000 idle). Processes share work/ on local disk, so no step
# waits on the Hub; pushes to HUMBLE_HF_REPO still happen as a backup.
#
#   export HF_TOKEN=...                       # never commit it
#   bash scripts/fire_parallel.sh             # PAR=1 runs the three repair seeds one after another (if OOM)
set -euo pipefail
cd "$(dirname "$0")/.."
: "${HF_TOKEN:?set HF_TOKEN}"
export HUMBLE_HOME=${HUMBLE_HOME:-$PWD/work}
export HUMBLE_MODEL_ID=${HUMBLE_MODEL_ID:-CohereLabs/tiny-aya-fire}
export HUMBLE_HF_REPO=${HUMBLE_HF_REPO:-icyBear02/humble-tiny-aya-fire}
export HUMBLE_LEGACY_REPO=none HUMBLE_PER_NODE=false
export HF_XET_HIGH_PERFORMANCE=1 HF_HUB_DISABLE_PROGRESS_BARS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PAR=${PAR:-3}
mkdir -p logs

# ---- setup: install while the model downloads
(pip install -q -U "huggingface_hub[hf_xet]" && hf download "$HUMBLE_MODEL_ID" >/dev/null) &
pip install -q uv && uv pip install --python "$(command -v python)" -q -e ".[gpu]"
wait
if [ ! -d "$HUMBLE_HOME/probes/squad3v" ]; then
  mkdir -p "$HUMBLE_HOME" && tar xzf data/humble_v3_shared.tar.gz -C "$HUMBLE_HOME"
  rm -rf "$HUMBLE_HOME/audits/squad3v/original"   # those are Tiny Aya Global's; this model makes its own
fi
(cd "$HUMBLE_HOME/probes/squad3v" && echo "$(cat TEST_LOCK.txt)  items_test.jsonl" | sha256sum -c -)

run() {  # run <job> <python expr>; stdout+stderr in logs/<job>.log, non-zero exit on error
  echo "$(date +%T) start $1"
  python -c "from humble import hub, experiments as E; hub.run_job('$1', lambda J: $2)" >"logs/$1.log" 2>&1 \
    || { echo "FAILED $1 (logs/$1.log)"; tail -20 "logs/$1.log"; return 1; }
  echo "$(date +%T) done  $1"
}

run env_check "E.env_check(J)"; tail -30 logs/env_check.log   # check the stop tokens printed

# ---- stage 1: original audits || naive training
( run orig_val "E.audit_candidate(J,'original',split='val')" && run orig_test "E.test_once(J,'original')" ) &
run naive31 "E.train_naive(J,seed=31)" &
wait

# ---- stage 2: naive val + gate (must BLOCK) || prevention arm
( run prevC31 "E.train_prevention(J,seed=31)" \
  && run prevC31_val "E.audit_candidate(J,'prevC_s31',adapter='prevC_s31',split='val')" \
  && run prevC31_test "E.test_once(J,'prevC_s31',adapter='prevC_s31')" ) > logs/_prevC.out 2>&1 &
PREV=$!
run naive31_val "E.audit_candidate(J,'naive_s31',adapter='naive_s31',split='val')"
run naive31_gate "E.gate_job(J,'naive_s31')"
DEC=$(python -c "import json,glob;print(json.load(open(glob.glob('$HUMBLE_HOME/audits/squad3v/naive_s31/val/gate_*.json')[0]))['decision'])")
echo "naive_s31 gate: $DEC"
if [ "$DEC" != "BLOCK" ]; then echo "naive_s31 PASSED the gate: tell Krishna before going on"; wait; exit 2; fi

# ---- stage 3: naive test || per-seed chains A -> testA -> B -> testB (B is matched to A of the same seed)
run naive31_test "E.test_once(J,'naive_s31',adapter='naive_s31')" &
seed_chain() {
  run armA$1 "E.arm_A(J,seed=$1)" && run testA$1 "E.test_arm(J,'A',$1)" \
  && run armB$1 "E.arm_B(J,seed=$1)" && run testB$1 "E.test_arm(J,'B',$1)"
}
for s in 31 32 33; do
  if [ "$PAR" -gt 1 ]; then seed_chain $s > "logs/_seed$s.out" 2>&1 & sleep 60; else seed_chain $s; fi
done
wait
cat logs/_prevC.out logs/_seed*.out 2>/dev/null | grep -E "FAILED|done" || true

# ---- stage 4: certificates, summary, report
F="[(lambda a: a.get('final_adapter') or a['final'])(E.arm_json(J,t,31)) for t in ('A','B')]"   # actual final round
run certify "E.certify_job(J,['naive_s31']+$F+['prevC_s31'])"
run summary "E.analysis(J,'e_repair_summary',split='test')"
run e_cat "E.analysis(J,'e_cat',split='test',names=['original','naive_s31']+$F+['prevC_s31'])"
python -c "from humble import hub; print('pending pushes flushed:', hub.flush_pending())"
python results/v3/make_report.py
echo "report: results/v3/RESULTS_${HUMBLE_MODEL_ID##*/}.md   (zip work/ as a backup)"
