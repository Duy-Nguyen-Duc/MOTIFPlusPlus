#!/bin/bash
# K sweep of the motion-aware kinematic graph (models.stage1.graph_temporal=true, docs/15): one chain per K
# (src/train_maniskill_kN_geod_chain.sh: stages 1-3, Transfer plain + steered, GLOBAL=1 -> Global plain + steered),
# at most SLOTS training chains at a time. K=5 is already trained, its chain only adds the two Global runs and takes no slot.
#   setsid nohup bash runs/tgraph_ksweep/queue.sh >> runs/tgraph_ksweep/queue.log 2>&1 &     (a copy of this file)
# Idempotent: finished stages are skipped and a chain that is running is never started twice, so after a container
# restart run the same command again. For chains not started yet: echo 3 > runs/tgraph_ksweep/SLOTS, echo 0 > runs/tgraph_ksweep/GPU.
# Results: eval_results/abl_tgraph_k<K>{,_cp_300,_all18,_cp_900}/SUMMARY.json, run folders runs/outputs_maniskill_k<K>_abl_tgraph.
set -u
ROOT=$(cd "$(dirname "$0")" && git rev-parse --show-toplevel 2>/dev/null || (cd "$(dirname "$0")/../.." && pwd)); cd "$ROOT"
Q=runs/tgraph_ksweep; KS=${KS:-1 3 10 50}; EXTRA=models.stage1.graph_temporal=true
mkdir -p $Q
exec 8> $Q/.lock
flock -n 8 || { echo "[$(date -u '+%F %T')] queue is already running, nothing launched"; exit 0; }
[ -f $Q/SLOTS ] || echo 2 > $Q/SLOTS
[ -f $Q/GPU ] || echo 1 > $Q/GPU
[ -f $Q/chain.sh ] || cp src/train_maniskill_kN_geod_chain.sh $Q/chain.sh      # one snapshot of the chain for the whole sweep
log() { echo "[$(date -u '+%F %T')] queue $*"; }
out() { echo runs/outputs_maniskill_k${1}_abl_tgraph; }
finished() { [ -f "$(out $1)/CHAIN_DONE" ] && [ -f eval_results/abl_tgraph_k${1}_cp_900/SUMMARY.json ]; }
alive() { [ -e "$(out $1)/.lock" ] && ! flock -n "$(out $1)/.lock" -c true; }   # the chain holds its lock while it runs
busy() { local n=0 k; for k in $KS; do alive $k && n=$((n + 1)); done; echo $n; }
launch() {
  local K=$1 O; O=$(out $1)
  mkdir -p "$O"; [ -f "$O/chain_sweep.sh" ] || cp $Q/chain.sh "$O/chain_sweep.sh"
  OUT=$O NAME=abl_tgraph_k$K DS=dataset_maniskill_prim_q S1_EXTRA=$EXTRA S3_EXTRA=$EXTRA GPU=$(cat $Q/GPU) K=$K \
    GLOBAL=1 CKPT_EVERY=60 MAX_TRIES=24 WAIT=900 setsid nohup bash "$O/chain_sweep.sh" >> "$O/chain.log" 2>&1 8>&- &
  log "K=$K launched on GPU $(cat $Q/GPU) (pid $!)"; sleep 10
}
finished 5 || alive 5 || launch 5
for K in $KS; do
  finished $K && { log "K=$K already finished"; continue; }
  alive $K && { log "K=$K already running"; continue; }
  until [ "$(busy)" -lt "$(cat $Q/SLOTS)" ]; do sleep 120; done
  launch $K
done
while [ "$(busy)" -gt 0 ] || alive 5; do sleep 300; done
for K in 5 $KS; do finished $K && log "K=$K finished" || log "K=$K NOT finished (see $(out $K)/chain.log)"; done
log "QUEUE DONE"
