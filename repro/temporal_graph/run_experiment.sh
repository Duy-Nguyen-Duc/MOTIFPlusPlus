#!/bin/bash
# Temporal-graph chain per K on one GPU: stages 1-3, Transfer plain + steered, Global plain + steered (same launch line
# as src/queue_tgraph_ksweep.sh). Resumable: rerun the same command after an interruption.
#   K=5 GPU=0 bash repro/temporal_graph/run_experiment.sh                       one K
#   KS="1 3 5 10 50" GPU=0 setsid nohup bash repro/temporal_graph/run_experiment.sh >> runs/tgraph_repro.log 2>&1 &   sweep
# Variables: KS or K (5), GPU (0), VARIANT (tgraph | tgraph_attn), MAX_TRIES (24), WAIT (900 s), MANISKILL_ROOT,
# VK_FALLBACK_ROOT (software renderer for the benchmark), CONDA_SH / CONDA_ENV (conda.sh path and env name for the chain).
set -u
ROOT=$(cd "$(dirname "$0")/../.." && pwd); cd "$ROOT"
KS=${KS:-${K:-5}}; GPU=${GPU:-0}; VARIANT=${VARIANT:-tgraph}
case $VARIANT in
  tgraph) EXTRA="models.stage1.graph_temporal=true";;
  tgraph_attn) EXTRA="models.stage1.graph_temporal=true models.stage1.graph_routing=attn";;
  *) echo "VARIANT must be tgraph or tgraph_attn"; exit 1;;
esac
[ -f demo_data_processed_prim_q/build_stats.json ] || { echo "no demo_data_processed_prim_q: run repro/temporal_graph/prepare_data.sh first"; exit 1; }
export MANISKILL_ROOT=${MANISKILL_ROOT:-$ROOT/../ManiSkill}
log() { echo "[$(date -u '+%F %T')] repro $*"; }
for K in $KS; do
  O=runs/outputs_maniskill_k${K}_abl_${VARIANT}; N=abl_${VARIANT}_k$K
  mkdir -p "$O"; [ -f "$O/chain_tgraph.sh" ] || cp src/train_maniskill_kN_geod_chain.sh "$O/chain_tgraph.sh"   # never edit a running copy
  log "K=$K -> $O (eval_results/$N{,_cp_300,_all18,_cp_900})"
  OUT=$O NAME=$N DS=dataset_maniskill_prim_q S1_EXTRA="$EXTRA" S3_EXTRA="$EXTRA" K=$K GPU=$GPU GLOBAL=1 CKPT_EVERY=60 \
    MAX_TRIES=${MAX_TRIES:-24} WAIT=${WAIT:-900} bash "$O/chain_tgraph.sh" >> "$O/chain.log" 2>&1
  [ -f "$O/CHAIN_DONE" ] || { log "K=$K did not finish (see $O/chain.log); rerun this command to resume"; exit 1; }
done
python - $VARIANT $KS <<'PY'
import json, os, sys
v, ks = sys.argv[1], sys.argv[2:]
print(f"{'K':>3} {'Transfer':>9} {'+steer':>7} {'Global':>8} {'+steer':>7}   (successes of 300 / 300 / 900 / 900)")
for k in ks:
    cells = []
    for sfx in ("", "_cp_300", "_all18", "_cp_900"):
        p = f"eval_results/abl_{v}_k{k}{sfx}/SUMMARY.json"
        cells.append(str(json.load(open(p))["successes"]) if os.path.exists(p) else "-")
    print(f"{k:>3} {cells[0]:>9} {cells[1]:>7} {cells[2]:>8} {cells[3]:>7}")
PY
