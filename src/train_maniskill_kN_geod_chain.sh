#!/bin/bash
# E-LoRA + geometry tokens + perception steering: K-shot cross-embodiment transfer on ManiSkill
# (K=5: 184/300 policy alone, 249/300 with steering). Stages, each resumable and retried:
#   0 frame cache -> 1 graph-routed motif tokenizer -> 2 motif predictor -> 3 E-LoRA policy + geometry tokens
#   -> benchmark (6 transfer pairs x 50 episodes) -> same checkpoint + perception steering.
#   K=5 GPU=0 bash src/train_maniskill_kN_geod_chain.sh       (run a copy; never edit a running script)
#   GLOBAL=1 adds the Global metric after the Transfer benchmark (all 18 pairs, the 6 transfer pairs reused, plain and
#   steered -> eval_results/<name>_all18, <name>_cp_900); CKPT_EVERY=60 keeps only the final stage-3 checkpoint.
#   One chain per run folder (lock): launching a chain that is already running is a no-op.
# Ablations (K=5, same chain, OUT/NAME + per-stage overrides):
#   no kinematic graph (learned per-robot vector instead of g_r, one E-LoRA node per robot):
#     OUT=runs/outputs_maniskill_k5_abl_nograph NAME=abl_nograph_k5 S1_EXTRA=models.stage1.morphology=embed S3_EXTRA=models.stage1.morphology=embed
#   no primitive labels (InfoNCE + adversary on task and episode progress, as MOTIF):
#     OUT=runs/outputs_maniskill_k5_abl_noprim NAME=abl_noprim_k5 S1_EXTRA="prim_conditioning=false infonce_cond=task infonce_sigma=0.02 infonce_min_weight=0"
#   no geometry tokens = the E-LoRA checkpoint without geometry_cfg (eval_results/graph_hard_elora{,_percept_cp_300})
#   temporal kinematic graph (nodes/edges also encode the window's joint trajectories; data/process/append_qpos.py first):
#     OUT=runs/outputs_maniskill_k5_abl_tgraph NAME=abl_tgraph_k5 DS=dataset_maniskill_prim_q S1_EXTRA=models.stage1.graph_temporal=true S3_EXTRA=models.stage1.graph_temporal=true
#   temporal graph + attention routing (each motif token attends over the robot's joint nodes; code -> joint map in outputs_stage1/routing/attn_epoch_*.json):
#     OUT=runs/outputs_maniskill_k5_abl_tgraph_attn NAME=abl_tgraph_attn_k5 DS=dataset_maniskill_prim_q S1_EXTRA="models.stage1.graph_temporal=true models.stage1.graph_routing=attn" S3_EXTRA="models.stage1.graph_temporal=true models.stage1.graph_routing=attn"
# Data, once (ManiSkill pd_ee_delta_pose demos -> LeRobot -> canonical rel-state actions -> primitive labels -> graphs):
#   python data/process/maniskill_to_lerobot.py --demo_root <demos> --robot <panda|xarm6|widowxai> --out_root demo_data/lerobot
#   python data/process/trajectory_canonicalization.py --dataset_path demo_data/lerobot --save_path demo_data_processed \
#       --in_rotation_type quat --out_action_type rel --out_rotation_type quat
#   python data/process/segment_primitives.py --h5 <demos>  &&  python data/process/label_primitives.py
#   python data/process/build_kinematic_graphs.py
set -u
K=${K:-5}; GPU=${GPU:-0}; SEED=42; MAX_TRIES=${MAX_TRIES:-8}; WAIT=${WAIT:-600}; GLOBAL=${GLOBAL:-0}; CKPT_EVERY=${CKPT_EVERY:-20}
ROOT=$(cd "$(dirname "$0")" && git rev-parse --show-toplevel 2>/dev/null || (cd "$(dirname "$0")/.." && pwd)); cd "$ROOT"   # also from a copy under runs/
OUT=${OUT:-runs/outputs_maniskill_k${K}_graph_hard}; NAME=${NAME:-graph_hard_elora_geod_k${K}}; CACHE=demo_data/frame_cache_full
S1_EXTRA=${S1_EXTRA:-}; S3_EXTRA=${S3_EXTRA:-}; DS=${DS:-dataset_maniskill_prim}
S1=$OUT/outputs_stage1; S2=$OUT/outputs_stage2; S3=$OUT/outputs_stage3_elora_geod
mkdir -p "$OUT"
exec 9> "$OUT/.lock"   # a second launch of the same run (another session, a resume script) does nothing; children get 9>&-
flock -n 9 || { echo "[$(date -u '+%F %T')] $OUT: this chain is already running, nothing launched"; exit 0; }
CONDA_SH=${CONDA_SH:-/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/miniconda3/etc/profile.d/conda.sh}; CONDA_ENV=${CONDA_ENV:-motif}
if [ -f "$CONDA_SH" ]; then source "$CONDA_SH" && conda activate "$CONDA_ENV"; fi   # else: launched inside an activated env
export CUDA_VISIBLE_DEVICES=$GPU PYTHONNOUSERSITE=1 PYTHONPATH="$ROOT"
log() { echo "[$(date -u '+%F %T')] K=$K gpu=$GPU $*"; }
run_stage() {   # label artifact cmd...: rerun until the artifact exists (training resumes from last.pth)
  local label=$1 art=$2 attempt rc; shift 2
  for attempt in $(seq 1 "$MAX_TRIES"); do
    [ -f "$art" ] && { log "$label: $art exists"; return 0; }
    log "$label attempt $attempt"
    "$@" >> "$OUT/$label.log" 2>&1 9>&-; rc=$?
    [ -f "$art" ] && { log "$label done (rc=$rc)"; return 0; }
    log "$label failed rc=$rc; retry in $WAIT s"; sleep "$WAIT"
  done
  log "$label FAILED after $MAX_TRIES attempts"; touch "$OUT/CHAIN_FAILED"; exit 1
}

# stage 0: lossless webp cache of every frame (read by stages 2 and 3 instead of decoding mp4s)
[ -f "$CACHE/PREXTRACT_DONE" ] || { python data/process/prextract_frames.py --dataset_root demo_data_processed \
    --out_root "$CACHE" --robots panda xarm6 widowxai --format webp --workers 12 9>&- && touch "$CACHE/PREXTRACT_DONE"; } \
    || { log "frame cache FAILED"; touch "$OUT/CHAIN_FAILED"; exit 1; }

# stage 1: graph-routed motif tokenizer, hard routing (src/train_stage_1_graph.py)
run_stage stage1 "$S1/epoch_0020.pth" \
  python src/train_stage_1_graph.py --config-name train_stage1_graph dataset=$DS K=$K seed=$SEED output_dir="$S1" \
    train.epochs=20 train.grad_accum_steps=16 dataset.drop_video=true +train.resume_from=auto \
    batch_size=8 num_robots=3 tasks_num=6 models.stage1.hard_training=true $S1_EXTRA

# stage 2: multimodal motif predictor on the frozen tokenizer (src/train_stage_2.py, config/models/model.yaml)
run_stage stage2 "$S2/epoch_0030.pth" \
  python src/train_stage_2.py dataset=$DS +K=$K seed=$SEED output_dir="$S2" \
    models.stage2.vq_vae_ckpt="$S1/epoch_0020.pth" train.epochs=30 train.grad_accum_steps=16 \
    +train.resume_from=auto batch_size=8 tasks_num=6

# stage 3: graph-routed E-LoRA policy with geometry tokens (src/train_stage_3.py, model/stage3_elora.py)
run_stage stage3 "$S3/epoch_0060.pth" \
  python src/train_stage_3.py dataset=$DS dataset.data_config_suffix=_geo \
    models=model_graph_elora_geod +K=$K seed=$SEED output_dir="$S3" \
    models.stage2.vq_vae_ckpt="$S1/epoch_0020.pth" models.stage3.mkl_ckpt="$S2/epoch_0030.pth" \
    train.epochs=60 train.grad_accum_steps=4 train.ckpt_interval=$CKPT_EVERY +train.resume_from=auto \
    batch_size=16 dataset.num_workers=10 tasks_num=6 $S3_EXTRA

# benchmark: the policy alone, then the same checkpoint with perception steering (eval/run_percept_steering.py)
run_stage eval "eval_results/$NAME/SUMMARY.json" \
  python eval/run_percept_steering.py --ckpt "$S3/epoch_0060.pth" --name $NAME --out eval_results/$NAME --steer 0
run_stage steer "eval_results/${NAME}_cp_300/SUMMARY.json" \
  python eval/run_percept_steering.py --ckpt "$S3/epoch_0060.pth" --name ${NAME}_cp_300 --out eval_results/${NAME}_cp_300 --steer 1.0

# Global metric: all 18 robot-task pairs, the 6 transfer pairs copied from the two runs above (same checkpoint, sha-checked)
if [ "$GLOBAL" = 1 ]; then
  run_stage global "eval_results/${NAME}_all18/SUMMARY.json" \
    python eval/run_percept_steering.py --ckpt "$S3/epoch_0060.pth" --name ${NAME}_all18 --out eval_results/${NAME}_all18 --steer 0 --all-pairs --reuse $NAME
  run_stage global_steer "eval_results/${NAME}_cp_900/SUMMARY.json" \
    python eval/run_percept_steering.py --ckpt "$S3/epoch_0060.pth" --name ${NAME}_cp_900 --out eval_results/${NAME}_cp_900 --steer 1.0 --all-pairs --reuse ${NAME}_cp_300
fi

rm -f "$S3/last.pth" "$S2/last.pth" "$S1/last.pth"
log "ALL DONE"; touch "$OUT/CHAIN_DONE"
