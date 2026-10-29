#!/bin/bash
# E-LoRA + geometry tokens + perception steering: K-shot cross-embodiment transfer on ManiSkill
# (K=5: 184/300 policy alone, 249/300 with steering). Stages, each resumable and retried:
#   0 frame cache -> 1 graph-routed motif tokenizer -> 2 motif predictor -> 3 E-LoRA policy + geometry tokens
#   -> benchmark (6 transfer pairs x 50 episodes) -> same checkpoint + perception steering.
#   K=5 GPU=0 bash src/train_maniskill_kN_geod_chain.sh       (run a copy; never edit a running script)
# Data, once (ManiSkill pd_ee_delta_pose demos -> LeRobot -> canonical rel-state actions -> primitive labels -> graphs):
#   python data/process/maniskill_to_lerobot.py --demo_root <demos> --robot <panda|xarm6|widowxai> --out_root demo_data/lerobot
#   python data/process/trajectory_canonicalization.py --dataset_path demo_data/lerobot --save_path demo_data_processed \
#       --in_rotation_type quat --out_action_type rel --out_rotation_type quat
#   python data/process/segment_primitives.py --h5 <demos>  &&  python data/process/label_primitives.py
#   python data/process/build_kinematic_graphs.py
set -u
K=${K:-5}; GPU=${GPU:-0}; SEED=42; MAX_TRIES=${MAX_TRIES:-8}; WAIT=${WAIT:-600}
ROOT=$(cd "$(dirname "$0")/.." && pwd); cd "$ROOT"
OUT=runs/outputs_maniskill_k${K}_graph_hard; NAME=graph_hard_elora_geod_k${K}; CACHE=demo_data/frame_cache_full
S1=$OUT/outputs_stage1; S2=$OUT/outputs_stage2; S3=$OUT/outputs_stage3_elora_geod
mkdir -p "$OUT"
source /pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/miniconda3/etc/profile.d/conda.sh && conda activate motif
export CUDA_VISIBLE_DEVICES=$GPU PYTHONNOUSERSITE=1 PYTHONPATH="$ROOT"
log() { echo "[$(date -u '+%F %T')] K=$K gpu=$GPU $*"; }
run_stage() {   # label artifact cmd...: rerun until the artifact exists (training resumes from last.pth)
  local label=$1 art=$2 attempt rc; shift 2
  for attempt in $(seq 1 "$MAX_TRIES"); do
    [ -f "$art" ] && { log "$label: $art exists"; return 0; }
    log "$label attempt $attempt"
    "$@" >> "$OUT/$label.log" 2>&1; rc=$?
    [ -f "$art" ] && { log "$label done (rc=$rc)"; return 0; }
    log "$label failed rc=$rc; retry in $WAIT s"; sleep "$WAIT"
  done
  log "$label FAILED after $MAX_TRIES attempts"; touch "$OUT/CHAIN_FAILED"; exit 1
}

# stage 0: lossless webp cache of every frame (read by stages 2 and 3 instead of decoding mp4s)
[ -f "$CACHE/PREXTRACT_DONE" ] || { python data/process/prextract_frames.py --dataset_root demo_data_processed \
    --out_root "$CACHE" --robots panda xarm6 widowxai --format webp --workers 12 && touch "$CACHE/PREXTRACT_DONE"; } \
    || { log "frame cache FAILED"; touch "$OUT/CHAIN_FAILED"; exit 1; }

# stage 1: graph-routed motif tokenizer, hard routing (src/train_stage_1_graph.py)
run_stage stage1 "$S1/epoch_0020.pth" \
  python src/train_stage_1_graph.py --config-name train_stage1_graph K=$K seed=$SEED output_dir="$S1" \
    train.epochs=20 train.grad_accum_steps=16 dataset.drop_video=true +train.resume_from=auto \
    batch_size=8 num_robots=3 tasks_num=6 models.stage1.hard_training=true

# stage 2: multimodal motif predictor on the frozen tokenizer (src/train_stage_2.py, config/models/model.yaml)
run_stage stage2 "$S2/epoch_0030.pth" \
  python src/train_stage_2.py dataset=dataset_maniskill_prim +K=$K seed=$SEED output_dir="$S2" \
    models.stage2.vq_vae_ckpt="$S1/epoch_0020.pth" train.epochs=30 train.grad_accum_steps=16 \
    +train.resume_from=auto batch_size=8 tasks_num=6

# stage 3: graph-routed E-LoRA policy with geometry tokens (src/train_stage_3.py, model/stage3_elora.py)
run_stage stage3 "$S3/epoch_0060.pth" \
  python src/train_stage_3.py dataset=dataset_maniskill_prim dataset.data_config_suffix=_geo \
    models=model_graph_elora_geod +K=$K seed=$SEED output_dir="$S3" \
    models.stage2.vq_vae_ckpt="$S1/epoch_0020.pth" models.stage3.mkl_ckpt="$S2/epoch_0030.pth" \
    train.epochs=60 train.grad_accum_steps=4 train.ckpt_interval=20 +train.resume_from=auto \
    batch_size=16 dataset.num_workers=10 tasks_num=6

# benchmark: the policy alone, then the same checkpoint with perception steering (eval/run_percept_steering.py)
run_stage eval "eval_results/$NAME/SUMMARY.json" \
  python eval/run_percept_steering.py --ckpt "$S3/epoch_0060.pth" --name $NAME --out eval_results/$NAME --steer 0
run_stage steer "eval_results/${NAME}_cp_300/SUMMARY.json" \
  python eval/run_percept_steering.py --ckpt "$S3/epoch_0060.pth" --name ${NAME}_cp_300 --out eval_results/${NAME}_cp_300 --steer 1.0

rm -f "$S3/last.pth" "$S2/last.pth" "$S1/last.pth"
log "ALL DONE"; touch "$OUT/CHAIN_DONE"
