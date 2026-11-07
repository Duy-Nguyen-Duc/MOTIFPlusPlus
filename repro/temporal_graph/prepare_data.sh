#!/bin/bash
# Data for the temporal-graph experiment; every step is skipped when its output exists:
#   generate (motion planning) -> replay (rgb, pd_ee_delta_pose) -> LeRobot -> canonical rel-state actions -> primitive
#   labels -> (graph check) -> joint trajectories -> frame cache.
#   MANISKILL_ROOT=../ManiSkill VK_FALLBACK_ROOT=<lavapipe root> bash repro/temporal_graph/prepare_data.sh
# Variables: RAW (demo_data/maniskill_h5), NUM_TRAJ (50 per robot-task pair), NUM_PROCS (10), REPLAY_ENVS (10),
# ROBOTS, TASKS, REBUILD_GRAPHS=1 (rebuild graphs + qpos layout from the fork's URDFs and diff against the committed files).
# To start from transferred demos put them at $RAW (generate + replay are then skipped).
set -eu
ROOT=$(cd "$(dirname "$0")/../.." && pwd); cd "$ROOT"
MANISKILL_ROOT=${MANISKILL_ROOT:-$ROOT/../ManiSkill}
RAW=${RAW:-demo_data/maniskill_h5}; NUM_TRAJ=${NUM_TRAJ:-50}; NUM_PROCS=${NUM_PROCS:-10}; REPLAY_ENVS=${REPLAY_ENVS:-10}
ROBOTS=${ROBOTS:-panda xarm6 widowxai}
TASKS=${TASKS:-PushCube-v1 PlaceSphere-v1 PullCube-v1 LiftPegUpright-v1 PickCube-v1 StackCube-v1}
H5=trajectory.rgb.pd_ee_delta_pose.physx_cpu.h5
export PYTHONPATH=$ROOT:$MANISKILL_ROOT PYTHONNOUSERSITE=1
if [ -n "${VK_FALLBACK_ROOT:-}" ]; then   # software renderer, same variables eval/run_percept_steering.py sets for its workers
  export VK_ICD_FILENAMES=$VK_FALLBACK_ROOT/lvp_icd.json \
         LD_LIBRARY_PATH=$VK_FALLBACK_ROOT/usr/lib/x86_64-linux-gnu:$VK_FALLBACK_ROOT/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}
fi
export MS_RENDER_BACKEND=${MS_RENDER_BACKEND:-pci:0} LP_NUM_THREADS=${LP_NUM_THREADS:-8} XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-$(mktemp -d)}
log() { echo "[$(date -u '+%F %T')] data $*"; }
gripper_idx() { case $1 in panda) echo 7,8;; xarm6) echo 6;; widowxai) echo 6,7;; *) echo "no gripper columns known for $1" >&2; exit 1;; esac; }

for r in $ROBOTS; do for t in $TASKS; do
  d=$RAW/$r/$t/motionplanning
  if [ ! -f "$d/trajectory.h5" ]; then
    log "generate $r $t ($NUM_TRAJ successful plans, $NUM_PROCS workers)"
    python "$MANISKILL_ROOT/mani_skill/examples/motionplanning/$r/run.py" --env-id "$t" --num-traj "$NUM_TRAJ" --num-procs "$NUM_PROCS" \
      --only-count-success --traj-name trajectory --obs-mode none --sim-backend cpu --shader default --record-dir "$RAW/$r"
  fi
  if [ ! -f "$d/$H5" ]; then
    log "replay $r $t (rgb, pd_ee_delta_pose, physx_cpu)"
    python "$MANISKILL_ROOT/mani_skill/trajectory/replay_trajectory.py" --traj-path "$d/trajectory.h5" -o rgb -c pd_ee_delta_pose \
      --save-traj --num-envs "$REPLAY_ENVS" -b physx_cpu
  fi
done; done

for r in $ROBOTS; do
  [ -f "demo_data/lerobot/$r/meta/episodes.jsonl" ] && continue
  log "convert $r -> demo_data/lerobot/$r"
  python data/process/maniskill_to_lerobot.py --demo_root "$RAW" --robot "$r" --out_root demo_data/lerobot --traj_name "$H5" \
    --tasks $TASKS --max_episodes_per_task "$NUM_TRAJ" --fps 20 --base_camera third_view_camera --wrist_camera hand_camera \
    --gripper_qpos_indices "$(gripper_idx "$r")" --gripper_norm dataset
done

done_all=1; for r in $ROBOTS; do [ -f "demo_data_processed/$r/meta/episodes.jsonl" ] || done_all=0; done
if [ $done_all = 0 ]; then
  log "canonicalize -> demo_data_processed"
  python data/process/trajectory_canonicalization.py --dataset_path demo_data/lerobot --save_path demo_data_processed \
    --in_rotation_type quat --out_action_type rel --out_rotation_type quat
fi

[ -f demo_data_processed_prim/segments.json ] || { log "segment primitives"; python data/process/segment_primitives.py --h5 "$RAW" --out demo_data_processed_prim; }
[ -f demo_data_processed_prim/PRIMITIVES_REPORT.json ] || { log "label primitives -> demo_data_processed_prim"; python data/process/label_primitives.py --robots $ROBOTS; }

if [ "${REBUILD_GRAPHS:-0}" = 1 ]; then
  log "rebuilding kinematic graphs + qpos layout from the fork's URDFs into demo_data/_graph_rebuild (committed assets untouched)"
  mkdir -p demo_data/_graph_rebuild
  MANISKILL_ASSETS=$MANISKILL_ROOT/mani_skill/assets/robots MANISKILL_DATA=${MS_ASSET_DIR:-$HOME/.maniskill}/data/robots \
    python data/process/build_kinematic_graphs.py --out demo_data/_graph_rebuild
  python data/process/build_qpos_layout.py --graph_dir demo_data/_graph_rebuild --robots $ROBOTS
  python - $ROBOTS <<'PY'
import json, sys, numpy as np
lay0, lay1 = json.load(open("assets/kinematic_graphs/qpos_layout.json")), json.load(open("demo_data/_graph_rebuild/qpos_layout.json"))
for r in sys.argv[1:]:
    a, b = json.load(open(f"assets/kinematic_graphs/{r}.json")), json.load(open(f"demo_data/_graph_rebuild/{r}.json"))
    same = (a["node_names"] == b["node_names"] and np.allclose(a["node_feats"], b["node_feats"]) and a["edge_src"] == b["edge_src"]
            and a["edge_dst"] == b["edge_dst"] and np.allclose(a["edge_feats"], b["edge_feats"]))
    l0, l1 = lay0[r], lay1[r]
    same_l = l0["joints"] == l1["joints"] and np.allclose(l0["lower"], l1["lower"]) and np.allclose(l0["upper"], l1["upper"])
    print(f"{r}: graph {'identical' if same else 'DIFFERS'} ({len(b['node_names'])} nodes), qpos layout {'identical' if same_l else 'DIFFERS'} ({len(l1['joints'])} dof)")
PY
fi

[ -f demo_data_processed_prim_q/build_stats.json ] || { log "append joint trajectories -> demo_data_processed_prim_q"; python data/process/append_qpos.py --h5 "$RAW" --robots $ROBOTS; }

if [ ! -f demo_data/frame_cache_full/PREXTRACT_DONE ]; then
  log "frame cache -> demo_data/frame_cache_full (webp)"
  python data/process/prextract_frames.py --dataset_root demo_data_processed --out_root demo_data/frame_cache_full \
    --robots $ROBOTS --format webp --workers "${CACHE_WORKERS:-12}" && touch demo_data/frame_cache_full/PREXTRACT_DONE
fi

log "summary (reference: 300 / 299 / 297 episodes, prim_q action width 30 / 30 / 31 with qpos at 18:30 / 18:30 / 19:31)"
python - $ROBOTS <<'PY'
import json, sys, os
st = json.load(open("demo_data_processed_prim_q/build_stats.json"))
for r in sys.argv[1:]:
    n = {k: sum(1 for _ in open(f"{d}/{r}/meta/episodes.jsonl")) for k, d in
         [("lerobot", "demo_data/lerobot"), ("processed", "demo_data_processed"), ("prim", "demo_data_processed_prim"), ("prim_q", "demo_data_processed_prim_q")]}
    mod = json.load(open(f"demo_data_processed_prim_q/{r}/meta/modality.json"))["action"]
    width = max(v["end"] for v in mod.values())
    print(f"{r}: episodes {n}, prim_q action width {width} (qpos {mod['qpos']['start']}:{mod['qpos']['end']}), "
          f"max |state - tcp| {st['robots'][r]['max_state_err']:.1e}, cache {'OK' if os.path.exists(f'demo_data/frame_cache_full/{r}/frames_wrist_image.json') else 'MISSING'}")
rep = json.load(open("demo_data_processed_prim/PRIMITIVES_REPORT.json"))["_all"]
print("derived slot mask == prior:", rep["agrees_with_prior"])
PY
log "DATA READY"
