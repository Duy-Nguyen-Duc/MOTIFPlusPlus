#!/bin/bash
# Verifies the installation for the temporal-graph experiment (run inside the `motif` env, from the repository root).
#   bash repro/temporal_graph/check_setup.sh
set -u
ROOT=$(cd "$(dirname "$0")/../.." && pwd); cd "$ROOT"
MANISKILL_ROOT=${MANISKILL_ROOT:-$ROOT/../ManiSkill}
PATCH=$ROOT/repro/temporal_graph/maniskill-renderer.patch
fail=0
ok() { echo "  OK   $*"; }
bad() { echo "  FAIL $*"; fail=1; }
export PYTHONPATH=$ROOT PYTHONNOUSERSITE=1

echo "python / torch"
TORCH=${TORCH:-2.6.0}   # same override as setup_env.sh (reference build 2.6.0+cu124)
TORCH=$TORCH python - <<'PY' && ok "torch $TORCH import + CUDA matmul" || bad "torch import / CUDA"
import os, sys, torch
assert sys.version_info[:2] == (3, 10), sys.version
assert torch.__version__.startswith(os.environ["TORCH"]), torch.__version__
print("   ", torch.__version__, "cuda available:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
assert torch.cuda.is_available()
x = torch.randn(256, 256, device="cuda"); assert torch.allclose((x @ x.T).cpu(), x.cpu() @ x.cpu().T, atol=1e-2)   # kernels exist for this GPU
if not torch.__version__.startswith("2.6.0"): print("    WARN not the reference torch build (2.6.0+cu124); results differ within the per-seed spread")
PY
python - <<'PY' && ok "repository imports + CPU forward of the temporal-graph stage 1" || bad "repository imports"
import torch, model.stage1_graph as s1, model.stage3_elora, model.geometry_tokens, data.data_config as dc
m = s1.build_model(dict(class_name="MOTIFStage1Graph", text_encoder="table", graph_temporal=True)).eval()
b = dict(rel_state=torch.randn(2, 32, 8), progress=torch.rand(2, 32), embodiment_id=torch.tensor([1, 0]),
         language=["Pick up the cube.", "Push the cube across the surface to the target location."], qpos=torch.randn(2, 32, 12))
with torch.no_grad(): out = m(b)
assert tuple(out["recon"].shape) == (2, 32, 8) and "panda_rel_prim_q" in dc.DATA_CONFIG_MAP
PY

echo "ManiSkill fork"
[ -d "$MANISKILL_ROOT" ] && ok "checkout at $MANISKILL_ROOT" || bad "no checkout at $MANISKILL_ROOT (set MANISKILL_ROOT)"
c=$(git -C "$MANISKILL_ROOT" rev-parse HEAD 2> /dev/null); [ "$c" = 52e1eb21dbe1fee29d18df223dc913d06b043481 ] && ok "commit $c" || bad "commit $c != 52e1eb21"
git -C "$MANISKILL_ROOT" apply --reverse --check "$PATCH" 2> /dev/null && ok "renderer patch applied" || bad "renderer patch not applied"
python -c "import mani_skill, os; p = os.path.realpath(mani_skill.__file__); assert p.startswith(os.path.realpath('$MANISKILL_ROOT')), p" 2> /dev/null && ok "mani_skill imports from the fork" || bad "mani_skill imports from elsewhere"
python -c "import eval_maniskill.eval_motif_maniskill" 2> /dev/null && ok "benchmark worker imports (pynvml, profiler)" || bad "eval_maniskill.eval_motif_maniskill does not import (pip install pynvml?)"
for r in panda_wristcam xarm6_robotiq_wristcam widowxai_wristcam; do
  python -c "import mani_skill.agents.robots; from mani_skill.agents.registration import REGISTERED_AGENTS as R; assert '$r' in R, list(R)[:5]" 2> /dev/null && ok "robot $r registered" || bad "robot $r missing"
done
A=${MS_ASSET_DIR:-$HOME/.maniskill}
[ -f "$A/data/robots/widowxai/wxai_base.urdf" ] && ok "widowxai asset" || bad "widowxai asset missing: python -m mani_skill.utils.download_asset -y widowxai"
[ -f "$A/data/robots/widowxai/meshes/base_link.stl.convex.stl" ] && ok "widowxai convex hulls" || bad "widowxai convex hulls missing: python data/process/make_convex_hulls.py --urdf $A/data/robots/widowxai/wxai_follower.urdf"
[ -d "$A/data/robots/xarm6" ] && ok "xarm6 asset" || bad "xarm6 asset missing: python -m mani_skill.utils.download_asset -y xarm6"

echo "repository assets"
for f in assets/kinematic_graphs/panda.json assets/kinematic_graphs/xarm6.json assets/kinematic_graphs/widowxai.json assets/kinematic_graphs/qpos_layout.json assets/wrist_camera/wrist_cameras.json config/dataset/dataset_maniskill_prim_q.yaml eval_maniskill/dataset_spec.json; do
  [ -f "$f" ] && ok "$f" || bad "$f missing"
done

echo "pretrained caches"
python - <<'PY' && ok "t5-base, Depth-Anything-V2-Metric-Indoor-Small, dinov2_vitb14_reg" || bad "pretrained caches (run setup_env.sh with internet or copy the caches, see README)"
import os; os.environ.setdefault("HF_HUB_OFFLINE", "1")
from transformers import T5EncoderModel, T5Tokenizer, AutoModelForDepthEstimation
import torch
T5EncoderModel.from_pretrained("t5-base"); T5Tokenizer.from_pretrained("t5-base")
AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf")
torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14_reg", skip_validation=True)
PY

echo "software Vulkan renderer (lavapipe)"
VK=${VK_FALLBACK_ROOT:-}
if [ -n "$VK" ]; then
  export VK_ICD_FILENAMES=$VK/lvp_icd.json LD_LIBRARY_PATH=$VK/usr/lib/x86_64-linux-gnu:$VK/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}
  [ -f "$VK/lvp_icd.json" ] && ok "VK_FALLBACK_ROOT=$VK" || bad "$VK/lvp_icd.json missing"
elif [ -n "${VK_ICD_FILENAMES:-}" ]; then ok "VK_ICD_FILENAMES=$VK_ICD_FILENAMES"
else echo "  WARN neither VK_FALLBACK_ROOT nor VK_ICD_FILENAMES set: rendering will use whatever Vulkan device the system has"; fi
export MS_RENDER_BACKEND=${MS_RENDER_BACKEND:-pci:0} XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-$(mktemp -d)} LP_NUM_THREADS=${LP_NUM_THREADS:-4}
python - <<'PY' 2> /dev/null && ok "PickCube-v1 / panda_wristcam renders 256x256 third-view + wrist rgb" || bad "ManiSkill rgb render (Vulkan / renderer setup)"
import gymnasium as gym, mani_skill.envs, numpy as np
env = gym.make("PickCube-v1", obs_mode="rgb", robot_uids="panda_wristcam", sim_backend="physx_cpu",   # as the benchmark builds it
               sim_config=dict(control_freq=20, sim_freq=100), sensor_configs=dict(shader_pack="default", width=256, height=256),
               control_mode="pd_ee_delta_pose")
obs, _ = env.reset(seed=2025)
cams = obs["sensor_data"]
for c in ("third_view_camera", "hand_camera"):
    assert tuple(np.asarray(cams[c]["rgb"]).shape[-3:]) == (256, 256, 3), (c, np.asarray(cams[c]["rgb"]).shape)
env.close()
PY

[ $fail = 0 ] && echo "ALL CHECKS PASSED" || { echo "SOME CHECKS FAILED"; exit 1; }
