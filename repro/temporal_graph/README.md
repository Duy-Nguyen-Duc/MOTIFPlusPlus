# Motion-aware kinematic graph: reproduction on another machine

Reproduces the **temporal (motion-aware) kinematic graph** experiment end to end: ManiSkill demos -> LeRobot ->
canonical rel-state actions -> primitive labels -> joint trajectories -> stages 1-3 -> Transfer and Global benchmarks,
plain and with perception steering. Everything runs through the existing repository scripts; the files in this folder
only pin the environment and chain the commands. Run everything from the repository root.

What the experiment is: `models.stage1.graph_temporal=true` makes every node of the kinematic graph also encode its
joint's trajectory over the 32-step window (limit-normalised position + per-step change, `model/stage1_graph.py`
`TaskConditionedMPNN.temporal_features`) and every edge the trajectories of both joints, so the motif router sees the
robot *moving*, not only its morphology. Stage 3 reuses the static graph (no joint positions at test time). Reference
result (seed 42, K=5, one A100-80GB, 2026-11-05): Transfer 218/300 plain, 246/300 steered; Global 719/900 plain,
706/900 steered (static graph: 184/249 and 684/710).

| | |
|---|---|
| Python / CUDA | 3.10.16, torch 2.6.0+cu124, driver >= 550 |
| GPU | one NVIDIA A100-SXM4-80GB; ~15 GB used (batch 16 in stage 3). A 24 GB card fits. |
| CPU | eval runs 3 simulator workers with 8 lavapipe threads each: >= 24 cores recommended |
| Disk | raw replayed demos 12 GB, frame cache 11 GB, LeRobot + processed + prim + prim_q 0.6 GB, one run folder ~3 GB |
| Time (K=5) | stage 1 6.5 h, stage 2 3 h, stage 3 15-17 h, Transfer eval 1.4 h + 1.1 h, Global eval 2.2 h + 2.2 h. Demo generation + replay: a few hours on 10 CPU workers (not re-measured). |

## 1. Install

```bash
git clone https://github.com/Duy-Nguyen-Duc/MOTIFPlusPlus.git && cd MOTIFPlusPlus
bash repro/temporal_graph/setup_env.sh        # conda env `motif`, pinned wheels, ManiSkill fork + renderer patch, assets, pretrained caches
bash repro/temporal_graph/check_setup.sh      # imports, fork commit, patch, assets, caches, GPU, Vulkan
```

`setup_env.sh` does, idempotently:

1. `conda create -n motif python=3.10.16`, `torch==2.6.0 torchvision==0.21.0` from the cu124 index,
   `pip install -r repro/temporal_graph/requirements.txt` (pins taken from the environment that produced the results).
2. ManiSkill fork `https://github.com/Johnathan218/ManiSkill.git` at commit `52e1eb21dbe1fee29d18df223dc913d06b043481`
   (3.0.0b21 + the `*_wristcam` robots and the motion-planning solutions for xArm6 / WidowX AI), cloned to
   `../ManiSkill` (override with `MANISKILL_ROOT`), `repro/temporal_graph/maniskill-renderer.patch` applied
   (env var `MS_RENDER_BACKEND` selects the renderer device, needed for software rendering), `pip install --no-deps -e`.
3. Robot assets `python -m mani_skill.utils.download_asset -y xarm6` and `... -y widowxai` into `~/.maniskill` (`MS_ASSET_DIR`).
4. Pretrained weights into the Hugging Face / torch hub caches: `t5-base` (text encoder of stages 1-3, 850 MB),
   `depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf` (geometry tokens, 95 MB), `facebookresearch/dinov2`
   `dinov2_vitb14_reg` (stage-2/3 image encoder, 331 MB). Offline machine: copy `~/.cache/huggingface/hub/models--t5-base`,
   `models--depth-anything--Depth-Anything-V2-Metric-Indoor-Small-hf`, `~/.cache/torch/hub/facebookresearch_dinov2_main`
   and `~/.cache/torch/hub/checkpoints/dinov2_vitb14_reg4_pretrain.pth` from this machine and set `HF_HUB_OFFLINE=1`.

Software Vulkan renderer (replay and evaluation render on the CPU with Mesa lavapipe, as the reference did):

* with root: `apt install libvulkan1 mesa-vulkan-drivers`, then `export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json`
  and leave `VK_FALLBACK_ROOT` empty;
* without root: `bash repro/temporal_graph/vulkan_lavapipe_norooot.sh <dir>` downloads the Ubuntu 22.04 packages with
  `apt-get download`, unpacks them with `dpkg -x` and writes `<dir>/root/lvp_icd.json`; then `export VK_FALLBACK_ROOT=<dir>/root`.
  (`eval/run_percept_steering.py` sets `VK_ICD_FILENAMES`, `LD_LIBRARY_PATH`, `MS_RENDER_BACKEND=pci:0` for its workers from that
  variable; `prepare_data.sh` does the same for the replay step.)

A GPU Vulkan driver also works for replay and evaluation (`MS_RENDER_BACKEND` unset) but renders slightly different pixels.

## 2. Data

```bash
export MANISKILL_ROOT=$PWD/../ManiSkill            # the patched fork
bash repro/temporal_graph/prepare_data.sh          # every step skips what already exists
```

| Step | Script | Output | Check |
|---|---|---|---|
| generate | fork `mani_skill/examples/motionplanning/{panda,xarm6,widowxai}/run.py --env-id T --num-traj 50 --num-procs 10 --only-count-success --traj-name trajectory --obs-mode none --sim-backend cpu --shader default` | `demo_data/maniskill_h5/<robot>/<task>/motionplanning/trajectory.h5` | 50 successful plans per pair (planner failures are discarded, so counts can differ by host) |
| replay | fork `mani_skill/trajectory/replay_trajectory.py -o rgb -c pd_ee_delta_pose --save-traj --num-envs 10 -b physx_cpu` | `trajectory.rgb.pd_ee_delta_pose.physx_cpu.h5` (256x256 `third_view_camera` + `hand_camera`, `obs/agent/qpos`, `obs/extra/tcp_pose`, `env_states/actors`) | reference kept 300 / 299 / 297 episodes (xArm6 LiftPeg 49, WidowX PlaceSphere 48 and PullCube 49) |
| convert | `data/process/maniskill_to_lerobot.py --fps 20 --base_camera third_view_camera --wrist_camera hand_camera --gripper_qpos_indices 7,8 / 6 / 6,7 --gripper_norm dataset` | `demo_data/lerobot/<robot>` | `observation.state` = tcp pose (p, q_wxyz) + gripper, `action` = the 7-d `pd_ee_delta_pose` command |
| canonicalize | `data/process/trajectory_canonicalization.py --in_rotation_type quat --out_action_type rel --out_rotation_type quat` | `demo_data_processed/<robot>` | rel-state actions in the first-frame frame + progress |
| segment + label | `data/process/segment_primitives.py --h5 demo_data/maniskill_h5` then `data/process/label_primitives.py` | `demo_data_processed_prim/{segments.json,<robot>}` | `PRIMITIVES_REPORT.json`: derived slot mask == prior |
| graphs | committed `assets/kinematic_graphs/{panda,xarm6,widowxai}.json` + `qpos_layout.json`; `REBUILD_GRAPHS=1` rebuilds them from the fork's URDFs and diffs | 15 / 21 / 12 nodes, 9 / 12 / 8 dof | `python eval/check_graph_nodes.py` samples nodes |
| joint trajectories | `data/process/append_qpos.py --h5 demo_data/maniskill_h5` | `demo_data_processed_prim_q/<robot>` (`action.qpos`, 12 columns, raw radians / metres) | `build_stats.json`: `max_state_err` 0, every column inside its limits up to ~0.05 rad |
| frame cache | `data/process/prextract_frames.py --format webp --workers 12` | `demo_data/frame_cache_full/<robot>` + `PREXTRACT_DONE` | 11 GB |

Shortcuts: copy this machine's `demo_data/maniskill_h5` (12 GB) and start at *convert* (identical demos), or copy
`demo_data_processed` + `demo_data_processed_prim` (0.3 GB) and start at *joint trajectories* (`append_qpos.py` still
needs the raw h5 for `obs/agent/qpos`, so `maniskill_h5` is required either way; the lean `--no-images` layout is not
available for ManiSkill). Regenerating the demos gives a different, statistically equivalent set: motion-planning
solutions are seeded per worker (seeds 0, 5, 10, ... per process) but planner acceptance depends on the host.

## 3. Train and evaluate

```bash
K=5 GPU=0 bash repro/temporal_graph/run_experiment.sh          # one K: stages 1-3, Transfer plain + steered, Global plain + steered
KS="1 3 5 10 50" GPU=0 bash repro/temporal_graph/run_experiment.sh   # the K sweep, sequential on one GPU
```

The runner copies `src/train_maniskill_kN_geod_chain.sh` into the run folder and launches it with the temporal-graph
settings, exactly as `src/queue_tgraph_ksweep.sh` does here:

```
OUT=runs/outputs_maniskill_k<K>_abl_tgraph NAME=abl_tgraph_k<K> DS=dataset_maniskill_prim_q
S1_EXTRA=models.stage1.graph_temporal=true S3_EXTRA=models.stage1.graph_temporal=true
K=<K> GPU=0 GLOBAL=1 CKPT_EVERY=60 MAX_TRIES=24 WAIT=900 bash runs/outputs_maniskill_k<K>_abl_tgraph/chain_tgraph.sh
```

| Stage | Entry point | Settings (from the chain) | Artifact |
|---|---|---|---|
| 0 frame cache | `data/process/prextract_frames.py` | webp, 12 workers | `demo_data/frame_cache_full/PREXTRACT_DONE` |
| 1 graph-routed motif tokenizer | `src/train_stage_1_graph.py --config-name train_stage1_graph` | `dataset=dataset_maniskill_prim_q K=<K> seed=42 train.epochs=20 train.grad_accum_steps=16 batch_size=8 dataset.drop_video=true models.stage1.hard_training=true models.stage1.graph_temporal=true`, window 32 | `outputs_stage1/epoch_0020.pth` |
| 2 motif predictor | `src/train_stage_2.py` (`config/models/model.yaml`) | `train.epochs=30 train.grad_accum_steps=16 batch_size=8`, frozen stage 1 | `outputs_stage2/epoch_0030.pth` |
| 3 E-LoRA policy + geometry tokens | `src/train_stage_3.py models=model_graph_elora_geod dataset.data_config_suffix=_geo` | `train.epochs=60 train.grad_accum_steps=4 batch_size=16 dataset.num_workers=10 models.stage1.graph_temporal=true`, window 16, depth-mode geometry tokens | `outputs_stage3_elora_geod/epoch_0060.pth` |
| Transfer | `eval/run_percept_steering.py --steer 0` then `--steer 1.0` | 6 pairs x 50 episodes, env seeds 2025-2074, policy seed 7, 4 denoising steps, 16-action chunks, lavapipe renderer | `eval_results/abl_tgraph_k<K>/SUMMARY.json`, `..._cp_300/SUMMARY.json` |
| Global | same with `--all-pairs --reuse <transfer run>` | 18 pairs x 50 episodes, the 6 transfer pairs copied (checkpoint sha-checked) | `eval_results/abl_tgraph_k<K>_all18/`, `..._cp_900/` |

K selects the **first K episodes in stored order** of each transfer pair (`modifiy_tasks_num` in
`config/dataset/dataset_maniskill_prim_q.yaml`: Panda Push / PlaceSphere, xArm6 Pull / LiftPeg, WidowX Pick / Stack);
the other 12 pairs keep all demos. Every stage resumes from `last.pth` and is retried `MAX_TRIES` times `WAIT` seconds
apart (disk-quota or renderer incidents), so after an interruption **run the same command again**; a chain whose run
folder is locked is a no-op. `CHAIN_DONE` / `CHAIN_FAILED` mark the end. Tables: `python eval/make_k_sweep_table.py`,
`python eval/make_global_table.py`, `python eval/make_ablation_table.py` (rows already exist for `abl_tgraph_k<K>`).

Static-graph control and the other ablations use the same chain with other overrides (header of
`src/train_maniskill_kN_geod_chain.sh`): no override = static graph (`graph_hard_elora_geod_k<K>`), `graph_routing=attn`
on top of the temporal graph = joint-attentive routing, `morphology=embed` = no graph, `prim_conditioning=false ...` = no
primitive labels.

## 4. What is and is not reproduced bit-exactly

Fixed: data pipeline, K selection, seeds (training 42, env 2025-2074, policy 7, SHA-256 request seeds), epochs, batch x
accumulation, window sizes, optimiser configs (`config/optim`), eval protocol, renderer (software lavapipe), pinned
package versions and the ManiSkill fork commit. Not bit-exact: regenerated demos (planner acceptance per host), GPU
kernels (cuDNN / index_add are nondeterministic; `_deterministic_kernels` only guards the stage-3 router), renderer
library versions (lavapipe 23.2.1 here). Expect the same numbers within the per-seed spread (a few successes out of 300),
not identical rollouts.

## 5. Files

| | |
|---|---|
| `repro/temporal_graph/setup_env.sh` | environment, fork, assets, pretrained caches |
| `repro/temporal_graph/check_setup.sh` | verifies the installation |
| `repro/temporal_graph/vulkan_lavapipe_norooot.sh` | software Vulkan renderer without root |
| `repro/temporal_graph/prepare_data.sh` | generate -> replay -> convert -> canonicalize -> segment -> label -> qpos -> frame cache |
| `repro/temporal_graph/run_experiment.sh` | the chain per K, sequential sweep, result summary |
| `repro/temporal_graph/requirements.txt`, `maniskill-renderer.patch` | pins |
| `model/stage1_graph.py` | `TemporalEncoder`, `TaskConditionedMPNN(temporal_dim, qpos_layout)`, `morphology_dynamic` |
| `data/process/build_qpos_layout.py`, `append_qpos.py`, `data/transform/concat.py`, `data/data_config.py` (`*_rel_prim_q`) | joint trajectories into `batch["qpos"]` |
| `config/dataset/dataset_maniskill_prim_q.yaml`, `config/models/model_graph_elora_geod.yaml` (`graph_temporal`, `graph_temporal_dim`, `qpos_layout`) | configuration |
| `src/train_maniskill_kN_geod_chain.sh`, `src/queue_tgraph_ksweep.sh`, `eval/run_percept_steering.py` | training chain, sweep queue, benchmark |
