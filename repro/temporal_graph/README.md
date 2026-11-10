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

## 4. Notes from the second reproduction (RunPod, RTX PRO 6000 Blackwell, 2026-11-07)

Results of that run (seed 42, demos regenerated on this machine, torch 2.8.0+cu128, two chains sharing one GPU):

| K | Transfer | +steer | Global | +steer | wall time (data ready -> CHAIN_DONE) |
|---|---|---|---|---|---|
| 5 (reference, A100) | 218/300 | 246/300 | 719/900 | 706/900 | ~34 h |
| 10 | 236/300 | 262/300 | 725/900 | 721/900 | 49 h |
| 50 | 269/300 | 273/300 | 746/900 | 716/900 | 57.5 h |

Steering on seen pairs again costs most on Panda PullCube (50 -> 23 at K=10, 50 -> 12 at K=50) and WidowX PlaceSphere (18 -> 0, 33 -> 1).

Problems met on a fresh machine and what now handles them (all scripts stay idempotent):

| Symptom | Cause | Fix |
|---|---|---|
| torch 2.6.0+cu124 cannot run on the GPU | Blackwell (sm_120) needs cu128 kernels | `TORCH=2.8.0 TORCHVISION=0.23.0 TORCHCODEC=0.7.0 CUDA_TAG=cu128 bash repro/temporal_graph/setup_env.sh`; `TORCH=2.8.0 bash repro/temporal_graph/check_setup.sh`. Not the reference build: expect the per-seed spread, not identical numbers. |
| `import mani_skill` fails: `No module named 'IPython'` / `'dacite'` | fork installed with `--no-deps`, both were missing from the pins | `ipython==8.39.0`, `dacite==1.9.2` added to `requirements.txt` |
| WidowX AI demos: `Could not load resource .../base_link.stl.convex.stl`, then `Found errors in the Convex mesh: The mesh is not watertight` | the WidowX asset ships no convex hulls (Panda / xArm6 do); mplib 0.1.1 does not generate them, and hulls with near-duplicate vertices (< 1e-5 m) collapse in its STL loader | `data/process/make_convex_hulls.py` (hull of vertices snapped to a 1e-4 grid), run by `setup_env.sh` after the asset download, checked by `check_setup.sh` |
| replay at 25-30 s per step, load average > 500 | each of the 10 replay-pool workers (and planners) started one BLAS/OpenMP thread per core (~440 threads per worker on 224 cores) | `prepare_data.sh` exports `OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=2`, as `eval/run_percept_steering.py` already did for its workers (~15x faster) |
| benchmark workers pointed at a non-existent `lvp_icd.json` when lavapipe is installed with root | `eval/run_percept_steering.py` always derived the ICD from `VK_FALLBACK_ROOT` | with `VK_FALLBACK_ROOT` unset the workers inherit `VK_ICD_FILENAMES` (root install) or the system Vulkan device |
| sapien hangs at start when no Vulkan ICD is visible | the container's NVIDIA driver has no Vulkan ICD | always export `VK_ICD_FILENAMES` (or `VK_FALLBACK_ROOT`) for generation, replay and evaluation |
| `label_primitives.py`: `FileNotFoundError: demo_data_processed/panda/meta/stats.json` | `meta/stats.json` is written lazily by `data/dataset.py` the first time a dataset is loaded for training, so it existed on the authors' disk but not on a fresh `prepare_data.sh` run | `label_primitives.py` computes it with `calculate_dataset_statistics` when it is missing |
| every benchmark worker exits with rc=1 after ~1 min: `ModuleNotFoundError: No module named 'pynvml'` | `eval_maniskill/eval_motif_maniskill.py` imports ManiSkill's benchmarking profiler, which imports `pynvml`; the fork is installed with `--no-deps`, so nothing pulled it in | `pynvml==13.0.1` pinned in `requirements.txt` (the eval runner retries by itself after 600 s once it is installed) |
| benchmark episodes 20x slower (env step ~1.2 s instead of 0.07 s) while `top` shows the CPUs mostly idle | the container has a cgroup CPU quota (`/sys/fs/cgroup/cpu.max`, 23.8 cores here) far below the visible core count; another job with ~150 runnable threads takes the fair share and our workers are throttled (`cpu.stat` nr_throttled climbs every period) | nothing to fix in the code; size thread counts to the quota and expect eval wall time to follow the neighbour's load |

`STOP_AFTER=replay ROBOTS=<robot> bash repro/temporal_graph/prepare_data.sh` runs one robot's generate + replay, so the three
robots can run in parallel on a many-core host; the plain `prepare_data.sh` afterwards does the rest.

## 5. What is and is not reproduced bit-exactly

Fixed: data pipeline, K selection, seeds (training 42, env 2025-2074, policy 7, SHA-256 request seeds), epochs, batch x
accumulation, window sizes, optimiser configs (`config/optim`), eval protocol, renderer (software lavapipe), pinned
package versions and the ManiSkill fork commit. Not bit-exact: regenerated demos (planner acceptance per host), GPU
kernels (cuDNN / index_add are nondeterministic; `_deterministic_kernels` only guards the stage-3 router), renderer
library versions (lavapipe 23.2.1 here). Expect the same numbers within the per-seed spread (a few successes out of 300),
not identical rollouts.

## 6. Files

| | |
|---|---|
| `repro/temporal_graph/setup_env.sh` | environment, fork, assets, pretrained caches |
| `repro/temporal_graph/check_setup.sh` | verifies the installation |
| `repro/temporal_graph/vulkan_lavapipe_norooot.sh` | software Vulkan renderer without root |
| `repro/temporal_graph/prepare_data.sh` | generate -> replay -> convert -> canonicalize -> segment -> label -> qpos -> frame cache |
| `repro/temporal_graph/run_experiment.sh` | the chain per K, sequential sweep, result summary |
| `repro/temporal_graph/requirements.txt`, `maniskill-renderer.patch` | pins |
| `data/process/make_convex_hulls.py` | convex collision hulls for the WidowX AI asset (motion planning) |
| `repro/temporal_graph/upload_checkpoints.py` | uploads the final stage 1/2/3 checkpoints, their configs and the eval summaries for K=10 / K=50 to a Hugging Face model repo (edit `HF_REPO` / `HF_TOKEN` at the top) |
| `model/stage1_graph.py` | `TemporalEncoder`, `TaskConditionedMPNN(temporal_dim, qpos_layout)`, `morphology_dynamic` |
| `data/process/build_qpos_layout.py`, `append_qpos.py`, `data/transform/concat.py`, `data/data_config.py` (`*_rel_prim_q`) | joint trajectories into `batch["qpos"]` |
| `config/dataset/dataset_maniskill_prim_q.yaml`, `config/models/model_graph_elora_geod.yaml` (`graph_temporal`, `graph_temporal_dim`, `qpos_layout`) | configuration |
| `src/train_maniskill_kN_geod_chain.sh`, `src/queue_tgraph_ksweep.sh`, `eval/run_percept_steering.py` | training chain, sweep queue, benchmark |
