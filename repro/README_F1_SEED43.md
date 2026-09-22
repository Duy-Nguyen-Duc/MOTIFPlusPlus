# F1 seed 43: reproduction on another server

This reproduces the **all-stage F1 seed-43 training protocol**, followed by **16-action execution between replans**, on the six transfer pairs only. It uses the existing repository code, not a separate implementation. Run commands from the repository root. Nothing has been staged or committed.

## 1. Environment

Reference: Linux x86-64, Python 3.10.16, PyTorch 2.6.0+cu124, torchvision 0.21.0+cu124. Use a CUDA-12.4-compatible NVIDIA driver and adequate shared memory for the configured data loaders (at least 1 GiB recommended).

```bash
conda create -n motif-f1 python=3.10.16 pip -y
conda activate motif-f1
export PYTHONNOUSERSITE=1
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r repro/requirements-f1.txt
git clone https://github.com/Johnathan218/ManiSkill.git ../ManiSkill
git -C ../ManiSkill checkout 52e1eb21dbe1fee29d18df223dc913d06b043481
git -C ../ManiSkill apply "$PWD/repro/maniskill-renderer.patch"
python -m pip install --no-deps -e ../ManiSkill
git clone --branch v0.7.6 --depth 1 https://github.com/facebookresearch/pytorch3d.git ../pytorch3d
export PYTHONPATH="$PWD:$PWD/src:$PWD/../ManiSkill:$PWD/../pytorch3d"
python -c 'import torch, mani_skill; from pytorch3d.transforms import quaternion_to_matrix; from data.schema import DatasetMetadata; print(torch.__version__)'
```

Only PyTorch3D's pure-Python transforms are needed; the launcher accepts the source checkout without compiling its CUDA extension. Pretrained DINOv2 and T5 models are downloaded by the existing model code; allow access on first use or transfer their Torch/Hugging Face caches. Preserve those caches if identical pretrained files are required. Optional FlashAttention is not required by this F1 path.

Generation/replay/evaluation used CPU PhysX and a software Vulkan renderer. Install your host's Vulkan loader and Mesa lavapipe package, then select its actual ICD path, for example:

```bash
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json
export MS_RENDER_BACKEND=pci:0
export LP_NUM_THREADS=8
export XDG_RUNTIME_DIR="$(mktemp -d)"
```

Alternatively, copy the old `vkfallback/root` directory and pass `--renderer-root /absolute/path/to/root`; the launcher adjusts its ICD path. Different renderer/library versions can change pixels. These are protocol-reproduction instructions, not a promise of bitwise-identical training or scores.

## 2. Generate, extract, transform, cache

Choose a **new** data directory. The launcher prints commands by default; add `--execute` to run. It refuses existing output targets rather than overwriting them.

```bash
F1_DATA=/absolute/path/f1-data
F1_RUN=/absolute/path/f1-seed43
F1_ARGS=(--data-root "$F1_DATA" --run "$F1_RUN" --maniskill "$PWD/../ManiSkill" --pytorch3d "$PWD/../pytorch3d" --gpu 0)
python repro/f1_seed43.py generate "${F1_ARGS[@]}" --execute
python repro/f1_seed43.py replay "${F1_ARGS[@]}" --execute
python repro/f1_seed43.py convert "${F1_ARGS[@]}" --execute
python repro/f1_seed43.py transform "${F1_ARGS[@]}" --execute
python repro/f1_seed43.py cache "${F1_ARGS[@]}" --execute
```

| Phase | Existing script / operation | Output |
|---|---|---|
| Generate | ManiSkill `mani_skill/examples/motionplanning/{panda,xarm6,widowxai}/run.py`; six tasks per robot, 50 successful demos, 10 workers | `demo_data/maniskill_h5/<robot>/<task>/motionplanning/trajectory.h5` and JSON |
| Replay | ManiSkill `mani_skill/trajectory/replay_trajectory.py`; RGB, `pd_ee_delta_pose`, `physx_cpu`, 10 environments | `trajectory.rgb.pd_ee_delta_pose.physx_cpu.h5` and JSON |
| Convert | `data/process/maniskill_to_lerobot.py` | `demo_data/lerobot/<robot>/{data,meta,videos}` |
| Transform | `data/process/trajectory_canonicalization.py` | `demo_data_processed/<robot>/{data,meta,videos}` |
| Cache | `data/process/prextract_frames.py`, Decord, K=5 episode mask | `demo_data/frame_cache/<robot>/` and `masked_episodes_K5.json` |

**Seed distinction:** 43 is the training seed, not a demo-generation flag. With this fork, 10 workers each generate five successes, starting at seeds 0,5,10,…,45 and incrementing after attempts. Failed plans are discarded. Task order is PushCube, PlaceSphere, PullCube, LiftPegUpright, PickCube, StackCube. Generation settings were recovered from logs, fork code and trajectory metadata.

Conversion uses 20 FPS, `third_view_camera` and `hand_camera`, native 256×256 RGB. Gripper state sums qpos indices Panda `[7,8]`, xArm6 `[6]`, WidowXAI `[6,7]`, then applies dataset-level normalization. Preserve the historical **legacy gripper convention**, including xArm polarity. Canonicalization uses quaternion input/output and relative trajectories: displacement from the initial position, `R_t R_0^T`, gripper change from its initial value, and progress `(t+1)/T`.

Historical replay retained **896 episodes**: Panda 300, xArm6 299 (LiftPeg 49), WidowXAI 297 (PlaceSphere 48, PullCube 49). All other pairs have 50. Planning/replay acceptance may vary on a different host; compare these counts. To reuse exactly the old demonstrations, transfer `demo_data/maniskill_h5` and start at convert, or transfer `demo_data_processed` and start at cache. Matching counts alone does not prove identical trajectories.

K=5 selects the **first five episodes in stored order** for each transfer pair; other tasks retain all episodes. Historical selected counts: **210/210/207 = 627**. Selection is not resampled with seed 43. Cache stores decoded native-resolution uint8 frames; crop, resize and augmentation happen in the training transforms.

## 3. Input-ready data and training

No additional dataset export is needed. `src/train_utils.py::build_dataset` reads processed Parquet/metadata through `data/dataset.py` and `data/frame_cache_dataset.py`, combines robots with `data/multi_lerobot_wrapper.py`, then applies `data/data_config.py` transforms and `data/collator.py`. Normalization statistics use the selected episodes. Stage 1 drops video; stages 2/3 read cached frames.

| Stage | Epochs | Batch × accumulation | Window | Main behavior |
|---|---:|---:|---:|---|
| 1 | 20 | 8 × 16 | 32 | Factorized T/R/G encoders; shared VQ; mixed reconstruction; task-balanced sampling |
| 2 | 30 | 8 × 16 | 32 | Images/text → teacher motifs; frozen Stage 1 in eval mode; task-balanced sampling |
| 3 | 60 | 16 × 4 | 16 | Frozen Stage 2; slot/time motif identity; flow-matching actions; no task balancing |

All stages use seed **43**, K=5, AdamW settings from the checked-in configs, and the original worker counts 16/16/10. Later JEPA, slot-alignment and gripper-convention changes remain disabled. Stage 2 does not add state conditioning. Stage 3 `skip_epochs=60` controls checkpoint saving; it does not skip training. Input widths: relative trajectory 8; Stage 3 actions padded to 10; images `[B,2,1,3,224,224]`.

```bash
python repro/f1_seed43.py train "${F1_ARGS[@]}"       # inspect commands
python repro/f1_seed43.py train "${F1_ARGS[@]}" --execute
```

The launcher copies `config/train_stage{1,2,3}_f1_seed43.yaml` into `$F1_RUN/configs`, relocates data/output/checkpoint paths, and runs all three stages sequentially. Fresh run directories only; interrupted runs require explicit inspection/resume, not blindly repeating this command. It does not change batch size or accumulation. Checkpoints are `outputs_stage1/epoch_0020.pth`, `outputs_stage2/epoch_0030.pth`, and `outputs_stage3/epoch_0060.pth` under `$F1_RUN`.

## 4. Evaluation: transfer pairs only, execute 16 actions

```bash
python repro/f1_seed43.py evaluate "${F1_ARGS[@]}" --execute
```

Pairs: Panda PushCube/PlaceSphere; xArm6 PullCube/LiftPegUpright; WidowXAI PickCube/StackCube. Each gets **50 episodes**, environment seeds **2025–2074**, policy seed **7**, and independent SHA-256-derived request seeds. The controller predicts 16 actions with **4 flow integration steps**, and executes **16 actions before replanning**, unless the episode ends. CPU simulation: control 20 Hz, physics 100 Hz; RGB 256×256. Episode budgets: 500 steps for Push/Pull/Place/Pick, 700 LiftPeg, 800 Stack. Success uses the simulator's explicit success signal.

Results: `$F1_RUN/evaluation16/*.json`, episode journals, and `$F1_RUN/SUMMARY16.json`. The launcher checks the run config before evaluation and records checkpoint identity. It evaluates sequentially to limit GPU usage.

**Do not use `analysis/eval_all_16` for this rerun:** its historical F1 mapping points to the seed-42 checkpoint. The actual historical all-stage seed-43 run is `runs/f1_seed43_full`; its recorded **144/300 = 48%** used **5-action execution**, not this requested 16-action protocol. That score is not a 16-action reproduction target.

## 5. Files to include in your commit

The following is the scoped candidate list, not an instruction to commit every dirty/untracked file. Existing edits predate this preparation; review their full diffs before committing. No data, checkpoints, caches, run folders or unrelated experiment directories belong in this commit.

| File | Purpose |
|---|---|
| `repro/README_F1_SEED43.md` | This guide and commit checklist |
| `repro/f1_seed43.py` | Portable generation → training → evaluation launcher; dry-run default |
| `repro/requirements-f1.txt` | Direct dependency pins for the existing F1 path |
| `repro/maniskill-renderer.patch` | Reapply the historical renderer-selection change to the pinned fork |
| `config/train_stage1_f1_seed43.yaml` | Stage 1 seed-43 settings |
| `config/train_stage2_f1_seed43.yaml` | Stage 2 seed-43 settings and teacher eval mode |
| `config/train_stage3_f1_seed43.yaml` | Stage 3 seed-43 settings and slot/time identity |
| `src/train_stage_1_factorized.py` | Factorized Stage 1 training entry point |
| `src/train_stage_2.py` | Stage 2 training, teacher handling and runtime fixes |
| `src/train_stage_3.py` | Stage 3 training and checkpoint handling |
| `model/stage1_factorized.py` | F1 T/R/G encoders and shared dictionary |
| `model/stage1.py` | Shared teacher/model support used by model imports |
| `model/stage2_3.py` | Stage 2 predictor and Stage 3 assembly/frozen model loading |
| `model/gripper_conventions.py` | Required import; F1 uses its legacy identity behavior |
| `model/slot_alignment.py` | Required Stage 1 import; its optional loss is disabled for F1 |
| `model/modules/blocks.py` | Attention implementation and masks used by F1 |
| `model/modules/flow_matching_action_head.py` | Slot/time identity and action flow head |
| `model/transforms.py` | Model input transforms and padding |
| `data/dataset.py` | Episode selection/statistics and trajectory-cache behavior |
| `data/data_config.py` | Embodiment-specific modalities and transforms |
| `data/collator.py` | Batch assembly |
| `data/transform/concat.py` | Modality concatenation and masks |
| `experiment/inference_wrapper.py` | Checkpoint loading and policy I/O |
| `eval_maniskill/eval_motif_maniskill.py` | Robot/action mappings and task protocol helpers |
| `analysis/h1_motif_identity_gpu2/evaluate.py` | Transfer-pair evaluator; explicit `--replan-steps 16` (default remains 5) |
| `.gitignore` | Existing exclusions for generated data and run artifacts; review separately |

Already tracked and unchanged, but required in the clone: the three `data/process/` scripts listed above; `src/train_utils.py`; `data/frame_cache_dataset.py`; `data/multi_lerobot_wrapper.py`; remaining imported model/data/util modules; `experiment/eval/service.py`; and `eval_maniskill/dataset_spec.json`. No additional commit entry is needed for unchanged files. The three seed-43 configs are self-contained: unrelated model/JEPA configs are not needed.

## Checks and limits

Compared the three stage settings against saved `runs/f1_seed43_full` configs. Relocation also fixes two inactive, stale seed-42 checkpoint fields; active seed-43 teacher/predictor paths remain correct. Reviewed later options to keep F1 behavior unchanged. Existing data checks verified 896 processed trajectories, the 627-episode mask, and sampled cache pixels against decoded video.

The helper and evaluator are syntax-checked; all seven command plans and config relocation are checked without training. Full clean-environment installation, GPU retraining and 300 simulator rollouts on the new server have **not** been run. No existing training job was changed. The larger superseded reproduction draft was moved out of this repository to `/tmp/f1-repro-audit.1ZKqqn/f1_seed43`, recoverable on this host; it is not part of the proposed commit.
