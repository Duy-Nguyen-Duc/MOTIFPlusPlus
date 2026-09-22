"""Run the F1 seed-43 data, training, and 16-action evaluation protocol.

By default, print commands only. Add --execute to run a phase.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
ROBOTS = ["panda", "xarm6", "widowxai"]
UIDS = ["panda_wristcam", "xarm6_robotiq_wristcam", "widowxai_wristcam"]
TASKS = ["PushCube-v1", "PlaceSphere-v1", "PullCube-v1", "LiftPegUpright-v1",
         "PickCube-v1", "StackCube-v1"]
PAIRS = [(robot, task) for i, robot in enumerate(UIDS) for task in TASKS[2*i:2*i+2]]
OLD_ROOT = "/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/cross_emb_2/NewTest/MOTIF"


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8*1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def configuration(stage, data, run):
    cfg = json.loads((ROOT / f"config/train_stage{stage}_f1_seed43.yaml").read_text())

    def relocate(value):
        if isinstance(value, dict):
            return {k: relocate(v) for k, v in value.items()}
        if isinstance(value, list):
            return [relocate(v) for v in value]
        if isinstance(value, str) and value.startswith(OLD_ROOT + "/"):
            suffix = value[len(OLD_ROOT)+1:]
            for prefix in ["runs/f1_seed43_full/", "runs/outputs_maniskill_k5_fact/"]:
                if suffix.startswith(prefix):
                    return str(run / suffix[len(prefix):])
            if suffix.startswith(("demo_data/", "demo_data_processed/")):
                return str(data / suffix)
            raise ValueError(f"Unexpected old-server path: {value}")
        return value

    cfg = relocate(cfg)
    assert cfg["seed"] == 43 and cfg["K"] == 5
    assert cfg["train"]["epochs"] == {1:20, 2:30, 3:60}[stage]
    assert cfg["dataset"]["batch_size"] == (16 if stage == 3 else 8)
    assert cfg["train"]["grad_accum_steps"] == (4 if stage == 3 else 16)
    assert not cfg.get("jepa", {}).get("enable", False)
    assert cfg.get("lambda_slot_infonce", 0) == 0
    assert cfg["models"]["stage1"].get("gripper_convention", "legacy") == "legacy"
    assert cfg["hydra"]["run"]["dir"] == "${output_dir}"
    return cfg


def environment(args):
    search = [ROOT, ROOT / "src", args.maniskill]
    if args.pytorch3d:
        search.append(args.pytorch3d)
    env = dict(os.environ, PYTHONNOUSERSITE="1", PYTHONUNBUFFERED="1",
               PYTHONPATH=os.pathsep.join(map(str, search)), CUDA_VISIBLE_DEVICES=args.gpu)
    if args.renderer_root:
        vk = args.renderer_root
        env.update(VK_ICD_FILENAMES=str(vk / "lvp_icd.json"), MS_RENDER_BACKEND="pci:0",
                   LP_NUM_THREADS="8", LD_LIBRARY_PATH=f"{vk}/usr/lib/x86_64-linux-gnu:{vk}/lib/x86_64-linux-gnu:"
                   + os.environ.get("LD_LIBRARY_PATH", ""))
    if args.phase == "evaluate":
        env.update(OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", LP_NUM_THREADS="8")
    return env


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("phase", choices=["generate", "replay", "convert", "transform", "cache", "train", "evaluate"])
    ap.add_argument("--data-root", type=Path, default=ROOT, help="Parent of demo_data and demo_data_processed")
    ap.add_argument("--run", type=Path, default=ROOT / "runs/f1_seed43_reproduced")
    ap.add_argument("--maniskill", type=Path, default=ROOT.parent / "ManiSkill")
    ap.add_argument("--pytorch3d", type=Path, help="Optional PyTorch3D v0.7.6 source checkout")
    ap.add_argument("--renderer-root", type=Path, help="Optional copied historical vkfallback/root")
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args()
    for name in ["data_root", "run", "maniskill", "pytorch3d", "renderer_root"]:
        if getattr(args, name) is not None:
            setattr(args, name, getattr(args, name).resolve())
    env = environment(args)
    data, run = args.data_root, args.run
    raw, intermediate, processed = data / "demo_data/maniskill_h5", data / "demo_data/lerobot", data / "demo_data_processed"
    commands = []
    outputs = []
    if args.phase in ["generate", "replay"]:
        for robot in ROBOTS:
            for task in TASKS:
                folder = raw / robot / task / "motionplanning"
                if args.phase == "generate":
                    commands.append([sys.executable, str(args.maniskill / f"mani_skill/examples/motionplanning/{robot}/run.py"),
                                     "--env-id", task, "--num-traj", "50", "--num-procs", "10", "--only-count-success",
                                     "--traj-name", "trajectory", "--obs-mode", "none", "--sim-backend", "cpu",
                                     "--shader", "default", "--record-dir", str(raw / robot)])
                    outputs.append(folder / "trajectory.h5")
                else:
                    commands.append([sys.executable, str(args.maniskill / "mani_skill/trajectory/replay_trajectory.py"),
                                     "--traj-path", str(folder / "trajectory.h5"), "-o", "rgb", "-c", "pd_ee_delta_pose",
                                     "--save-traj", "--num-envs", "10", "-b", "physx_cpu"])
                    outputs.append(folder / "trajectory.rgb.pd_ee_delta_pose.physx_cpu.h5")
    elif args.phase == "convert":
        for robot, joints in zip(ROBOTS, ["7,8", "6", "6,7"]):
            commands.append([sys.executable, str(ROOT / "data/process/maniskill_to_lerobot.py"),
                             "--demo_root", str(raw), "--robot", robot, "--out_root", str(intermediate),
                             "--traj_name", "trajectory.rgb.pd_ee_delta_pose.physx_cpu.h5", "--tasks", *TASKS,
                             "--max_episodes_per_task", "50", "--fps", "20", "--base_camera", "third_view_camera",
                             "--wrist_camera", "hand_camera", "--gripper_qpos_indices", joints, "--gripper_norm", "dataset"])
            outputs.append(intermediate / robot)
    elif args.phase == "transform":
        commands.append([sys.executable, str(ROOT / "data/process/trajectory_canonicalization.py"),
                         "--dataset_path", str(intermediate), "--save_path", str(processed),
                         "--in_rotation_type", "quat", "--out_action_type", "rel", "--out_rotation_type", "quat"])
        outputs.append(processed)
    elif args.phase == "cache":
        mask_path = data / "demo_data/frame_cache/masked_episodes_K5.json"
        if args.execute:
            if mask_path.parent.exists():
                raise FileExistsError(f"Cache already exists: {mask_path.parent}")
            cfg = configuration(1, data, run)
            masks = {}
            for i, robot in enumerate(ROBOTS):
                grouped = {}
                for line in (processed / robot / "meta/episodes.jsonl").read_text().splitlines():
                    row = json.loads(line)
                    grouped.setdefault(row["tasks"][0], []).append(row["episode_index"])
                limits = cfg["dataset"]["modifiy_tasks_num"][i]
                assert set(limits) <= set(grouped)
                masks[robot] = sorted(e for task, ids in grouped.items() for e in ids[:limits.get(task, len(ids))])
            save(mask_path, masks)
        commands.append([sys.executable, str(ROOT / "data/process/prextract_frames.py"),
                         "--dataset_root", str(processed), "--out_root", str(mask_path.parent),
                         "--video_backend", "decord", "--episodes_json", str(mask_path)])
    elif args.phase == "train":
        if args.execute:
            run.mkdir(parents=True, exist_ok=False)
            for stage in [1, 2, 3]:
                save(run / f"configs/stage{stage}.yaml", configuration(stage, data, run))
        for stage in [1, 2, 3]:
            script = "train_stage_1_factorized.py" if stage == 1 else f"train_stage_{stage}.py"
            commands.append([sys.executable, str(ROOT / "src" / script), "--config-path", str(run / "configs"),
                             "--config-name", f"stage{stage}"])
    elif args.phase == "evaluate":
        checkpoint = run / "outputs_stage3/epoch_0060.pth"
        manifest = run / "evaluation16_manifest.json"
        if args.execute:
            cfg = json.loads((checkpoint.parent / "config.json").read_text())
            expected = configuration(3, data, run)
            expected.pop("hydra")  # Hydra settings are not stored in training config.json.
            assert cfg == expected, "Run config differs from F1 seed-43 settings or relocated paths"
            if (run / "evaluation16").exists():
                raise FileExistsError("Use a fresh evaluation directory; do not mix episode journals")
            save(manifest, dict(checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint),
                                code_manifest_sha256=sha(ROOT / "analysis/h1_motif_identity_gpu2/evaluate.py"),
                                seed=43, replan_steps=16))
        for robot, task in PAIRS:
            commands.append([sys.executable, str(ROOT / "analysis/h1_motif_identity_gpu2/evaluate.py"),
                             "--checkpoint", str(checkpoint), "--manifest", str(manifest), "--robot", robot,
                             "--task", task, "--out", str(run / "evaluation16"), "--episodes", "50",
                             "--seed-base", "2025", "--policy-seed", "7", "--replan-steps", "16"])
    for command in commands:
        print(shlex.join(command), flush=True)
    if not args.execute:
        return
    for target in outputs:
        if target.exists():
            raise FileExistsError(f"Refusing to overwrite existing output: {target}")
    if args.phase == "convert":
        for robot in ROBOTS:
            for task in TASKS:
                path = raw / robot / task / "motionplanning/trajectory.rgb.pd_ee_delta_pose.physx_cpu.h5"
                if not path.is_file():
                    raise FileNotFoundError(path)  # Prevent converter's fallback to another HDF5.
    if args.renderer_root:
        icd = args.renderer_root / "lvp_icd.json"
        save(icd, {"ICD": {"api_version":"1.1.204", "library_path":str(args.renderer_root / "usr/lib/x86_64-linux-gnu/libvulkan_lvp.so")},
                   "file_format_version":"1.0.0"})
    for command in commands:
        subprocess.run(command, cwd=ROOT, env=env, check=True)
    if args.phase == "transform":
        for robot in ROBOTS:
            action = json.loads((processed / robot / "meta/modality.json").read_text())["action"]
            assert action["rel_state_gripper"]["end"] - action["rel_state_position"]["start"] == 8
    if args.phase == "evaluate":
        rows = [json.loads((run / "evaluation16" / f"{r}__{t}.json").read_text()) for r,t in PAIRS]
        assert all(x["num_per_task"] == 50 and x["replan_steps"] == 16 for x in rows)
        successes = sum(x["successes"] for x in rows)
        save(run / "SUMMARY16.json", dict(successes=successes, episodes=300, percent=successes/3, pairs=rows))


if __name__ == "__main__":
    main()
