"""
ManiSkill -> GR00T-LeRobot converter.

Converts replayed ManiSkill demonstration trajectories (.h5 + .json, as produced by
`mani_skill/trajectory/replay_trajectory.py -o rgb -c pd_ee_delta_pose --save_traj`)
into the GR00T-compatible LeRobot v2.1 layout consumed by `data.dataset.LeRobotSingleDataset`.

All tasks belonging to one robot are merged into a SINGLE dataset directory, because
`config/dataset/dataset.yaml` takes one `dataset_path` per embodiment and
`modifiy_tasks_num` is keyed by task description within that dataset.

Output layout (per robot):

    <out_root>/<robot>/
      meta/info.json
      meta/modality.json
      meta/tasks.jsonl
      meta/episodes.jsonl
      meta/episodes_stats.jsonl
      data/chunk-000/episode_%06d.parquet
      videos/chunk-000/observation.images.image/episode_%06d.mp4
      videos/chunk-000/observation.images.wrist_image/episode_%06d.mp4

`observation.state` is written as the 8-dim end-effector pose expected by
`trajectory_canonicalization.py --in_rotation_type quat`:

    [x, y, z, qw, qx, qy, qz, gripper]

`action` is the replayed `pd_ee_delta_pose` action (7 dims), which is what
`PANDAWithRELDataConfig` splits into action.position / action.rotation / action.gripper.

Run canonicalization afterwards:

    python data/process/trajectory_canonicalization.py \
        --dataset_path <out_root> --save_path <canonicalized_root> \
        --in_rotation_type quat --out_action_type rel --out_rotation_type quat

Usage:
    # inspect one trajectory file to discover obs keys / camera names
    python data/process/maniskill_to_lerobot.py --inspect path/to/trajectory.h5

    # convert one robot, merging all six tasks
    python data/process/maniskill_to_lerobot.py \
        --demo_root ./demos --robot panda --out_root ./raw_lerobot \
        --tasks PushCube-v1 PlaceSphere-v1 PullCube-v1 \
                LiftPegUpright-v1 PickCube-v1 StackCube-v1 \
        --max_episodes_per_task 50 --fps 20
"""

import argparse
import json
import os
import shutil
from pathlib import Path

import h5py
import imageio.v2 as imageio
import numpy as np
import pandas as pd
from tqdm import tqdm

# Natural-language instruction per ManiSkill env id. These strings become the rows of
# meta/tasks.jsonl and are the keys `modifiy_tasks_num` must match exactly.
TASK_DESCRIPTIONS = {
    "PushCube-v1": "Push the cube across the surface to the target location.",
    "PlaceSphere-v1": "Pick up the sphere and place it precisely into the target.",
    "PullCube-v1": "Pull the cube toward the target marker.",
    "LiftPegUpright-v1": "Lift the peg and stand it upright.",
    "PickCube-v1": "Pick up the cube.",
    "StackCube-v1": "Stack the cube on top of the other cube.",
}

CHUNK_SIZE = 1000
CHUNK_DIR = "chunk-000"

# Video keys, matching PANDAWithRELDataConfig.video_keys == ["video.image", "video.wrist_image"]
BASE_VIDEO_KEY = "observation.images.image"
WRIST_VIDEO_KEY = "observation.images.wrist_image"

# Camera-name heuristics, applied to the group names under obs/sensor_data.
WRIST_CAMERA_HINTS = ("hand", "wrist", "eye_in_hand", "gripper")
BASE_CAMERA_HINTS = ("base", "front", "third", "render", "external", "side", "overhead")


# --------------------------------------------------------------------------------------
# HDF5 inspection helpers
# --------------------------------------------------------------------------------------
def describe_h5(path: str, max_depth: int = 6) -> None:
    """Print the group/dataset tree of a ManiSkill trajectory file."""

    def walk(node, prefix="", depth=0):
        if depth > max_depth:
            return
        for name, item in node.items():
            if isinstance(item, h5py.Group):
                print(f"{prefix}{name}/")
                walk(item, prefix + "  ", depth + 1)
            else:
                print(f"{prefix}{name}  shape={item.shape} dtype={item.dtype}")

    with h5py.File(path, "r") as f:
        traj_keys = sorted(f.keys())
        print(f"{path}: {len(traj_keys)} trajectories")
        if not traj_keys:
            return
        first = traj_keys[0]
        print(f"--- structure of {first} ---")
        walk(f[first], prefix="  ")


def _h5_group_names(node, path: str) -> list[str]:
    """Names of the direct children of `path`, or [] if `path` is absent."""
    cur = node
    for part in path.split("/"):
        if not part:
            continue
        if part not in cur:
            return []
        cur = cur[part]
    return sorted(cur.keys()) if isinstance(cur, h5py.Group) else []


def _pick_camera(names: list[str], hints: tuple, exclude: str | None = None) -> str | None:
    """Choose a camera whose name matches one of `hints`, else fall back to any remaining one."""
    candidates = [n for n in names if n != exclude]
    for hint in hints:
        for name in candidates:
            if hint in name.lower():
                return name
    return candidates[0] if candidates else None


def detect_layout(traj: h5py.Group, args) -> dict:
    """Work out where EE pose, gripper and camera images live in a ManiSkill trajectory."""
    cameras = _h5_group_names(traj, "obs/sensor_data")

    wrist_cam = args.wrist_camera or _pick_camera(cameras, WRIST_CAMERA_HINTS)
    base_cam = args.base_camera or _pick_camera(cameras, BASE_CAMERA_HINTS, exclude=wrist_cam)
    if base_cam is None or wrist_cam is None:
        raise ValueError(
            f"Could not resolve two cameras from {cameras}. "
            f"Pass --base_camera/--wrist_camera explicitly."
        )

    extra_keys = _h5_group_names(traj, "obs/extra")
    if args.tcp_pose_key:
        tcp_key = args.tcp_pose_key
    elif "tcp_pose" in extra_keys:
        tcp_key = "obs/extra/tcp_pose"
    else:
        raise ValueError(
            f"No tcp_pose found under obs/extra (saw {extra_keys}). "
            f"Pass --tcp_pose_key, e.g. obs/extra/tcp_pose."
        )

    # Gripper: prefer an explicit width signal, else derive from the trailing qpos joints.
    gripper_key, gripper_mode = None, None
    if args.gripper_qpos_indices:
        gripper_key, gripper_mode = (args.gripper_key or "obs/agent/qpos"), "qpos_indices"
    elif args.gripper_key:
        gripper_key, gripper_mode = args.gripper_key, "direct"
    else:
        for cand in ("gripper_width", "gripper_pos", "gripper_qpos"):
            if cand in extra_keys:
                gripper_key, gripper_mode = f"obs/extra/{cand}", "direct"
                break
        if gripper_key is None:
            gripper_key, gripper_mode = "obs/agent/qpos", "qpos_tail"

    return {
        "base_camera": base_cam,
        "wrist_camera": wrist_cam,
        "tcp_pose_key": tcp_key,
        "gripper_key": gripper_key,
        "gripper_mode": gripper_mode,
        "cameras": cameras,
    }


def _read(traj: h5py.Group, path: str) -> np.ndarray:
    cur = traj
    for part in path.split("/"):
        if not part:
            continue
        cur = cur[part]
    return np.asarray(cur)


# --------------------------------------------------------------------------------------
# Per-episode extraction
# --------------------------------------------------------------------------------------
def extract_episode(traj: h5py.Group, layout: dict, args) -> dict | None:
    """Pull the aligned (T,) arrays for one trajectory, or None if it should be skipped."""
    actions = np.asarray(traj["actions"], dtype=np.float32)  # (T, A)
    n_steps = actions.shape[0]
    if n_steps < args.min_length:
        return None

    if args.only_success and "success" in traj:
        success = np.asarray(traj["success"])
        if not bool(np.asarray(success).reshape(-1)[-1]):
            return None

    # ManiSkill stores T+1 observations for T actions; keep the leading T so that
    # observation[t] is the state the policy saw before emitting action[t].
    tcp_pose = _read(traj, layout["tcp_pose_key"]).astype(np.float32)[:n_steps]
    if tcp_pose.shape[-1] != 7:
        raise ValueError(f"Expected 7-dim tcp_pose (xyz + wxyz quat), got {tcp_pose.shape}")

    raw_gripper = _read(traj, layout["gripper_key"]).astype(np.float32)[:n_steps]
    if layout["gripper_mode"] == "qpos_indices":
        # Explicit qpos columns. Required whenever the driven gripper joints are not the
        # trailing entries of qpos -- e.g. the xArm6 Robotiq appends passive mimic joints
        # (left/right_inner_finger_joint) after the driven outer knuckle joints, so a
        # trailing slice would read a follower signal instead of the commanded gripper.
        gripper = raw_gripper[:, args.gripper_qpos_indices].sum(axis=-1, keepdims=True)
    elif layout["gripper_mode"] == "qpos_tail":
        gripper = raw_gripper[:, -args.gripper_joints:].sum(axis=-1, keepdims=True)
    else:
        gripper = raw_gripper.reshape(n_steps, -1)[:, :1]

    base_rgb = _read(traj, f"obs/sensor_data/{layout['base_camera']}/rgb")[:n_steps]
    wrist_rgb = _read(traj, f"obs/sensor_data/{layout['wrist_camera']}/rgb")[:n_steps]

    return {
        "actions": actions,
        "tcp_pose": tcp_pose,
        "gripper": gripper.astype(np.float32),
        "base_rgb": np.ascontiguousarray(base_rgb, dtype=np.uint8),
        "wrist_rgb": np.ascontiguousarray(wrist_rgb, dtype=np.uint8),
        "length": n_steps,
    }


def normalize_gripper(episodes: list[dict], mode: str, lo: float, hi: float) -> None:
    """Map the raw gripper signal to [0, 1] in place, so that arms with different
    finger kinematics share a comparable gripper dimension."""
    if mode == "none":
        return
    if mode == "fixed":
        g_lo, g_hi = lo, hi
    else:  # "dataset": use the observed range across the whole robot's dataset
        all_g = np.concatenate([ep["gripper"] for ep in episodes], axis=0)
        g_lo, g_hi = float(all_g.min()), float(all_g.max())
    span = g_hi - g_lo
    if span < 1e-8:
        print(f"  [warn] gripper range is degenerate ({g_lo}, {g_hi}); leaving unnormalized")
        return
    for ep in episodes:
        ep["gripper"] = np.clip((ep["gripper"] - g_lo) / span, 0.0, 1.0).astype(np.float32)
    print(f"  gripper normalized from [{g_lo:.5f}, {g_hi:.5f}] to [0, 1]")


# --------------------------------------------------------------------------------------
# Writers
# --------------------------------------------------------------------------------------
def write_video(path: Path, frames: np.ndarray, fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        path.as_posix(),
        fps=fps,
        codec="libx264",
        pixelformat="yuv420p",
        macro_block_size=1,
        ffmpeg_log_level="error",
    )
    try:
        for frame in frames:
            writer.append_data(frame)
    finally:
        writer.close()


def build_modality_json(rot_dim: int, action_dim: int, action_rot_dim: int = 3) -> dict:
    """State/action slice map. Canonicalization later appends the rel_state_* and
    progress entries; only the original action slices belong here.

    `pd_ee_delta_pose` actions are [3 position, 3 axis-angle rotation, G gripper], where G
    is 1 for panda / xarm6_robotiq but 2 for widowxai (which drives its left and right
    carriage joints independently). The rotation slice is therefore pinned at 3 dims and
    the gripper takes the remainder -- deriving rotation as `action_dim - 1` instead would
    silently absorb widowxai's second gripper channel into the rotation.
    """
    return {
        "state": {
            "position": {"start": 0, "end": 3},
            "rotation": {"start": 3, "end": 3 + rot_dim, "rotation_type": "quaternion"},
            "gripper": {"start": 3 + rot_dim, "end": 4 + rot_dim},
        },
        "action": {
            "position": {"start": 0, "end": 3, "absolute": False},
            "rotation": {"start": 3, "end": 3 + action_rot_dim, "absolute": False},
            "gripper": {"start": 3 + action_rot_dim, "end": action_dim, "absolute": False},
        },
        "video": {
            "image": {"original_key": BASE_VIDEO_KEY},
            "wrist_image": {"original_key": WRIST_VIDEO_KEY},
        },
        "annotation": {"human.task_description": {}},
    }


def build_info_json(
    robot: str,
    n_episodes: int,
    total_frames: int,
    n_tasks: int,
    fps: int,
    height: int,
    width: int,
    state_dim: int,
    action_dim: int,
) -> dict:
    def video_feature():
        return {
            "dtype": "video",
            "shape": [height, width, 3],
            "names": ["height", "width", "channel"],
            "info": {
                "video.fps": float(fps),
                "video.height": height,
                "video.width": width,
                "video.channels": 3,
                "video.codec": "h264",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "has_audio": False,
            },
        }

    return {
        "codebase_version": "v2.1",
        "robot_type": robot,
        "total_episodes": n_episodes,
        "total_frames": total_frames,
        "total_tasks": n_tasks,
        "total_videos": n_episodes * 2,
        "total_chunks": 1,
        "chunks_size": CHUNK_SIZE,
        "fps": fps,
        "splits": {"train": f"0:{n_episodes}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "observation.state": {
                "dtype": "float32",
                "shape": [state_dim],
                "names": ["state"],
            },
            "action": {"dtype": "float32", "shape": [action_dim], "names": ["action"]},
            BASE_VIDEO_KEY: video_feature(),
            WRIST_VIDEO_KEY: video_feature(),
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
            "annotation.human.task_description": {
                "dtype": "int64",
                "shape": [1],
                "names": None,
            },
        },
    }


def episode_stats(state: np.ndarray, action: np.ndarray, length: int) -> dict:
    def stats(arr):
        return {
            "min": np.min(arr, axis=0).tolist(),
            "max": np.max(arr, axis=0).tolist(),
            "mean": np.mean(arr, axis=0).tolist(),
            "std": np.std(arr, axis=0).tolist(),
            "count": [length],
        }

    return {"observation.state": stats(state), "action": stats(action)}


# --------------------------------------------------------------------------------------
# Main conversion
# --------------------------------------------------------------------------------------
def convert_robot(args) -> Path:
    out_dir = Path(args.out_root) / args.robot
    if out_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{out_dir} exists; pass --overwrite to replace it.")
        shutil.rmtree(out_dir)
    (out_dir / "data" / CHUNK_DIR).mkdir(parents=True, exist_ok=True)
    (out_dir / "meta").mkdir(parents=True, exist_ok=True)

    # ---- 1. Load every task's trajectories -------------------------------------------
    per_task: dict[str, list[dict]] = {}
    layout = None
    for env_id in args.tasks:
        h5_path = Path(args.demo_root) / args.robot / env_id / "motionplanning" / args.traj_name
        if not h5_path.exists():
            matches = sorted(
                (Path(args.demo_root) / args.robot / env_id / "motionplanning").glob("*.h5")
            )
            if not matches:
                raise FileNotFoundError(f"No .h5 found for {args.robot}/{env_id} at {h5_path}")
            h5_path = matches[0]

        with h5py.File(h5_path, "r") as f:
            traj_names = sorted(f.keys(), key=lambda k: int(k.split("_")[-1]))
            if layout is None:
                layout = detect_layout(f[traj_names[0]], args)
                print(f"  detected layout: {json.dumps(layout, indent=2)}")
            kept = []
            for name in tqdm(traj_names, desc=f"  reading {env_id}", leave=False):
                if len(kept) >= args.max_episodes_per_task:
                    break
                ep = extract_episode(f[name], layout, args)
                if ep is not None:
                    kept.append(ep)
        if len(kept) < args.max_episodes_per_task:
            print(
                f"  [warn] {env_id}: only {len(kept)} usable episodes "
                f"(< {args.max_episodes_per_task})"
            )
        per_task[env_id] = kept
        print(f"  {env_id}: {len(kept)} episodes from {h5_path.name}")

    # ---- 2. Normalize the gripper across the whole robot ------------------------------
    all_eps = [ep for eps in per_task.values() for ep in eps]
    if not all_eps:
        raise RuntimeError(f"No usable episodes found for {args.robot}")
    normalize_gripper(all_eps, args.gripper_norm, args.gripper_lo, args.gripper_hi)

    # ---- 3. Write tasks.jsonl ---------------------------------------------------------
    task_rows, task_index_of = [], {}
    for i, env_id in enumerate(args.tasks):
        description = TASK_DESCRIPTIONS.get(env_id, env_id)
        task_index_of[env_id] = i
        task_rows.append({"task_index": i, "task": description})
    with open(out_dir / "meta" / "tasks.jsonl", "w") as f:
        for row in task_rows:
            f.write(json.dumps(row) + "\n")

    # ---- 4. Write parquet + videos ----------------------------------------------------
    episode_rows, stats_rows = [], []
    episode_index, global_index = 0, 0
    height = width = None
    state_dim = action_dim = None

    for env_id in args.tasks:
        description = TASK_DESCRIPTIONS.get(env_id, env_id)
        task_index = task_index_of[env_id]
        for ep in tqdm(per_task[env_id], desc=f"  writing {env_id}", leave=False):
            length = ep["length"]
            # observation.state = [xyz, quat_wxyz, gripper] -> 8 dims
            state = np.concatenate([ep["tcp_pose"], ep["gripper"]], axis=-1).astype(np.float32)
            action = ep["actions"].astype(np.float32)
            if state_dim is None:
                state_dim, action_dim = state.shape[-1], action.shape[-1]
                height, width = ep["base_rgb"].shape[1], ep["base_rgb"].shape[2]

            timestamps = (np.arange(length, dtype=np.float32) / args.fps).astype(np.float32)
            frame = pd.DataFrame(
                {
                    "observation.state": [state[t] for t in range(length)],
                    "action": [action[t] for t in range(length)],
                    "timestamp": timestamps,
                    "annotation.human.task_description": np.full(length, task_index, dtype=np.int64),
                    "task_index": np.full(length, task_index, dtype=np.int64),
                    "episode_index": np.full(length, episode_index, dtype=np.int64),
                    "index": np.arange(global_index, global_index + length, dtype=np.int64),
                    "next.done": np.array([t == length - 1 for t in range(length)], dtype=bool),
                }
            )
            frame.to_parquet(
                out_dir / "data" / CHUNK_DIR / f"episode_{episode_index:06d}.parquet",
                index=False,
            )

            write_video(
                out_dir / "videos" / CHUNK_DIR / BASE_VIDEO_KEY / f"episode_{episode_index:06d}.mp4",
                ep["base_rgb"],
                args.fps,
            )
            write_video(
                out_dir
                / "videos"
                / CHUNK_DIR
                / WRIST_VIDEO_KEY
                / f"episode_{episode_index:06d}.mp4",
                ep["wrist_rgb"],
                args.fps,
            )

            episode_rows.append(
                {"episode_index": episode_index, "tasks": [description], "length": length}
            )
            stats_rows.append(
                {
                    "episode_index": episode_index,
                    "stats": episode_stats(state, action, length),
                }
            )
            episode_index += 1
            global_index += length

    if episode_index > CHUNK_SIZE:
        raise ValueError(
            f"{episode_index} episodes exceeds chunk size {CHUNK_SIZE}; "
            f"trajectory_canonicalization.py only walks data/chunk-000."
        )

    # ---- 5. Write remaining metadata --------------------------------------------------
    with open(out_dir / "meta" / "episodes.jsonl", "w") as f:
        for row in episode_rows:
            f.write(json.dumps(row) + "\n")
    with open(out_dir / "meta" / "episodes_stats.jsonl", "w") as f:
        for row in stats_rows:
            f.write(json.dumps(row) + "\n")
    with open(out_dir / "meta" / "modality.json", "w") as f:
        json.dump(build_modality_json(4, action_dim), f, indent=4)
    with open(out_dir / "meta" / "info.json", "w") as f:
        json.dump(
            build_info_json(
                args.robot,
                episode_index,
                global_index,
                len(args.tasks),
                args.fps,
                height,
                width,
                state_dim,
                action_dim,
            ),
            f,
            indent=4,
        )

    print(
        f"  wrote {episode_index} episodes / {global_index} frames to {out_dir} "
        f"(state_dim={state_dim}, action_dim={action_dim}, {width}x{height} @ {args.fps}fps)"
    )
    return out_dir


def parse_args():
    p = argparse.ArgumentParser(
        description="Convert replayed ManiSkill demonstrations to the GR00T-LeRobot layout."
    )
    p.add_argument("--inspect", type=str, default=None, help="Print an .h5 tree and exit")
    p.add_argument("--demo_root", type=str, help="Root holding <robot>/<env_id>/motionplanning/")
    p.add_argument("--robot", type=str, help="Robot directory name, e.g. panda / xarm6 / widowxai")
    p.add_argument("--out_root", type=str, help="Root to write <robot>/ into")
    p.add_argument("--tasks", type=str, nargs="+", default=list(TASK_DESCRIPTIONS))
    p.add_argument("--traj_name", type=str, default="trajectory.rgb.pd_ee_delta_pose.physx_cpu.h5")
    p.add_argument("--max_episodes_per_task", type=int, default=50)
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--min_length", type=int, default=2)
    p.add_argument("--only_success", action="store_true", default=True)
    p.add_argument("--keep_failures", dest="only_success", action="store_false")
    p.add_argument("--overwrite", action="store_true")

    p.add_argument("--base_camera", type=str, default=None)
    p.add_argument("--wrist_camera", type=str, default=None)
    p.add_argument("--tcp_pose_key", type=str, default=None)
    p.add_argument("--gripper_key", type=str, default=None)
    p.add_argument("--gripper_joints", type=int, default=2,
                   help="Trailing qpos entries summed into a gripper width")
    p.add_argument("--gripper_qpos_indices", type=lambda s: [int(x) for x in s.split(",")],
                   default=None,
                   help="Explicit comma-separated qpos column indices to sum into the "
                        "gripper width, e.g. '6' for xarm6_robotiq (whose driven joints "
                        "are not the trailing qpos entries). Overrides --gripper_joints.")
    p.add_argument("--gripper_norm", choices=["dataset", "fixed", "none"], default="dataset")
    p.add_argument("--gripper_lo", type=float, default=0.0)
    p.add_argument("--gripper_hi", type=float, default=1.0)
    return p.parse_args()


def main():
    args = parse_args()
    if args.inspect:
        describe_h5(args.inspect)
        return
    for required in ("demo_root", "robot", "out_root"):
        if getattr(args, required) is None:
            raise ValueError(f"--{required} is required (or use --inspect)")
    print(f"Converting {args.robot}: {len(args.tasks)} tasks, "
          f"<= {args.max_episodes_per_task} episodes each")
    convert_robot(args)


if __name__ == "__main__":
    main()
