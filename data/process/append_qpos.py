# demo_data_processed_prim_q: the prim dataset plus the robot's joint positions per frame for the motion-aware
# kinematic graph (models.stage1.graph_temporal). Copies demo_data_processed_prim/<robot> (videos symlinked) and
# appends action.qpos = obs/agent/qpos of the raw ManiSkill h5 (raw, zero-padded to --width, column order =
# assets/kinematic_graphs/qpos_layout.json); frame t == h5 index t, checked via observation.state == obs/extra/tcp_pose.
#   python data/process/append_qpos.py --h5 <root with <robot>/<task>/motionplanning/*.h5>
import argparse
import json
import os
import shutil
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from data.process.label_primitives import fix_stats

H5_NAME = "trajectory.rgb.pd_ee_delta_pose.physx_cpu.h5"
TASK_OF = {"Push the cube across the surface to the target location.": "PushCube-v1",
           "Pick up the sphere and place it precisely into the target.": "PlaceSphere-v1",
           "Pull the cube toward the target marker.": "PullCube-v1",
           "Lift the peg and stand it upright.": "LiftPegUpright-v1",
           "Pick up the cube.": "PickCube-v1",
           "Stack the cube on top of the other cube.": "StackCube-v1"}


def build_robot(src: Path, dst: Path, h5_root: Path, robot: str, width: int, dof: int) -> dict:
    s, d = src / robot, dst / robot
    (d / "meta").mkdir(parents=True, exist_ok=True)
    for f in (s / "meta").glob("*"):
        if f.is_file():
            shutil.copy2(f, d / "meta" / f.name)
    if not (d / "videos").exists():
        os.symlink((s / "videos").resolve(), d / "videos")

    eps = [json.loads(l) for l in open(s / "meta/episodes.jsonl")]
    rank, files = Counter(), {}
    stats = {"episodes": 0, "max_state_err": 0.0, "qpos_min": None, "qpos_max": None}
    for e in eps:
        task = TASK_OF[e["tasks"][0]]
        r = rank[task]
        rank[task] += 1
        if task not in files:
            files[task] = h5py.File(h5_root / robot / task / "motionplanning" / H5_NAME, "r")
        g = files[task][f"traj_{r}"]
        pq = next((s / "data").rglob(f"episode_{e['episode_index']:06d}.parquet"))
        df = pd.read_parquet(pq)
        act = np.stack(df["action"].to_numpy()).astype(np.float32)
        st = np.stack(df["observation.state"].to_numpy()).astype(np.float64)
        T = len(act)
        q = g["obs/agent/qpos"][:].astype(np.float32)
        tcp = g["obs/extra/tcp_pose"][:].astype(np.float64)
        assert q.shape == (T + 1, dof), (robot, task, r, q.shape, T, dof)
        err = float(np.abs(st[:, :7] - tcp[:T]).max())
        assert err < 1e-5, (robot, task, r, err)
        stats["max_state_err"] = max(stats["max_state_err"], err)
        qp = np.zeros((T, width), np.float32)
        qp[:, :dof] = q[:T]
        lo, hi = q[:T].min(0), q[:T].max(0)
        stats["qpos_min"] = lo if stats["qpos_min"] is None else np.minimum(stats["qpos_min"], lo)
        stats["qpos_max"] = hi if stats["qpos_max"] is None else np.maximum(stats["qpos_max"], hi)
        df["action"] = list(np.concatenate([act, qp], axis=1))
        out = d / "data" / pq.parent.name / pq.name
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out, index=False)
        stats["episodes"] += 1
    for f in files.values():
        f.close()

    mp = d / "meta" / "modality.json"
    mod = json.load(open(mp))
    w0 = max(v["end"] for v in mod["action"].values())
    mod["action"]["qpos"] = {"start": w0, "end": w0 + width, "absolute": True}
    json.dump(mod, open(mp, "w"), indent=4)
    fix_stats(d, robot)
    stats["qpos_min"], stats["qpos_max"] = stats["qpos_min"].round(3).tolist(), stats["qpos_max"].round(3).tolist()
    print(f"{robot}: {stats['episodes']} episodes, action width {w0} -> {w0 + width}, dof {dof}, "
          f"max |state - tcp| {stats['max_state_err']:.1e}")
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="demo_data_processed_prim")
    ap.add_argument("--dst", default="demo_data_processed_prim_q")
    ap.add_argument("--h5", required=True, help="root with <robot>/<task>/motionplanning/" + H5_NAME)
    ap.add_argument("--layout", default="assets/kinematic_graphs/qpos_layout.json")
    ap.add_argument("--robots", nargs="+", default=["panda", "xarm6", "widowxai"])
    ap.add_argument("--width", type=int, default=12, help="padded qpos columns (>= every robot's dof)")
    args = ap.parse_args()
    layout = json.load(open(args.layout))
    src, dst = Path(args.src), Path(args.dst)
    all_stats = {}
    for r in args.robots:
        dof = len(layout[r]["joints"])
        assert dof <= args.width, f"{r}: {dof} dof > width {args.width}"
        all_stats[r] = build_robot(src, dst, Path(args.h5), r, args.width, dof)
    json.dump({"width": args.width, "layout": args.layout, "h5": str(args.h5), "robots": all_stats},
              open(dst / "build_stats.json", "w"), indent=2)


if __name__ == "__main__":
    main()
