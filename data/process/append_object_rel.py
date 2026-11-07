# demo_data_processed_prim_obj: the prim dataset plus the object trajectory for the causal effect loss. Copies
# demo_data_processed_prim/<robot> (videos symlinked) and appends action.obj_rel (object xyz relative to the first
# frame, metres) and action.obj_valid (1 if the side-car meta/object_states.parquet had the episode).
#   python data/process/append_object_rel.py --src demo_data_processed_prim --dst demo_data_processed_prim_obj --sidecar-root <root>
import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from data.process.label_primitives import fix_stats

DEFAULT_SIDECAR = ("/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/cross_emb_2/NewTest/MOTIF/"
                   "demo_data_processed_objanchor")


def build_robot(src: Path, dst: Path, sidecar: Path, robot: str, ee_to_obj: bool = False) -> dict:
    s, d = src / robot, dst / robot
    obj = pd.read_parquet(sidecar / robot / "meta" / "object_states.parquet")
    cols = ["obj_x", "obj_y", "obj_z", "ee_to_obj_x", "ee_to_obj_y", "ee_to_obj_z"]
    lut = {int(ep): g.sort_values("frame")[cols].to_numpy(np.float32)
           for ep, g in obj.groupby("episode_index")}

    (d / "meta").mkdir(parents=True, exist_ok=True)
    for f in (s / "meta").glob("*"):
        if f.is_file():
            shutil.copy2(f, d / "meta" / f.name)
    if not (d / "videos").exists():
        os.symlink((s / "videos").resolve(), d / "videos")

    stats = {"episodes": 0, "missing": [], "length_mismatch": [], "max_disp_m": 0.0}
    for pq in sorted((s / "data").rglob("episode_*.parquet")):
        ep = int(pq.stem.split("_")[-1])
        out = d / "data" / pq.parent.name / pq.name
        out.parent.mkdir(parents=True, exist_ok=True)
        df = pd.read_parquet(pq)
        act = np.stack(df["action"].to_numpy()).astype(np.float32)
        T = len(act)
        valid = np.ones((T, 1), np.float32)
        if ep not in lut:
            stats["missing"].append(ep)
            rel, valid = np.zeros((T, 3), np.float32), np.zeros((T, 1), np.float32)
            ee = np.zeros((T, 3), np.float32)
        else:
            o = lut[ep]
            if len(o) != T:
                stats["length_mismatch"].append((ep, len(o), T))
                o = (np.concatenate([o, np.repeat(o[-1:], T - len(o), axis=0)])
                     if len(o) < T else o[:T])
            rel = o[:, :3] - o[:1, :3]
            ee = o[:, 3:6]
            stats["max_disp_m"] = max(stats["max_disp_m"], float(np.abs(rel).max()))
        df["action"] = list(np.concatenate([act, rel, valid] + ([ee] if ee_to_obj else []), axis=1))
        df.to_parquet(out, index=False)
        stats["episodes"] += 1

    mp = d / "meta" / "modality.json"
    mod = json.load(open(mp))
    width = max(v["end"] for v in mod["action"].values())
    mod["action"]["obj_rel"] = {"start": width, "end": width + 3}
    mod["action"]["obj_valid"] = {"start": width + 3, "end": width + 4, "absolute": True}
    new_width = width + 4
    if ee_to_obj:
        # gripper->object offset of every frame (metres, world axes), raw like obj_rel
        mod["action"]["ee_to_obj"] = {"start": width + 4, "end": width + 7, "absolute": True}
        new_width = width + 7
    json.dump(mod, open(mp, "w"), indent=4)
    fix_stats(d, robot)
    print(f"{robot}: {stats['episodes']} episodes, action width {width} -> {new_width}, "
          f"missing {len(stats['missing'])}, length mismatches {len(stats['length_mismatch'])}, "
          f"max |obj displacement| {stats['max_disp_m']:.3f} m")
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="demo_data_processed_prim")
    ap.add_argument("--dst", default="demo_data_processed_prim_obj")
    ap.add_argument("--sidecar-root", default=DEFAULT_SIDECAR)
    ap.add_argument("--robots", nargs="+", default=["panda", "xarm6", "widowxai"])
    ap.add_argument("--ee-to-obj", action="store_true",
                    help="also append action.ee_to_obj (gripper->object offset per frame, metres)")
    args = ap.parse_args()
    src, dst, sc = Path(args.src), Path(args.dst), Path(args.sidecar_root)
    all_stats = {r: build_robot(src, dst, sc, r, args.ee_to_obj) for r in args.robots}
    json.dump(all_stats, open(dst / "build_stats.json", "w"), indent=2)

    # end-to-end check on one episode
    robot = args.robots[0]
    obj = pd.read_parquet(sc / robot / "meta" / "object_states.parquet")
    ep = int(obj["episode_index"].iloc[0])
    pq = next((dst / robot / "data").rglob(f"episode_{ep:06d}.parquet"))
    act = np.stack(pd.read_parquet(pq)["action"].to_numpy())
    mod = json.load(open(dst / robot / "meta" / "modality.json"))["action"]
    rows = obj[obj["episode_index"] == ep].sort_values("frame")
    ref = rows[["obj_x", "obj_y", "obj_z"]].to_numpy(np.float32)
    s0 = mod["obj_rel"]["start"]
    err = np.abs(act[:, s0:s0 + 3] - (ref - ref[:1])).max()
    if args.ee_to_obj:
        s1 = mod["ee_to_obj"]["start"]
        ee_ref = rows[["ee_to_obj_x", "ee_to_obj_y", "ee_to_obj_z"]].to_numpy(np.float32)
        err = max(err, np.abs(act[:, s1:s1 + 3] - ee_ref).max())
    print(f"verify ep {ep}: max|appended - side-car| = {err:.3g} ({'OK' if err < 1e-5 else 'MISMATCH'})")


if __name__ == "__main__":
    main()
