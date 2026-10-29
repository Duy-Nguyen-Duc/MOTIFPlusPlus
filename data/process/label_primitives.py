import argparse
import json
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from data.dataset import calculate_dataset_statistics
from model.primitive_tying import DEFAULT_SLOT_MASK, PRIMITIVES, SLOT_NAMES

# Copies demo_data_processed/<robot> (videos symlinked) and appends two absolute action columns per frame:
# action.prim_id (0 REACH 1 PRESHAPE 2 GRASP 3 CARRY 4 REORIENT 5 PUSH 6 RELEASE) and action.prim_prog in [0, 1],
# from the segments of data/process/segment_primitives.py. data/transform/concat.py routes them to
# batch["prim_id"] / batch["prim_prog"]; they never enter the action target or the normaliser.
MERGE = {"REACH": 0, "APPROACH": 0, "CLOSE_EMPTY": 1, "GRASP": 2, "LIFT": 3, "TRANSPORT": 3,
         "LOWER": 3, "REORIENT": 4, "PUSH": 5, "RELEASE": 6, "RETREAT": 6}
TASK_OF = {"Push the cube": "PushCube-v1", "Pick up the sphere": "PlaceSphere-v1",
           "Pull the cube": "PullCube-v1", "Lift the peg": "LiftPegUpright-v1",
           "Pick up the cube": "PickCube-v1", "Stack the cube": "StackCube-v1"}
DEFAULT_SEGMENTS = "demo_data_processed_prim/segments.json"
THR_T_MM, THR_R_DEG, THR_G_FRAC, G_MOVE_EPS = 1.0, 0.2, 0.2, 0.01   # energy thresholds of the derived slot mask


def task_id(instr: str) -> str:
    for k, v in TASK_OF.items():
        if instr.startswith(k):
            return v
    raise KeyError(instr)


def frame_labels(items, T):
    lab = np.full(T, -1, np.int64)
    for it in sorted(items, key=lambda i: i["t0"]):            # moves first, gripper events overwrite
        if it["kind"] == "move":
            lab[it["t0"]:min(it["t1"], T - 1) + 1] = MERGE[it["label"]]
    for it in sorted(items, key=lambda i: i["t0"]):
        if it["kind"] == "gripper":
            lab[it["t0"]:min(it["t1"], T - 1) + 1] = MERGE[it["label"]]
    first = np.flatnonzero(lab >= 0)
    assert len(first), "episode without segments"
    lab[:first[0]] = lab[first[0]]
    for t in range(1, T):                                       # gaps and tail continue the previous primitive
        if lab[t] < 0:
            lab[t] = lab[t - 1]
    prog = np.zeros(T, np.float32)
    s = 0
    for t in range(1, T + 1):
        if t == T or lab[t] != lab[s]:
            n = t - s
            prog[s:t] = np.arange(n) / max(n - 1, 1)
            s = t
    return lab, prog


def quat_step_deg(q):
    """Angle between consecutive wxyz quaternions, [T-1] degrees."""
    q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-9)
    d = np.abs((q[1:] * q[:-1]).sum(1)).clip(0.0, 1.0)
    return np.degrees(2.0 * np.arccos(d))


def episode_stats(df: pd.DataFrame, col: str) -> dict:
    a = np.vstack([np.asarray(x, dtype=np.float32) for x in df[col]])
    return {"min": a.min(0).tolist(), "max": a.max(0).tolist(), "mean": a.mean(0).tolist(),
            "std": a.std(0).tolist(), "count": [len(a)]}


def fix_stats(d: Path, robot: str):
    meta = d / "meta"
    parquets = sorted((d / "data").rglob("episode_*.parquet"))
    width = int(np.asarray(pd.read_parquet(parquets[0])["action"].iloc[0]).shape[0])
    declared = max(v["end"] for v in json.load(open(meta / "modality.json"))["action"].values())
    assert width == declared, f"{robot}: parquet action width {width} != modality.json {declared}"
    stats = calculate_dataset_statistics(parquets)
    json.dump(stats, open(meta / "stats.json", "w"), indent=4)
    by_ep = {int(p.stem.split("_")[-1]): {c: episode_stats(pd.read_parquet(p), c)
                                           for c in ("observation.state", "action")}
             for p in parquets}
    lines = []
    for line in open(meta / "episodes_stats.jsonl"):
        row = json.loads(line)
        if row["episode_index"] in by_ep:
            row["stats"].update(by_ep[row["episode_index"]])
        lines.append(json.dumps(row))
    open(meta / "episodes_stats.jsonl", "w").write("\n".join(lines) + "\n")
    info = json.load(open(meta / "info.json"))
    info["features"]["action"]["shape"] = [width]
    json.dump(info, open(meta / "info.json", "w"), indent=4)
    (meta / "temp_stats.json").unlink(missing_ok=True)
    print(f"{robot}: stats rewritten for action width {width}")


def build_robot(src: Path, dst: Path, robot: str, seg: dict) -> dict:
    s, d = src / robot, dst / robot
    (d / "meta").mkdir(parents=True, exist_ok=True)
    for f in (s / "meta").glob("*"):
        if f.is_file():
            shutil.copy2(f, d / "meta" / f.name)
    if not (d / "videos").exists():
        os.symlink((s / "videos").resolve(), d / "videos")

    st_stats = json.load(open(s / "meta/stats.json"))["observation.state"]
    g_range = max(float(st_stats["max"][7]) - float(st_stats["min"][7]), 1e-6)

    eps = [json.loads(l) for l in open(s / "meta/episodes.jsonl")]
    rank = Counter()
    frames = Counter()                          # (task, prim) -> frames
    chains = Counter()                          # (task, chain) -> episodes
    energy = defaultdict(lambda: np.zeros(4))   # prim -> [sum T mm, sum R deg, sum G moving, n]
    for e in eps:
        task = task_id(e["tasks"][0])
        r = rank[task]
        rank[task] += 1
        ep = seg[f"{robot}/{task}"][r]           # LeRobot episode order == h5 trajectory order per task
        assert ep["key"] == f"traj_{r}", (robot, task, r, ep["key"])
        pq = next((s / "data").rglob(f"episode_{e['episode_index']:06d}.parquet"))
        df = pd.read_parquet(pq)
        act = np.stack(df["action"].to_numpy()).astype(np.float32)
        T = len(act)
        assert T == e["length"] == ep["T"], (robot, task, r, T, e["length"], ep["T"])
        lab, prog = frame_labels(ep["items"], T)
        df["action"] = list(np.concatenate([act, lab[:, None].astype(np.float32), prog[:, None]], axis=1))
        out = d / "data" / pq.parent.name / pq.name
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out, index=False)

        for n in np.unique(lab):
            frames[(task, PRIMITIVES[n])] += int((lab == n).sum())
        chain = [PRIMITIVES[lab[0]]]
        for a, b in zip(lab[:-1], lab[1:]):
            if a != b:
                chain.append(PRIMITIVES[b])
        chains[(task, " ".join(chain))] += 1

        st = np.stack(df["observation.state"].to_numpy()).astype(np.float64)
        v_mm = np.linalg.norm(np.diff(st[:, :3], axis=0), axis=1) * 1000.0
        w_deg = quat_step_deg(st[:, 3:7])
        g_mov = (np.abs(np.diff(st[:, 7])) / g_range > G_MOVE_EPS).astype(np.float64)
        for t in range(1, T):                   # step t-1 -> t carries the label of frame t
            acc = energy[int(lab[t])]
            acc[0] += v_mm[t - 1]; acc[1] += w_deg[t - 1]; acc[2] += g_mov[t - 1]; acc[3] += 1

    mp = d / "meta/modality.json"
    mod = json.load(open(mp))
    width = max(v["end"] for v in mod["action"].values())
    mod["action"]["prim_id"] = {"start": width, "end": width + 1, "absolute": True}
    mod["action"]["prim_prog"] = {"start": width + 1, "end": width + 2, "absolute": True}
    json.dump(mod, open(mp, "w"), indent=4)
    fix_stats(d, robot)

    report = {
        "action_width_before": width,
        "frames_per_task_primitive": {f"{t}/{n}": c for (t, n), c in sorted(frames.items())},
        "chains": {f"{t}: {c}": n for (t, c), n in sorted(chains.items(), key=lambda kv: (kv[0][0], -kv[1]))},
        "energy": {PRIMITIVES[p]: {"T_mm_per_step": a[0] / max(a[3], 1), "R_deg_per_step": a[1] / max(a[3], 1),
                                   "G_moving_frac": a[2] / max(a[3], 1), "steps": int(a[3])}
                   for p, a in sorted(energy.items())},
    }
    print(f"{robot}: {len(eps)} episodes, action {width} -> {width + 2}")
    return report


def derive_mask(energy: dict):
    mask = []
    for n in PRIMITIVES:
        e = energy.get(n)
        if e is None:
            mask.append([None, None, None]); continue
        mask.append([int(e["T_mm_per_step"] > THR_T_MM), int(e["R_deg_per_step"] > THR_R_DEG),
                     int(e["G_moving_frac"] > THR_G_FRAC)])
    return mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="demo_data_processed")
    ap.add_argument("--dst", default="demo_data_processed_prim")
    ap.add_argument("--segments", default=DEFAULT_SEGMENTS)
    ap.add_argument("--robots", nargs="+", default=["panda", "xarm6", "widowxai"])
    args = ap.parse_args()
    src, dst = Path(args.src), Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)
    seg_path = Path(args.segments)
    seg = json.load(open(seg_path))
    if seg_path.resolve() != (dst / "segments.json").resolve():
        shutil.copy2(seg_path, dst / "segments.json")

    report = {}
    pooled = defaultdict(lambda: np.zeros(4))
    for robot in args.robots:
        report[robot] = build_robot(src, dst, robot, seg)
        for n, e in report[robot]["energy"].items():
            k = e["steps"]
            pooled[n] += np.array([e["T_mm_per_step"] * k, e["R_deg_per_step"] * k, e["G_moving_frac"] * k, k])
    energy_all = {n: {"T_mm_per_step": a[0] / a[3], "R_deg_per_step": a[1] / a[3],
                      "G_moving_frac": a[2] / a[3], "steps": int(a[3])} for n, a in pooled.items()}
    derived = derive_mask(energy_all)
    print("\n== per-step channel energy per primitive (all robots) ==")
    print(f"{'primitive':10s} {'T mm/step':>10s} {'R deg/step':>11s} {'G moving':>9s} {'steps':>7s}   derived  prior")
    for p, n in enumerate(PRIMITIVES):
        e = energy_all.get(n)
        if e is None:
            continue
        print(f"{n:10s} {e['T_mm_per_step']:10.3f} {e['R_deg_per_step']:11.3f} {e['G_moving_frac']:9.3f} "
              f"{e['steps']:7d}   {derived[p]}  {DEFAULT_SLOT_MASK[p]}")
    agree = all(derived[p] == DEFAULT_SLOT_MASK[p] for p in range(len(PRIMITIVES)) if PRIMITIVES[p] in energy_all)
    print("derived mask == prior in model/primitive_tying.py:", agree)
    report["_all"] = {"energy": energy_all, "derived_slot_mask": derived, "prior_slot_mask": DEFAULT_SLOT_MASK,
                      "slots": SLOT_NAMES, "primitives": PRIMITIVES, "agrees_with_prior": agree,
                      "thresholds": {"T_mm": THR_T_MM, "R_deg": THR_R_DEG, "G_frac": THR_G_FRAC, "G_eps": G_MOVE_EPS}}
    json.dump(report, open(dst / "PRIMITIVES_REPORT.json", "w"), indent=2)

    robot = args.robots[0]
    e = json.loads(open(dst / robot / "meta/episodes.jsonl").readline())
    pq = next((dst / robot / "data").rglob(f"episode_{e['episode_index']:06d}.parquet"))
    act = np.stack(pd.read_parquet(pq)["action"].to_numpy())
    print("verify", robot, e["tasks"][0][:24], "action", act.shape,
          "prims", [PRIMITIVES[int(i)] for i in np.unique(act[:, -2])],
          "prog range", float(act[:, -1].min()), float(act[:, -1].max()))


if __name__ == "__main__":
    main()
