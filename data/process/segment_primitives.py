import argparse
import json
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
from scipy.signal import find_peaks
from scipy.spatial.transform import Rotation as R

H5 = Path("demo_data/maniskill_h5")        # <robot>/<task>/motionplanning/trajectory.rgb.pd_ee_delta_pose.physx_cpu.h5
OUT = Path("demo_data_processed_prim")     # receives segments.json (input of label_primitives.py)

ROBOTS = ["panda", "xarm6", "widowxai"]
TASKS = ["PickCube-v1", "PushCube-v1", "PullCube-v1", "PlaceSphere-v1", "StackCube-v1", "LiftPegUpright-v1"]
OBJ = {"PickCube-v1": "cube", "PushCube-v1": "cube", "PullCube-v1": "cube",
       "PlaceSphere-v1": "sphere", "StackCube-v1": "cubeA", "LiftPegUpright-v1": "peg"}
GRIP_OPEN = {"panda": 1.0, "xarm6": 0.0, "widowxai": 1.0}

V_MOVE = 4e-4                 # m/step: the arm counts as moving above 0.4 mm/step
W_MOVE = np.deg2rad(0.15)
MIN_SEG_LEN = 5e-3            # segments shorter than 5 mm and 3 deg are settling after gripper events
MIN_SEG_ROT = np.deg2rad(3)


def quat_wxyz_to_rot(q):
    return R.from_quat(np.concatenate([q[..., 1:4], q[..., :1]], axis=-1))


def load(robot, task):
    fn = H5 / robot / task / "motionplanning/trajectory.rgb.pd_ee_delta_pose.physx_cpu.h5"
    f = h5py.File(fn, "r")
    eps = []
    for k in sorted(f.keys(), key=lambda s: int(s.split("_")[1])):
        g = f[k]
        actors = g["env_states/actors"]
        d = dict(
            key=k,
            tcp=g["obs/extra/tcp_pose"][:].astype(np.float64),
            act=g["actions"][:].astype(np.float64),
            obj=actors[OBJ[task]][:, :7].astype(np.float64),
            grasped=g["obs/extra/is_grasped"][:] if "is_grasped" in g["obs/extra"] else None,
        )
        eps.append(d)
    return eps


def runs(mask):
    """(start, end_exclusive) runs where mask is True."""
    out, s = [], None
    for i, m in enumerate(mask):
        if m and s is None:
            s = i
        elif not m and s is not None:
            out.append((s, i)); s = None
    if s is not None:
        out.append((s, len(mask)))
    return out


def peg_tilt_deg(obj_q):
    """Angle between the peg's long axis (local x) and world z; 90 = lying, 0 = upright."""
    ax = quat_wxyz_to_rot(obj_q).apply(np.array([1.0, 0, 0]))
    return np.degrees(np.arccos(np.clip(np.abs(ax[:, 2]), 0, 1)))


def segment_episode(robot, task, ep):
    tcp, act, obj = ep["tcp"], ep["act"], ep["obj"]
    T = act.shape[0]
    p, rot = tcp[:, :3], quat_wxyz_to_rot(tcp[:, 3:7])
    v = np.linalg.norm(np.diff(p, axis=0), axis=1)
    w = (rot[1:] * rot[:-1].inv()).magnitude()
    grip = act[:, 6]
    closed = np.abs(grip - GRIP_OPEN[robot]) > 1e-3

    # trim the post-motion creep tail of the joint-pos -> ee-delta conversion
    active = (v > V_MOVE) | (w > W_MOVE)
    gchg = np.flatnonzero(np.diff(closed.astype(int)) != 0) + 1
    last = max(np.flatnonzero(active).max() if active.any() else 0, gchg.max() if len(gchg) else 0)
    t_end = min(T, last + 6)
    creep = T - t_end

    # gripper events
    events = []
    if closed[0]:
        events.append((0, "close"))
    for t in gchg:
        if t < t_end:
            events.append((int(t), "close" if closed[t] else "open"))

    # arm motion segments: one bell-shaped speed profile per planner call, split at deep valleys
    segs = []
    for s, e in runs(active[:t_end]):
        sp = v[s:e] / 0.01 + w[s:e] / np.deg2rad(2)
        pk, _ = find_peaks(np.r_[0.0, sp, 0.0], prominence=0.1)
        pk = pk - 1
        cuts = [s]
        for p1, p2 in zip(pk[:-1], pk[1:]):
            j = p1 + int(np.argmin(sp[p1:p2 + 1]))
            if sp[j] < 0.15 and sp[j] < 0.3 * min(sp[p1], sp[p2]):
                cuts.append(s + j)
        cuts.append(e)
        for a, b in zip(cuts[:-1], cuts[1:]):
            path = v[a:b].sum()
            rang = (rot[b] * rot[a].inv()).magnitude()
            if path < MIN_SEG_LEN and rang < MIN_SEG_ROT:
                continue
            segs.append((a, b))

    # labels from tcp, gripper and object signals only
    obj_p = obj[:, :3]
    table_z = obj_p[0, 2]
    items = []
    for t, kind in events:
        d_obj = np.linalg.norm(p[t] - obj_p[t])
        items.append(dict(t0=t, t1=t + 1, kind="gripper", event=kind, tcp_obj_dist=float(d_obj),
                          grip_cmd=float(grip[min(t, T - 1)])))
    for a, b in segs:
        dp = p[b] - p[a]
        path = float(v[a:b].sum())
        rel = rot[b] * rot[a].inv()
        rv = rel.as_rotvec()
        rang = float(np.linalg.norm(rv))
        axis = rv / (rang + 1e-9)
        dobj = obj_p[b] - obj_p[a]
        rel_change = np.linalg.norm((obj_p[b] - p[b]) - (obj_p[a] - p[a]))
        is_closed = bool(closed[min((a + b) // 2, T - 1)])
        if ep["grasped"] is not None:
            held = bool(ep["grasped"][a + 1:b + 1].mean() > 0.5)
        else:
            held = bool(is_closed and (obj_p[a:b + 1, 2].max() - table_z) > 0.01)
        moved = np.linalg.norm(dobj) > 0.005 or (task == "LiftPegUpright-v1" and held)
        horiz = np.linalg.norm(dp[:2])
        if rang > np.deg2rad(5) and path > 0.01:
            ft, fr = np.cumsum(v[a:b]) / v[a:b].sum(), np.cumsum(w[a:b]) / w[a:b].sum()
            desync = float(np.abs(ft - fr).max())
        else:
            desync = None
        items.append(dict(rot_trans_desync=desync,
            t0=int(a), t1=int(b), kind="move", gripper_closed=is_closed, held=held,
            dp=dp.tolist(), dist=float(np.linalg.norm(dp)), path=path,
            straightness=float(np.linalg.norm(dp) / (path + 1e-9)), dz=float(dp[2]), dxy=float(horiz),
            rot_deg=float(np.degrees(rang)), rot_axis_world=axis.tolist(),
            yaw_frac=float(abs(axis[2])), dur=int(b - a),
            obj_disp=float(np.linalg.norm(dobj)), obj_dz=float(dobj[2]), rel_change=float(rel_change),
            tcp_obj_dist_start=float(np.linalg.norm(p[a] - obj_p[a])),
            tcp_obj_dist_end=float(np.linalg.norm(p[b] - obj_p[b])),
            obj_moved=bool(moved),
        ))
    items.sort(key=lambda d: (d["t0"], 0 if d["kind"] == "gripper" else 1))

    moves = [it for it in items if it["kind"] == "move"]
    opens = [it["t0"] for it in items if it["kind"] == "gripper" and it["event"] == "open"]
    for it in items:
        if it["kind"] == "gripper" and it["event"] == "close":
            nxt = min([t for t in opens if t > it["t0"]], default=10 ** 9)
            it["grasp"] = any(m["held"] and it["t0"] <= m["t0"] < nxt for m in moves)
    has_grasp = any(it.get("grasp") for it in items)

    released = False
    for it in items:
        if it["kind"] == "gripper":
            if it["event"] == "close":
                it["label"] = "GRASP" if it["grasp"] else "CLOSE_EMPTY"
            else:
                it["label"] = "RELEASE"; released = True
            continue
        it["obj_disturbed"] = False
        if it["held"]:
            dz, dxy, dist, rd = it["dz"], it["dxy"], it["dist"], it["rot_deg"]
            if rd > 12 and dist < 0.04:
                it["label"] = "REORIENT"
            elif dz > 0 and dz > 1.5 * dxy:
                it["label"] = "LIFT"
            elif dz < 0 and -dz > 1.5 * dxy:
                it["label"] = "LOWER"
            else:
                it["label"] = "TRANSPORT"
        elif released:
            it["label"] = "RETREAT"
        elif it["obj_moved"] and not has_grasp:
            it["label"] = "PUSH"
        else:
            it["label"] = "FREE"
            it["obj_disturbed"] = bool(it["obj_moved"])
    # free moves before the first interaction: a short last one is the APPROACH, the rest REACH
    contact = next((i for i, it in enumerate(items) if it["label"] in ("GRASP", "PUSH", "LIFT", "TRANSPORT",
                                                                      "REORIENT", "LOWER")), len(items))
    free_idx = [i for i in range(contact) if items[i]["label"] == "FREE"]
    for i in free_idx:
        items[i]["label"] = "REACH"
    if len(free_idx) >= 2 and items[free_idx[-1]]["dist"] < 0.12:
        items[free_idx[-1]]["label"] = "APPROACH"
    for it in items:
        if it["label"] == "FREE":
            it["label"] = "REACH"
    grammar = [it["label"] for it in items]
    coarse = []
    for g in grammar:
        g = "CARRY" if g in ("LIFT", "TRANSPORT", "LOWER") else g
        if not coarse or coarse[-1] != g or g in ("GRASP", "RELEASE"):
            coarse.append(g)

    extra = {}
    if task == "LiftPegUpright-v1":
        tilt = peg_tilt_deg(obj[:, 3:7])
        extra["peg_tilt_start"] = float(tilt[0])
        extra["peg_tilt_end"] = float(tilt[t_end])
        for it in items:
            it["peg_tilt_start"] = float(tilt[it["t0"]]); it["peg_tilt_end"] = float(tilt[min(it["t1"], len(tilt) - 1)])
    if task in ("PushCube-v1", "PullCube-v1"):
        extra["obj_dx"] = float(obj_p[t_end, 0] - obj_p[0, 0])   # cube travel along +x (away from the base)
    return dict(key=ep["key"], T=int(T), t_end=int(t_end), creep=int(creep), items=items, grammar=grammar, coarse=coarse,
                disturbed=any(it.get("obj_disturbed") for it in items), **extra)


def main():
    global H5, OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", default=str(H5), help="root with <robot>/<task>/motionplanning/*.h5")
    ap.add_argument("--out", default=str(OUT), help="directory that receives segments.json")
    args = ap.parse_args()
    H5, OUT = Path(args.h5), Path(args.out)
    OUT.mkdir(parents=True, exist_ok=True)
    allres = {}
    for robot in ROBOTS:
        for task in TASKS:
            eps = load(robot, task)
            allres[f"{robot}/{task}"] = [segment_episode(robot, task, ep) for ep in eps]
            g = Counter(" ".join(r["grammar"]) for r in allres[f"{robot}/{task}"])
            print(f"{robot:9s} {task:18s} n={len(eps)} variants={len(g)} top={g.most_common(1)[0][1]}  {g.most_common(1)[0][0]}",
                  flush=True)
    with open(OUT / "segments.json", "w") as fh:
        json.dump(allres, fh)


if __name__ == "__main__":
    main()
