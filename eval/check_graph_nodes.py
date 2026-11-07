"""Sanity check of the kinematic graphs (assets/kinematic_graphs/<robot>.json).

Three checks, all printed as text (tee into eval_results/analysis/ to keep them):

  1. structure   every node/edge of the saved JSON against (a) a fresh parse of the
                 URDF and (b) a hand-written table of what each node is MEANT to be
                 (INTENT below: coarse role, gripper flag, leaf flag). Any disagreement
                 is listed as a MISMATCH.
  2. sample      a seeded random sample of nodes, printed as "cards": decoded feature
                 vector, parent / children, URDF line, intended role.
  3. learned     (optional, --ckpt) the trained stage-1 GNN's node embeddings: for every
                 node the nearest node of the OTHER robots (cosine) and whether its
                 coarse role agrees; g_r similarity between robots and across tasks.

Usage (repo root, CPU is enough):
  python eval/check_graph_nodes.py [--seed 0] [--per-robot 2] [--nodes xarm6/eef_joint ...] \
      [--ckpt runs/.../outputs_stage1/epoch_0020.pth ...]
"""
from __future__ import annotations

import argparse
import math
import os
import random
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from model.kinematic_graph import (EDGE_FEATURE_NAMES, JOINT_TYPES, NODE_FEATURE_NAMES,  # noqa: E402
                                   KinematicGraph, parse_urdf, _quat_rotate)

ROLES = ("base", "arm", "mount", "gripper_act", "gripper_fixed", "sensor")

# What each node is meant to be. role: base = root link; arm = actuated arm joint;
# mount = fixed joint inside the arm/wrist (flange, world mount); gripper_act = actuated
# (or mimic) gripper joint; gripper_fixed = fixed joint of the hand (TCP, pads, mount of
# the gripper body); sensor = camera mount (not part of the manipulation chain).
# (role, is_gripper, is_leaf)
INTENT = {
    "panda": {
        "panda_link0": ("base", 0, 0),
        **{f"panda_joint{i}": ("arm", 0, 0) for i in range(1, 8)},
        "panda_joint8": ("mount", 0, 0),
        "panda_hand_joint": ("gripper_fixed", 1, 0),
        "panda_hand_tcp_joint": ("gripper_fixed", 1, 1),
        "panda_leftfinger_pad_joint": ("gripper_fixed", 1, 1),
        "panda_rightfinger_pad_joint": ("gripper_fixed", 1, 1),
        "panda_finger_joint1": ("gripper_act", 1, 0),
        "panda_finger_joint2": ("gripper_act", 1, 0),
    },
    "xarm6": {
        "world": ("base", 0, 0),
        "world_joint": ("mount", 0, 0),
        **{f"joint{i}": ("arm", 0, 0) for i in range(1, 7)},
        "camera_link_joint": ("sensor", 0, 1),
        "gripper_fix": ("gripper_fixed", 1, 0),
        "eef_joint": ("gripper_fixed", 1, 1),          # TCP of the robotiq gripper
        **{f"{s}_outer_knuckle_joint": ("gripper_act", 1, 0) for s in ("left", "right")},
        **{f"{s}_outer_finger_joint": ("gripper_fixed", 1, 0) for s in ("left", "right")},
        **{f"{s}_inner_knuckle_joint": ("gripper_act", 1, 1) for s in ("left", "right")},
        **{f"{s}_inner_finger_joint": ("gripper_act", 1, 0) for s in ("left", "right")},
        **{f"{s}_inner_finger_pad_joint": ("gripper_fixed", 1, 1) for s in ("left", "right")},
    },
    "widowxai": {
        "base_link": ("base", 0, 0),
        **{f"joint_{i}": ("arm", 0, 0) for i in range(6)},
        "right_carriage_joint": ("gripper_act", 1, 0),
        "left_carriage_joint": ("gripper_act", 1, 0),
        "left_gripper_joint": ("gripper_fixed", 1, 1),
        "right_gripper_joint": ("gripper_fixed", 1, 1),
        "ee_gripper": ("gripper_fixed", 1, 1),         # TCP
    },
}

F = {n: i for i, n in enumerate(NODE_FEATURE_NAMES)}


def decode(feat):
    jt = JOINT_TYPES[int(np.argmax(feat[:4]))]
    scale = 0.1 if jt == "prismatic" else math.pi
    return dict(type=jt, lower=feat[F["lower"]] * scale, upper=feat[F["upper"]] * scale,
                velocity=feat[F["velocity"]], effort=feat[F["effort"]],
                axis=feat[F["axis_x"]:F["axis_z"] + 1], depth=feat[F["depth"]],
                is_mimic=int(feat[F["is_mimic"]]), is_gripper=int(feat[F["is_gripper"]]),
                is_base=int(feat[F["is_base"]]), is_leaf=int(feat[F["is_leaf"]]))


def role_from_feats(feat):
    """Coarse role implied by the stored flags (what the GNN actually sees)."""
    d = decode(feat)
    if d["is_base"]:
        return "base"
    movable = d["type"] != "fixed"
    if d["is_gripper"]:
        return "gripper_act" if movable else "gripper_fixed"
    return "arm" if movable else "mount"


def children_of(g: KinematicGraph):
    ch = {i: [] for i in range(g.num_nodes)}
    par = {}
    for s, d, e in zip(g.edge_src, g.edge_dst, g.edge_feats):
        if e[-1] == 0.0:                      # forward edge only
            ch[s].append(d)
            par[d] = s
    return ch, par


def urdf_line(urdf_path, joint_name):
    try:
        import xml.etree.ElementTree as ET
        root = ET.parse(urdf_path).getroot()
    except Exception as ex:                    # noqa: BLE001
        return f"(URDF unreadable: {ex})"
    for j in root.findall("joint"):
        if j.get("name") == joint_name:
            o, lim, ax, mim = j.find("origin"), j.find("limit"), j.find("axis"), j.find("mimic")
            return (f"<joint name={joint_name} type={j.get('type')} parent={j.find('parent').get('link')} "
                    f"child={j.find('child').get('link')} xyz={o.get('xyz') if o is not None else None} "
                    f"rpy={o.get('rpy') if o is not None else None} "
                    f"limit={dict(lim.attrib) if lim is not None else None} "
                    f"axis={ax.get('xyz') if ax is not None else None} "
                    f"mimic={mim.get('joint') if mim is not None else None}>")
    return "(root link, no joint element)"


# --------------------------------------------------------------------------- 1. structure
def check_structure(robot, g: KinematicGraph):
    problems = []
    ch, par = children_of(g)
    intent = INTENT[robot]
    # (a) fresh URDF parse == saved JSON
    if os.path.isfile(g.urdf):
        fresh = parse_urdf(g.urdf, robot=robot, bidirectional=True)
        if fresh.node_names != g.node_names:
            problems.append(f"node names differ from fresh URDF parse: {fresh.node_names}")
        elif not np.allclose(fresh.node_feats, g.node_feats, atol=1e-9):
            bad = [g.node_names[i] for i in range(g.num_nodes)
                   if not np.allclose(fresh.node_feats[i], g.node_feats[i], atol=1e-9)]
            problems.append(f"node features differ from fresh URDF parse at {bad}")
        if (fresh.edge_src, fresh.edge_dst) != (g.edge_src, g.edge_dst) or \
                not np.allclose(fresh.edge_feats, g.edge_feats, atol=1e-9):
            problems.append("edges differ from fresh URDF parse")
    else:
        problems.append(f"URDF not found, skipped fresh-parse comparison: {g.urdf}")
    # (b) intended labels
    missing = [n for n in g.node_names if n not in intent]
    extra = [n for n in intent if n not in g.node_names]
    if missing:
        problems.append(f"nodes without an INTENT entry: {missing}")
    if extra:
        problems.append(f"INTENT entries without a node: {extra}")
    for i, name in enumerate(g.node_names):
        d = decode(g.node_feats[i])
        role = role_from_feats(g.node_feats[i])
        if name in intent:
            want_role, want_grip, want_leaf = intent[name]
            # 'sensor' is not representable by the flags: it comes out as 'mount'
            ok_role = role == want_role or (want_role == "sensor" and role == "mount")
            if not ok_role:
                problems.append(f"MISMATCH {name}: stored flags imply role '{role}', intended '{want_role}'")
            if d["is_gripper"] != want_grip:
                problems.append(f"MISMATCH {name}: is_gripper={d['is_gripper']}, intended {want_grip}")
            if d["is_leaf"] != want_leaf:
                problems.append(f"MISMATCH {name}: is_leaf={d['is_leaf']}, intended {want_leaf}")
        # structural invariants
        if d["is_base"] != int(i == 0):
            problems.append(f"{name}: is_base={d['is_base']} but index {i}")
        if d["is_leaf"] != int(len(ch[i]) == 0):
            problems.append(f"{name}: is_leaf={d['is_leaf']} but {len(ch[i])} children")
        if i > 0 and par.get(i) is None:
            problems.append(f"{name}: no parent edge")
        if i > 0:
            dp = decode(g.node_feats[par[i]])["depth"]
            if not d["depth"] > dp:
                problems.append(f"{name}: depth {d['depth']:.3f} not > parent depth {dp:.3f}")
        if d["type"] == "fixed" and any(abs(a) > 0 for a in d["axis"]):
            problems.append(f"{name}: fixed joint with non-zero axis {d['axis']}")
        if d["type"] != "fixed" and abs(np.linalg.norm(d["axis"]) - 1) > 1e-6:
            problems.append(f"{name}: movable joint axis not unit {d['axis']}")
        if d["type"] != "fixed" and not d["upper"] > d["lower"]:
            problems.append(f"{name}: empty limit range [{d['lower']}, {d['upper']}]")
    # edges: N-1 forward, same number reversed, reverse = inverse transform
    fwd = [k for k, e in enumerate(g.edge_feats) if e[-1] == 0.0]
    rev = [k for k, e in enumerate(g.edge_feats) if e[-1] == 1.0]
    if len(fwd) != g.num_nodes - 1:
        problems.append(f"{len(fwd)} forward edges for {g.num_nodes} nodes (expect N-1)")
    if len(rev) != len(fwd):
        problems.append(f"{len(rev)} reverse edges vs {len(fwd)} forward")
    for k in fwd:
        kr = k + 1
        if kr >= len(g.edge_feats) or g.edge_feats[kr][-1] != 1.0 or \
                (g.edge_src[kr], g.edge_dst[kr]) != (g.edge_dst[k], g.edge_src[k]):
            problems.append(f"edge {k} has no matching reverse edge at {kr}")
            continue
        t, q = g.edge_feats[k][:3], g.edge_feats[k][3:7]
        ti, qi = g.edge_feats[kr][:3], g.edge_feats[kr][3:7]
        if abs(np.linalg.norm(q) - 1) > 1e-6:
            problems.append(f"edge {k}: quaternion not unit")
        # q * (qi * t + ti) should be 0 : apply inverse then forward
        back = np.array(_quat_rotate(q, ti)) + np.array(t)
        if np.linalg.norm(back) > 1e-6:
            problems.append(f"edge {k}: reverse edge is not the inverse transform (residual {np.linalg.norm(back):.2e})")
        if abs(g.edge_feats[k][7] - np.linalg.norm(t)) > 1e-6:
            problems.append(f"edge {k}: length feature != |xyz|")
    return problems


# --------------------------------------------------------------------------- 2. sample
def node_card(robot, g: KinematicGraph, i, urdf_ok=True):
    ch, par = children_of(g)
    name = g.node_names[i]
    d = decode(g.node_feats[i])
    role = role_from_feats(g.node_feats[i])
    want = INTENT[robot].get(name)
    lines = [f"### {robot} node {i}: `{name}`",
             f"- stored features: type={d['type']}, limits=[{d['lower']:.3f}, {d['upper']:.3f}] "
             f"({'m' if d['type'] == 'prismatic' else 'rad'}), vel={d['velocity']:.3f}, "
             f"effort={d['effort']:.3f}, axis={[round(a, 3) for a in d['axis']]}, depth={d['depth']:.3f}, "
             f"mimic={d['is_mimic']}, gripper={d['is_gripper']}, base={d['is_base']}, leaf={d['is_leaf']}",
             f"- parent: {g.node_names[par[i]] if i in par else '(none, root)'}; "
             f"children: {[g.node_names[c] for c in ch[i]]}",
             f"- role implied by flags: **{role}**; intended: **{want[0] if want else '?'}** "
             f"(gripper {want[1] if want else '?'}, leaf {want[2] if want else '?'}) -> "
             f"{'OK' if want and (role == want[0] or (want[0] == 'sensor' and role == 'mount')) and d['is_gripper'] == want[1] and d['is_leaf'] == want[2] else 'MISMATCH'}"]
    if i > 0:
        k = next(k for k, (s, dd, e) in enumerate(zip(g.edge_src, g.edge_dst, g.edge_feats))
                 if dd == i and e[-1] == 0.0)
        e = g.edge_feats[k]
        lines.append(f"- link from parent (edge feats): xyz={[round(x, 4) for x in e[:3]]}, "
                     f"quat(wxyz)={[round(x, 4) for x in e[3:7]]}, length={e[7]:.4f} m")
    if urdf_ok:
        lines.append(f"- URDF: `{urdf_line(g.urdf, name)}`")
    return "\n".join(lines)


# --------------------------------------------------------------------------- 3. learned
def learned_check(ckpt_path, graphs, samples):
    import torch
    torch.set_grad_enabled(False)
    from model.stage1_graph import TASK_IDS, build_model
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]["models"]["stage1"]
    cfg["graph_dir"] = os.path.join(REPO, "assets/kinematic_graphs")
    model = build_model(cfg).eval()
    # Older checkpoints carry trainer-side modules (discriminator.*) and lack the
    # effect head; only the GNN / task embedder / routing weights matter here.
    res = model.load_state_dict(ck["model"], strict=False)
    needed = [k for k in model.state_dict() if k.startswith(("gnn.", "task_embed.", "route_query.", "vq."))]
    lost = [k for k in needed if k in res.missing_keys]
    assert not lost, f"GNN/routing weights missing from checkpoint: {lost}"
    robots = model.robot_names
    gid = model.gnn.gid.numpy()
    names, roles = [], []
    for r in robots:
        g = graphs[r]
        names += [f"{r}/{n}" for n in g.node_names]
        roles += [INTENT[r].get(n, ("?",))[0] for n in g.node_names]
    roles = np.array(roles)
    out = [f"## Learned node embeddings: `{os.path.relpath(ckpt_path, REPO)}` (epoch {ck.get('epoch')})"]
    tasks = list(TASK_IDS)
    G, N = [], []
    for t in tasks:
        e_task = model.task_embed([t], "cpu")
        g, n = model.gnn(e_task[0])
        G.append(g); N.append(n)
    G = torch.stack(G)                                     # [T, R, D]
    N = torch.stack(N)                                     # [T, Nn, D]
    # g_r similarity
    Gn = torch.nn.functional.normalize(G, dim=-1)
    out.append("### g_r (morphology embedding) cosine similarity")
    out.append("between robots, averaged over the 6 instructions:")
    for a in range(len(robots)):
        for b in range(a + 1, len(robots)):
            out.append(f"- {robots[a]} vs {robots[b]}: {(Gn[:, a] * Gn[:, b]).sum(-1).mean():.3f}")
    out.append("same robot, between instructions (min over task pairs):")
    for r_i, r in enumerate(robots):
        s = Gn[:, r_i] @ Gn[:, r_i].T
        iu = np.triu_indices(len(tasks), 1)
        out.append(f"- {r}: {s.numpy()[iu].min():.3f} (mean {s.numpy()[iu].mean():.3f})")
    # nearest cross-robot node, averaged embedding over tasks
    Nm = torch.nn.functional.normalize(N.mean(0), dim=-1)  # [Nn, D]
    S = (Nm @ Nm.T).numpy()
    agree, per_role = [], {}
    nn_of = {}
    for i in range(len(names)):
        mask = gid != gid[i]
        cand = np.where(mask)[0]
        order = cand[np.argsort(-S[i, cand])]
        nn_of[i] = order[:3]
        ok = roles[order[0]] == roles[i] or {roles[order[0]], roles[i]} <= {"mount", "sensor"}
        agree.append(ok)
        per_role.setdefault(roles[i], []).append(ok)
    out.append(f"### Nearest node of another robot has the same intended role: "
               f"{np.mean(agree) * 100:.0f}% of {len(names)} nodes")
    for role in ROLES:
        if role in per_role:
            v = per_role[role]
            out.append(f"- {role}: {sum(v)}/{len(v)}")
    out.append("### Sampled nodes: top-3 cross-robot neighbours (cosine, task-averaged)")
    base = 0
    offsets = {}
    for r in robots:
        offsets[r] = base
        base += graphs[r].num_nodes
    for r, i in samples:
        gi = offsets[r] + i
        nb = ", ".join(f"{names[j]} ({roles[j]}, {S[gi, j]:.2f})" for j in nn_of[gi])
        out.append(f"- {names[gi]} [{roles[gi]}] -> {nb}")
    # does the task condition change node embeddings? per-node cosine across tasks
    Nn_ = torch.nn.functional.normalize(N, dim=-1)
    cross = torch.einsum("tnd,snd->tsn", Nn_, Nn_)
    iu = np.triu_indices(len(tasks), 1)
    out.append(f"### Task dependence: node embedding cosine between instructions, mean "
               f"{cross.numpy()[iu].mean():.3f}, min {cross.numpy()[iu].min():.3f} "
               f"(1.0 = task embedding ignored)")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph-dir", default=os.path.join(REPO, "assets/kinematic_graphs"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--per-robot", type=int, default=2)
    ap.add_argument("--ckpt", nargs="*", default=[])
    ap.add_argument("--nodes", nargs="*", default=[], help="extra named nodes, e.g. xarm6/eef_joint")
    args = ap.parse_args()
    robots = list(INTENT)
    graphs = {r: KinematicGraph.load(os.path.join(args.graph_dir, f"{r}.json")) for r in robots}

    print("# Kinematic graph node check\n")
    print("## 1. Structure: JSON vs URDF vs intended labels")
    total = 0
    for r in robots:
        g = graphs[r]
        probs = check_structure(r, g)
        total += len(probs)
        roles = [role_from_feats(f) for f in g.node_feats]
        counts = {k: roles.count(k) for k in ROLES if roles.count(k)}
        print(f"- {r}: {g.num_nodes} nodes, {len(g.edge_src)} edges ({len(g.edge_src)//2} links x2), roles {counts} "
              f"-> {'OK' if not probs else f'{len(probs)} problem(s)'}")
        for p in probs:
            print(f"    - {p}")
    print(f"\n{total} problem(s) in total.\n")

    print(f"## 2. Sampled nodes (seed {args.seed}, {args.per_robot} per robot)\n")
    rng = random.Random(args.seed)
    samples = []
    for r in robots:
        for i in sorted(rng.sample(range(graphs[r].num_nodes), args.per_robot)):
            samples.append((r, i))
    for spec in args.nodes:
        r, n = spec.split("/", 1)
        samples.append((r, graphs[r].node_names.index(n)))
    for r, i in samples:
        print(node_card(r, graphs[r], i, urdf_ok=os.path.isfile(graphs[r].urdf)))
        print()

    for ck in args.ckpt:
        print(learned_check(ck, graphs, samples))
        print()


if __name__ == "__main__":
    main()
