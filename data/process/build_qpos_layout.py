# assets/kinematic_graphs/qpos_layout.json: per robot graph, the joint behind each column of ManiSkill's obs/agent/qpos
# with the URDF limits (read by model/stage1_graph.py, graph_temporal). Column order = PhysX articulation order =
# breadth-first over the URDF link tree, siblings in file order, fixed joints skipped (== agent.robot.active_joints).
#   python data/process/build_qpos_layout.py [--robots panda xarm6 widowxai]
import argparse
import json
import math
import os
import xml.etree.ElementTree as ET
from collections import deque
from pathlib import Path

from model.kinematic_graph import KinematicGraph


def parse_joints(urdf: str):
    root = ET.parse(urdf).getroot()
    joints = []
    for j in root.findall("joint"):
        jt, lim = j.get("type", "fixed"), j.find("limit")
        lo = float(lim.get("lower", 0.0)) if lim is not None else 0.0
        hi = float(lim.get("upper", 0.0)) if lim is not None else 0.0
        if jt == "continuous" or hi <= lo:
            lo, hi = (-math.pi, math.pi) if jt != "prismatic" else (0.0, 0.1)
        joints.append(dict(name=j.get("name"), type=jt, parent=j.find("parent").get("link"),
                           child=j.find("child").get("link"), lower=lo, upper=hi, mimic=j.find("mimic") is not None))
    links = [l.get("name") for l in root.findall("link")]
    return joints, links


def active_joint_order(joints, links):
    children = {l: [] for l in links}
    for j in joints:
        children[j["parent"]].append(j)
    root_link = [l for l in links if l not in {j["child"] for j in joints}]
    assert len(root_link) == 1, root_link
    order, queue = [], deque(root_link)
    while queue:
        for j in children[queue.popleft()]:
            if j["type"] != "fixed":
                order.append(j)
            queue.append(j["child"])
    return order


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph_dir", default="assets/kinematic_graphs")
    ap.add_argument("--robots", nargs="+", default=["panda", "xarm6", "widowxai"])
    args = ap.parse_args()
    layout = {}
    for r in args.robots:
        g = KinematicGraph.load(os.path.join(args.graph_dir, f"{r}.json"))
        order = active_joint_order(*parse_joints(g.urdf))
        missing = [j["name"] for j in order if j["name"] not in g.node_names]
        assert not missing, f"{r}: qpos joints without a graph node: {missing}"
        layout[r] = {"joints": [j["name"] for j in order], "type": [j["type"] for j in order],
                     "lower": [j["lower"] for j in order], "upper": [j["upper"] for j in order],
                     "mimic": [j["mimic"] for j in order], "urdf": g.urdf}
        print(f"{r}: {len(order)} dof {layout[r]['joints']}")
    out = Path(args.graph_dir) / "qpos_layout.json"
    json.dump(layout, open(out, "w"), indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
