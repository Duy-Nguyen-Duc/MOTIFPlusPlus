from __future__ import annotations

import json
import math
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

import numpy as np

JOINT_TYPES = ["revolute", "continuous", "prismatic", "fixed"]
NODE_FEATURE_NAMES = (
    [f"type_{t}" for t in JOINT_TYPES]
    + ["lower", "upper", "range", "velocity", "effort", "axis_x", "axis_y", "axis_z",
       "depth", "is_mimic", "is_gripper", "is_base", "is_leaf"]
)
EDGE_FEATURE_NAMES = ["dx", "dy", "dz", "qw", "qx", "qy", "qz", "length", "reverse"]
GRIPPER_WORDS = ("finger", "gripper", "knuckle", "jaw", "hand", "tcp", "tool", "carriage", "pad")


@dataclass
class KinematicGraph:
    robot: str
    urdf: str
    node_names: List[str]
    node_feats: List[List[float]]      # [N, d_v]
    edge_src: List[int]                # [E]
    edge_dst: List[int]
    edge_feats: List[List[float]]      # [E, d_e]
    node_feature_names: List[str]
    edge_feature_names: List[str]

    @property
    def num_nodes(self) -> int:
        return len(self.node_names)

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=1)

    @staticmethod
    def load(path: str) -> "KinematicGraph":
        with open(path) as f:
            return KinematicGraph(**json.load(f))


def _floats(s: Optional[str], n: int, default: float = 0.0) -> List[float]:
    if not s:
        return [default] * n
    v = [float(x) for x in s.split()]
    return (v + [default] * n)[:n]


def _rpy_to_quat(r: float, p: float, y: float) -> List[float]:
    cr, sr = math.cos(r / 2), math.sin(r / 2)
    cp, sp = math.cos(p / 2), math.sin(p / 2)
    cy, sy = math.cos(y / 2), math.sin(y / 2)
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    return [qw, qx, qy, qz]


def _quat_conj(q):
    return [q[0], -q[1], -q[2], -q[3]]


def _quat_rotate(q, v):
    w, x, y, z = q
    # v' = q v q*
    t = [2 * (y * v[2] - z * v[1]), 2 * (z * v[0] - x * v[2]), 2 * (x * v[1] - y * v[0])]
    return [v[0] + w * t[0] + (y * t[2] - z * t[1]),
            v[1] + w * t[1] + (z * t[0] - x * t[2]),
            v[2] + w * t[2] + (x * t[1] - y * t[0])]


def parse_urdf(path: str, robot: str = None, bidirectional: bool = True) -> KinematicGraph:
    """Joints -> nodes, consecutive-joint links -> edges 

    Node 0 is the root link ("base", a fixed pseudo-joint). With `bidirectional`
    every edge also gets a reversed copy (inverse transform, reverse flag = 1) so
    messages reach the base and the fingertips see the arm. 
    """
    root = ET.parse(path).getroot()
    robot = robot or root.get("name", os.path.basename(path))
    joints = []
    for j in root.findall("joint"):
        o = j.find("origin")
        lim = j.find("limit")
        ax = j.find("axis")
        joints.append(dict(
            name=j.get("name"), type=j.get("type", "fixed"),
            parent=j.find("parent").get("link"), child=j.find("child").get("link"),
            xyz=_floats(o.get("xyz") if o is not None else None, 3),
            rpy=_floats(o.get("rpy") if o is not None else None, 3),
            lower=float(lim.get("lower", 0.0)) if lim is not None else 0.0,
            upper=float(lim.get("upper", 0.0)) if lim is not None else 0.0,
            velocity=float(lim.get("velocity", 0.0)) if lim is not None else 0.0,
            effort=float(lim.get("effort", 0.0)) if lim is not None else 0.0,
            axis=_floats(ax.get("xyz") if ax is not None else None, 3),
            mimic=j.find("mimic") is not None,
        ))
    links = [l.get("name") for l in root.findall("link")]
    children = {j["child"] for j in joints}
    roots = [l for l in links if l not in children]
    if len(roots) != 1:
        raise ValueError(f"{path}: expected one root link, got {roots}")
    root_link = roots[0]

    # node 0 = root link (base pseudo-joint); node k = joint k-1
    node_names = [root_link] + [j["name"] for j in joints]
    joint_of_child_link = {j["child"]: i + 1 for i, j in enumerate(joints)}
    joint_of_child_link[root_link] = 0
    parent_node = [None] + [joint_of_child_link[j["parent"]] for j in joints]

    depth = [0] * len(node_names)
    for i in range(1, len(node_names)):
        d, p = 0, parent_node[i]
        while p is not None and p != 0:
            d += 1
            p = parent_node[p]
        depth[i] = d + 1
    max_depth = max(1, max(depth))
    has_child = {parent_node[i] for i in range(1, len(node_names))}

    def is_gripper(name: str) -> bool:
        n = name.lower()
        return any(w in n for w in GRIPPER_WORDS)

    node_feats = []
    for i, name in enumerate(node_names):
        if i == 0:
            jt, lo, up, vel, eff, axis, mimic = "fixed", 0.0, 0.0, 0.0, 0.0, [0.0] * 3, False
        else:
            j = joints[i - 1]
            jt, lo, up, vel, eff, axis, mimic = (j["type"], j["lower"], j["upper"], j["velocity"],
                                                  j["effort"], j["axis"], j["mimic"])
        if jt == "continuous":
            lo, up = -math.pi, math.pi
        prismatic = jt == "prismatic"
        scale = 0.1 if prismatic else math.pi
        feats = [1.0 if jt == t else 0.0 for t in JOINT_TYPES]
        feats += [lo / scale, up / scale, (up - lo) / scale,
                  min(vel / (1.0 if prismatic else 10.0), 2.0), min(eff / 100.0, 2.0)]
        feats += axis if jt != "fixed" else [0.0] * 3
        feats += [depth[i] / max_depth, float(mimic), float(is_gripper(name) and i > 0),
                  float(i == 0), float(i not in has_child)]
        node_feats.append([float(x) for x in feats])

    edge_src, edge_dst, edge_feats = [], [], []
    for i in range(1, len(node_names)):
        j = joints[i - 1]
        p = parent_node[i]
        q = _rpy_to_quat(*j["rpy"])
        length = float(np.linalg.norm(j["xyz"]))
        edge_src.append(p); edge_dst.append(i)
        edge_feats.append([*j["xyz"], *q, length, 0.0])
        if bidirectional:
            qi = _quat_conj(q)
            t_inv = [-x for x in _quat_rotate(qi, j["xyz"])]
            edge_src.append(i); edge_dst.append(p)
            edge_feats.append([*t_inv, *qi, length, 1.0])

    return KinematicGraph(robot=robot, urdf=os.path.abspath(path), node_names=node_names,
                          node_feats=node_feats, edge_src=edge_src, edge_dst=edge_dst,
                          edge_feats=[[float(x) for x in e] for e in edge_feats],
                          node_feature_names=list(NODE_FEATURE_NAMES),
                          edge_feature_names=list(EDGE_FEATURE_NAMES))


def load_graphs(graph_dir: str, robots: List[str]) -> Dict[str, KinematicGraph]:
    return {r: KinematicGraph.load(os.path.join(graph_dir, f"{r}.json")) for r in robots}
