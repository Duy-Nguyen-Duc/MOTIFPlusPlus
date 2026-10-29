import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from model.kinematic_graph import parse_urdf  # noqa: E402

# ManiSkill URDFs of the demo robots (panda_v2, xarm6_robotiq, wxai_base); override with --urdf robot=/path.
MS = os.environ.get("MANISKILL_ASSETS",
                    "/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/cross_emb/NewTest/ManiSkill/mani_skill/assets/robots")
MS_DATA = os.environ.get("MANISKILL_DATA", "/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/.maniskill/data/robots")
DEFAULT_URDFS = {
    "panda": f"{MS}/panda/panda_v2.urdf",
    "xarm6": f"{MS}/xarm6/xarm6_robotiq.urdf",
    "widowxai": f"{MS_DATA}/widowxai/wxai_base.urdf",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="assets/kinematic_graphs")
    ap.add_argument("--urdf", action="append", default=[], help="robot=/path/to.urdf (repeatable)")
    ap.add_argument("--directed", action="store_true", help="no reversed edges (paper-literal graph)")
    args = ap.parse_args()
    urdfs = dict(DEFAULT_URDFS)
    for spec in args.urdf:
        r, p = spec.split("=", 1)
        urdfs[r] = p
    os.makedirs(args.out, exist_ok=True)
    for robot, path in urdfs.items():
        g = parse_urdf(path, robot=robot, bidirectional=not args.directed)
        g.save(os.path.join(args.out, f"{robot}.json"))
        movable = sum(1 for f in g.node_feats if f[3] == 0.0)
        grip = sum(1 for f in g.node_feats if f[-3] == 1.0)
        print(f"{robot:9s} nodes {g.num_nodes:3d} (movable {movable:2d}, gripper-named {grip:2d})  "
              f"edges {len(g.edge_src):3d}  <- {path}")
        print("          ", " ".join(g.node_names))


if __name__ == "__main__":
    main()
