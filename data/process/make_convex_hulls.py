"""Convex collision hulls (<mesh>.convex.stl) for a robot asset whose download ships none.

mplib 0.1.1 (the pinned motion planner) loads every URDF collision mesh as `<filename>.convex.stl` and does not create the
file itself. The Panda (repository) and xArm6 (asset zip) ship these hulls; the WidowX AI asset
(TrossenRobotics/ManiSkill-WidowX_AI v0.1.0) does not, so its motion-planning demos fail with
"Could not load resource .../base_link.stl.convex.stl". This writes the missing hulls next to the meshes, as trimesh
convex hulls of the unscaled mesh (the URDF scale is applied by the loader). Hull vertices are first snapped to a
`--merge` grid (1e-4 m): the planner's STL loader merges vertices closer than ~1e-5, and a hull with such near-duplicates
collapses into degenerate faces that FCL rejects ("The mesh is not watertight"). Idempotent.

    python data/process/make_convex_hulls.py --urdf ~/.maniskill/data/robots/widowxai/wxai_follower.urdf [--urdf ...]
"""
import argparse
import os
import xml.etree.ElementTree as ET

import numpy as np
import trimesh


def convex_hull(mesh: trimesh.Trimesh, merge: float) -> trimesh.Trimesh:
    pts = np.unique(np.round(mesh.vertices / merge) * merge, axis=0) if merge > 0 else mesh.vertices
    hull = trimesh.Trimesh(pts).convex_hull
    hull.merge_vertices(); hull.update_faces(hull.nondegenerate_faces()); hull.fix_normals()
    assert hull.is_watertight and hull.is_volume, "hull not watertight"
    return hull


def hulls_for_urdf(urdf: str, force: bool = False, merge: float = 1e-4) -> tuple[int, int]:
    root = ET.parse(urdf).getroot()
    base = os.path.dirname(os.path.abspath(urdf))
    written = skipped = 0
    for mesh in root.iter("mesh"):
        fn = mesh.get("filename", "")
        if fn.startswith("package://"):
            fn = fn.split("package://", 1)[1].split("/", 1)[1]
        src = fn if os.path.isabs(fn) else os.path.join(base, fn)
        dst = src + ".convex.stl"
        if os.path.exists(dst) and not force:
            skipped += 1
            continue
        hull = convex_hull(trimesh.load(src, force="mesh"), merge)
        hull.export(dst, file_type="stl")
        written += 1
        print(f"  {os.path.relpath(dst, base)}: {len(hull.vertices)} vertices")
    return written, skipped


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--urdf", action="append", required=True, help="URDF whose collision meshes need hulls (repeatable)")
    ap.add_argument("--force", action="store_true", help="rewrite existing hulls")
    ap.add_argument("--merge", type=float, default=1e-4, help="vertex snapping grid (model units) before the hull; 0 disables")
    args = ap.parse_args()
    for u in args.urdf:
        u = os.path.expanduser(u)
        w, s = hulls_for_urdf(u, args.force, args.merge)
        print(f"{u}: {w} hulls written, {s} already present")
