"""Open a Teeth3DS+ scan in a 3D viewer, coloured by its labels.

Examples:
    python visualize_scan.py                       # first upper-jaw scan found in data/raw
    python visualize_scan.py --mode teeth          # one colour per tooth
    python visualize_scan.py data/raw/.../XXXX_upper.obj

Viewer controls: left-drag rotate, scroll zoom, ctrl+left-drag pan, Q to close.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh

GINGIVA = np.array([0.85, 0.45, 0.50])
TOOTH = np.array([0.95, 0.95, 0.88])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("obj", nargs="?", type=Path, help="path to a .obj scan")
    p.add_argument("--root", type=Path, default=Path("data/raw"))
    p.add_argument("--mode", choices=["binary", "teeth"], default="binary",
                   help="binary: tooth vs gingiva; teeth: one colour per tooth (FDI label)")
    args = p.parse_args()

    obj = args.obj or next(iter(sorted(args.root.rglob("*_upper.obj"))), None)
    if obj is None:
        raise SystemExit(f"No *_upper.obj found under {args.root}")

    # process=False keeps the vertex order identical to the file, so labels line up
    mesh = trimesh.load(obj, process=False, force="mesh")
    labels = np.asarray(json.loads(obj.with_suffix(".json").read_text())["labels"])
    if len(labels) != len(mesh.vertices):
        raise SystemExit(f"{len(labels)} labels but {len(mesh.vertices)} vertices")

    tooth_ids = sorted(set(labels.tolist()) - {0})
    print(f"Scan:     {obj.name}")
    print(f"Vertices: {len(mesh.vertices):,}   Faces: {len(mesh.faces):,}")
    print(f"Teeth:    {len(tooth_ids)}  (FDI labels {tooth_ids})")
    print(f"Tooth vertices: {(labels > 0).mean():.1%}   Gingiva: {(labels == 0).mean():.1%}")

    colors = np.tile(GINGIVA, (len(labels), 1))
    if args.mode == "binary":
        colors[labels > 0] = TOOTH
    else:
        rng = np.random.default_rng(0)
        for t in tooth_ids:
            colors[labels == t] = rng.uniform(0.2, 1.0, 3)

    m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(mesh.vertices),
                                  o3d.utility.Vector3iVector(mesh.faces))
    m.vertex_colors = o3d.utility.Vector3dVector(colors)
    m.compute_vertex_normals()  # needed for shading
    o3d.visualization.draw_geometries([m], window_name=obj.name, mesh_show_back_face=True)


if __name__ == "__main__":
    main()