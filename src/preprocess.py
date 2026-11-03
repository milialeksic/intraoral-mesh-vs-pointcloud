"""Preprocess Teeth3DS+ upper-jaw scans into graphs for tooth/gingiva segmentation.

For every *_upper.obj (with its .json labels) under --raw:
  1. load mesh + per-vertex labels (vertex order preserved)
  2. simplify the mesh to --target-faces triangles (quadric decimation)
  3. transfer labels to the new vertices (nearest original vertex), make them binary
  4. compute vertex normals on the simplified mesh
  5. normalize: center at 0, scale into the unit sphere
  6. build undirected graph edges from the triangles
  7. save one .pt file per scan (torch_geometric Data)
Finally writes split.json (train/val/test, fixed seed) used by both models.

Example:
    python preprocess.py                      # data/raw -> data/processed
    python preprocess.py --target-faces 40000 --limit 5
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import open3d as o3d
import torch
import trimesh
from scipy.spatial import cKDTree
from torch_geometric.data import Data
from tqdm import tqdm


def find_scans(raw: Path, jaw: str) -> list[Path]:
    """All meshes of the given jaw that have a label file next to them."""
    return [o for o in sorted(raw.rglob(f"*_{jaw}.obj")) if o.with_suffix(".json").exists()]


def simplify(vertices: np.ndarray, faces: np.ndarray, target_faces: int):
    """Quadric decimation + cleanup. Returns vertices, faces, vertex normals."""
    m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices),
                                  o3d.utility.Vector3iVector(faces))
    if len(faces) > target_faces:
        m = m.simplify_quadric_decimation(target_number_of_triangles=target_faces)
    m.remove_degenerate_triangles()
    m.remove_duplicated_triangles()
    m.remove_duplicated_vertices()
    m.remove_unreferenced_vertices()
    m.compute_vertex_normals()
    v = np.asarray(m.vertices, dtype=np.float32)
    f = np.asarray(m.triangles, dtype=np.int64)
    n = np.nan_to_num(np.asarray(m.vertex_normals, dtype=np.float32))
    return v, f, n


def faces_to_edges(faces: np.ndarray) -> torch.Tensor:
    """Triangle (a, b, c) -> edges a-b, b-c, c-a, in both directions, without duplicates."""
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    e = np.concatenate([e, e[:, ::-1]])
    e = np.unique(e, axis=0)
    return torch.from_numpy(e.T.copy()).long()  # shape [2, num_edges]


def process_scan(obj: Path, target_faces: int) -> Data:
    # 1. load (process=False keeps vertex order identical to the file, so labels line up)
    mesh = trimesh.load(obj, process=False, force="mesh")
    v0 = np.asarray(mesh.vertices, dtype=np.float64)
    f0 = np.asarray(mesh.faces, dtype=np.int64)
    labels0 = np.asarray(json.loads(obj.with_suffix(".json").read_text())["labels"])
    if len(labels0) != len(v0):
        raise ValueError(f"{len(labels0)} labels but {len(v0)} vertices")

    # 2. + 4. simplify and compute normals
    v, f, n = simplify(v0, f0, target_faces)

    # 3. label transfer: each new vertex takes the label of the closest original vertex
    _, nearest = cKDTree(v0).query(v)
    y = (labels0[nearest] > 0).astype(np.int64)  # tooth = 1, gingiva = 0

    # 5. normalize positions (normals are directions, they stay unchanged)
    center = v.mean(axis=0)
    v = v - center
    scale = np.linalg.norm(v, axis=1).max()
    v = v / scale

    # 6. mesh edges
    edge_index = faces_to_edges(f)

    return Data(
        pos=torch.from_numpy(v).float(),
        normal=torch.from_numpy(n).float(),
        y=torch.from_numpy(y),
        edge_index=edge_index,
        face=torch.from_numpy(f.T.copy()).long(),  # [3, num_faces], for exporting heatmaps
        center=torch.from_numpy(center).float(),
        scale=torch.tensor(float(scale)),
        scan_id=obj.stem,
        num_nodes=len(v),
    )


def make_split(names: list[str], seed: int, val: float, test: float) -> dict:
    names = sorted(names)
    random.Random(seed).shuffle(names)
    n_test = round(len(names) * test)
    n_val = round(len(names) * val)
    return {
        "seed": seed,
        "test": sorted(names[:n_test]),
        "val": sorted(names[n_test:n_test + n_val]),
        "train": sorted(names[n_test + n_val:]),
    }


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--raw", type=Path, default=Path("data/raw"))
    p.add_argument("--out", type=Path, default=Path("data/processed"))
    p.add_argument("--jaw", choices=["upper", "lower"], default="upper")
    p.add_argument("--target-faces", type=int, default=20000)
    p.add_argument("--limit", type=int, default=0, help="process only the first N scans (for testing)")
    p.add_argument("--force", action="store_true", help="re-process scans that already have a .pt file")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val", type=float, default=0.15)
    p.add_argument("--test", type=float, default=0.15)
    args = p.parse_args(argv)

    scans = find_scans(args.raw, args.jaw)
    if args.limit:
        scans = scans[:args.limit]
    if not scans:
        raise SystemExit(f"No labelled *_{args.jaw}.obj files under {args.raw}")
    args.out.mkdir(parents=True, exist_ok=True)

    failed = []
    for obj in tqdm(scans, desc="Preprocessing"):
        out_file = args.out / f"{obj.stem}.pt"
        if out_file.exists() and not args.force:
            continue
        try:
            torch.save(process_scan(obj, args.target_faces), out_file)
        except Exception as e:  # keep going, report at the end
            failed.append((obj.name, str(e)))

    # summary over everything in the output folder
    files = sorted(args.out.glob(f"*_{args.jaw}.pt"))
    data = [torch.load(f, weights_only=False) for f in files]
    if not data:
        raise SystemExit("Nothing was processed successfully.")
    n_vert = np.array([d.num_nodes for d in data])
    n_edge = np.array([d.edge_index.shape[1] // 2 for d in data])
    tooth = np.concatenate([d.y.numpy() for d in data]).mean()
    print(f"\nScans processed: {len(data)}   failed: {len(failed)}")
    print(f"Vertices per scan: mean {n_vert.mean():,.0f}  (min {n_vert.min():,}, max {n_vert.max():,})")
    print(f"Mesh edges per scan: mean {n_edge.mean():,.0f}")
    print(f"Tooth vertices: {tooth:.1%}   Gingiva: {1 - tooth:.1%}")
    for name, err in failed:
        print(f"  failed: {name}: {err}")

    # split over all successfully processed scans
    split = make_split([f.stem for f in files], args.seed, args.val, args.test)
    (args.out / "split.json").write_text(json.dumps(split, indent=2))
    print(f"Split: {len(split['train'])} train / {len(split['val'])} val / {len(split['test'])} test "
          f"-> {args.out / 'split.json'}")


if __name__ == "__main__":
    main()