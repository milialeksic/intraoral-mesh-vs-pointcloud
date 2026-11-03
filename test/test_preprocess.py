"""Tests for preprocess.py: edges, label transfer, normalization, split."""
import json

import numpy as np
import pytest
import torch
import trimesh

import preprocess


def test_faces_to_edges_single_triangle():
    edges = preprocess.faces_to_edges(np.array([[0, 1, 2]]))
    pairs = set(map(tuple, edges.T.tolist()))
    assert pairs == {(0, 1), (1, 0), (1, 2), (2, 1), (0, 2), (2, 0)}


def test_faces_to_edges_shared_edge_not_duplicated():
    # two triangles sharing edge 1-2 -> 5 undirected edges -> 10 directed
    edges = preprocess.faces_to_edges(np.array([[0, 1, 2], [1, 3, 2]]))
    assert edges.shape == (2, 10)


def test_simplify_reduces_faces_and_returns_unit_normals():
    sphere = trimesh.creation.icosphere(subdivisions=4)  # 5120 faces
    v, f, n = preprocess.simplify(sphere.vertices, sphere.faces, target_faces=1000)
    assert len(f) <= 1000
    assert f.max() < len(v)
    assert np.allclose(np.linalg.norm(n, axis=1), 1, atol=1e-4)


def test_process_scan(tmp_path, synthetic_scan):
    info = synthetic_scan(tmp_path, "S001", subdivisions=4)
    d = preprocess.process_scan(info["obj"], target_faces=1000)
    n = d.num_nodes

    # shapes and types
    assert d.pos.shape == (n, 3) and d.normal.shape == (n, 3) and d.y.shape == (n,)
    assert d.face.shape[0] == 3 and d.face.shape[1] <= 1000
    assert set(d.y.unique().tolist()) <= {0, 1}

    # normalization: centred, inside the unit sphere, farthest point at distance 1
    assert torch.allclose(d.pos.mean(0), torch.zeros(3), atol=1e-5)
    assert d.pos.norm(dim=1).max().item() == pytest.approx(1.0, abs=1e-5)

    # positions can be mapped back to the original coordinates
    original = d.pos * d.scale + d.center
    radius = (original - torch.tensor(info["offset"]).float()).norm(dim=1)
    assert torch.allclose(radius, torch.full_like(radius, info["scale"]), rtol=0.05)

    # label transfer: tooth = upper half of the sphere (allow a few errors at the boundary)
    upper = (original[:, 2] > info["offset"][2]).long()
    assert (d.y == upper).float().mean().item() > 0.95

    # mesh edges: valid indices, symmetric, no self-loops
    src, dst = d.edge_index
    assert d.edge_index.max().item() < n
    assert set(zip(src.tolist(), dst.tolist())) == set(zip(dst.tolist(), src.tolist()))
    assert not (src == dst).any()


def test_process_scan_label_count_mismatch(tmp_path, synthetic_scan):
    info = synthetic_scan(tmp_path, "BAD")
    js = info["obj"].with_suffix(".json")
    data = json.loads(js.read_text())
    data["labels"] = data["labels"][:-5]
    js.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="labels"):
        preprocess.process_scan(info["obj"], target_faces=600)


def test_make_split_is_deterministic_disjoint_and_complete():
    names = [f"scan_{i}" for i in range(40)]
    a = preprocess.make_split(names, seed=42, val=0.15, test=0.15)
    b = preprocess.make_split(list(reversed(names)), seed=42, val=0.15, test=0.15)
    assert a == b
    parts = [set(a["train"]), set(a["val"]), set(a["test"])]
    assert set().union(*parts) == set(names)
    assert sum(len(p) for p in parts) == len(names)
    assert len(a["test"]) == 6 and len(a["val"]) == 6


def test_main_writes_files_and_split(processed_dir):
    files = sorted(processed_dir.glob("*_upper.pt"))
    assert len(files) == 10
    split = json.loads((processed_dir / "split.json").read_text())
    assert sorted(split["train"] + split["val"] + split["test"]) == [f.stem for f in files]