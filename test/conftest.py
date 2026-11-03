"""Shared test helpers: synthetic scans that look like Teeth3DS+ files.

A synthetic scan is a sphere mesh with a random scale and offset; vertices in its
upper half are labelled as a tooth (FDI 11), the lower half as gingiva (0).
Everything runs on CPU in seconds, no real data needed.

Run all tests from the repo root:  pytest tests -q
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import trimesh

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))


def write_obj(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    """Plain .obj writer (v / f lines, 1-based faces), so vertex order is exactly ours."""
    lines = [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in vertices]
    lines += [f"f {a + 1} {b + 1} {c + 1}" for a, b, c in faces]
    path.write_text("\n".join(lines) + "\n")


def make_synthetic_scan(folder: Path, name: str, subdivisions: int = 3, seed: int = 0) -> dict:
    """Write <name>_upper.obj + .json into folder. Returns the generation parameters."""
    sphere = trimesh.creation.icosphere(subdivisions=subdivisions)
    rng = np.random.default_rng(seed)
    scale = rng.uniform(5, 10)
    offset = rng.normal(0, 20, 3)
    vertices = sphere.vertices * scale + offset
    labels = np.where(sphere.vertices[:, 2] > 0, 11, 0)

    folder.mkdir(parents=True, exist_ok=True)
    obj = folder / f"{name}_upper.obj"
    write_obj(obj, vertices, sphere.faces)
    obj.with_suffix(".json").write_text(json.dumps(
        {"id_patient": name, "jaw": "upper", "labels": labels.tolist()}))
    return {"obj": obj, "scale": scale, "offset": offset, "labels": labels}


@pytest.fixture
def synthetic_scan():
    """The make_synthetic_scan function, for tests that need their own scan.
    (Tests should not import from conftest directly; pytest provides fixtures instead.)"""
    return make_synthetic_scan


@pytest.fixture(scope="session")
def raw_dir(tmp_path_factory) -> Path:
    """10 synthetic scans in a raw-data folder."""
    raw = tmp_path_factory.mktemp("raw")
    for i in range(10):
        make_synthetic_scan(raw / f"patient_{i}", f"P{i:03d}", seed=i)
    return raw


@pytest.fixture(scope="session")
def processed_dir(raw_dir, tmp_path_factory) -> Path:
    """Synthetic scans run through preprocess.py (+ split.json)."""
    import preprocess

    out = tmp_path_factory.mktemp("processed")
    preprocess.main(["--raw", str(raw_dir), "--out", str(out), "--target-faces", "600"])
    return out


TINY_MODEL_ARGS = ["--hidden", "16", "--emb", "32", "--k", "8", "--layers", "2"]


@pytest.fixture(scope="session")
def trained_runs(processed_dir, tmp_path_factory) -> dict:
    """A tiny model of each kind, trained for 2 epochs on CPU."""
    import train

    runs = {}
    for graph in ("knn", "mesh"):
        base = tmp_path_factory.mktemp(f"run_{graph}")
        train.main(["--graph", graph, "--data", str(processed_dir), "--out", str(base / "run"),
                    "--results-dir", str(base / "results"), "--epochs", "2", "--batch-size", "2",
                    "--device", "cpu", *TINY_MODEL_ARGS])
        runs[graph] = base
    return runs