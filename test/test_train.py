"""Tests for train.py: metrics, augmentation, dataset and a tiny end-to-end run."""
import json

import pytest
import torch

import train


def test_confusion_and_metrics():
    target = torch.tensor([0, 0, 1, 1])
    pred = torch.tensor([0, 1, 1, 1])
    cm = train.confusion(pred, target)
    assert cm.tolist() == [[1, 1], [0, 2]]
    m = train.metrics_from_confusion(cm)
    assert m["iou_gingiva"] == pytest.approx(1 / 2)   # 1 / (1 + 1 + 0)
    assert m["iou_tooth"] == pytest.approx(2 / 3)     # 2 / (2 + 1 + 0)
    assert m["miou"] == pytest.approx((1 / 2 + 2 / 3) / 2)
    assert m["acc"] == pytest.approx(3 / 4)


def test_perfect_prediction():
    y = torch.tensor([0, 1, 1, 0, 1])
    m = train.metrics_from_confusion(train.confusion(y, y))
    assert m["miou"] == pytest.approx(1.0) and m["acc"] == pytest.approx(1.0)


def test_boundary_mask_path_graph():
    # 0 - 1 - 2 - 3 with labels 0 0 1 1 -> boundary vertices are 1 and 2
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]])
    y = torch.tensor([0, 0, 1, 1])
    assert train.boundary_mask(y, edge_index).tolist() == [False, True, True, False]


def test_random_rotation_is_a_rotation():
    r = torch.from_numpy(train.random_rotation(10.0))
    assert torch.allclose(r @ r.T, torch.eye(3, dtype=r.dtype), atol=1e-10)
    assert torch.det(r).item() == pytest.approx(1.0)


def test_augment_keeps_labels_and_unit_normals_and_original(processed_dir):
    d = torch.load(next(processed_dir.glob("*_upper.pt")), weights_only=False)
    original_pos = d.pos.clone()
    a = train.augment(d)
    assert torch.equal(a.y, d.y)
    assert torch.equal(a.edge_index, d.edge_index)
    assert not torch.allclose(a.pos, d.pos)
    assert torch.allclose(a.normal.norm(dim=1), d.normal.norm(dim=1), atol=1e-5)
    assert torch.equal(d.pos, original_pos)  # the stored scan itself is not modified


def test_dataset_uses_split(processed_dir):
    split = train.load_split(processed_dir)
    ds = train.ScanDataset(processed_dir, split["test"])
    assert len(ds) == len(split["test"])
    assert ds[0].scan_id in split["test"]


@pytest.mark.parametrize("graph", ["knn", "mesh"])
def test_end_to_end_training(trained_runs, graph):
    base = trained_runs[graph]
    assert (base / "run" / "best.pt").exists()
    results = json.loads((base / "results" / f"{graph}_metrics.json").read_text())
    history = json.loads((base / "results" / f"{graph}_history.json").read_text())

    assert results["graph"] == graph
    assert len(history) == 2
    for key in ("iou_gingiva", "iou_tooth", "miou", "acc", "boundary_acc"):
        assert 0.0 <= results["test"][key] <= 1.0
    assert results["scans"] == {"train": 6, "val": 2, "test": 2}


def test_checkpoint_can_be_reloaded(trained_runs, processed_dir):
    ckpt = torch.load(trained_runs["mesh"] / "run" / "best.pt", weights_only=False)
    model = train.build_model(ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])
    loader = train.DataLoader(train.ScanDataset(processed_dir, train.load_split(processed_dir)["val"]),
                              batch_size=2)
    metrics = train.evaluate(model, loader, torch.device("cpu"))
    assert metrics["miou"] == pytest.approx(ckpt["val"]["miou"])  # same weights, same data -> same score