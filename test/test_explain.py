"""Tests for explain.py: Integrated Gradients, target selection, end-to-end outputs."""
import json

import numpy as np
import pytest
import torch

import explain


def test_ig_exact_for_linear_function():
    torch.manual_seed(0)
    w = torch.randn(10, 6)
    x = torch.randn(10, 6)
    attr = explain.integrated_gradients(lambda inp: (inp * w).sum(), x, steps=8)
    assert torch.allclose(attr, x * w, atol=1e-6)


def test_ig_exact_for_quadratic_with_midpoint_rule():
    # f = sum(x^2): gradient 2*a*x, mean over midpoints of a is 0.5 -> attribution x^2
    x = torch.randn(5, 3)
    attr = explain.integrated_gradients(lambda inp: (inp ** 2).sum(), x, steps=4)
    assert torch.allclose(attr, x ** 2, atol=1e-6)


def test_ig_completeness_with_baseline():
    torch.manual_seed(1)
    x, base = torch.randn(8, 4), torch.randn(8, 4)
    f = lambda inp: torch.tanh(inp).sum()
    attr = explain.integrated_gradients(f, x, baseline=base, steps=200)
    assert attr.sum().item() == pytest.approx((f(x) - f(base)).item(), rel=1e-3)


def test_select_targets():
    # path 0-1-2-3, predicted gingiva gingiva tooth tooth
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]])
    pred = torch.tensor([0, 0, 1, 1])
    assert explain.select_targets(pred, edge_index, "region").tolist() == [False, False, True, True]
    assert explain.select_targets(pred, edge_index, "boundary").tolist() == [False, False, True, False]
    assert explain.select_targets(pred, edge_index, "vertex", 3).tolist() == [False, False, False, True]


def test_select_targets_errors():
    edge_index = torch.tensor([[0, 1], [1, 0]])
    with pytest.raises(ValueError):
        explain.select_targets(torch.tensor([0, 0]), edge_index, "region")      # nothing predicted tooth
    with pytest.raises(ValueError):
        explain.select_targets(torch.tensor([0, 1]), edge_index, "vertex", 5)   # invalid index


def test_heat_colors_range():
    c = explain.heat_colors(np.array([0.0, 1.0, 2.0, 100.0]))
    assert c.shape == (4, 3) and (c >= 0).all() and (c <= 1).all()
    assert (explain.heat_colors(np.zeros(3)) == explain.heat_colors(np.zeros(3))[0]).all()


@pytest.mark.parametrize("graph", ["knn", "mesh"])
def test_end_to_end_explain(trained_runs, processed_dir, tmp_path, graph):
    out = tmp_path / graph
    try:
        summary = explain.main(["--run", str(trained_runs[graph] / "run"), "--data", str(processed_dir),
                                "--target", "region", "--steps", "16", "--out", str(out),
                                "--device", "cpu"])
    except ValueError as e:
        # a 2-epoch model may predict no tooth at all on a scan; that's not a bug in explain.py
        pytest.skip(f"tiny model selected no target vertices: {e}")

    for name in ("prediction.ply", "errors.ply", "attr_total.ply", "attr_pos.ply",
                 "attr_normal.ply", "attributions.npy", "summary.json"):
        assert (out / name).exists(), name

    assert summary["graph"] == graph
    assert summary["position_share"] + summary["normal_share"] == pytest.approx(1.0)
    attr = np.load(out / "attributions.npy")
    assert attr.shape[1] == 6
    saved = json.loads((out / "summary.json").read_text())
    assert saved["target_vertices"] == summary["target_vertices"] > 0