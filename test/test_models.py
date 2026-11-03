"""Tests for models.py: k-NN graph and the EdgeSegNet model."""
import pytest
import torch

from models import EdgeSegNet, build_model, knn_graph, node_features


def brute_force_neighbours(x, k):
    return torch.cdist(x, x).topk(k, dim=1, largest=False).indices


def neighbour_sets(edge_index, n):
    sets = [set() for _ in range(n)]
    for j, i in edge_index.T.tolist():
        sets[i].add(j)
    return sets


def test_knn_graph_matches_brute_force_per_graph():
    torch.manual_seed(0)
    x = torch.randn(50, 3)
    batch = torch.tensor([0] * 30 + [1] * 20)
    k = 5
    edges = neighbour_sets(knn_graph(x, k, batch), 50)

    for start, n in [(0, 30), (30, 20)]:
        expected = brute_force_neighbours(x[start:start + n], k) + start
        for i in range(n):
            assert edges[start + i] == set(expected[i].tolist())


def test_knn_graph_includes_self_and_stays_within_graph():
    torch.manual_seed(1)
    x = torch.randn(40, 6)
    batch = torch.tensor([0] * 25 + [1] * 15)
    ei = knn_graph(x, 4, batch)
    assert ei.shape == (2, 40 * 4)
    assert (batch[ei[0]] == batch[ei[1]]).all()          # never connects different scans
    for i, nbrs in enumerate(neighbour_sets(ei, 40)):
        assert i in nbrs                                  # own nearest neighbour


def test_knn_graph_chunking_gives_same_result():
    torch.manual_seed(2)
    x = torch.randn(100, 3)
    batch = torch.zeros(100, dtype=torch.long)
    a = neighbour_sets(knn_graph(x, 6, batch, chunk_size=7), 100)
    b = neighbour_sets(knn_graph(x, 6, batch, chunk_size=1000), 100)
    assert a == b


def test_knn_graph_small_graph_uses_all_points():
    x = torch.randn(3, 3)
    ei = knn_graph(x, 10, torch.zeros(3, dtype=torch.long))
    assert ei.shape == (2, 9)


def test_knn_graph_rejects_unsorted_batch():
    with pytest.raises(ValueError):
        knn_graph(torch.randn(4, 3), 2, torch.tensor([0, 1, 0, 1]))


def ring_edges(n, offset=0):
    """Simple ring graph i <-> i+1 as a stand-in for mesh edges."""
    i = torch.arange(n)
    j = (i + 1) % n
    return torch.stack([torch.cat([i, j]), torch.cat([j, i])]) + offset


def batch_of_two(n1=30, n2=20):
    torch.manual_seed(3)
    x = torch.randn(n1 + n2, 6)
    edge_index = torch.cat([ring_edges(n1), ring_edges(n2, n1)], dim=1)
    batch = torch.tensor([0] * n1 + [1] * n2)
    return x, edge_index, batch


@pytest.mark.parametrize("graph", ["knn", "mesh"])
def test_forward_shape_and_backward(graph):
    x, edge_index, batch = batch_of_two()
    model = EdgeSegNet(graph=graph, hidden=16, emb_dim=32, k=5)
    logits = model(x, edge_index, batch)
    assert logits.shape == (50, 2)
    logits.sum().backward()
    assert all(p.grad is not None for p in model.parameters())


def test_mesh_model_uses_edges_knn_model_ignores_them():
    x, edge_index, batch = batch_of_two()
    other_edges = edge_index[:, torch.randperm(edge_index.shape[1])[: edge_index.shape[1] // 2]]

    mesh = EdgeSegNet(graph="mesh", hidden=16, emb_dim=32).eval()
    knn = EdgeSegNet(graph="knn", hidden=16, emb_dim=32, k=5).eval()
    with torch.no_grad():
        assert not torch.allclose(mesh(x, edge_index, batch), mesh(x, other_edges, batch))
        assert torch.allclose(knn(x, edge_index, batch), knn(x, other_edges, batch))


def test_scans_in_a_batch_do_not_influence_each_other():
    # predictions for scan 0 must not change when scan 1 changes
    x, edge_index, batch = batch_of_two()
    for graph in ("knn", "mesh"):
        model = EdgeSegNet(graph=graph, hidden=16, emb_dim=32, k=5).eval()
        x2 = x.clone()
        x2[30:] += 5.0
        with torch.no_grad():
            a = model(x, edge_index, batch)[:30]
            b = model(x2, edge_index, batch)[:30]
        assert torch.allclose(a, b, atol=1e-5), graph


def test_config_roundtrip():
    model = EdgeSegNet(graph="knn", hidden=8, emb_dim=16, k=3, num_layers=2)
    clone = build_model(model.config())
    clone.load_state_dict(model.state_dict())  # same architecture -> loads cleanly
    assert clone.config() == model.config()


def test_invalid_graph_name():
    with pytest.raises(ValueError):
        EdgeSegNet(graph="voxel")


def test_node_features():
    class D:
        pos = torch.ones(4, 3)
        normal = torch.zeros(4, 3)
    f = node_features(D)
    assert f.shape == (4, 6) and (f[:, :3] == 1).all() and (f[:, 3:] == 0).all()