"""EdgeConv segmentation network with two graph definitions.

graph="knn"  -> DGCNN-style: neighbours = k nearest points in the current
               feature space, recomputed before every layer (dynamic graph)
graph="mesh" -> neighbours = vertices connected by a mesh edge (fixed graph)

Everything else (layers, sizes, head) is identical, so the two models differ
only in how the graph is defined.

The k-NN search is plain PyTorch (chunked torch.cdist + topk), so the
torch_cluster package is not needed.

For explainability, forward() can return the graph each layer used
(return_graphs=True) and can reuse given graphs instead of building new ones
(graphs=[...]). Training uses neither, so it behaves exactly as before.
"""
import torch
from torch import nn
from torch_geometric.nn import MLP, EdgeConv, global_max_pool

GRAPHS = ("knn", "mesh")


@torch.no_grad()
def knn_graph(x: torch.Tensor, k: int, batch: torch.Tensor, chunk_size: int = 2048) -> torch.Tensor:
    """k nearest neighbours of every node, computed separately for each graph in the batch.

    Returns edge_index [2, N*k] with row 0 = neighbour (source), row 1 = centre node
    (target), the convention EdgeConv uses for messages j -> i. Each node is its own
    nearest neighbour (distance 0), as in DGCNN. `batch` must be sorted (PyG does this).
    Distances are computed in chunks of `chunk_size` query nodes to limit memory.
    """
    if batch.numel() > 1 and not bool((batch[1:] >= batch[:-1]).all()):
        raise ValueError("batch vector must be sorted")
    counts = torch.bincount(batch)
    starts = torch.cumsum(counts, 0) - counts
    sources, targets = [], []
    for start, n in zip(starts.tolist(), counts.tolist()):
        if n == 0:
            continue
        xb = x[start:start + n]
        kk = min(k, n)
        for s in range(0, n, chunk_size):
            q = xb[s:s + chunk_size]
            nbr = torch.cdist(q, xb).topk(kk, dim=1, largest=False).indices  # [len(q), kk]
            centre = torch.arange(s, s + len(q), device=x.device).unsqueeze(1).expand(-1, kk)
            sources.append(nbr.reshape(-1) + start)
            targets.append(centre.reshape(-1) + start)
    return torch.stack([torch.cat(sources), torch.cat(targets)])


class EdgeSegNet(nn.Module):
    """Per-vertex segmentation: EdgeConv layers -> global max-pooled feature -> MLP head."""

    def __init__(self, graph: str = "mesh", in_channels: int = 6, num_classes: int = 2,
                 hidden: int = 64, num_layers: int = 3, k: int = 20, emb_dim: int = 512,
                 dropout: float = 0.5, knn_chunk: int = 2048):
        super().__init__()
        if graph not in GRAPHS:
            raise ValueError(f"graph must be one of {GRAPHS}, got {graph!r}")
        self.graph, self.k, self.knn_chunk = graph, k, knn_chunk
        self._config = dict(graph=graph, in_channels=in_channels, num_classes=num_classes,
                            hidden=hidden, num_layers=num_layers, k=k, emb_dim=emb_dim,
                            dropout=dropout, knn_chunk=knn_chunk)

        # EdgeConv: h_i' = max_j MLP([h_i, h_j - h_i])
        in_dims = [in_channels] + [hidden] * (num_layers - 1)
        self.convs = nn.ModuleList(
            EdgeConv(MLP([2 * c, hidden, hidden], plain_last=False), aggr="max") for c in in_dims
        )
        cat_dim = hidden * num_layers
        self.global_mlp = MLP([cat_dim, emb_dim], plain_last=False)
        self.head = MLP([cat_dim + emb_dim, 256, 128, num_classes], dropout=dropout)

    def config(self) -> dict:
        return dict(self._config)

    def _edges(self, h: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        if self.graph == "knn":
            return knn_graph(h, self.k, batch, self.knn_chunk)  # recomputed for every layer
        return edge_index

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor,
                graphs: list[torch.Tensor] | None = None, return_graphs: bool = False):
        """x: [N, in_channels] node features; returns logits [N, num_classes].
        edge_index (mesh edges) is ignored by the knn model.

        graphs: optional list with one edge_index per layer; when given, these are used
            instead of building the graphs (so the knn graph does not change with x).
        return_graphs: if True, returns (logits, graphs actually used by each layer).
        """
        if graphs is not None and len(graphs) != len(self.convs):
            raise ValueError(f"expected {len(self.convs)} graphs, got {len(graphs)}")
        feats, used, h = [], [], x
        for i, conv in enumerate(self.convs):
            e = graphs[i] if graphs is not None else self._edges(h, edge_index, batch)
            used.append(e)
            h = conv(h, e)
            feats.append(h)
        h = torch.cat(feats, dim=1)                                  # local features, all layers
        g = global_max_pool(self.global_mlp(h), batch)[batch]        # whole-scan summary per node
        logits = self.head(torch.cat([h, g], dim=1))
        return (logits, used) if return_graphs else logits


def node_features(data) -> torch.Tensor:
    """Input features per vertex: position (3) + normal (3)."""
    return torch.cat([data.pos, data.normal], dim=1)


def build_model(config: dict) -> EdgeSegNet:
    return EdgeSegNet(**config)