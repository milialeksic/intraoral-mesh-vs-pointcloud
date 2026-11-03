"""Integrated Gradients for a trained EdgeSegNet on one scan.

    python src/explain.py --run runs/mesh                      # first test scan, all predicted tooth vertices
    python src/explain.py --run runs/knn --target boundary     # only predicted gum-line vertices
    python src/explain.py --run runs/mesh --scan XXXX_upper --target vertex --vertex 123

Explained score: sum over the target vertices of (tooth logit - gingiva logit).
Baseline: all-zero input.

Fixed graphs: the model is run once on the real scan and the graph of every
layer is stored. All IG steps (and the baseline) reuse these graphs. For the
mesh model this changes nothing (its graph is always the mesh). For the knn
model it is necessary: otherwise the k-NN graphs of layers 2-3 change along the
path from zero to the scan, which makes the score jump and breaks IG. The
attributions therefore explain the prediction given the neighbours the model
actually used for this scan.

Outputs in <run>/explain/<scan>_<target>/ :
  prediction.ply   tooth (white) / gingiva (pink)
  errors.ply       correct (grey) / wrong (red)
  attr_total.ply   |attribution| heatmap, all 6 features
  attr_pos.ply     heatmap from position features only
  attr_normal.ply  heatmap from normal features only
  attributions.npy signed attributions [N, 6]
  summary.json     position vs normal share, completeness check
Open the .ply files in MeshLab.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import trimesh
from matplotlib import colormaps

from models import build_model, node_features
from train import load_split

TOOTH = np.array([0.95, 0.95, 0.88])
GINGIVA = np.array([0.85, 0.45, 0.50])
GREY = np.array([0.75, 0.75, 0.75])
RED = np.array([0.90, 0.10, 0.10])


def integrated_gradients(fn, x: torch.Tensor, baseline: torch.Tensor | None = None,
                         steps: int = 64) -> torch.Tensor:
    """IG with the midpoint rule: (x - baseline) * mean gradient along the straight path.
    fn maps an input like x to a scalar."""
    if baseline is None:
        baseline = torch.zeros_like(x)
    grad_sum = torch.zeros_like(x)
    for a in (torch.arange(steps, dtype=x.dtype, device=x.device) + 0.5) / steps:
        inp = (baseline + a * (x - baseline)).detach().requires_grad_(True)
        (g,) = torch.autograd.grad(fn(inp), inp)
        grad_sum += g
    return (x - baseline) * grad_sum / steps


def select_targets(pred: torch.Tensor, edge_index: torch.Tensor, mode: str, vertex: int | None = None):
    """region: predicted tooth vertices; boundary: predicted tooth vertices with a predicted-gingiva
    mesh neighbour; vertex: a single vertex."""
    if mode == "region":
        mask = pred == 1
    elif mode == "boundary":
        src, dst = edge_index
        mask = torch.zeros_like(pred, dtype=torch.bool)
        mask[src[(pred[src] == 1) & (pred[dst] == 0)]] = True
    elif mode == "vertex":
        if vertex is None or not 0 <= vertex < len(pred):
            raise ValueError("--vertex must be a valid vertex index")
        mask = torch.zeros_like(pred, dtype=torch.bool)
        mask[vertex] = True
    else:
        raise ValueError(f"unknown target mode {mode!r}")
    if not mask.any():
        raise ValueError(f"no vertices selected for target {mode!r}")
    return mask


def heat_colors(values: np.ndarray, clip_percentile: float = 99.0) -> np.ndarray:
    hi = np.percentile(values, clip_percentile)
    v = np.clip(values / hi, 0, 1) if hi > 0 else np.zeros_like(values)
    return colormaps["inferno"](v)[:, :3]


def export_ply(path: Path, vertices: np.ndarray, faces: np.ndarray, colors: np.ndarray) -> None:
    rgba = np.concatenate([(colors * 255).astype(np.uint8),
                           np.full((len(colors), 1), 255, np.uint8)], axis=1)
    trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=rgba, process=False).export(path)


def load_model(ckpt_path: Path, device):
    ckpt = torch.load(ckpt_path, weights_only=False, map_location=device)
    model = build_model(ckpt["config"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    return model.eval()


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True, help="training run folder containing best.pt")
    p.add_argument("--data", type=Path, default=Path("data/processed"))
    p.add_argument("--scan", default=None, help="scan id; default: first test scan")
    p.add_argument("--target", choices=["region", "boundary", "vertex"], default="region")
    p.add_argument("--vertex", type=int, default=None)
    p.add_argument("--steps", type=int, default=64)
    p.add_argument("--out", type=Path, default=None, help="default: <run>/explain/<scan>_<target>")
    p.add_argument("--device", default="auto")
    return p.parse_args(argv)


def main(argv=None) -> dict:
    args = parse_args(argv)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    model = load_model(args.run / "best.pt", device)
    scan = args.scan or load_split(args.data)["test"][0]
    data = torch.load(args.data / f"{scan}.pt", weights_only=False)
    out = args.out or args.run / "explain" / f"{scan}_{args.target}"
    out.mkdir(parents=True, exist_ok=True)

    x = node_features(data).to(device)
    edge_index = data.edge_index.to(device)
    batch = torch.zeros(len(x), dtype=torch.long, device=device)

    # one normal forward pass on the real scan: prediction + the graph each layer used
    with torch.no_grad():
        logits, graphs = model(x, edge_index, batch, return_graphs=True)
    pred = logits.argmax(1)
    mask = select_targets(pred, edge_index, args.target, args.vertex)

    def score(inp):
        logits = model(inp, edge_index, batch, graphs=graphs)  # same graphs at every IG step
        return (logits[mask, 1] - logits[mask, 0]).sum()

    attr = integrated_gradients(score, x, steps=args.steps)
    with torch.no_grad():
        expected = (score(x) - score(torch.zeros_like(x))).item()
    total = attr.sum().item()

    a = attr.detach().cpu().numpy()
    abs_pos, abs_nrm = np.abs(a[:, :3]).sum(1), np.abs(a[:, 3:]).sum(1)
    pos_share = float(abs_pos.sum() / max(abs_pos.sum() + abs_nrm.sum(), 1e-12))

    # export meshes in original scan coordinates
    verts = (data.pos * data.scale + data.center).numpy()
    faces = data.face.T.numpy()
    y, p = data.y.numpy(), pred.cpu().numpy()
    export_ply(out / "prediction.ply", verts, faces, np.where(p[:, None] == 1, TOOTH, GINGIVA))
    export_ply(out / "errors.ply", verts, faces, np.where((p == y)[:, None], GREY, RED))
    export_ply(out / "attr_total.ply", verts, faces, heat_colors(abs_pos + abs_nrm))
    export_ply(out / "attr_pos.ply", verts, faces, heat_colors(abs_pos))
    export_ply(out / "attr_normal.ply", verts, faces, heat_colors(abs_nrm))
    np.save(out / "attributions.npy", a)

    summary = {
        "scan": scan,
        "graph": model.graph,
        "target": args.target,
        "target_vertices": int(mask.sum().item()),
        "graphs_fixed_to_real_scan": True,
        "position_share": pos_share,
        "normal_share": 1 - pos_share,
        "completeness": {"sum_of_attributions": total, "score_minus_baseline": expected,
                         "relative_error": abs(total - expected) / max(abs(expected), 1e-12)},
        "accuracy_on_scan": float((p == y).mean()),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Scan {scan} ({model.graph} model), target '{args.target}': {summary['target_vertices']} vertices")
    print(f"Attribution share: position {pos_share:.1%}, normals {1 - pos_share:.1%}")
    print(f"Completeness: attributions sum {total:.3f} vs score difference {expected:.3f} "
          f"(relative error {summary['completeness']['relative_error']:.2%}; small = IG is reliable, "
          f"raise --steps if large)")
    print(f"Outputs in {out}")
    return summary


if __name__ == "__main__":
    main()