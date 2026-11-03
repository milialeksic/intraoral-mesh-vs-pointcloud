"""Train EdgeSegNet for tooth/gingiva segmentation on preprocessed Teeth3DS+ scans.

    python src/train.py --graph knn     # DGCNN (dynamic k-NN graph)
    python src/train.py --graph mesh    # mesh GNN (mesh edges)

Uses data/processed/split.json: trains on "train", picks the best epoch by
validation mIoU, then reports metrics on "test" once at the end.
Outputs: runs/<graph>/best.pt (checkpoint) and results/<graph>_metrics.json,
results/<graph>_history.json (small, meant to be committed).
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from models import EdgeSegNet, build_model, node_features

CLASSES = ("gingiva", "tooth")


# ---------------------------------------------------------------- data

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_split(data_dir: Path) -> dict:
    return json.loads((Path(data_dir) / "split.json").read_text())


def random_rotation(max_deg: float) -> np.ndarray:
    """Small random rotation: up to +-max_deg around each axis."""
    ax, ay, az = np.deg2rad(np.random.uniform(-max_deg, max_deg, 3))
    rx = np.array([[1, 0, 0], [0, np.cos(ax), -np.sin(ax)], [0, np.sin(ax), np.cos(ax)]])
    ry = np.array([[np.cos(ay), 0, np.sin(ay)], [0, 1, 0], [-np.sin(ay), 0, np.cos(ay)]])
    rz = np.array([[np.cos(az), -np.sin(az), 0], [np.sin(az), np.cos(az), 0], [0, 0, 1]])
    return rz @ ry @ rx


def augment(data, max_rot_deg: float = 10.0, scale_range=(0.9, 1.1), jitter: float = 0.002):
    """Random small rotation (positions and normals), scaling and jitter. Labels unchanged."""
    d = data.clone()
    r = torch.from_numpy(random_rotation(max_rot_deg)).float()
    s = float(np.random.uniform(*scale_range))
    d.pos = (d.pos @ r.T) * s + jitter * torch.randn_like(d.pos)
    d.normal = d.normal @ r.T
    return d


class ScanDataset(torch.utils.data.Dataset):
    """Preprocessed scans (.pt files), loaded into memory."""

    def __init__(self, data_dir: Path, names: list[str], augment: bool = False):
        self.items = [torch.load(Path(data_dir) / f"{n}.pt", weights_only=False) for n in names]
        self.augment = augment

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i):
        return augment(self.items[i]) if self.augment else self.items[i]


# ---------------------------------------------------------------- metrics

def confusion(pred: torch.Tensor, target: torch.Tensor, num_classes: int = 2) -> torch.Tensor:
    """cm[t, p] = number of vertices with true class t predicted as p."""
    idx = target * num_classes + pred
    return torch.bincount(idx, minlength=num_classes ** 2).reshape(num_classes, num_classes)


def metrics_from_confusion(cm: torch.Tensor) -> dict:
    cm = cm.double()
    tp = cm.diag()
    iou = tp / (cm.sum(0) + cm.sum(1) - tp).clamp(min=1)
    out = {f"iou_{c}": iou[i].item() for i, c in enumerate(CLASSES)}
    out["miou"] = iou.mean().item()
    out["acc"] = (tp.sum() / cm.sum().clamp(min=1)).item()
    return out


def boundary_mask(y: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """Vertices with at least one mesh neighbour of a different class (the gum line)."""
    src, dst = edge_index
    mask = torch.zeros_like(y, dtype=torch.bool)
    mask[src[y[src] != y[dst]]] = True
    return mask


@torch.no_grad()
def evaluate(model: EdgeSegNet, loader, device) -> dict:
    model.eval()
    cm = torch.zeros(2, 2, dtype=torch.long)
    b_correct = b_total = 0
    for d in loader:
        d = d.to(device)
        pred = model(node_features(d), d.edge_index, d.batch).argmax(1)
        cm += confusion(pred, d.y).cpu()
        b = boundary_mask(d.y, d.edge_index)
        b_correct += (pred[b] == d.y[b]).sum().item()
        b_total += b.sum().item()
    out = metrics_from_confusion(cm)
    out["boundary_acc"] = b_correct / max(b_total, 1)
    return out


# ---------------------------------------------------------------- training

def train_one_epoch(model, loader, optimizer, device) -> float:
    model.train()
    total, n = 0.0, 0
    for d in tqdm(loader, desc="train", leave=False):
        d = d.to(device)
        logits = model(node_features(d), d.edge_index, d.batch)
        loss = F.cross_entropy(logits, d.y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item() * d.num_graphs
        n += d.num_graphs
    return total / max(n, 1)


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--graph", choices=["knn", "mesh"], required=True)
    p.add_argument("--data", type=Path, default=Path("data/processed"))
    p.add_argument("--out", type=Path, default=None, help="default: runs/<graph>")
    p.add_argument("--results-dir", type=Path, default=Path("results"))
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--emb", type=int, default=512)
    p.add_argument("--k", type=int, default=20)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--knn-chunk", type=int, default=2048, help="lower this if the knn model runs out of memory")
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    args = p.parse_args(argv)
    args.out = args.out or Path("runs") / args.graph
    return args


def main(argv=None) -> dict:
    args = parse_args(argv)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    set_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    split = load_split(args.data)
    train_ds = ScanDataset(args.data, split["train"], augment=not args.no_augment)
    val_ds = ScanDataset(args.data, split["val"])
    test_ds = ScanDataset(args.data, split["test"])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)
    print(f"Device: {device}   scans: {len(train_ds)} train / {len(val_ds)} val / {len(test_ds)} test")

    model = EdgeSegNet(graph=args.graph, hidden=args.hidden, num_layers=args.layers, k=args.k,
                       emb_dim=args.emb, dropout=args.dropout, knn_chunk=args.knn_chunk).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.graph}   parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    history, best_miou, best_epoch, epoch_times = [], -1.0, -1, []
    ckpt_path = args.out / "best.pt"
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        loss = train_one_epoch(model, train_loader, optimizer, device)
        scheduler.step()
        epoch_times.append(time.time() - t0)
        val = evaluate(model, val_loader, device)
        history.append({"epoch": epoch, "train_loss": loss, **{f"val_{k}": v for k, v in val.items()}})
        improved = val["miou"] > best_miou
        if improved:
            best_miou, best_epoch = val["miou"], epoch
            torch.save({"config": model.config(), "state_dict": model.state_dict(), "epoch": epoch,
                        "val": val, "args": {k: str(v) for k, v in vars(args).items()}}, ckpt_path)
        print(f"epoch {epoch:3d}  loss {loss:.4f}  val mIoU {val['miou']:.4f}  "
              f"boundary acc {val['boundary_acc']:.4f}  {epoch_times[-1]:.1f}s" + ("  *" if improved else ""))

    # final test evaluation with the best checkpoint
    ckpt = torch.load(ckpt_path, weights_only=False, map_location=device)
    model = build_model(ckpt["config"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    test = evaluate(model, test_loader, device)

    results = {
        "graph": args.graph,
        "best_epoch": best_epoch,
        "val": ckpt["val"],
        "test": test,
        "parameters": n_params,
        "mean_epoch_seconds": float(np.mean(epoch_times)),
        "peak_gpu_memory_mb": (torch.cuda.max_memory_allocated(device) / 2**20) if device.type == "cuda" else None,
        "scans": {"train": len(train_ds), "val": len(val_ds), "test": len(test_ds)},
        "args": {k: str(v) for k, v in vars(args).items()},
    }
    (args.results_dir / f"{args.graph}_metrics.json").write_text(json.dumps(results, indent=2))
    (args.results_dir / f"{args.graph}_history.json").write_text(json.dumps(history, indent=2))
    print(f"\nBest epoch {best_epoch}.  TEST: mIoU {test['miou']:.4f}  IoU tooth {test['iou_tooth']:.4f}  "
          f"IoU gingiva {test['iou_gingiva']:.4f}  acc {test['acc']:.4f}  boundary acc {test['boundary_acc']:.4f}")
    return results


if __name__ == "__main__":
    main()