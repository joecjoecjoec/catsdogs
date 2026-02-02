# train.py
from __future__ import annotations

import argparse
import copy
import json
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models


# -------------------------
# Reproducibility
# -------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        # Apple MPS seed
        torch.mps.manual_seed(seed)


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# -------------------------
# Data
# -------------------------
def build_dataloaders(
    data_dir: Path,
    img_size: int,
    batch_size: int,
    num_workers: int,
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    train_dir = data_dir / "train"
    val_dir = data_dir / "validation"
    test_dir = data_dir / "test"

    assert train_dir.exists(), f"Missing: {train_dir}"
    assert val_dir.exists(), f"Missing: {val_dir}"
    assert test_dir.exists(), f"Missing: {test_dir}"

    train_tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    eval_tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    train_ds = datasets.ImageFolder(train_dir, transform=train_tfms)
    val_ds = datasets.ImageFolder(val_dir, transform=eval_tfms)
    test_ds = datasets.ImageFolder(test_dir, transform=eval_tfms)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    class_names = train_ds.classes  # e.g. ["cats", "dogs"]
    return train_loader, val_loader, test_loader, class_names


# -------------------------
# Model
# -------------------------
def build_mobilenetv3_small(num_classes: int) -> nn.Module:
    m = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    in_features = m.classifier[-1].in_features
    m.classifier[-1] = nn.Linear(in_features, num_classes)
    return m


# -------------------------
# Train / Eval
# -------------------------
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> Tuple[float, float]:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)

        total_loss += float(loss.item()) * y.size(0)
        pred = logits.argmax(dim=1)
        correct += int((pred == y).sum().item())
        total += int(y.size(0))

    return total_loss / max(total, 1), correct / max(total, 1)


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer,
                    criterion: nn.Module, device: torch.device) -> Tuple[float, float]:
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)

        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item()) * y.size(0)
        pred = logits.argmax(dim=1)
        correct += int((pred == y).sum().item())
        total += int(y.size(0))

    return total_loss / max(total, 1), correct / max(total, 1)


# -------------------------
# Experiment
# -------------------------
def run_experiment(
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    class_names: List[str],
    device: torch.device,
    lr: float,
    weight_decay: float,
    epochs: int,
    seed: int,
    early_stop_patience: int,
    min_delta: float,
    out_dir: Path,
) -> Dict:
    set_seed(seed)
    num_classes = len(class_names)

    model = build_mobilenetv3_small(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_state = None
    best_val_loss = float("inf")
    best_val_acc = -1.0
    best_epoch = None
    no_improve = 0

    t0 = time.time()

    print("\n" + "=" * 60)
    print(f"RUN: lr={lr} | wd={weight_decay} | epochs={epochs} | seed={seed} | device={device.type}")
    print("=" * 60)

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        improved = val_loss < (best_val_loss - min_delta)
        if improved:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1

        print(
            f"epoch {epoch:02d}: "
            f"train loss={tr_loss:.4f} acc={tr_acc:.3f} | "
            f"val loss={val_loss:.4f} acc={val_acc:.3f} | "
            f"best_val_loss={best_val_loss:.4f} (epoch={best_epoch}) | "
            f"no_improve={no_improve}/{early_stop_patience}"
        )

        if no_improve >= early_stop_patience:
            print(f"EARLY STOP at epoch {epoch}. Best epoch={best_epoch} (best_val_loss={best_val_loss:.4f}).")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss, test_acc = evaluate(model, test_loader, criterion, device)

    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / f"mobilenetv3_lr{lr}_wd{weight_decay}_seed{seed}.pt"
    torch.save(
        {
            "arch": "mobilenet_v3_small",
            "lr": lr,
            "weight_decay": weight_decay,
            "seed": seed,
            "best_epoch": best_epoch,
            "class_names": class_names,
            "state_dict": model.state_dict(),
        },
        ckpt_path,
    )

    elapsed = time.time() - t0
    print("---- SUMMARY ----")
    print(
        f"seed={seed} lr={lr} wd={weight_decay} "
        f"best_epoch={best_epoch} best_val_loss={best_val_loss:.4f} best_val_acc={best_val_acc:.4f} "
        f"test_acc={test_acc:.4f} test_loss={test_loss:.4f} "
        f"time={elapsed:.1f}s | saved={ckpt_path}"
    )

    return {
        "seed": seed,
        "lr": lr,
        "weight_decay": weight_decay,
        "epochs": epochs,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_acc": best_val_acc,
        "test_acc": test_acc,
        "test_loss": test_loss,
        "time_sec": elapsed,
        "ckpt_path": str(ckpt_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data/processed/cats_dogs_70_30")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--early-stop-patience", type=int, default=3)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--out-dir", type=str, default="models")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    data_dir = (base_dir / args.data_dir).resolve()
    out_dir = (base_dir / args.out_dir).resolve()

    device = get_device()
    print(f"device: {device}")

    train_loader, val_loader, test_loader, class_names = build_dataloaders(
        data_dir=data_dir,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    results = []
    for seed in args.seeds:
        res = run_experiment(
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            class_names=class_names,
            device=device,
            lr=args.lr,
            weight_decay=args.weight_decay,
            epochs=args.epochs,
            seed=seed,
            early_stop_patience=args.early_stop_patience,
            min_delta=args.min_delta,
            out_dir=out_dir,
        )
        results.append(res)

    # summary
    print("\nSUMMARY (per seed):")
    for r in results:
        print(
            f"seed={r['seed']} | best_epoch={r['best_epoch']} | "
            f"test_acc={r['test_acc']:.4f} | best_val_loss={r['best_val_loss']:.4f} | "
            f"best_val_acc={r['best_val_acc']:.4f} | time={r['time_sec']:.1f}s"
        )

    # write json
    summary_path = out_dir / "results_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "arch": "mobilenet_v3_small",
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "epochs": args.epochs,
                "early_stop_patience": args.early_stop_patience,
                "min_delta": args.min_delta,
                "seeds": args.seeds,
                "results": results,
            },
            f,
            indent=2,
        )
    print(f"\nSaved run summary to: {summary_path}")


if __name__ == "__main__":
    main()