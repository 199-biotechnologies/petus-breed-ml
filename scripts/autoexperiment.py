#!/usr/bin/env python3
"""
Auto-experiment loop — Karpathy-style autonomous hyperparameter search.

Runs short training experiments (5 epochs each), logs results to TSV,
keeps the best config. Inspired by autoresearch pattern.

Usage:
    python scripts/autoexperiment.py --backbone convnextv2_tiny --budget 5
    python scripts/autoexperiment.py --backbone dinov3_vitb --budget 5 --sweep all
"""

import os
import sys
import json
import time
import csv
import argparse
from datetime import datetime
from copy import deepcopy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torchvision import datasets
from torch.utils.data import DataLoader

from src.backbones import *  # noqa
from src.registry import get_backbone
from src.dataset import get_transforms
from src.train import train_model, get_device

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(project_root, "data")
models_dir = os.path.join(project_root, "models")
results_file = os.path.join(project_root, "experiments.tsv")


# ─── Experiment definitions ───

SWEEPS = {
    "backbone_lr": {
        "param": "backbone_lr_mult",
        "values": [0.001, 0.005, 0.01, 0.02, 0.05, 0.1],
        "description": "Backbone learning rate multiplier",
    },
    "label_smoothing": {
        "param": "label_smoothing",
        "values": [0.0, 0.05, 0.1, 0.15, 0.2],
        "description": "Label smoothing factor",
    },
    "weight_decay": {
        "param": "weight_decay",
        "values": [0.01, 0.03, 0.05, 0.1],
        "description": "Weight decay",
    },
    "mixup": {
        "param": "mixup_alpha",
        "values": [0.0, 0.2, 0.5, 0.8, 1.0],
        "description": "MixUp alpha",
    },
    "lr": {
        "param": "lr",
        "values": [3e-4, 5e-4, 1e-3, 2e-3, 3e-3],
        "description": "Head learning rate",
    },
}


def build_loaders(backbone_name, batch_size=128):
    """Build train/val loaders for a backbone."""
    backbone = get_backbone(backbone_name)
    preprocess_config = backbone.get_preprocess_config()
    del backbone

    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")

    train_transform = get_transforms(preprocess_config, is_train=True)
    val_transform = get_transforms(preprocess_config, is_train=False)

    train_ds = datasets.ImageFolder(train_dir, transform=train_transform)
    val_ds = datasets.ImageFolder(val_dir, transform=val_transform)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=False, drop_last=True, persistent_workers=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=False, persistent_workers=True,
    )

    return train_loader, val_loader


def run_experiment(backbone_name, train_loader, val_loader, budget_epochs, **overrides):
    """Run a single short training experiment and return val accuracy."""
    # Default config
    config = {
        "epochs": budget_epochs + 1,  # +1 for warmup
        "warmup_epochs": 1,
        "lr": 1e-3,
        "backbone_lr_mult": 0.01,
        "label_smoothing": 0.1,
        "mixup_alpha": 0.8,
        "cutmix_alpha": 1.0,
        "mix_prob": 0.5,
        "no_aug_final_epochs": 0,  # No final stage in short experiments
        "unfreeze_warmup_epochs": 1,
        "early_stop_patience": 999,  # Disable early stopping
        "output_dir": os.path.join(models_dir, "experiments"),
        "time_limit_minutes": 30.0,
    }
    config.update(overrides)

    os.makedirs(config["output_dir"], exist_ok=True)

    result = train_model(
        backbone_name=backbone_name,
        train_loader=train_loader,
        val_loader=val_loader,
        **config,
    )

    # Clear memory
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    return result


def log_result(experiment_id, backbone, param_name, param_value, baseline_acc, result_acc, verdict):
    """Append result to TSV log."""
    file_exists = os.path.exists(results_file)

    with open(results_file, "a", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        if not file_exists:
            writer.writerow([
                "timestamp", "id", "backbone", "param", "value",
                "baseline_top1", "result_top1", "delta", "verdict",
            ])
        writer.writerow([
            datetime.now().isoformat(timespec="seconds"),
            experiment_id,
            backbone,
            param_name,
            param_value,
            f"{baseline_acc:.2f}",
            f"{result_acc:.2f}",
            f"{result_acc - baseline_acc:+.2f}",
            verdict,
        ])


def run_sweep(backbone_name, sweep_name, budget_epochs, batch_size):
    """Run a parameter sweep."""
    sweep = SWEEPS[sweep_name]
    param = sweep["param"]
    values = sweep["values"]

    print(f"\n{'='*60}")
    print(f"SWEEP: {sweep['description']} ({param})")
    print(f"Values: {values}")
    print(f"Budget: {budget_epochs} epochs per experiment")
    print(f"{'='*60}")

    train_loader, val_loader = build_loaders(backbone_name, batch_size)

    # Run baseline first
    print(f"\n--- Baseline ({param}=default) ---")
    baseline_result = run_experiment(backbone_name, train_loader, val_loader, budget_epochs)
    baseline_acc = baseline_result["best_top1"]
    print(f"  Baseline: {baseline_acc:.2f}%")

    log_result(
        f"{sweep_name}_baseline", backbone_name, param, "default",
        baseline_acc, baseline_acc, "BASELINE",
    )

    best_value = "default"
    best_acc = baseline_acc

    for i, value in enumerate(values):
        print(f"\n--- Experiment {i+1}/{len(values)}: {param}={value} ---")
        start = time.time()

        override = {param: value}
        result = run_experiment(backbone_name, train_loader, val_loader, budget_epochs, **override)
        acc = result["best_top1"]
        elapsed = time.time() - start

        delta = acc - baseline_acc
        verdict = "WIN" if delta > 0.3 else ("LOSE" if delta < -0.3 else "NEUTRAL")

        log_result(
            f"{sweep_name}_{i}", backbone_name, param, value,
            baseline_acc, acc, verdict,
        )

        print(f"  Result: {acc:.2f}% (Δ {delta:+.2f}%) — {verdict} [{elapsed/60:.1f}min]")

        if acc > best_acc:
            best_acc = acc
            best_value = value

    print(f"\n{'='*60}")
    print(f"SWEEP RESULT: {param}")
    print(f"  Best value: {best_value} ({best_acc:.2f}%)")
    print(f"  Improvement over baseline: {best_acc - baseline_acc:+.2f}%")
    print(f"{'='*60}")

    return best_value, best_acc


def main():
    parser = argparse.ArgumentParser(description="Auto-experiment loop")
    parser.add_argument("--backbone", default="convnextv2_tiny")
    parser.add_argument("--budget", type=int, default=5, help="Epochs per experiment")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--sweep", default="backbone_lr",
                        choices=list(SWEEPS.keys()) + ["all"],
                        help="Which parameter to sweep")
    args = parser.parse_args()

    print(f"Auto-Experiment Loop")
    print(f"Backbone: {args.backbone}")
    print(f"Budget: {args.budget} epochs per experiment")
    print(f"Results log: {results_file}")

    if args.sweep == "all":
        # Run all sweeps sequentially, using best from each
        best_config = {}
        for sweep_name in SWEEPS:
            best_val, best_acc = run_sweep(args.backbone, sweep_name, args.budget, args.batch_size)
            best_config[SWEEPS[sweep_name]["param"]] = best_val

        print(f"\n{'='*60}")
        print(f"ALL SWEEPS COMPLETE")
        print(f"{'='*60}")
        print(f"Best config:")
        for k, v in best_config.items():
            print(f"  {k}: {v}")
    else:
        run_sweep(args.backbone, args.sweep, args.budget, args.batch_size)

    print(f"\nFull results in: {results_file}")


if __name__ == "__main__":
    main()
