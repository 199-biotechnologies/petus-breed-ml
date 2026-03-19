#!/usr/bin/env python3
"""
Train all registered backbone models on Stanford Dogs.

Usage:
    python scripts/train_all.py                    # Train all backbones
    python scripts/train_all.py --backbones efficientnetv2_s siglip2_vitb
    python scripts/train_all.py --epochs 20 --batch-size 32
"""

import os
import sys
import argparse

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.registry import list_backbones, get_backbone
from src.backbones import *  # noqa — trigger @register decorators
from src.dataset import get_dataloaders, get_breed_names
from src.train import train_model


def main():
    parser = argparse.ArgumentParser(description="Train breed classifiers")
    parser.add_argument("--backbones", nargs="+", default=None,
                        help="Specific backbones to train (default: all)")
    parser.add_argument("--data-dir", default="data", help="Dataset directory")
    parser.add_argument("--output-dir", default="models", help="Output directory")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--time-limit", type=float, default=60.0,
                        help="Time limit per model in minutes")
    args = parser.parse_args()

    # Resolve paths relative to project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, args.data_dir)
    output_dir = os.path.join(project_root, args.output_dir)

    available = list_backbones()
    print(f"Registered backbones: {available}")

    backbones = args.backbones or available
    print(f"Will train: {backbones}\n")

    results = []
    for backbone_name in backbones:
        if backbone_name not in available:
            print(f"WARNING: '{backbone_name}' not registered. Skipping.")
            continue

        # Get backbone-specific preprocessing
        backbone = get_backbone(backbone_name)
        preprocess_config = backbone.get_preprocess_config()
        del backbone  # Free memory before training

        # Build dataloaders with backbone-aware transforms
        # Use train/val/test split directories
        train_dir = os.path.join(data_dir, "train")
        val_dir = os.path.join(data_dir, "val")
        test_dir = os.path.join(data_dir, "test")

        from src.dataset import get_transforms
        from torchvision import datasets
        from torch.utils.data import DataLoader

        train_transform = get_transforms(preprocess_config, is_train=True)
        val_transform = get_transforms(preprocess_config, is_train=False)

        train_ds = datasets.ImageFolder(train_dir, transform=train_transform)
        breed_names = [c.replace("_", " ") for c in train_ds.classes]

        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True,
            num_workers=4, pin_memory=True, drop_last=True,
            persistent_workers=True,
        )

        # Val loader for checkpoint selection (NOT test)
        val_loader = None
        if os.path.isdir(val_dir):
            val_ds = datasets.ImageFolder(val_dir, transform=val_transform)
            val_loader = DataLoader(
                val_ds, batch_size=args.batch_size, shuffle=False,
                num_workers=4, pin_memory=True, persistent_workers=True,
            )
            print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}, Breeds: {len(breed_names)}")
        else:
            print(f"  WARNING: No val dir found. Using test for validation.")

        # Test loader kept separate — only for final benchmark
        test_ds = datasets.ImageFolder(test_dir, transform=val_transform)
        test_loader = DataLoader(
            test_ds, batch_size=args.batch_size, shuffle=False,
            num_workers=4, pin_memory=True, persistent_workers=True,
        )

        result = train_model(
            backbone_name=backbone_name,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,  # backward compat fallback
            epochs=args.epochs,
            warmup_epochs=args.warmup_epochs,
            lr=args.lr,
            output_dir=output_dir,
            time_limit_minutes=args.time_limit,
        )
        results.append(result)

        # Clear GPU/MPS memory between models
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*60}")
    print("Training Summary")
    print(f"{'='*60}")
    for r in results:
        print(f"  {r['backbone']:20s}  Top-1: {r['best_top1']:.1f}%")


if __name__ == "__main__":
    main()
