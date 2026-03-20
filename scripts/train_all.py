#!/usr/bin/env python3
"""
Train all registered backbone models on Stanford Dogs (v3 recipe).

Usage:
    python scripts/train_all.py                    # Train all backbones with ArcFace + progressive resize
    python scripts/train_all.py --backbones convnextv2_tiny
    python scripts/train_all.py --no-arcface       # Fallback to CE loss
    python scripts/train_all.py --no-progressive    # Skip progressive resizing
    python scripts/train_all.py --epochs 50 --batch-size 64
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.registry import list_backbones, get_backbone
from src.backbones import *  # noqa — trigger @register decorators
from src.dataset import get_transforms
from src.train import train_model, _build_loaders


def main():
    parser = argparse.ArgumentParser(description="Train breed classifiers (v3)")
    parser.add_argument("--backbones", nargs="+", default=None,
                        help="Specific backbones to train (default: all)")
    parser.add_argument("--data-dir", default="data", help="Dataset directory")
    parser.add_argument("--output-dir", default="models", help="Output directory")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--time-limit", type=float, default=180.0,
                        help="Time limit per model in minutes")
    # ArcFace
    parser.add_argument("--no-arcface", action="store_true", help="Use CE instead of ArcFace")
    parser.add_argument("--arcface-scale", type=float, default=30.0)
    parser.add_argument("--arcface-margin", type=float, default=0.3)
    # Progressive resizing
    parser.add_argument("--no-progressive", action="store_true", help="Skip progressive resizing")
    parser.add_argument("--resize-to", type=int, default=336, help="Target resolution")
    parser.add_argument("--resize-at-epoch", type=int, default=20,
                        help="Epoch to switch resolution")
    parser.add_argument("--resize-batch-size", type=int, default=32,
                        help="Batch size after resolution bump")
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, args.data_dir)
    output_dir = os.path.join(project_root, args.output_dir)

    available = list_backbones()
    print(f"Registered backbones: {available}")

    backbones = args.backbones or available
    use_arcface = not args.no_arcface
    use_progressive = not args.no_progressive

    print(f"Will train: {backbones}")
    print(f"Loss: {'ArcFace (s={}, m={})'.format(args.arcface_scale, args.arcface_margin) if use_arcface else 'CrossEntropy'}")
    if use_progressive:
        print(f"Progressive resize: 224 → {args.resize_to} at epoch {args.resize_at_epoch}")
    print()

    results = []
    for backbone_name in backbones:
        if backbone_name not in available:
            print(f"WARNING: '{backbone_name}' not registered. Skipping.")
            continue

        backbone = get_backbone(backbone_name)
        preprocess_config = backbone.get_preprocess_config()
        del backbone

        # Build initial dataloaders at 224px
        from torchvision import datasets
        from torch.utils.data import DataLoader

        train_dir = os.path.join(data_dir, "train")
        val_dir = os.path.join(data_dir, "val")
        test_dir = os.path.join(data_dir, "test")

        train_transform = get_transforms(preprocess_config, is_train=True)
        val_transform = get_transforms(preprocess_config, is_train=False)

        train_ds = datasets.ImageFolder(train_dir, transform=train_transform)
        breed_names = [c.replace("_", " ") for c in train_ds.classes]

        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True,
            num_workers=4, pin_memory=True, drop_last=True, persistent_workers=True,
        )

        val_loader = None
        if os.path.isdir(val_dir):
            val_ds = datasets.ImageFolder(val_dir, transform=val_transform)
            val_loader = DataLoader(
                val_ds, batch_size=args.batch_size, shuffle=False,
                num_workers=4, pin_memory=True, persistent_workers=True,
            )
            print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}, Breeds: {len(breed_names)}")

        test_loader = None
        if os.path.isdir(test_dir):
            test_ds = datasets.ImageFolder(test_dir, transform=val_transform)
            test_loader = DataLoader(
                test_ds, batch_size=args.batch_size, shuffle=False,
                num_workers=4, pin_memory=True, persistent_workers=True,
            )

        result = train_model(
            backbone_name=backbone_name,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            epochs=args.epochs,
            warmup_epochs=args.warmup_epochs,
            lr=args.lr,
            output_dir=output_dir,
            time_limit_minutes=args.time_limit,
            # ArcFace
            use_arcface=use_arcface,
            arcface_scale=args.arcface_scale,
            arcface_margin=args.arcface_margin,
            # Progressive resizing
            prog_resize_to=args.resize_to if use_progressive else None,
            prog_resize_at_epoch=args.resize_at_epoch if use_progressive else None,
            prog_resize_batch_size=args.resize_batch_size if use_progressive else None,
            data_dir=data_dir,
        )
        results.append(result)

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*60}")
    print("Training Summary (v3 recipe)")
    print(f"{'='*60}")
    for r in results:
        print(f"  {r['backbone']:20s}  Top-1: {r['best_top1']:.1f}%")


if __name__ == "__main__":
    main()
