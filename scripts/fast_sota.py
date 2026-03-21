#!/usr/bin/env python3
"""
Fast SOTA: Train remaining models with ArcFace at 224px + build ensemble.
ConvNeXt already done at 336px (91.8%). Focus on EfficientNet + DINOv3 at 224px.
"""

import os
import sys
import json
import time
import gc

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as torchF
from torch.utils.data import DataLoader
from torchvision import datasets

from src.backbones import *  # noqa
from src.registry import get_backbone
from src.dataset import get_transforms, clean_breed_name
from src.train import (
    BreedClassifier, train_model, train_one_epoch,
    evaluate, get_device, load_model, NUM_CLASSES,
)
from src.ensemble import collect_predictions


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")


def clear_gpu():
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def make_loaders(backbone_name, img_size=224, batch_size=64):
    bb = get_backbone(backbone_name, pretrained=False)
    pc = bb.get_preprocess_config()
    pc["input_size"] = img_size
    del bb

    train_t = get_transforms(pc, is_train=True)
    val_t = get_transforms(pc, is_train=False)

    train_ds = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=train_t)
    val_ds = datasets.ImageFolder(os.path.join(DATA_DIR, "val"), transform=val_t)
    test_ds = datasets.ImageFolder(os.path.join(DATA_DIR, "test"), transform=val_t)

    kw = dict(num_workers=4, pin_memory=True, persistent_workers=True)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True, **kw)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **kw)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, **kw)
    return train_loader, val_loader, test_loader


def train_arcface_224(backbone_name, epochs=30, batch_size=64, lr=1e-3):
    """Train a single model with ArcFace at 224px."""
    print(f"\n{'='*60}")
    print(f"TRAINING: {backbone_name} (ArcFace, 224px, {epochs} epochs)")
    print(f"{'='*60}")

    train_loader, val_loader, test_loader = make_loaders(backbone_name, 224, batch_size)

    result = train_model(
        backbone_name=backbone_name,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        epochs=epochs,
        warmup_epochs=2,
        lr=lr,
        backbone_lr_mult=0.01,
        output_dir=MODELS_DIR,
        time_limit_minutes=120.0,
        use_arcface=True,
        arcface_scale=30.0,
        arcface_margin=0.3,
        early_stop_patience=10,
        data_dir=DATA_DIR,
    )

    # Test set eval
    device = get_device()
    ckpt_path = os.path.join(MODELS_DIR, f"{backbone_name}_best.pt")
    model = load_model(backbone_name, ckpt_path, device)
    criterion = nn.CrossEntropyLoss()
    test_metrics = evaluate(model, test_loader, criterion, device)
    print(f"\n{backbone_name} TEST: Top1={test_metrics['top1_acc']:.1f}% Top5={test_metrics['top5_acc']:.1f}%")

    del model, train_loader, val_loader, test_loader
    clear_gpu()
    return {**result, "test_top1": test_metrics["top1_acc"], "test_top5": test_metrics["top5_acc"]}


def build_ensemble():
    """Build ensemble from all available models."""
    device = get_device()
    print(f"\n{'='*60}")
    print(f"BUILDING ENSEMBLE")
    print(f"{'='*60}")

    # Use ConvNeXt at 336px, others at 224px
    model_configs = {}
    for fname in os.listdir(MODELS_DIR):
        if not fname.endswith("_best.pt"):
            continue
        backbone = fname.replace("_best.pt", "")
        if backbone.endswith("_336"):
            continue  # Skip, we'll use base name with 336 resolution
        model_configs[backbone] = {
            "path": os.path.join(MODELS_DIR, fname),
            "img_size": 224,
        }

    # Override ConvNeXt to use 336px checkpoint if it exists
    ckpt_336 = os.path.join(MODELS_DIR, "convnextv2_tiny_336_best.pt")
    if os.path.exists(ckpt_336) and "convnextv2_tiny" in model_configs:
        model_configs["convnextv2_tiny"]["path"] = ckpt_336
        model_configs["convnextv2_tiny"]["img_size"] = 336

    print(f"Models: {list(model_configs.keys())}")
    for name, cfg in model_configs.items():
        print(f"  {name}: {cfg['img_size']}px")

    criterion = nn.CrossEntropyLoss()
    all_val_probs = {}
    all_test_probs = {}
    val_labels = None
    test_labels = None
    individual_results = {}

    for backbone_name, cfg in sorted(model_configs.items()):
        print(f"\n  Loading {backbone_name} ({cfg['img_size']}px)...")
        model = load_model(backbone_name, cfg["path"], device)
        pc = model.get_preprocess_config()
        pc["input_size"] = cfg["img_size"]

        val_t = get_transforms(pc, is_train=False)
        val_ds = datasets.ImageFolder(os.path.join(DATA_DIR, "val"), transform=val_t)
        test_ds = datasets.ImageFolder(os.path.join(DATA_DIR, "test"), transform=val_t)

        batch = 32 if cfg["img_size"] > 224 else 64
        kw = dict(num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=batch, shuffle=False, **kw)
        test_loader = DataLoader(test_ds, batch_size=batch, shuffle=False, **kw)

        val_probs, vl = collect_predictions(model, val_loader, device)
        test_probs, tl = collect_predictions(model, test_loader, device)

        if val_labels is None:
            val_labels = vl
            test_labels = tl

        all_val_probs[backbone_name] = val_probs
        all_test_probs[backbone_name] = test_probs

        val_acc = 100.0 * np.mean(val_probs.argmax(axis=1) == val_labels)
        test_acc = 100.0 * np.mean(test_probs.argmax(axis=1) == test_labels)
        test_top5 = 100.0 * np.mean(
            np.any(np.argsort(test_probs, axis=1)[:, -5:] == test_labels[:, None], axis=1)
        )
        print(f"  {backbone_name}: Val {val_acc:.1f}%, Test {test_acc:.1f}% (Top5: {test_top5:.1f}%)")
        individual_results[backbone_name] = {
            "val_top1": round(val_acc, 2),
            "test_top1": round(test_acc, 2),
            "test_top5": round(test_top5, 2),
        }

        del model
        clear_gpu()

    # Average ensemble
    print(f"\n--- ENSEMBLE ({len(all_test_probs)} models) ---")
    avg_test = np.mean(list(all_test_probs.values()), axis=0)
    avg_test_top1 = 100.0 * np.mean(avg_test.argmax(axis=1) == test_labels)
    avg_test_top5 = 100.0 * np.mean(
        np.any(np.argsort(avg_test, axis=1)[:, -5:] == test_labels[:, None], axis=1)
    )
    print(f"  Average: Test Top1={avg_test_top1:.1f}%, Top5={avg_test_top5:.1f}%")

    # Weighted by val accuracy
    val_accs = {n: np.mean(p.argmax(axis=1) == val_labels) for n, p in all_val_probs.items()}
    total = sum(val_accs.values())
    weights = {n: a / total for n, a in val_accs.items()}
    weighted_test = sum(weights[n] * all_test_probs[n] for n in weights)
    w_top1 = 100.0 * np.mean(weighted_test.argmax(axis=1) == test_labels)
    w_top5 = 100.0 * np.mean(
        np.any(np.argsort(weighted_test, axis=1)[:, -5:] == test_labels[:, None], axis=1)
    )
    print(f"  Weighted: Test Top1={w_top1:.1f}%, Top5={w_top5:.1f}%")
    print(f"  Weights: {', '.join(f'{n}={w:.3f}' for n, w in weights.items())}")

    return {
        "individual": individual_results,
        "avg_ensemble_test_top1": round(avg_test_top1, 2),
        "avg_ensemble_test_top5": round(avg_test_top5, 2),
        "weighted_ensemble_test_top1": round(w_top1, 2),
        "weighted_ensemble_test_top5": round(w_top5, 2),
        "weights": {k: round(v, 3) for k, v in weights.items()},
    }


def main():
    start = time.time()

    # Train EfficientNetV2-S with ArcFace
    eff_result = train_arcface_224("efficientnetv2_s", epochs=30, batch_size=64)
    print(f"\nEfficientNet done. {(time.time()-start)/60:.0f}min elapsed")

    # Train DINOv3 with ArcFace (more epochs for ViT)
    dino_result = train_arcface_224("dinov3_vitb", epochs=50, batch_size=64)
    print(f"\nDINOv3 done. {(time.time()-start)/60:.0f}min elapsed")

    # Build ensemble (ConvNeXt 336px + EfficientNet 224px + DINOv3 224px)
    ensemble_result = build_ensemble()

    # Save results
    results = {
        "efficientnetv2_s": eff_result,
        "dinov3_vitb": dino_result,
        "ensemble": ensemble_result,
        "total_minutes": round((time.time() - start) / 60, 1),
    }
    path = os.path.join(MODELS_DIR, "fast_sota_results.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"FAST SOTA COMPLETE — {results['total_minutes']:.0f} minutes")
    print(f"Results: {path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
