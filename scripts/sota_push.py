#!/usr/bin/env python3
"""
SOTA push: Train all models with ArcFace + high-res fine-tune + ensemble.

This is the comprehensive training pipeline that:
1. Trains ConvNeXt V2 Tiny with ArcFace at 224px (fast convergence)
2. Fine-tunes the best ConvNeXt checkpoint at 336px (high-res features)
3. Trains EfficientNetV2-S with ArcFace at 224px → 336px
4. Trains DINOv3 ViT-B with ArcFace at 224px (50 epochs, ViTs need more)
5. Builds calibrated ensemble from all models
6. Benchmarks everything on held-out test set

Expected: individual models 92-94%, ensemble 95%+
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

from src.backbones import *  # noqa — trigger @register decorators
from src.registry import get_backbone
from src.dataset import get_transforms, clean_breed_name
from src.train import (
    BreedClassifier, train_model, _build_loaders, train_one_epoch,
    evaluate, get_device, load_model, NUM_CLASSES, _save_checkpoint,
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
    """Build train/val/test loaders for a backbone at given resolution."""
    bb = get_backbone(backbone_name, pretrained=False)
    preprocess_config = bb.get_preprocess_config()
    preprocess_config["input_size"] = img_size
    del bb

    train_transform = get_transforms(preprocess_config, is_train=True)
    val_transform = get_transforms(preprocess_config, is_train=False)

    train_ds = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=train_transform)
    val_ds = datasets.ImageFolder(os.path.join(DATA_DIR, "val"), transform=val_transform)
    test_ds = datasets.ImageFolder(os.path.join(DATA_DIR, "test"), transform=val_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=4, pin_memory=True, persistent_workers=True)

    return train_loader, val_loader, test_loader


def finetune_highres(backbone_name, source_ckpt, target_size=336, epochs=10,
                     lr=2e-4, batch_size=16, backbone_lr_mult=0.005):
    """Fine-tune an existing checkpoint at higher resolution.

    Loads the best 224px model and continues training at 336px with:
    - Very low LR (2e-4 head, 1e-6 backbone)
    - Short schedule (10 epochs)
    - No MixUp/CutMix (ArcFace doesn't need it)
    - Small batch size for memory
    """
    device = get_device()
    print(f"\n{'='*60}")
    print(f"HIGH-RES FINE-TUNE: {backbone_name} at {target_size}px")
    print(f"Source: {source_ckpt}")
    print(f"{'='*60}")

    # Load existing model
    ckpt = torch.load(source_ckpt, map_location=device, weights_only=False)
    model = BreedClassifier(backbone_name, use_arcface=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')}, val: {ckpt.get('val_top1', '?'):.1f}%")

    # Build high-res loaders
    train_loader, val_loader, test_loader = make_loaders(backbone_name, target_size, batch_size)
    print(f"Loaders: {target_size}px, batch {batch_size}")

    # CE criterion for eval
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Eval at new resolution before training
    pre_metrics = evaluate(model, val_loader, criterion, device)
    print(f"Pre-finetune val @ {target_size}px: {pre_metrics['top1_acc']:.1f}%")

    # Unfreeze everything with very low backbone LR
    model.unfreeze_backbone()
    param_groups = model.get_param_groups(lr, backbone_lr_mult)
    optimizer = torch.optim.AdamW(param_groups, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-7)

    best_val_acc = pre_metrics["top1_acc"]
    save_tag = f"{backbone_name}_{target_size}"

    for epoch in range(epochs):
        epoch_start = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            scheduler=None, mixup_alpha=0, cutmix_alpha=0, mix_prob=0,
        )
        scheduler.step()
        val_metrics = evaluate(model, val_loader, criterion, device)
        epoch_time = time.time() - epoch_start

        improved = ""
        if val_metrics["top1_acc"] > best_val_acc:
            best_val_acc = val_metrics["top1_acc"]
            _save_checkpoint(model, save_tag, epoch + 1, val_metrics, MODELS_DIR)
            # Also overwrite the base checkpoint if we beat it
            _save_checkpoint(model, backbone_name, epoch + 1, val_metrics, MODELS_DIR)
            improved = " *NEW BEST*"

        bb_lr = optimizer.param_groups[0]['lr']
        head_lr = optimizer.param_groups[-1]['lr']
        print(f"  E{epoch+1} [{target_size}px]: Train {train_acc:.1f}% | "
              f"Val T1={val_metrics['top1_acc']:.1f}% T5={val_metrics['top5_acc']:.1f}% | "
              f"LR bb={bb_lr:.1e} head={head_lr:.1e} | {epoch_time:.0f}s{improved}")

    # Final test set eval
    test_metrics = evaluate(model, test_loader, criterion, device)
    print(f"\n{save_tag} TEST: Top1={test_metrics['top1_acc']:.1f}% Top5={test_metrics['top5_acc']:.1f}%")

    del model
    clear_gpu()

    return {"backbone": save_tag, "best_val_top1": best_val_acc,
            "test_top1": test_metrics["top1_acc"], "test_top5": test_metrics["top5_acc"]}


def train_and_finetune(backbone_name, epochs_224=30, epochs_336=10,
                       batch_224=64, batch_336=16, lr=1e-3):
    """Full pipeline: train at 224px, then fine-tune at 336px."""
    print(f"\n{'#'*60}")
    print(f"# TRAINING: {backbone_name}")
    print(f"# Phase A: {epochs_224} epochs @ 224px (batch {batch_224})")
    print(f"# Phase B: {epochs_336} epochs @ 336px (batch {batch_336})")
    print(f"{'#'*60}")

    # Phase A: Train at 224px
    train_loader, val_loader, test_loader = make_loaders(backbone_name, 224, batch_224)

    result_224 = train_model(
        backbone_name=backbone_name,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        epochs=epochs_224,
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

    del train_loader, val_loader, test_loader
    clear_gpu()

    # Phase B: Fine-tune at 336px
    ckpt_path = os.path.join(MODELS_DIR, f"{backbone_name}_best.pt")
    result_336 = finetune_highres(
        backbone_name, ckpt_path,
        target_size=336, epochs=epochs_336, batch_size=batch_336,
    )

    clear_gpu()
    return result_224, result_336


def build_ensemble():
    """Build and evaluate calibrated ensemble from all trained models."""
    device = get_device()
    print(f"\n{'#'*60}")
    print(f"# ENSEMBLE BUILDING")
    print(f"{'#'*60}")

    # Find all trained models
    checkpoints = {}
    for fname in os.listdir(MODELS_DIR):
        if fname.endswith("_best.pt") and not fname.startswith("ensemble"):
            backbone_name = fname.replace("_best.pt", "")
            # Skip 336-specific files, use the base name (which points to best overall)
            if "_336" not in backbone_name:
                checkpoints[backbone_name] = os.path.join(MODELS_DIR, fname)

    if len(checkpoints) < 2:
        print("Need at least 2 models for ensemble. Skipping.")
        return None

    # Evaluate at 336px if those models exist, else 224px
    all_val_probs = {}
    all_test_probs = {}
    val_labels = None
    test_labels = None

    criterion = nn.CrossEntropyLoss()

    for backbone_name, ckpt_path in sorted(checkpoints.items()):
        print(f"\n  Loading {backbone_name}...")
        model = load_model(backbone_name, ckpt_path, device)

        # Check what resolution the model was trained at
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        # Use 336px if available, else native
        preprocess_config = model.get_preprocess_config()
        img_size = preprocess_config.get("input_size", 224)

        # Check if a 336px checkpoint exists
        ckpt_336 = os.path.join(MODELS_DIR, f"{backbone_name}_336_best.pt")
        if os.path.exists(ckpt_336):
            print(f"  Using 336px checkpoint")
            model = load_model(backbone_name, ckpt_336, device)
            img_size = 336

        val_transform = get_transforms({**preprocess_config, "input_size": img_size}, is_train=False)
        test_transform = val_transform

        val_ds = datasets.ImageFolder(os.path.join(DATA_DIR, "val"), transform=val_transform)
        test_ds = datasets.ImageFolder(os.path.join(DATA_DIR, "test"), transform=test_transform)

        val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
        test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

        # Collect predictions
        val_probs, vl = collect_predictions(model, val_loader, device)
        test_probs, tl = collect_predictions(model, test_loader, device)

        if val_labels is None:
            val_labels = vl
            test_labels = tl

        all_val_probs[backbone_name] = val_probs
        all_test_probs[backbone_name] = test_probs

        # Individual model accuracy
        val_acc = 100.0 * np.mean(val_probs.argmax(axis=1) == val_labels)
        test_acc = 100.0 * np.mean(test_probs.argmax(axis=1) == test_labels)
        print(f"  {backbone_name}: Val {val_acc:.1f}%, Test {test_acc:.1f}%")

        del model
        clear_gpu()

    # Simple average ensemble
    print(f"\n  Simple average ensemble ({len(all_test_probs)} models):")
    avg_val_probs = np.mean(list(all_val_probs.values()), axis=0)
    avg_test_probs = np.mean(list(all_test_probs.values()), axis=0)

    avg_val_acc = 100.0 * np.mean(avg_val_probs.argmax(axis=1) == val_labels)
    avg_test_acc = 100.0 * np.mean(avg_test_probs.argmax(axis=1) == test_labels)
    avg_test_top5 = 100.0 * np.mean(
        np.any(np.argsort(avg_test_probs, axis=1)[:, -5:] == test_labels[:, None], axis=1)
    )
    print(f"  Average Ensemble Val: {avg_val_acc:.1f}%")
    print(f"  Average Ensemble Test: Top1={avg_test_acc:.1f}%, Top5={avg_test_top5:.1f}%")

    # Weighted ensemble: weight by val accuracy
    val_accs = {}
    for name, probs in all_val_probs.items():
        val_accs[name] = np.mean(probs.argmax(axis=1) == val_labels)

    total_acc = sum(val_accs.values())
    weights = {name: acc / total_acc for name, acc in val_accs.items()}
    print(f"\n  Weighted ensemble (by val accuracy):")
    for name, w in weights.items():
        print(f"    {name}: weight={w:.3f}")

    weighted_test_probs = sum(weights[name] * all_test_probs[name] for name in weights)
    weighted_test_acc = 100.0 * np.mean(weighted_test_probs.argmax(axis=1) == test_labels)
    weighted_test_top5 = 100.0 * np.mean(
        np.any(np.argsort(weighted_test_probs, axis=1)[:, -5:] == test_labels[:, None], axis=1)
    )
    print(f"  Weighted Ensemble Test: Top1={weighted_test_acc:.1f}%, Top5={weighted_test_top5:.1f}%")

    return {
        "models": list(checkpoints.keys()),
        "individual_val_accs": {k: round(v * 100, 1) for k, v in val_accs.items()},
        "avg_ensemble_test_top1": round(avg_test_acc, 2),
        "avg_ensemble_test_top5": round(avg_test_top5, 2),
        "weighted_ensemble_test_top1": round(weighted_test_acc, 2),
        "weighted_ensemble_test_top5": round(weighted_test_top5, 2),
    }


def main():
    start = time.time()
    all_results = {}

    # ─── 1. ConvNeXt V2 Tiny (our best single model) ───
    r224, r336 = train_and_finetune(
        "convnextv2_tiny",
        epochs_224=30,   # ArcFace converges fast
        epochs_336=10,   # Short high-res finetune
        batch_224=64,
        batch_336=16,    # Memory-safe for 336px
    )
    all_results["convnextv2_tiny"] = {"224": r224, "336": r336}
    print(f"\nConvNeXt done. Elapsed: {(time.time()-start)/60:.0f}min")

    # ─── 2. EfficientNetV2-S ───
    r224, r336 = train_and_finetune(
        "efficientnetv2_s",
        epochs_224=30,
        epochs_336=10,
        batch_224=64,
        batch_336=16,
    )
    all_results["efficientnetv2_s"] = {"224": r224, "336": r336}
    print(f"\nEfficientNet done. Elapsed: {(time.time()-start)/60:.0f}min")

    # ─── 3. DINOv3 ViT-B (ViTs need more epochs) ───
    r224, r336 = train_and_finetune(
        "dinov3_vitb",
        epochs_224=50,   # ViTs converge slower
        epochs_336=10,
        batch_224=64,
        batch_336=16,
    )
    all_results["dinov3_vitb"] = {"224": r224, "336": r336}
    print(f"\nDINOv3 done. Elapsed: {(time.time()-start)/60:.0f}min")

    # ─── 4. Build ensemble ───
    ensemble_results = build_ensemble()
    if ensemble_results:
        all_results["ensemble"] = ensemble_results

    # ─── Save everything ───
    total_time = (time.time() - start) / 60
    all_results["total_time_minutes"] = round(total_time, 1)

    results_path = os.path.join(MODELS_DIR, "sota_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"SOTA PUSH COMPLETE — {total_time:.0f} minutes")
    print(f"{'='*60}")
    print(f"Results saved to: {results_path}")

    # Summary
    for model_name, result in all_results.items():
        if model_name in ("total_time_minutes",):
            continue
        if model_name == "ensemble":
            print(f"\n  ENSEMBLE: Test {result['weighted_ensemble_test_top1']:.1f}%")
        elif "336" in result:
            r = result["336"]
            print(f"  {model_name}: Test {r.get('test_top1', '?'):.1f}% (336px)")


if __name__ == "__main__":
    main()
