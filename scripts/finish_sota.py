#!/usr/bin/env python3
"""
Finish SOTA: Train DINOv3 with ArcFace + build 3-model ensemble.
ConvNeXt (91.8% val 336px) and EfficientNet (86.0% val 224px) already done.
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
from torch.utils.data import DataLoader
from torchvision import datasets

from src.backbones import *  # noqa
from src.registry import get_backbone
from src.dataset import get_transforms, clean_breed_name
from src.train import (
    BreedClassifier, train_model, evaluate, get_device, load_model, NUM_CLASSES,
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


def main():
    start = time.time()
    device = get_device()

    # ─── 1. Train DINOv3 with ArcFace at 224px ───
    print(f"\n{'='*60}")
    print(f"TRAINING: dinov3_vitb (ArcFace, 224px, 50 epochs)")
    print(f"{'='*60}")

    bb = get_backbone("dinov3_vitb", pretrained=False)
    pc = bb.get_preprocess_config()
    del bb

    train_t = get_transforms(pc, is_train=True)
    val_t = get_transforms(pc, is_train=False)
    kw = dict(num_workers=4, pin_memory=True, persistent_workers=True)

    train_ds = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=train_t)
    val_ds = datasets.ImageFolder(os.path.join(DATA_DIR, "val"), transform=val_t)
    test_ds = datasets.ImageFolder(os.path.join(DATA_DIR, "test"), transform=val_t)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, drop_last=True, **kw)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, **kw)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, **kw)

    dino_result = train_model(
        backbone_name="dinov3_vitb",
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        epochs=50,
        warmup_epochs=2,
        lr=1e-3,
        backbone_lr_mult=0.01,
        output_dir=MODELS_DIR,
        time_limit_minutes=180.0,
        use_arcface=True,
        arcface_scale=30.0,
        arcface_margin=0.3,
        early_stop_patience=12,  # ViTs need more patience
        data_dir=DATA_DIR,
    )

    # Test eval for DINOv3
    model = load_model("dinov3_vitb", os.path.join(MODELS_DIR, "dinov3_vitb_best.pt"), device)
    criterion = nn.CrossEntropyLoss()
    dino_test = evaluate(model, test_loader, criterion, device)
    print(f"\nDINOv3 TEST: Top1={dino_test['top1_acc']:.1f}% Top5={dino_test['top5_acc']:.1f}%")
    del model
    clear_gpu()

    print(f"\nDINOv3 done. {(time.time()-start)/60:.0f}min elapsed")

    # ─── 2. Test eval for EfficientNet (was skipped due to bug) ───
    print(f"\n{'='*60}")
    print(f"TESTING: efficientnetv2_s")
    print(f"{'='*60}")

    eff_bb = get_backbone("efficientnetv2_s", pretrained=False)
    eff_pc = eff_bb.get_preprocess_config()
    del eff_bb
    eff_val_t = get_transforms(eff_pc, is_train=False)
    eff_test_ds = datasets.ImageFolder(os.path.join(DATA_DIR, "test"), transform=eff_val_t)
    eff_test_loader = DataLoader(eff_test_ds, batch_size=64, shuffle=False, **kw)

    eff_model = load_model("efficientnetv2_s", os.path.join(MODELS_DIR, "efficientnetv2_s_best.pt"), device)
    eff_test = evaluate(eff_model, eff_test_loader, criterion, device)
    print(f"EfficientNet TEST: Top1={eff_test['top1_acc']:.1f}% Top5={eff_test['top5_acc']:.1f}%")
    del eff_model
    clear_gpu()

    # ─── 3. Build 3-model ensemble ───
    print(f"\n{'='*60}")
    print(f"BUILDING 3-MODEL ENSEMBLE")
    print(f"{'='*60}")

    model_configs = {
        "convnextv2_tiny": {"img_size": 336, "path": os.path.join(MODELS_DIR, "convnextv2_tiny_best.pt")},
        "efficientnetv2_s": {"img_size": 224, "path": os.path.join(MODELS_DIR, "efficientnetv2_s_best.pt")},
        "dinov3_vitb": {"img_size": 224, "path": os.path.join(MODELS_DIR, "dinov3_vitb_best.pt")},
    }

    all_val_probs = {}
    all_test_probs = {}
    val_labels = None
    test_labels = None

    for name, cfg in model_configs.items():
        print(f"\n  Collecting predictions: {name} ({cfg['img_size']}px)")
        model = load_model(name, cfg["path"], device)
        mpc = model.get_preprocess_config()
        mpc["input_size"] = cfg["img_size"]

        vt = get_transforms(mpc, is_train=False)
        v_ds = datasets.ImageFolder(os.path.join(DATA_DIR, "val"), transform=vt)
        t_ds = datasets.ImageFolder(os.path.join(DATA_DIR, "test"), transform=vt)

        bs = 32 if cfg["img_size"] > 224 else 64
        v_loader = DataLoader(v_ds, batch_size=bs, shuffle=False, num_workers=4, pin_memory=True)
        t_loader = DataLoader(t_ds, batch_size=bs, shuffle=False, num_workers=4, pin_memory=True)

        vp, vl = collect_predictions(model, v_loader, device)
        tp, tl = collect_predictions(model, t_loader, device)

        if val_labels is None:
            val_labels = vl
            test_labels = tl

        all_val_probs[name] = vp
        all_test_probs[name] = tp

        vacc = 100.0 * np.mean(vp.argmax(axis=1) == val_labels)
        tacc = 100.0 * np.mean(tp.argmax(axis=1) == test_labels)
        print(f"  {name}: Val {vacc:.1f}%, Test {tacc:.1f}%")

        del model
        clear_gpu()

    # Average ensemble
    avg_test = np.mean(list(all_test_probs.values()), axis=0)
    avg_top1 = 100.0 * np.mean(avg_test.argmax(axis=1) == test_labels)
    avg_top5 = 100.0 * np.mean(
        np.any(np.argsort(avg_test, axis=1)[:, -5:] == test_labels[:, None], axis=1)
    )
    print(f"\n  AVERAGE ENSEMBLE Test: Top1={avg_top1:.1f}%, Top5={avg_top5:.1f}%")

    # Weighted ensemble
    val_accs = {n: np.mean(p.argmax(axis=1) == val_labels) for n, p in all_val_probs.items()}
    total = sum(val_accs.values())
    weights = {n: a / total for n, a in val_accs.items()}
    weighted_test = sum(weights[n] * all_test_probs[n] for n in weights)
    w_top1 = 100.0 * np.mean(weighted_test.argmax(axis=1) == test_labels)
    w_top5 = 100.0 * np.mean(
        np.any(np.argsort(weighted_test, axis=1)[:, -5:] == test_labels[:, None], axis=1)
    )
    print(f"  WEIGHTED ENSEMBLE Test: Top1={w_top1:.1f}%, Top5={w_top5:.1f}%")
    for n, w in weights.items():
        print(f"    {n}: weight={w:.3f}")

    # Save
    results = {
        "dinov3_train": dino_result,
        "dinov3_test": {"top1": round(dino_test["top1_acc"], 2), "top5": round(dino_test["top5_acc"], 2)},
        "efficientnet_test": {"top1": round(eff_test["top1_acc"], 2), "top5": round(eff_test["top5_acc"], 2)},
        "ensemble_avg": {"top1": round(avg_top1, 2), "top5": round(avg_top5, 2)},
        "ensemble_weighted": {"top1": round(w_top1, 2), "top5": round(w_top5, 2)},
        "weights": {k: round(v, 3) for k, v in weights.items()},
        "total_minutes": round((time.time() - start) / 60, 1),
    }
    with open(os.path.join(MODELS_DIR, "final_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"DONE — {results['total_minutes']:.0f} minutes")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
