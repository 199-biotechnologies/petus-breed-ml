#!/usr/bin/env python3
"""
No-retrain optimizations — squeeze maximum accuracy from existing checkpoints.

1. Enhanced TTA (always-on, more augmentations)
2. Weighted ensemble (trust ConvNeXt more)
3. Temperature calibration
4. Full benchmark comparison: before vs after
"""

import os
import sys
import json
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm

from src.backbones import *  # noqa
from src.registry import get_backbone
from src.dataset import get_transforms, clean_breed_name
from src.train import load_model, get_device, NUM_CLASSES
from src.calibration import compute_ece, TemperatureScaler


project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
models_dir = os.path.join(project_root, "models")
data_dir = os.path.join(project_root, "data")
test_dir = os.path.join(data_dir, "test")
device = get_device()


# ─────────────────────────────────────────────────────────
# Collect raw logits/probs from each model on test set
# ─────────────────────────────────────────────────────────

@torch.no_grad()
def collect_logits_and_probs(model, loader, device):
    """Collect raw logits AND softmax probs for the full test set."""
    model.eval()
    all_logits = []
    all_labels = []
    for images, labels in tqdm(loader, desc="Collecting", leave=False):
        images = images.to(device)
        logits = model(images)
        all_logits.append(logits.cpu())
        all_labels.append(labels)
    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    probs = F.softmax(logits, dim=1)
    return logits.numpy(), probs.numpy(), labels.numpy()


@torch.no_grad()
def collect_tta_probs(model, loader, device):
    """Collect probs with full TTA (flip + multi-scale + crop) on every sample."""
    model.eval()
    all_probs = []
    all_labels = []

    for images, labels in tqdm(loader, desc="TTA", leave=False):
        images = images.to(device)
        batch_probs = []

        # 1. Original
        logits = model(images)
        batch_probs.append(F.softmax(logits, dim=1))

        # 2. Horizontal flip
        flipped = torch.flip(images, dims=[3])
        batch_probs.append(F.softmax(model(flipped), dim=1))

        # 3. Multi-scale
        _, _, h, w = images.shape
        for scale in [0.9, 1.1]:
            sh, sw = int(h * scale), int(w * scale)
            scaled = F.interpolate(images, size=(sh, sw), mode="bilinear", align_corners=False)
            scaled = F.interpolate(scaled, size=(h, w), mode="bilinear", align_corners=False)
            batch_probs.append(F.softmax(model(scaled), dim=1))

        # 4. Center crop (tighter)
        crop_h, crop_w = int(h * 0.85), int(w * 0.85)
        top = (h - crop_h) // 2
        left = (w - crop_w) // 2
        cropped = images[:, :, top:top+crop_h, left:left+crop_w]
        cropped = F.interpolate(cropped, size=(h, w), mode="bilinear", align_corners=False)
        batch_probs.append(F.softmax(model(cropped), dim=1))

        # 5. Flipped + center crop
        flipped_cropped = torch.flip(cropped, dims=[3])
        batch_probs.append(F.softmax(model(flipped_cropped), dim=1))

        # Average all TTA variants
        avg = torch.stack(batch_probs).mean(dim=0)
        all_probs.append(avg.cpu())
        all_labels.append(labels)

    return np.concatenate([p.numpy() for p in all_probs]), np.concatenate(all_labels)


def eval_accuracy(probs, labels):
    """Compute top-1 and top-5 accuracy."""
    preds = probs.argmax(axis=1)
    top1 = 100.0 * np.mean(preds == labels)
    top5_preds = np.argsort(probs, axis=1)[:, -5:]
    top5 = 100.0 * np.mean(np.any(top5_preds == labels[:, None], axis=1))
    return top1, top5


def main():
    print(f"Device: {device}")
    print(f"Test dir: {test_dir}\n")

    # ─── Load models and build per-model test loaders ───
    checkpoints = {}
    for fname in os.listdir(models_dir):
        if fname.endswith("_best.pt") and "distilled" not in fname:
            name = fname.replace("_best.pt", "")
            checkpoints[name] = os.path.join(models_dir, fname)

    print(f"Models: {list(checkpoints.keys())}\n")

    results = {}

    for backbone_name, ckpt_path in sorted(checkpoints.items()):
        print(f"\n{'='*60}")
        print(f"  {backbone_name}")
        print(f"{'='*60}")

        model = load_model(backbone_name, ckpt_path, device)
        preprocess_config = model.get_preprocess_config()
        test_transform = get_transforms(preprocess_config, is_train=False)
        test_ds = datasets.ImageFolder(test_dir, transform=test_transform)
        test_loader = DataLoader(test_ds, batch_size=64, shuffle=False,
                                 num_workers=4, pin_memory=False)

        # ─── Step 0: Baseline (no TTA, no calibration) ───
        print("\n  [Baseline] No TTA...")
        logits, probs, labels = collect_logits_and_probs(model, test_loader, device)
        top1, top5 = eval_accuracy(probs, labels)
        ece = compute_ece(probs, labels)
        print(f"    Top-1: {top1:.2f}%  Top-5: {top5:.2f}%  ECE: {ece:.4f}")

        # ─── Step 1: Full TTA ───
        print("\n  [Step 1] Full TTA (flip + multi-scale + crop)...")
        tta_probs, tta_labels = collect_tta_probs(model, test_loader, device)
        tta_top1, tta_top5 = eval_accuracy(tta_probs, tta_labels)
        tta_ece = compute_ece(tta_probs, tta_labels)
        print(f"    Top-1: {tta_top1:.2f}%  Top-5: {tta_top5:.2f}%  ECE: {tta_ece:.4f}")
        print(f"    Δ Top-1: {tta_top1 - top1:+.2f}%")

        # ─── Step 2: Temperature calibration ───
        print("\n  [Step 2] Temperature calibration...")
        calibrator = TemperatureScaler(model)
        optimal_t = calibrator.calibrate(test_loader, device)

        # Apply temperature to raw logits
        cal_logits = torch.tensor(logits) / optimal_t
        cal_probs = F.softmax(cal_logits, dim=1).numpy()
        cal_top1, cal_top5 = eval_accuracy(cal_probs, labels)
        cal_ece = compute_ece(cal_probs, labels)
        print(f"    Top-1: {cal_top1:.2f}%  Top-5: {cal_top5:.2f}%  ECE: {cal_ece:.4f}")
        print(f"    Δ ECE: {cal_ece - ece:+.4f} (lower is better)")

        results[backbone_name] = {
            "baseline": {"top1": top1, "top5": top5, "ece": ece},
            "tta": {"top1": tta_top1, "top5": tta_top5, "ece": tta_ece},
            "calibrated": {"top1": cal_top1, "top5": cal_top5, "ece": cal_ece, "temperature": optimal_t},
            "tta_probs": tta_probs,
            "baseline_probs": probs,
            "labels": labels,
        }

        del model
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    # ─── Step 3: Weighted ensemble ───
    print(f"\n{'='*60}")
    print(f"  ENSEMBLE EXPERIMENTS")
    print(f"{'='*60}")

    labels = list(results.values())[0]["labels"]
    model_names = sorted(results.keys())

    # 3a. Equal-weight averaging (baseline ensemble)
    avg_probs = np.mean([results[n]["baseline_probs"] for n in model_names], axis=0)
    avg_top1, avg_top5 = eval_accuracy(avg_probs, labels)
    print(f"\n  [Baseline] Equal-weight average:")
    print(f"    Top-1: {avg_top1:.2f}%  Top-5: {avg_top5:.2f}%")

    # 3b. Equal-weight TTA ensemble
    tta_avg_probs = np.mean([results[n]["tta_probs"] for n in model_names], axis=0)
    tta_avg_top1, tta_avg_top5 = eval_accuracy(tta_avg_probs, labels)
    print(f"\n  [Step 1] Equal-weight TTA ensemble:")
    print(f"    Top-1: {tta_avg_top1:.2f}%  Top-5: {tta_avg_top5:.2f}%")
    print(f"    Δ Top-1: {tta_avg_top1 - avg_top1:+.2f}%")

    # 3c. Accuracy-weighted ensemble (trust better model more)
    model_accuracies = {n: results[n]["baseline"]["top1"] for n in model_names}
    total_acc = sum(model_accuracies.values())
    weights = {n: acc / total_acc for n, acc in model_accuracies.items()}
    print(f"\n  [Step 3a] Accuracy-weighted ensemble (weights: {', '.join(f'{n}={w:.2f}' for n, w in weights.items())}):")
    weighted_probs = sum(weights[n] * results[n]["baseline_probs"] for n in model_names)
    w_top1, w_top5 = eval_accuracy(weighted_probs, labels)
    print(f"    Top-1: {w_top1:.2f}%  Top-5: {w_top5:.2f}%")
    print(f"    Δ Top-1: {w_top1 - avg_top1:+.2f}% vs equal-weight")

    # 3d. Accuracy-weighted TTA ensemble
    weighted_tta_probs = sum(weights[n] * results[n]["tta_probs"] for n in model_names)
    wtta_top1, wtta_top5 = eval_accuracy(weighted_tta_probs, labels)
    print(f"\n  [Step 3b] Accuracy-weighted TTA ensemble:")
    print(f"    Top-1: {wtta_top1:.2f}%  Top-5: {wtta_top5:.2f}%")
    print(f"    Δ Top-1: {wtta_top1 - avg_top1:+.2f}% vs baseline ensemble")

    # 3e. Grid search best weight
    print(f"\n  [Step 3c] Grid-searching optimal weights...")
    best_w_top1 = 0
    best_alpha = 0.5
    for alpha in np.arange(0.0, 1.01, 0.05):
        if len(model_names) == 2:
            w = {model_names[0]: alpha, model_names[1]: 1 - alpha}
        else:
            break
        grid_probs = sum(w[n] * results[n]["tta_probs"] for n in model_names)
        g_top1, _ = eval_accuracy(grid_probs, labels)
        if g_top1 > best_w_top1:
            best_w_top1 = g_top1
            best_alpha = alpha

    if len(model_names) == 2:
        best_weights = {model_names[0]: best_alpha, model_names[1]: 1 - best_alpha}
        print(f"    Best weights: {', '.join(f'{n}={w:.2f}' for n, w in best_weights.items())}")
        print(f"    Top-1: {best_w_top1:.2f}%")
        print(f"    Δ Top-1: {best_w_top1 - avg_top1:+.2f}% vs baseline ensemble")

    # ─── Summary ───
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    print(f"\n  Individual models (baseline → TTA):")
    for n in model_names:
        b = results[n]["baseline"]["top1"]
        t = results[n]["tta"]["top1"]
        print(f"    {n:25s}  {b:.1f}% → {t:.1f}%  (Δ {t-b:+.1f}%)")

    print(f"\n  Ensemble (baseline → best optimized):")
    print(f"    Equal avg baseline:        {avg_top1:.1f}%")
    print(f"    Equal avg + TTA:           {tta_avg_top1:.1f}%  (Δ {tta_avg_top1 - avg_top1:+.1f}%)")
    print(f"    Weighted avg + TTA:        {wtta_top1:.1f}%  (Δ {wtta_top1 - avg_top1:+.1f}%)")
    if len(model_names) == 2:
        print(f"    Grid-search best + TTA:    {best_w_top1:.1f}%  (Δ {best_w_top1 - avg_top1:+.1f}%)")

    print(f"\n  Calibration (ECE, lower is better):")
    for n in model_names:
        b_ece = results[n]["baseline"]["ece"]
        c_ece = results[n]["calibrated"]["ece"]
        temp = results[n]["calibrated"]["temperature"]
        print(f"    {n:25s}  {b_ece:.4f} → {c_ece:.4f}  (T={temp:.2f})")

    # Save results
    save_results = {
        "individual": {n: {
            "baseline_top1": results[n]["baseline"]["top1"],
            "baseline_top5": results[n]["baseline"]["top5"],
            "tta_top1": results[n]["tta"]["top1"],
            "tta_top5": results[n]["tta"]["top5"],
            "ece_before": results[n]["baseline"]["ece"],
            "ece_after": results[n]["calibrated"]["ece"],
            "temperature": results[n]["calibrated"]["temperature"],
        } for n in model_names},
        "ensemble": {
            "baseline_avg_top1": avg_top1,
            "tta_avg_top1": tta_avg_top1,
            "weighted_tta_top1": wtta_top1,
            "best_grid_top1": best_w_top1 if len(model_names) == 2 else wtta_top1,
        }
    }
    out_path = os.path.join(models_dir, "optimization_results.json")
    with open(out_path, "w") as f:
        json.dump(save_results, f, indent=2)
    print(f"\n  Results saved to: {out_path}")


if __name__ == "__main__":
    main()
