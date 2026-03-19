#!/usr/bin/env python3
"""
Build stacking ensemble from trained backbone models.

1. Load all trained models
2. Collect predictions on test set
3. Fit meta-learner
4. Evaluate ensemble
5. Save ensemble

Usage:
    python scripts/build_ensemble.py
    python scripts/build_ensemble.py --models-dir models --data-dir data
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets

from src.backbones import *  # noqa — trigger @register
from src.registry import get_backbone
from src.dataset import get_transforms
from src.train import load_model, get_device, NUM_CLASSES
from src.ensemble import StackingEnsemble, collect_predictions


def main():
    parser = argparse.ArgumentParser(description="Build stacking ensemble")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output", default="models/ensemble.pkl")
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(project_root, args.models_dir)
    data_dir = os.path.join(project_root, args.data_dir)
    output_path = os.path.join(project_root, args.output)

    device = get_device()
    print(f"Device: {device}")

    # Find all trained model checkpoints
    checkpoints = {}
    for fname in os.listdir(models_dir):
        if fname.endswith("_best.pt") and not fname.endswith("_distilled_best.pt"):
            backbone_name = fname.replace("_best.pt", "")
            checkpoints[backbone_name] = os.path.join(models_dir, fname)

    if len(checkpoints) < 2:
        print(f"Need at least 2 trained models, found {len(checkpoints)}: {list(checkpoints.keys())}")
        sys.exit(1)

    print(f"Found {len(checkpoints)} models: {list(checkpoints.keys())}\n")

    # Collect predictions from each model
    model_predictions_train = {}
    model_predictions_test = {}
    labels_train = None
    labels_test = None

    for backbone_name, ckpt_path in sorted(checkpoints.items()):
        print(f"\nProcessing: {backbone_name}")
        model = load_model(backbone_name, ckpt_path, device)
        preprocess_config = model.get_preprocess_config()

        # Build test loader with this backbone's preprocessing
        test_dir = os.path.join(data_dir, "test")
        train_dir = os.path.join(data_dir, "train")

        test_transform = get_transforms(preprocess_config, is_train=False)
        train_transform = get_transforms(preprocess_config, is_train=False)  # No aug for meta-learner

        test_ds = datasets.ImageFolder(test_dir, transform=test_transform)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                                 num_workers=4, pin_memory=True)

        train_ds = datasets.ImageFolder(train_dir, transform=train_transform)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False,
                                  num_workers=4, pin_memory=True)

        # Collect predictions
        test_probs, test_labels = collect_predictions(model, test_loader, device)
        train_probs, train_labels = collect_predictions(model, train_loader, device)

        model_predictions_test[backbone_name] = test_probs
        model_predictions_train[backbone_name] = train_probs

        if labels_test is None:
            labels_test = test_labels
            labels_train = train_labels

        # Per-model accuracy
        top1 = 100.0 * np.mean(test_probs.argmax(axis=1) == test_labels)
        print(f"  {backbone_name} standalone: {top1:.1f}% top-1")

        del model
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    # Build ensemble
    print(f"\n{'='*60}")
    print("Fitting stacking meta-learner...")
    print(f"{'='*60}")

    ensemble = StackingEnsemble()

    # Fit on train set
    fit_metrics = ensemble.fit(model_predictions_train, labels_train)

    # Evaluate on test set
    test_metrics = ensemble.evaluate(model_predictions_test, labels_test)
    print(f"\nEnsemble Test Results:")
    print(f"  Top-1: {test_metrics['top1_acc']:.1f}%")
    print(f"  Top-5: {test_metrics['top5_acc']:.1f}%")

    # Compare with simple averaging
    avg_probs = np.mean(list(model_predictions_test.values()), axis=0)
    avg_top1 = 100.0 * np.mean(avg_probs.argmax(axis=1) == labels_test)
    print(f"\nSimple averaging baseline: {avg_top1:.1f}% top-1")
    print(f"Stacking improvement: +{test_metrics['top1_acc'] - avg_top1:.1f}%")

    # Save
    ensemble.save(output_path)
    print(f"\nEnsemble saved to: {output_path}")


if __name__ == "__main__":
    main()
