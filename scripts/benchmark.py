#!/usr/bin/env python3
"""
Full evaluation suite: per-model metrics, ensemble, calibration, confusion analysis.

Usage:
    python scripts/benchmark.py
    python scripts/benchmark.py --image /path/to/dog.jpg   # Single image inference
"""

import os
import sys
import argparse
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets

from src.backbones import *  # noqa
from src.registry import get_backbone, list_backbones
from src.dataset import get_transforms, clean_breed_name
from src.train import load_model, get_device, evaluate, NUM_CLASSES
from src.ensemble import StackingEnsemble, collect_predictions
from src.calibration import compute_ece, reliability_diagram_data
from src.active_learning import find_confused_pairs, find_hard_examples, prioritize_breeds_for_collection
from src.inference import predict_single, predict_ensemble, print_predictions


def benchmark_models(models_dir: str, data_dir: str, batch_size: int = 64):
    """Run full benchmark on all trained models."""
    device = get_device()
    test_dir = os.path.join(data_dir, "test")

    # Find models
    checkpoints = {}
    for fname in os.listdir(models_dir):
        if fname.endswith("_best.pt") and "distilled" not in fname:
            backbone_name = fname.replace("_best.pt", "")
            checkpoints[backbone_name] = os.path.join(models_dir, fname)

    results = {}

    for backbone_name, ckpt_path in sorted(checkpoints.items()):
        print(f"\n{'='*60}")
        print(f"Benchmarking: {backbone_name}")
        print(f"{'='*60}")

        model = load_model(backbone_name, ckpt_path, device)
        preprocess_config = model.get_preprocess_config()

        test_transform = get_transforms(preprocess_config, is_train=False)
        test_ds = datasets.ImageFolder(test_dir, transform=test_transform)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                                 num_workers=4, pin_memory=True)

        breed_names = [clean_breed_name(c) for c in test_ds.classes]

        # Standard eval
        criterion = torch.nn.CrossEntropyLoss()
        metrics = evaluate(model, test_loader, criterion, device)

        # Collect probs for detailed analysis
        probs, labels = collect_predictions(model, test_loader, device)

        # ECE
        ece = compute_ece(probs, labels)

        # Confused pairs
        preds = probs.argmax(axis=1)
        confused = find_confused_pairs(preds, labels, probs, breed_names, top_k=10)

        # Hard breeds
        breed_stats = find_hard_examples(probs, labels, breed_names)
        priorities = prioritize_breeds_for_collection(breed_stats, top_k=5)

        results[backbone_name] = {
            "top1_acc": round(metrics["top1_acc"], 2),
            "top5_acc": round(metrics["top5_acc"], 2),
            "ece": round(ece, 4),
            "top_confused_pairs": confused[:5],
            "hardest_breeds": priorities,
        }

        print(f"  Top-1: {metrics['top1_acc']:.1f}%")
        print(f"  Top-5: {metrics['top5_acc']:.1f}%")
        print(f"  ECE:   {ece:.4f}")
        print(f"\n  Top confused pairs:")
        for c in confused[:5]:
            print(f"    {c['breed_a']} ↔ {c['breed_b']}: {c['confusion_count']} errors")
        print(f"\n  Hardest breeds:")
        for p in priorities:
            print(f"    {p['breed']:30s} acc={p['accuracy']:.0%}")

        del model
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    return results


def single_image_inference(image_path: str, models_dir: str, data_dir: str):
    """Run inference on a single image with all available models."""
    device = get_device()
    test_dir = os.path.join(data_dir, "test")

    # Get breed names
    breed_names = sorted([
        clean_breed_name(d) for d in os.listdir(test_dir)
        if os.path.isdir(os.path.join(test_dir, d))
    ])

    checkpoints = {}
    for fname in os.listdir(models_dir):
        if fname.endswith("_best.pt"):
            backbone_name = fname.replace("_best.pt", "").replace("_distilled", " (distilled)")
            checkpoints[backbone_name] = os.path.join(models_dir, fname)

    print(f"\nImage: {image_path}")
    print(f"Models: {list(checkpoints.keys())}\n")

    models = {}
    for backbone_name, ckpt_path in sorted(checkpoints.items()):
        clean_name = backbone_name.replace(" (distilled)", "")
        model = load_model(clean_name, ckpt_path, device)
        models[backbone_name] = model

        results = predict_single(model, image_path, breed_names, device)
        print_predictions(results, title=f"{backbone_name}")

    if len(models) > 1:
        results = predict_ensemble(models, image_path, breed_names, device)
        print_predictions(results, title="Ensemble (average)")


def main():
    parser = argparse.ArgumentParser(description="Benchmark breed classifiers")
    parser.add_argument("--image", type=str, help="Single image for inference")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output", default="models/benchmark_results.json")
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(project_root, args.models_dir)
    data_dir = os.path.join(project_root, args.data_dir)

    if args.image:
        single_image_inference(args.image, models_dir, data_dir)
    else:
        results = benchmark_models(models_dir, data_dir, args.batch_size)

        # Save results
        output_path = os.path.join(project_root, args.output)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
