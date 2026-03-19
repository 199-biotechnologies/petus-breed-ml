#!/usr/bin/env python3
"""
Download Stanford Dogs dataset (120 breeds, ~20K images).

Uses HuggingFace `amaye15/stanford-dogs` (reliable mirror with proper splits).
"""

import os
import sys

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"Data directory: {DATA_DIR}\n")

    train_dir = os.path.join(DATA_DIR, "train")
    test_dir = os.path.join(DATA_DIR, "test")

    # Check if already downloaded
    if os.path.isdir(train_dir) and os.path.isdir(test_dir):
        n_train = sum(len(f) for _, _, f in os.walk(train_dir))
        n_test = sum(len(f) for _, _, f in os.walk(test_dir))
        if n_train > 1000 and n_test > 1000:
            breeds = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
            print(f"Dataset already present: {len(breeds)} breeds, {n_train} train, {n_test} test")
            return

    from datasets import load_dataset

    print("Downloading Stanford Dogs from HuggingFace (amaye15/stanford-dogs)...")
    ds = load_dataset("amaye15/stanford-dogs")

    label_names = ds["train"].features["label"].names
    print(f"  {len(label_names)} breeds")

    for split_name, split_data, split_dir in [
        ("train", ds["train"], train_dir),
        ("test", ds["test"], test_dir),
    ]:
        print(f"\n  Saving {split_name} ({len(split_data)} images)...")
        for i, example in enumerate(split_data):
            label_idx = example["label"]
            breed_name = label_names[label_idx].replace(" ", "_")
            breed_dir = os.path.join(split_dir, breed_name)
            os.makedirs(breed_dir, exist_ok=True)

            img = example["pixel_values"]
            img_path = os.path.join(breed_dir, f"{i:05d}.jpg")
            if not os.path.exists(img_path):
                if img.mode != "RGB":
                    img = img.convert("RGB")
                img.save(img_path, "JPEG", quality=95)

            if (i + 1) % 2000 == 0:
                print(f"    {i+1}/{len(split_data)}")

    # Final count
    n_train = sum(len(f) for _, _, f in os.walk(train_dir))
    n_test = sum(len(f) for _, _, f in os.walk(test_dir))
    breeds = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
    print(f"\nDone! {len(breeds)} breeds, {n_train} train, {n_test} test")
    print(f"Sample breeds: {', '.join(sorted(breeds)[:5])}...")


if __name__ == "__main__":
    main()
