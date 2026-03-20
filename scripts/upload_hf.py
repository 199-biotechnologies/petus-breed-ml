#!/usr/bin/env python3
"""
Upload trained breed classifier to HuggingFace Hub.

Usage:
    python scripts/upload_hf.py --backbone convnextv2_tiny --repo dboris/petus-breed-classifier
    python scripts/upload_hf.py --backbone convnextv2_tiny --repo dboris/petus-breed-classifier --private
"""

import os
import sys
import json
import argparse
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from huggingface_hub import HfApi, create_repo


MODEL_CARD_TEMPLATE = """---
license: apache-2.0
tags:
- image-classification
- dog-breeds
- fine-grained
- arcface
- convnext
- pytorch
datasets:
- stanford-dogs
metrics:
- accuracy
pipeline_tag: image-classification
model-index:
- name: {model_name}
  results:
  - task:
      type: image-classification
    dataset:
      name: Stanford Dogs
      type: stanford-dogs
    metrics:
    - name: Top-1 Accuracy (Val)
      type: accuracy
      value: {val_top1}
    - name: Top-5 Accuracy (Val)
      type: accuracy
      value: {val_top5}
---

# {model_name}

Dog breed classifier trained on Stanford Dogs (120 breeds) using **{backbone}** backbone with **ArcFace** angular margin loss and progressive resizing.

## Model Details

| Property | Value |
|----------|-------|
| Backbone | {backbone} |
| Loss | ArcFace (s={arcface_s}, m={arcface_m}) |
| Parameters | {params} |
| Input Size | {input_size}px |
| Val Top-1 | **{val_top1}%** |
| Val Top-5 | **{val_top5}%** |
| Training | 2-phase (frozen head → unfrozen backbone) |
| Progressive Resize | 224 → 336px |

## Training Recipe (v3)

1. **Phase 1**: Frozen backbone, train ArcFace head only (2 epochs)
2. **Phase 2**: Unfreeze backbone with 1/100th LR, cosine annealing (48 epochs)
   - 3-epoch linear LR warmup after unfreeze
   - Progressive resize from 224→336 mid-training
   - ArcFace angular margin loss (no MixUp/CutMix needed)
   - Early stopping with patience=10

## Usage

```python
import torch
from torchvision import transforms
from PIL import Image

# Load model
checkpoint = torch.load("convnextv2_tiny_best.pt", map_location="cpu")

# Preprocess
transform = transforms.Compose([
    transforms.Resize(384),  # 336 * 1.14
    transforms.CenterCrop(336),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

image = Image.open("dog.jpg").convert("RGB")
input_tensor = transform(image).unsqueeze(0)

# Inference
model.eval()
with torch.no_grad():
    logits = model(input_tensor)
    pred = logits.argmax(dim=1).item()
    confidence = logits.softmax(dim=1).max().item()
```

## Breeds

120 dog breeds from the Stanford Dogs dataset (synsets from ImageNet).

## Citation

```bibtex
@misc{{petus-breed-ml,
  author = {{199 Biotechnologies}},
  title = {{Petus Breed Classifier}},
  year = {{2026}},
  url = {{https://github.com/199-biotechnologies/petus-breed-ml}}
}}
```

## License

Apache 2.0
"""


def main():
    parser = argparse.ArgumentParser(description="Upload model to HuggingFace")
    parser.add_argument("--backbone", required=True, help="Backbone name (e.g. convnextv2_tiny)")
    parser.add_argument("--repo", required=True, help="HF repo (e.g. dboris/petus-breed-classifier)")
    parser.add_argument("--models-dir", default="models", help="Models directory")
    parser.add_argument("--private", action="store_true", help="Make repo private")
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(project_root, args.models_dir)

    # Load checkpoint
    ckpt_path = os.path.join(models_dir, f"{args.backbone}_best.pt")
    if not os.path.exists(ckpt_path):
        print(f"ERROR: Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    print(f"Loaded checkpoint: {ckpt_path}")
    print(f"  Epoch: {ckpt.get('epoch', '?')}")
    print(f"  Val Top-1: {ckpt.get('val_top1', '?'):.1f}%")
    print(f"  Val Top-5: {ckpt.get('val_top5', '?'):.1f}%")

    # Load history if available
    hist_path = os.path.join(models_dir, f"{args.backbone}_history.json")
    history = None
    if os.path.exists(hist_path):
        with open(hist_path) as f:
            history = json.load(f)

    # Count params from checkpoint
    total_params = sum(v.numel() for v in ckpt["model_state_dict"].values())

    # Create HF repo
    api = HfApi()
    try:
        create_repo(args.repo, private=args.private, exist_ok=True)
        print(f"Repo created/exists: {args.repo}")
    except Exception as e:
        print(f"Repo creation: {e}")

    # Generate model card
    model_card = MODEL_CARD_TEMPLATE.format(
        model_name=f"Petus Breed Classifier ({args.backbone})",
        backbone=args.backbone,
        val_top1=f"{ckpt.get('val_top1', 0):.1f}",
        val_top5=f"{ckpt.get('val_top5', 0):.1f}",
        params=f"{total_params:,}",
        input_size=336,
        arcface_s=30.0,
        arcface_m=0.3,
    )

    # Upload files
    with tempfile.TemporaryDirectory() as tmpdir:
        # Write model card
        readme_path = os.path.join(tmpdir, "README.md")
        with open(readme_path, "w") as f:
            f.write(model_card)

        # Upload checkpoint
        print(f"\nUploading checkpoint ({os.path.getsize(ckpt_path) / 1e6:.1f}MB)...")
        api.upload_file(
            path_or_fileobj=ckpt_path,
            path_in_repo=f"{args.backbone}_best.pt",
            repo_id=args.repo,
        )

        # Upload history
        if history:
            api.upload_file(
                path_or_fileobj=hist_path,
                path_in_repo=f"{args.backbone}_history.json",
                repo_id=args.repo,
            )

        # Upload model card
        api.upload_file(
            path_or_fileobj=readme_path,
            path_in_repo="README.md",
            repo_id=args.repo,
        )

        # Upload source code for reproducibility
        src_files = [
            "src/train.py",
            "src/losses.py",
            "src/dataset.py",
            "src/augmentations.py",
            "src/heads/mlp_head.py",
            "src/backbones/timm_backbone.py",
            "src/registry.py",
        ]
        for src_file in src_files:
            src_path = os.path.join(project_root, src_file)
            if os.path.exists(src_path):
                api.upload_file(
                    path_or_fileobj=src_path,
                    path_in_repo=src_file,
                    repo_id=args.repo,
                )

    print(f"\nDone! Model published at: https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
