"""
2-phase training pipeline for dog breed classification.

Adapted from unblurml/src/train.py:
- Phase 1: Frozen backbone, train MLP head only, OneCycleLR warmup
- Phase 2: Unfreeze backbone, differential LR (backbone 0.1×), CosineAnnealingLR
- Label smoothing, MixUp/CutMix at batch level
- MPS-optimized for M4 Max
"""

import os
import json
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from .registry import get_backbone, list_backbones
from .heads.mlp_head import MLPHead
from .augmentations import mixup_data, cutmix_data, mixup_criterion


NUM_CLASSES = 120


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class BreedClassifier(nn.Module):
    """Backbone + MLP head — the full model for a single backbone."""

    def __init__(self, backbone_name: str, num_classes: int = NUM_CLASSES, pretrained: bool = True):
        super().__init__()
        self.backbone = get_backbone(backbone_name, pretrained=pretrained)
        self.head = MLPHead(self.backbone.embed_dim, num_classes)
        self.backbone_name = backbone_name

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)

    def freeze_backbone(self):
        self.backbone.freeze()

    def unfreeze_backbone(self, **kwargs):
        self.backbone.unfreeze(**kwargs)

    def get_param_groups(self, lr: float, backbone_lr_mult: float = 0.1) -> list[dict]:
        groups = self.backbone.get_param_groups(lr, backbone_lr_mult)
        groups.append({"params": list(self.head.parameters()), "lr": lr})
        return groups

    def get_preprocess_config(self) -> dict:
        return self.backbone.get_preprocess_config()


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    scheduler=None,
    mixup_alpha: float = 0.2,
    cutmix_alpha: float = 1.0,
    mix_prob: float = 0.5,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="Train", leave=False)
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        # MixUp / CutMix
        r = np.random.random()
        if r < mix_prob / 2:
            images, targets_a, targets_b, lam = mixup_data(images, labels, mixup_alpha)
            use_mix = True
        elif r < mix_prob:
            images, targets_a, targets_b, lam = cutmix_data(images, labels, cutmix_alpha)
            use_mix = True
        else:
            use_mix = False

        optimizer.zero_grad()
        outputs = model(images)

        if use_mix:
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        else:
            loss = criterion(outputs, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix(
            loss=f"{loss.item():.3f}",
            acc=f"{100.*correct/total:.1f}%",
            lr=f"{optimizer.param_groups[-1]['lr']:.1e}",
        )

    return total_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    model.eval()
    total_loss = 0.0
    correct = 0
    correct_top5 = 0
    total = 0

    for images, labels in tqdm(loader, desc="Eval", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()

        _, top5_pred = outputs.topk(5, 1, True, True)
        correct_top5 += top5_pred.eq(labels.view(-1, 1).expand_as(top5_pred)).sum().item()
        total += labels.size(0)

    return {
        "loss": total_loss / total,
        "top1_acc": 100.0 * correct / total,
        "top5_acc": 100.0 * correct_top5 / total,
    }


def train_model(
    backbone_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 40,
    warmup_epochs: int = 2,
    lr: float = 1e-3,
    backbone_lr_mult: float = 0.01,  # 1/100th — Codex+Gemini recommendation
    label_smoothing: float = 0.1,
    mixup_alpha: float = 0.8,
    cutmix_alpha: float = 1.0,
    mix_prob: float = 0.5,
    no_aug_final_epochs: int = 5,  # Turn off MixUp/CutMix for last N epochs
    unfreeze_warmup_epochs: int = 3,  # Linear warmup after unfreeze
    early_stop_patience: int = 8,
    output_dir: str = "models",
    time_limit_minutes: float = 120.0,
    # Keep test_loader for backward compat but don't use for selection
    test_loader: DataLoader = None,
) -> dict:
    """Train with improved recipe (Codex+Gemini recommendations).

    Key improvements over v1:
    - Val loader for checkpoint selection (not test)
    - Much lower backbone LR (1/100th of head, was 1/10th)
    - Linear warmup after unfreeze (prevents gradient shock)
    - No MixUp/CutMix during initial unfreeze epochs
    - MixUp/CutMix turned off for final epochs (sharpen features)
    - Early stopping with patience
    """
    # Handle backward compat: if val_loader not provided, use test_loader with warning
    if val_loader is None and test_loader is not None:
        print("  WARNING: Using test_loader for validation (no val_loader provided)")
        val_loader = test_loader

    device = get_device()
    print(f"\n{'='*60}")
    print(f"Training: {backbone_name} (v2 recipe)")
    print(f"Device: {device}")
    print(f"Backbone LR mult: {backbone_lr_mult} (1/{int(1/backbone_lr_mult)}th of head)")
    print(f"{'='*60}")

    model = BreedClassifier(backbone_name)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    os.makedirs(output_dir, exist_ok=True)

    best_val_acc = 0.0
    patience_counter = 0
    history = []
    start_time = time.time()
    time_limit_sec = time_limit_minutes * 60

    # ─── Phase 1: Frozen backbone, train head only (no augmentation) ───
    print(f"\n[Phase 1] Frozen backbone — training head ({warmup_epochs} epochs)")
    model.freeze_backbone()
    head_params = [p for p in model.parameters() if p.requires_grad]
    warmup_optimizer = optim.AdamW(head_params, lr=lr, weight_decay=0.01)

    for epoch in range(warmup_epochs):
        warmup_scheduler = optim.lr_scheduler.OneCycleLR(
            warmup_optimizer,
            max_lr=lr,
            steps_per_epoch=len(train_loader),
            epochs=1,
            pct_start=0.3,
        )

        epoch_start = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, warmup_optimizer, device,
            scheduler=warmup_scheduler, mixup_alpha=0, cutmix_alpha=0, mix_prob=0,
        )
        val_metrics = evaluate(model, val_loader, criterion, device)
        epoch_time = time.time() - epoch_start

        record = {
            "epoch": epoch + 1, "phase": 1,
            "train_loss": train_loss, "train_acc": train_acc,
            **val_metrics, "epoch_time": epoch_time,
        }
        history.append(record)
        print(
            f"  Epoch {epoch+1}: Train {train_acc:.1f}% | "
            f"Val T1={val_metrics['top1_acc']:.1f}% T5={val_metrics['top5_acc']:.1f}% | "
            f"{epoch_time:.0f}s"
        )

        if val_metrics["top1_acc"] > best_val_acc:
            best_val_acc = val_metrics["top1_acc"]
            _save_checkpoint(model, backbone_name, epoch + 1, val_metrics, output_dir)

    # ─── Phase 2: Unfreeze backbone with careful LR recipe ───
    remaining = epochs - warmup_epochs
    print(f"\n[Phase 2] Unfrozen backbone — {remaining} epochs")
    print(f"  Unfreeze warmup: {unfreeze_warmup_epochs} epochs (no MixUp/CutMix)")
    print(f"  Final no-aug stage: last {no_aug_final_epochs} epochs")

    model.unfreeze_backbone()
    param_groups = model.get_param_groups(lr, backbone_lr_mult)
    # Store initial LR for warmup reference
    for pg in param_groups:
        pg['initial_lr'] = pg['lr']
    # Use weight_decay=0.05 for better regularization (Gemini rec)
    optimizer = optim.AdamW(param_groups, weight_decay=0.05)

    # Cosine schedule with warmup
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=remaining, eta_min=1e-6)

    for epoch in range(remaining):
        elapsed = time.time() - start_time
        if elapsed > time_limit_sec:
            print(f"\nTime limit reached ({time_limit_minutes}min). Stopping.")
            break

        epoch_num = warmup_epochs + epoch + 1

        # Decide MixUp/CutMix schedule:
        # - First unfreeze_warmup_epochs: NO augmentation (prevent gradient shock)
        # - Middle epochs: full augmentation
        # - Last no_aug_final_epochs: NO augmentation (sharpen features)
        if epoch < unfreeze_warmup_epochs:
            # Gentle unfreeze warmup — no batch augmentation
            ep_mix_prob = 0
            ep_mixup = 0
            ep_cutmix = 0
            phase_label = "2a-warmup"
        elif epoch >= remaining - no_aug_final_epochs:
            # Final refinement — no batch augmentation
            ep_mix_prob = 0
            ep_mixup = 0
            ep_cutmix = 0
            phase_label = "2c-refine"
        else:
            # Full training with augmentation
            ep_mix_prob = mix_prob
            ep_mixup = mixup_alpha
            ep_cutmix = cutmix_alpha
            phase_label = "2b-train"

        # Linear LR warmup during unfreeze warmup phase
        if epoch < unfreeze_warmup_epochs:
            warmup_factor = (epoch + 1) / unfreeze_warmup_epochs
            for pg in optimizer.param_groups:
                pg['lr'] = pg['initial_lr'] * warmup_factor if 'initial_lr' in pg else pg['lr']

        epoch_start = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            scheduler=None,
            mixup_alpha=ep_mixup, cutmix_alpha=ep_cutmix, mix_prob=ep_mix_prob,
        )
        scheduler.step()
        val_metrics = evaluate(model, val_loader, criterion, device)
        epoch_time = time.time() - epoch_start

        record = {
            "epoch": epoch_num, "phase": phase_label,
            "train_loss": train_loss, "train_acc": train_acc,
            **val_metrics, "epoch_time": epoch_time,
        }
        history.append(record)

        improved = ""
        if val_metrics["top1_acc"] > best_val_acc:
            best_val_acc = val_metrics["top1_acc"]
            _save_checkpoint(model, backbone_name, epoch_num, val_metrics, output_dir)
            improved = f" *NEW BEST*"
            patience_counter = 0
        else:
            patience_counter += 1

        bb_lr = optimizer.param_groups[0]['lr']
        head_lr = optimizer.param_groups[-1]['lr']
        print(
            f"  E{epoch_num} [{phase_label}]: Train {train_acc:.1f}% | "
            f"Val T1={val_metrics['top1_acc']:.1f}% T5={val_metrics['top5_acc']:.1f}% | "
            f"LR bb={bb_lr:.1e} head={head_lr:.1e} | {epoch_time:.0f}s{improved}"
        )

        # Early stopping
        if patience_counter >= early_stop_patience:
            print(f"\n  Early stopping: no improvement for {early_stop_patience} epochs")
            break

    # Save history
    hist_path = os.path.join(output_dir, f"{backbone_name}_history.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)

    total_time = time.time() - start_time
    print(f"\n{backbone_name} complete — {total_time/60:.1f}min, best val top1: {best_val_acc:.1f}%")

    return {"backbone": backbone_name, "best_top1": best_val_acc, "history": history}


def _save_checkpoint(model, backbone_name, epoch, val_metrics, output_dir):
    save_path = os.path.join(output_dir, f"{backbone_name}_best.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "backbone_name": backbone_name,
        "epoch": epoch,
        "val_top1": val_metrics["top1_acc"],
        "val_top5": val_metrics["top5_acc"],
        "num_classes": NUM_CLASSES,
    }, save_path)


def load_model(backbone_name: str, checkpoint_path: str, device: torch.device = None) -> BreedClassifier:
    """Load a trained model from checkpoint."""
    if device is None:
        device = get_device()
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = BreedClassifier(backbone_name, num_classes=ckpt.get("num_classes", NUM_CLASSES), pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model
