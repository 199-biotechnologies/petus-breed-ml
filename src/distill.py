"""
Knowledge distillation: ensemble teacher → single student model.

Uses KL divergence soft loss + hard CE loss.
Teacher = stacking ensemble (or any model producing soft targets)
Student = EfficientNetV2-S (lightweight, fast for deployment)
"""

import os
import time
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from .train import BreedClassifier, get_device, evaluate, NUM_CLASSES


class DistillationLoss(nn.Module):
    """Combined distillation loss: soft KL + hard CE."""

    def __init__(self, temperature: float = 4.0, alpha_soft: float = 0.7):
        super().__init__()
        self.temperature = temperature
        self.alpha_soft = alpha_soft
        self.alpha_hard = 1.0 - alpha_soft

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        # Soft loss: KL divergence between soft targets
        soft_student = F.log_softmax(student_logits / self.temperature, dim=1)
        soft_teacher = F.softmax(teacher_logits / self.temperature, dim=1)
        soft_loss = F.kl_div(soft_student, soft_teacher, reduction="batchmean") * (self.temperature ** 2)

        # Hard loss: standard cross-entropy
        hard_loss = F.cross_entropy(student_logits, labels)

        return self.alpha_soft * soft_loss + self.alpha_hard * hard_loss


@torch.no_grad()
def collect_teacher_logits(
    teacher_models: list[BreedClassifier],
    loader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Collect averaged teacher logits over the full dataset.

    Args:
        teacher_models: list of trained teacher models
        loader: data loader
        device: compute device
    Returns:
        (teacher_logits, labels) — both on CPU
    """
    all_logits = []
    all_labels = []

    for images, labels in tqdm(loader, desc="Collecting teacher logits", leave=False):
        images = images.to(device)
        batch_logits = []
        for model in teacher_models:
            model.eval()
            logits = model(images)
            batch_logits.append(logits)
        # Average logits across teachers
        avg_logits = torch.stack(batch_logits).mean(dim=0)
        all_logits.append(avg_logits.cpu())
        all_labels.append(labels)

    return torch.cat(all_logits), torch.cat(all_labels)


def distill(
    teacher_models: list[BreedClassifier],
    student_backbone: str = "efficientnetv2_s",
    train_loader: DataLoader = None,
    test_loader: DataLoader = None,
    epochs: int = 20,
    lr: float = 5e-4,
    temperature: float = 4.0,
    alpha_soft: float = 0.7,
    output_dir: str = "models",
) -> dict:
    """Distill ensemble knowledge into a single student model."""
    device = get_device()
    print(f"\n{'='*60}")
    print(f"Knowledge Distillation")
    print(f"Teacher: {len(teacher_models)} models → Student: {student_backbone}")
    print(f"Temperature: {temperature}, Alpha (soft): {alpha_soft}")
    print(f"{'='*60}")

    # Pre-collect teacher logits for efficiency
    print("\nPre-collecting teacher logits...")
    teacher_logits, teacher_labels = collect_teacher_logits(teacher_models, train_loader, device)

    # Create student
    student = BreedClassifier(student_backbone, pretrained=True)
    student = student.to(device)

    # Phase 1: Train head only
    student.freeze_backbone()
    criterion = DistillationLoss(temperature=temperature, alpha_soft=alpha_soft)
    ce_criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    head_params = [p for p in student.parameters() if p.requires_grad]
    optimizer = optim.AdamW(head_params, lr=lr, weight_decay=0.01)

    print("\n[Phase 1] Training student head (5 epochs)...")
    for epoch in range(min(5, epochs)):
        student.train()
        total_loss = 0
        correct = 0
        total = 0
        batch_idx = 0

        for images, labels in tqdm(train_loader, desc=f"Distill E{epoch+1}", leave=False):
            images = images.to(device)
            labels = labels.to(device)

            # Get pre-computed teacher logits for this batch
            start_idx = batch_idx * train_loader.batch_size
            end_idx = start_idx + images.size(0)
            t_logits = teacher_logits[start_idx:end_idx].to(device)

            optimizer.zero_grad()
            s_logits = student(images)
            loss = criterion(s_logits, t_logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            _, predicted = s_logits.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
            batch_idx += 1

        val = evaluate(student, test_loader, ce_criterion, device)
        print(f"  Epoch {epoch+1}: Loss {total_loss/total:.4f} | Val T1={val['top1_acc']:.1f}%")

    # Phase 2: Unfreeze and fine-tune
    student.unfreeze_backbone()
    param_groups = student.get_param_groups(lr * 0.5, backbone_lr_mult=0.05)
    optimizer = optim.AdamW(param_groups, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs - 5, eta_min=1e-6)

    best_acc = 0.0
    print(f"\n[Phase 2] Full fine-tuning ({epochs - 5} epochs)...")

    for epoch in range(5, epochs):
        student.train()
        total_loss = 0
        total = 0
        batch_idx = 0

        for images, labels in tqdm(train_loader, desc=f"Distill E{epoch+1}", leave=False):
            images = images.to(device)
            labels = labels.to(device)

            start_idx = batch_idx * train_loader.batch_size
            end_idx = start_idx + images.size(0)
            t_logits = teacher_logits[start_idx:end_idx].to(device)

            optimizer.zero_grad()
            s_logits = student(images)
            loss = criterion(s_logits, t_logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            total += labels.size(0)
            batch_idx += 1

        scheduler.step()
        val = evaluate(student, test_loader, ce_criterion, device)
        print(f"  Epoch {epoch+1}: Loss {total_loss/total:.4f} | Val T1={val['top1_acc']:.1f}% T5={val['top5_acc']:.1f}%")

        if val["top1_acc"] > best_acc:
            best_acc = val["top1_acc"]
            save_path = os.path.join(output_dir, f"{student_backbone}_distilled_best.pt")
            torch.save({
                "model_state_dict": student.state_dict(),
                "backbone_name": student_backbone,
                "val_top1": val["top1_acc"],
                "val_top5": val["top5_acc"],
                "num_classes": NUM_CLASSES,
                "distilled": True,
            }, save_path)
            print(f"  -> Saved distilled model: {best_acc:.1f}%")

    print(f"\nDistillation complete. Best student accuracy: {best_acc:.1f}%")
    return {"student": student_backbone, "best_top1": best_acc}
