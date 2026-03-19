"""
Active learning: hard example mining and confused breed pair detection.

Identifies which breeds the model confuses most often,
enabling targeted data collection or specialized training.
"""

import numpy as np
from collections import defaultdict


def find_confused_pairs(
    predictions: np.ndarray,
    labels: np.ndarray,
    probabilities: np.ndarray,
    breed_names: list[str],
    top_k: int = 20,
) -> list[dict]:
    """Find breed pairs that are most commonly confused.

    Args:
        predictions: (N,) predicted class indices
        labels: (N,) ground truth class indices
        probabilities: (N, C) prediction probabilities
        breed_names: list of breed name strings
        top_k: number of confused pairs to return

    Returns:
        List of dicts with breed_a, breed_b, count, avg_confidence
    """
    confusion_counts = defaultdict(lambda: {"count": 0, "confidences": []})

    for pred, label, probs in zip(predictions, labels, probabilities):
        if pred != label:
            pair = (min(label, pred), max(label, pred))
            confusion_counts[pair]["count"] += 1
            confusion_counts[pair]["confidences"].append(probs[pred])

    # Sort by count
    sorted_pairs = sorted(confusion_counts.items(), key=lambda x: -x[1]["count"])

    results = []
    for (a, b), data in sorted_pairs[:top_k]:
        results.append({
            "breed_a": breed_names[a],
            "breed_b": breed_names[b],
            "breed_a_idx": int(a),
            "breed_b_idx": int(b),
            "confusion_count": data["count"],
            "avg_wrong_confidence": float(np.mean(data["confidences"])),
        })

    return results


def find_hard_examples(
    probabilities: np.ndarray,
    labels: np.ndarray,
    breed_names: list[str],
    confidence_threshold: float = 0.5,
) -> dict:
    """Find hard-to-classify examples per breed.

    Returns:
        Dict mapping breed names to their hardness metrics
    """
    predictions = probabilities.argmax(axis=1)
    confidences = probabilities.max(axis=1)

    breed_stats = {}
    for breed_idx, breed_name in enumerate(breed_names):
        mask = labels == breed_idx
        if mask.sum() == 0:
            continue

        breed_probs = probabilities[mask]
        breed_preds = predictions[mask]
        breed_confs = confidences[mask]
        breed_labels = labels[mask]

        correct = (breed_preds == breed_labels)
        accuracy = correct.mean()
        avg_confidence = breed_confs.mean()
        low_confidence = (breed_confs < confidence_threshold).sum()

        # Which breeds does this breed get confused with?
        wrong_mask = ~correct
        if wrong_mask.sum() > 0:
            wrong_preds = breed_preds[wrong_mask]
            confused_with = defaultdict(int)
            for wp in wrong_preds:
                confused_with[breed_names[wp]] += 1
            top_confused = sorted(confused_with.items(), key=lambda x: -x[1])[:5]
        else:
            top_confused = []

        breed_stats[breed_name] = {
            "accuracy": float(accuracy),
            "avg_confidence": float(avg_confidence),
            "n_samples": int(mask.sum()),
            "n_low_confidence": int(low_confidence),
            "top_confused_with": top_confused,
        }

    return breed_stats


def prioritize_breeds_for_collection(breed_stats: dict, top_k: int = 10) -> list[str]:
    """Rank breeds by need for additional training data.

    Criteria: low accuracy + low confidence = high priority.
    """
    scored = []
    for breed, stats in breed_stats.items():
        # Priority score: lower accuracy and confidence = higher priority
        score = (1 - stats["accuracy"]) * 0.6 + (1 - stats["avg_confidence"]) * 0.4
        scored.append((breed, score, stats))

    scored.sort(key=lambda x: -x[1])

    return [
        {"breed": breed, "priority_score": round(score, 3), **stats}
        for breed, score, stats in scored[:top_k]
    ]
