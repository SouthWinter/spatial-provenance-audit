"""Metric primitives for RECAP."""

from __future__ import annotations

import math


def mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def rank_norm(values: list[float]) -> list[float]:
    if not values:
        return []
    if len(values) == 1:
        return [0.0]
    pairs = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(pairs):
        j = i + 1
        while j < len(pairs) and pairs[j][1] == pairs[i][1]:
            j += 1
        avg_rank = (i + j - 1) / 2.0
        for k in range(i, j):
            ranks[pairs[k][0]] = avg_rank / (len(values) - 1)
        i = j
    return ranks


def roc_auc(labels: list[float], scores: list[float]) -> float:
    if not labels:
        return 0.0
    positives = sum(1 for label in labels if label > 0.5)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return 0.5

    pairs = sorted(zip(scores, labels), key=lambda item: item[0])
    ranks = [0.0] * len(pairs)
    i = 0
    while i < len(pairs):
        j = i + 1
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[k] = avg_rank
        i = j

    rank_sum_pos = sum(rank for rank, (_, label) in zip(ranks, pairs) if label > 0.5)
    return (rank_sum_pos - positives * (positives + 1) / 2.0) / (positives * negatives)


def average_precision(labels: list[float], scores: list[float]) -> float:
    if not labels:
        return 0.0
    positives = sum(1 for label in labels if label > 0.5)
    if positives == 0:
        return 0.0
    ordered = sorted(zip(scores, labels), key=lambda item: item[0], reverse=True)
    true_positives = 0.0
    precision_sum = 0.0
    for rank, (_, label) in enumerate(ordered, start=1):
        if label > 0.5:
            true_positives += 1.0
            precision_sum += true_positives / rank
    return precision_sum / positives


def aurc(labels: list[float], risks: list[float]) -> float:
    if not labels:
        return 0.0
    ordered = [label for _, label in sorted(zip(risks, labels), key=lambda item: item[0])]
    cumulative_errors = 0.0
    curve = []
    for i, label in enumerate(ordered, start=1):
        cumulative_errors += label
        curve.append(cumulative_errors / i)
    return mean(curve)


def binary_entropy(margin: float) -> float:
    p = 1.0 / (1.0 + math.exp(-max(min(margin, 60.0), -60.0)))
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -(p * math.log(p) + (1.0 - p) * math.log(1.0 - p))

