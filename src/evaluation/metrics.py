from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    adjusted_mutual_info_score,
)


# IR metrics

def precision_at_k(rel: list[int], k: int) -> float:
    """
    Precision@k for a single query.
    rel: binary relevance vector for the top-k ranks (1=relevant, 0=non-relevant).
    k:   cutoff (uses rel[:k]).
    """
    if k <= 0:
        return 0.0
    return sum(rel[:k]) / k


def recall_at_k(rel: list[int], k: int, num_relevant: int) -> float:
    """
    Recall@k for a single query.
    num_relevant: |R(d)| = number of gold-standard relevant items for the query (excluding self).
    Returns NaN if num_relevant == 0 (caller may skip in aggregation).
    """
    if num_relevant == 0:
        return float("nan")
    return sum(rel[:k]) / num_relevant


def average_precision_at_k(rel: list[int], k: int) -> float:
    """
    AP@k for a single query (binary relevance).
    Uses the standard definition: mean of precision@i over relevant hits within top-k.
    Returns 0.0 if there are no hits within top-k.
    """
    hits = 0
    sum_prec = 0.0
    for i, r in enumerate(rel[:k], start=1):
        if r:
            hits += 1
            sum_prec += hits / i
    if hits == 0:
        return 0.0
    return sum_prec / hits


def dcg_at_k(rel: list[int], k: int) -> float:
    """
    Discounted Cumulative Gain@k (binary gains).
    """
    total = 0.0
    for i, r in enumerate(rel[:k], start=1):
        if r:
            total += 1.0 / math.log2(i + 1)
    return total


def idcg_at_k(num_relevant: int, k: int) -> float:
    """
    Ideal DCG@k for binary relevance with num_relevant available relevant items.
    """
    ideal_hits = min(k, max(0, num_relevant))
    ideal = 0.0
    for i in range(1, ideal_hits + 1):
        ideal += 1.0 / math.log2(i + 1)
    return ideal


def ndcg_at_k(rel: list[int], k: int, num_relevant: int | None = None) -> float:
    """
    nDCG@k for a single query (binary relevance).
    If num_relevant is provided, IDCG uses that count; otherwise it assumes
    num_relevant = sum(rel), which is fine for per-query evaluation.
    Returns NaN if IDCG==0 (e.g., no relevant items exist for the query).
    """
    dcg = dcg_at_k(rel, k)
    if num_relevant is None:
        num_relevant = sum(rel)
    idcg = idcg_at_k(num_relevant, k)
    if idcg == 0:
        return float("nan")
    return dcg / idcg


def f1_at_k(rel: list[int], k: int, num_relevant: int) -> float:
    """
    F1@k = harmonic mean of P@k and R@k. Included for completeness.
    Returns NaN if num_relevant==0 and recall is undefined.
    """
    p = precision_at_k(rel, k)
    r = recall_at_k(rel, k, num_relevant)
    if math.isnan(r) or (p + r) == 0:
        return float("nan")
    return 2 * p * r / (p + r)


def binary_rel_vector(retrieved_uris: list[str], gold_relevant: set[str], k: int) -> list[int]:
    """
    Build a binary relevance vector y[1..k] against the *gold standard* relevant set.
    retrieved_uris: ranked list returned by a method (top-k already filtered & self removed).
    gold_relevant:  set of URIs considered relevant for the query (same label).
    """
    rel = []
    for u in retrieved_uris[:k]:
        rel.append(1 if u in gold_relevant else 0)
    # If fewer than k retrieved (shouldn't happen with typical getters), pad with zeros
    if len(rel) < k:
        rel.extend([0] * (k - len(rel)))
    return rel


# helpers

def mean_ignore_nan(values: Iterable[float]) -> float:
    """
    Arithmetic mean ignoring NaNs; returns NaN if all are NaN or the iterable is empty.
    """
    xs = [v for v in values if not math.isnan(v)]
    return float(np.mean(xs)) if xs else float("nan")


@dataclass(slots=True)
class IRResults:
    """
    Container for aggregated IR scores at a given k.
    """
    k: int
    precision: float
    recall: float
    map: float
    ndcg: float
    f1: float | None
    queries_evaluated: int


# Clustering metrics

def clustering_scores(true_labels: list[str], pred_labels: list[int]) -> dict[str, float]:
    """
    Compare predicted cluster assignments to gold-standard labels.
    Returns ARI, NMI, and AMI (all in [0,1], chance-corrected for ARI/AMI).
    """
    return {
        "ARI": adjusted_rand_score(true_labels, pred_labels),
        "NMI": normalized_mutual_info_score(true_labels, pred_labels),
        "AMI": adjusted_mutual_info_score(true_labels, pred_labels),
    }
