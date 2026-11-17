"""Capacity-constrained targeting diagnostics on saved development predictions."""
import math
import numpy as np

def targeting_metrics(y, p, ids, budget):
    if not 0 < budget <= 1:
        raise ValueError('budget must be in (0, 1]')
    y, p, ids = np.asarray(y), np.asarray(p), np.asarray(ids, dtype=str)
    if not len(y) or len(y) != len(p) or len(y) != len(ids):
        raise ValueError('Aligned nonempty arrays required')
    if not np.isin(y, [0, 1]).all() or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError('Binary labels and finite probabilities required')
    if len(set(ids)) != len(ids):
        raise ValueError('Unique IDs required within a validation fold')
    order = np.lexsort((ids, -p))
    k = max(1, math.ceil(len(y)*budget))
    hits = int(y[order[:k]].sum())
    rate = float(y.mean())
    return dict(n=len(y), selected=k, positives=int(y.sum()), hits=hits,
        precision=hits/k, recall=hits/int(y.sum()) if y.sum() else None,
        lift=(hits/k)/rate if rate else None, prevalence=rate,
        expected_hits=float(p[order[:k]].sum()), threshold=float(p[order[k-1]]))
