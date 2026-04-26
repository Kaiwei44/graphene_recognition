from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression

from .training_utils import class_balanced_weights


@dataclass
class PiecewiseLinearBaseline:
    split_layer: int
    gate: LogisticRegression
    low_model: LinearRegression
    high_model: LinearRegression

    def gate_probability(self, x: np.ndarray) -> np.ndarray:
        return self.gate.predict_proba(x)[:, 1]

    def predict_score(self, x: np.ndarray) -> np.ndarray:
        low = self.low_model.predict(x)
        high = self.high_model.predict(x)
        use_high = self.gate_probability(x) >= 0.5
        return np.where(use_high, high, low).astype(np.float32)


def fit_piecewise_baseline(x: np.ndarray, y: np.ndarray, split_layer: int) -> PiecewiseLinearBaseline:
    weights = class_balanced_weights(y)
    y_gate = (y > split_layer).astype(np.int64)
    if len(np.unique(y_gate)) < 2:
        raise ValueError("Need samples on both sides of split_layer for residual baseline")

    gate = LogisticRegression(max_iter=2000, C=10.0)
    gate.fit(x, y_gate, sample_weight=weights)

    low_mask = y <= split_layer
    high_mask = y > split_layer
    low_model = LinearRegression()
    high_model = LinearRegression()
    low_model.fit(x[low_mask], y[low_mask].astype(np.float32), sample_weight=weights[low_mask])
    high_model.fit(x[high_mask], y[high_mask].astype(np.float32), sample_weight=weights[high_mask])
    return PiecewiseLinearBaseline(split_layer=split_layer, gate=gate, low_model=low_model, high_model=high_model)

