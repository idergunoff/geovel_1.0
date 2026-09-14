"""Testable objective evaluator used by genetic feature selection."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import logging
from time import perf_counter
from typing import Callable, Iterable, Sequence

import numpy as np
from sklearn.base import clone


INVALID_SCORE = -1.0e12
LOGGER = logging.getLogger(__name__)


@dataclass
class EvaluationTelemetry:
    elapsed: float = 0.0
    model_fits: int = 0
    cache_hits: int = 0
    unique_masks: int = 0
    evaluations: int = 0
    score: float = INVALID_SCORE
    feature_count: int = 0
    seed: int = 0


def _normalise_mask(features: Sequence[object], n_features: int) -> np.ndarray:
    """Convert both plain booleans and Platypus ``Binary(1)`` values."""
    mask = np.asarray([
        bool(value[0]) if isinstance(value, (list, tuple, np.ndarray)) else bool(value)
        for value in features
    ], dtype=bool)
    if mask.size != n_features:
        raise ValueError(f"Expected {n_features} feature flags, got {mask.size}")
    return mask


def _seed_estimator(estimator, seed: int):
    """Set every unset sklearn random_state, including nested estimators."""
    params = estimator.get_params(deep=True)
    random_states = {
        name: seed for name, value in params.items()
        if name.endswith("random_state") and value is None
    }
    if random_states:
        estimator.set_params(**random_states)
    return estimator


class FeatureMaskEvaluator:
    """Evaluate masks consistently and expose baseline performance counters."""

    def __init__(
        self,
        X,
        y,
        estimator,
        folds: Iterable[tuple[np.ndarray, np.ndarray]],
        *,
        objective_count: int,
        seed: int,
        invalid_score: float = INVALID_SCORE,
        progress: Callable[[int, int], None] | None = None,
    ):
        if objective_count not in (1, 2):
            raise ValueError("objective_count must be 1 or 2")
        self.X = np.asarray(X)
        self.y = np.asarray(y).reshape(-1)
        self.estimator = estimator
        self.folds = list(folds)
        self.objective_count = objective_count
        self.invalid_score = float(invalid_score)
        self.progress = progress
        self.telemetry = EvaluationTelemetry(seed=int(seed))
        self._seen_masks: set[tuple[bool, ...]] = set()

    def __call__(self, features: Sequence[object]) -> list[float]:
        started = perf_counter()
        mask = _normalise_mask(features, self.X.shape[1])
        key = tuple(mask.tolist())
        self._seen_masks.add(key)
        count = int(mask.sum())
        scores: list[float] = []

        if count:
            selected = self.X[:, mask]
            if not self.folds:
                LOGGER.warning("Feature mask cannot be evaluated: no valid CV folds")
            for number, (train_idx, test_idx) in enumerate(self.folds, start=1):
                fitted = _seed_estimator(clone(self.estimator), self.telemetry.seed)
                fitted.fit(selected[train_idx], self.y[train_idx])
                self.telemetry.model_fits += 1
                score = float(fitted.score(selected[test_idx], self.y[test_idx]))
                if np.isfinite(score):
                    scores.append(score)
                if self.progress:
                    self.progress(number, len(self.folds))

        score = float(np.mean(scores)) if scores else self.invalid_score
        self.telemetry.elapsed += perf_counter() - started
        self.telemetry.evaluations += 1
        self.telemetry.unique_masks = len(self._seen_masks)
        self.telemetry.score = score
        self.telemetry.feature_count = count
        result = [score]
        if self.objective_count == 2:
            result.append(count)
        return result

    def metrics(self) -> dict[str, float | int]:
        return asdict(self.telemetry)

    def restore_metrics(self, metrics: dict | None, masks: Iterable[Sequence[object]] = ()) -> None:
        """Continue cumulative telemetry when a genetic run is resumed."""
        if metrics:
            for name in asdict(self.telemetry):
                if name in metrics:
                    setattr(self.telemetry, name, metrics[name])
        for features in masks:
            mask = _normalise_mask(features, self.X.shape[1])
            self._seen_masks.add(tuple(mask.tolist()))
        self.telemetry.unique_masks = max(
            self.telemetry.unique_masks, len(self._seen_masks)
        )
