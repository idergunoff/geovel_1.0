"""Testable objective evaluator used by genetic feature selection."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import logging
from time import perf_counter
from typing import Callable, Iterable, Sequence

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import f_classif, f_regression, mutual_info_classif
from sklearn.linear_model import Lasso, LogisticRegression


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


@dataclass(frozen=True)
class CachedEvaluation:
    score: float
    feature_count: int
    fold_scores: tuple[float, ...]


class ScoreEarlyStopping:
    """Stop a maximisation loop after a configurable score plateau."""

    def __init__(self, patience: int = 10, min_delta: float = 0.001):
        if patience < 1:
            raise ValueError("patience must be positive")
        self.patience = patience
        self.min_delta = float(min_delta)
        self.best = -np.inf
        self.stale_generations = 0

    def update(self, scores: Iterable[float]) -> bool:
        finite = [float(score) for score in scores if np.isfinite(score)]
        current = max(finite, default=-np.inf)
        if current > self.best + self.min_delta:
            self.best = current
            self.stale_generations = 0
        else:
            self.stale_generations += 1
        return self.stale_generations >= self.patience


def build_seed_masks(X, y, *, task: str, seed: int) -> list[list[bool]]:
    """Build deterministic, diverse masks for the first GA population."""
    X = np.asarray(X)
    y = np.asarray(y).reshape(-1)
    if task not in {"classification", "regression"}:
        raise ValueError("task must be 'classification' or 'regression'")
    n_features = X.shape[1]
    if not n_features:
        return []
    rankings = []
    score_functions = (
        (f_classif, mutual_info_classif) if task == "classification" else (f_regression,)
    )
    for score_function in score_functions:
        try:
            scores = score_function(X, y)
            scores = scores[0] if isinstance(scores, tuple) else scores
            rankings.append(np.argsort(np.nan_to_num(scores, nan=-np.inf))[::-1])
        except (TypeError, ValueError):
            LOGGER.debug("Could not create univariate seed mask", exc_info=True)
    try:
        if task == "classification":
            forest = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=1)
            sparse = LogisticRegression(penalty="l1", solver="liblinear", random_state=seed)
        else:
            forest = RandomForestRegressor(n_estimators=100, random_state=seed, n_jobs=1)
            sparse = Lasso(alpha=0.01, random_state=seed, max_iter=5000)
        forest.fit(X, y)
        rankings.append(np.argsort(forest.feature_importances_)[::-1])
        sparse.fit(X, y)
        coefficients = np.asarray(sparse.coef_)
        rankings.append(np.argsort(np.max(np.abs(np.atleast_2d(coefficients)), axis=0))[::-1])
    except (TypeError, ValueError):
        LOGGER.debug("Could not create model-based seed masks", exc_info=True)

    masks = [tuple([True] * n_features)]
    sizes = sorted({max(1, n_features // 4), max(1, n_features // 2)})
    for ranking in rankings:
        for size in sizes:
            mask = np.zeros(n_features, dtype=bool)
            mask[ranking[:size]] = True
            masks.append(tuple(mask.tolist()))
    return [list(mask) for mask in dict.fromkeys(masks)]


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
        feature_names: Sequence[str] | None = None,
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
        self._cache: dict[tuple[bool, ...], CachedEvaluation] = {}
        self.fingerprint = self._make_fingerprint(feature_names)

    def _make_fingerprint(self, feature_names: Sequence[str] | None) -> str:
        """Identify every immutable input that affects an exact evaluation."""
        digest = hashlib.sha256()
        for array in (self.X, self.y):
            contiguous = np.ascontiguousarray(array)
            digest.update(str(contiguous.dtype).encode())
            digest.update(str(contiguous.shape).encode())
            if contiguous.dtype.hasobject:
                digest.update(json.dumps(contiguous.tolist(), default=repr).encode())
            else:
                digest.update(contiguous.tobytes())
        for train_idx, test_idx in self.folds:
            digest.update(np.ascontiguousarray(train_idx).tobytes())
            digest.update(np.ascontiguousarray(test_idx).tobytes())
        # sklearn's estimator representation is stable across processes, unlike
        # repr() of individual callable-valued parameters (which may contain an
        # address and would invalidate a cache after every restart).
        digest.update(repr(self.estimator).encode())
        digest.update(json.dumps(list(feature_names or ())).encode())
        digest.update(str(self.telemetry.seed).encode())
        return digest.hexdigest()

    def __call__(self, features: Sequence[object]) -> list[float]:
        started = perf_counter()
        mask = _normalise_mask(features, self.X.shape[1])
        key = tuple(mask.tolist())
        self._seen_masks.add(key)
        count = int(mask.sum())
        cached = self._cache.get(key)
        if cached is not None:
            self.telemetry.cache_hits += 1
            self.telemetry.evaluations += 1
            self.telemetry.unique_masks = len(self._seen_masks)
            self.telemetry.score = cached.score
            self.telemetry.feature_count = cached.feature_count
            self.telemetry.elapsed += perf_counter() - started
            result = [cached.score]
            if self.objective_count == 2:
                result.append(cached.feature_count)
            return result
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
                # UI callbacks stay outside the expensive per-fold loop.
                self.progress(len(self.folds), len(self.folds))

        score = float(np.mean(scores)) if scores else self.invalid_score
        self._cache[key] = CachedEvaluation(score, count, tuple(scores))
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

    def checkpoint_cache(self) -> dict:
        """Return a pickle-friendly exact cache tied to this evaluator."""
        return {
            "schema_version": 1,
            "fingerprint": self.fingerprint,
            "entries": [
                (list(mask), asdict(value)) for mask, value in self._cache.items()
            ],
        }

    def restore_cache(self, payload: dict | None) -> bool:
        """Restore cache only when schema and immutable inputs match exactly."""
        if not payload:
            return False
        if payload.get("schema_version") != 1 or payload.get("fingerprint") != self.fingerprint:
            LOGGER.warning("Ignoring incompatible feature-mask evaluation cache")
            return False
        restored = {}
        for mask, value in payload.get("entries", ()):
            key = tuple(bool(item) for item in mask)
            if len(key) != self.X.shape[1]:
                return False
            restored[key] = CachedEvaluation(
                float(value["score"]), int(value["feature_count"]),
                tuple(float(score) for score in value.get("fold_scores", ())),
            )
        self._cache = restored
        self._seen_masks.update(restored)
        self.telemetry.unique_masks = max(self.telemetry.unique_masks, len(self._seen_masks))
        return True

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
