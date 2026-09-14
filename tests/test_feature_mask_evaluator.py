import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut

from feature_mask_evaluator import (
    FeatureMaskEvaluator,
    INVALID_SCORE,
    ScoreEarlyStopping,
    build_seed_masks,
)


def make_evaluator(*, objective_count=1, folds=None, seed=17):
    X = np.array([[0, 1], [0, 2], [1, 1], [1, 2], [0, 3], [1, 3]])
    y = np.array([0, 0, 1, 1, 0, 1])
    groups = np.array([0, 0, 1, 1, 2, 2])
    if folds is None:
        folds = LeaveOneGroupOut().split(X, y, groups)
    model = RandomForestClassifier(n_estimators=8)
    return FeatureMaskEvaluator(
        X, y, model, folds, objective_count=objective_count, seed=seed
    )


def test_empty_mask_respects_single_objective_contract():
    evaluator = make_evaluator()
    assert evaluator([[False], [False]]) == [INVALID_SCORE]
    assert evaluator.metrics()["model_fits"] == 0


def test_no_valid_folds_has_finite_worst_score():
    evaluator = make_evaluator(objective_count=2, folds=[])
    result = evaluator([True, False])
    assert result == [INVALID_SCORE, 1]
    assert np.isfinite(result[0])


def test_same_mask_and_seed_are_reproducible_and_measured():
    first = make_evaluator(seed=42)
    second = make_evaluator(seed=42)
    assert first([True, True]) == second([True, True])
    metrics = first.metrics()
    assert metrics["model_fits"] == 3
    assert metrics["unique_masks"] == 1
    assert metrics["feature_count"] == 2
    assert metrics["seed"] == 42
    assert metrics["elapsed"] >= 0


def test_checkpoint_metrics_can_be_continued():
    evaluator = make_evaluator()
    evaluator.restore_metrics({"model_fits": 9, "evaluations": 3}, [[True], [False]])
    evaluator([True, False])
    assert evaluator.metrics()["model_fits"] == 12
    assert evaluator.metrics()["evaluations"] == 4


def test_duplicate_mask_uses_exact_cache_without_more_fits():
    evaluator = make_evaluator()
    expected = evaluator([True, False])
    fits = evaluator.metrics()["model_fits"]
    assert evaluator([[True], [False]]) == expected
    assert evaluator.metrics()["model_fits"] == fits
    assert evaluator.metrics()["cache_hits"] == 1


def test_cache_checkpoint_requires_matching_fingerprint():
    source = make_evaluator(seed=42)
    expected = source([True, False])
    payload = source.checkpoint_cache()

    matching = make_evaluator(seed=42)
    assert matching.restore_cache(payload)
    assert matching([True, False]) == expected
    assert matching.metrics()["model_fits"] == 0

    different_seed = make_evaluator(seed=43)
    assert not different_seed.restore_cache(payload)


def test_score_early_stopping_resets_only_for_meaningful_improvement():
    stopping = ScoreEarlyStopping(patience=2, min_delta=0.01)
    assert not stopping.update([0.5])
    assert not stopping.update([0.505])
    assert stopping.update([0.509])
    assert not stopping.update([0.52])


def test_seed_masks_are_deterministic_and_include_all_features():
    X = np.arange(48, dtype=float).reshape(12, 4)
    y = np.array([0, 1] * 6)
    first = build_seed_masks(X, y, task="classification", seed=7)
    second = build_seed_masks(X, y, task="classification", seed=7)
    assert first == second
    assert [True, True, True, True] in first
    assert all(any(mask) and len(mask) == X.shape[1] for mask in first)
