import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut

from feature_mask_evaluator import FeatureMaskEvaluator, INVALID_SCORE


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
