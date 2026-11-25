import pickle

import numpy as np
import pytest
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from spatial_lgbm import (
    SpatialLGBMClassifier,
    SpatialLGBMRegressor,
    profile_measure_coordinates,
    spatial_cv_params,
    spatial_training_note,
)


def test_profile_coordinates_follow_training_rows_and_keep_missing_values():
    def load_profiles(profile_ids):
        assert set(profile_ids) == {7, 9, 11}
        return [(9, None, None), (7, '[1, 2, 3]', '[4, 5, 6]')]

    coordinates = profile_measure_coordinates(
        ['7_2_2', '9_3_0', '7_1_0', '11_3_0', '7_2_6', '7_2', '7_2_-1'], load_profiles,
    )
    np.testing.assert_array_equal(coordinates[[0, 2]], [[3, 6], [1, 4]])
    assert np.isnan(coordinates[[1, 3, 4, 5, 6]]).all()


def test_spatial_loss_reduces_jumps_and_pipeline_predicts_after_pickle():
    positions = np.arange(100)
    X = positions.reshape(-1, 1)
    y = 20 + np.sin(positions / 18) * 2 + (-1.0) ** positions * 2
    xy = np.column_stack((positions, np.zeros(len(X))))
    baseline = LGBMRegressor(n_estimators=80, min_child_samples=2, verbosity=-1, n_jobs=1)
    smoothed = SpatialLGBMRegressor(baseline, spatial_loss=8, spatial_radius=4)
    pipe = Pipeline([('scaler', StandardScaler()), ('model', smoothed)])
    fit_params = {'model__spatial_coordinates': xy}

    cv = KFold(3, shuffle=True, random_state=1)
    result = cross_validate(
        pipe, X, y, cv=cv, return_estimator=True, error_score='raise', **spatial_cv_params(fit_params),
    )
    for (train, _), fitted in zip(cv.split(X), result['estimator']):
        # Each pair within four metres is connected on this one-metre grid.
        distances = np.abs(positions[train, None] - positions[train])
        expected_pairs = np.count_nonzero(np.triu((0 < distances) & (distances < 4)))
        assert fitted['model'].spatial_pair_count_ == expected_pairs

    prediction = pipe.fit(X, y, **fit_params).predict(X)
    reference = baseline.fit(X, y).predict(X)
    assert np.mean(np.diff(prediction) ** 2) < np.mean(np.diff(reference) ** 2)
    np.testing.assert_allclose(pickle.loads(pickle.dumps(pipe)).predict(X), prediction)


def test_no_coordinate_pairs_uses_native_objective():
    X = np.arange(40).reshape(-1, 1)
    y = X.ravel() ** 2
    xy = np.full((len(X), 2), np.nan)
    xy[:2] = [[0, 0], [100, 100]]
    model = LGBMRegressor(n_estimators=5, min_child_samples=2, verbosity=-1, n_jobs=1)
    smoothed = SpatialLGBMRegressor(model, spatial_loss=8, spatial_radius=4).fit(X, y, spatial_coordinates=xy)

    assert smoothed.spatial_pair_count_ == 0
    assert '0 пар' in spatial_training_note(smoothed)
    np.testing.assert_array_equal(smoothed.predict(X), model.fit(X, y).predict(X))


@pytest.mark.parametrize('class_count', [2, 3])
def test_classifier_smooths_probabilities_and_preserves_labels_through_cv_and_pickle(class_count):
    positions = np.arange(120)
    X = positions.reshape(-1, 1)
    xy = np.column_stack((positions, np.zeros(len(X))))
    y = np.array(['sand', 'clay', 'silt'])[positions % class_count]
    baseline = LGBMClassifier(n_estimators=60, min_child_samples=2, verbosity=-1, n_jobs=1)
    smoothed = SpatialLGBMClassifier(baseline, spatial_loss=4, spatial_radius=2)
    pipe = Pipeline([('scaler', StandardScaler()), ('model', smoothed)])
    fit_params = {'model__spatial_coordinates': xy}

    cv = StratifiedKFold(3, shuffle=True, random_state=2)
    result = cross_validate(
        pipe, X, y, cv=cv, return_estimator=True, error_score='raise', **spatial_cv_params(fit_params),
    )
    for (train, _), fitted in zip(cv.split(X, y), result['estimator']):
        expected_pairs = np.count_nonzero(np.diff(positions[train]) == 1)
        assert fitted['model'].spatial_pair_count_ == expected_pairs

    probability = pipe.fit(X, y, **fit_params).predict_proba(X)
    reference = baseline.fit(X, y).predict_proba(X)
    assert np.mean(np.diff(probability, axis=0) ** 2) < np.mean(np.diff(reference, axis=0) ** 2)
    np.testing.assert_allclose(probability.sum(axis=1), 1)
    np.testing.assert_array_equal(pipe.classes_, baseline.classes_)
    restored = pickle.loads(pickle.dumps(pipe))
    np.testing.assert_allclose(restored.predict_proba(X), probability)
    np.testing.assert_array_equal(restored.predict(X), pipe.classes_[probability.argmax(axis=1)])


def test_classifier_without_neighbors_uses_native_probabilities():
    X = np.arange(40).reshape(-1, 1)
    y = np.where(X.ravel() < 20, 'clay', 'sand')
    xy = np.full((len(X), 2), np.nan)
    model = LGBMClassifier(n_estimators=5, min_child_samples=2, verbosity=-1, n_jobs=1)
    smoothed = SpatialLGBMClassifier(model, spatial_loss=4, spatial_radius=2).fit(X, y, spatial_coordinates=xy)

    assert '0 пар' in spatial_training_note(smoothed)
    np.testing.assert_allclose(smoothed.predict_proba(X), model.fit(X, y).predict_proba(X))
