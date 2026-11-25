"""Spatial regularization for GEOVEL's LightGBM models."""

import json
from inspect import signature

import numpy as np
from lightgbm import LGBMClassifier
from scipy.sparse import csr_matrix, diags
from scipy.special import expit, softmax
from scipy.spatial import cKDTree
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone, is_classifier
from sklearn.model_selection import cross_validate
from sklearn.utils.validation import check_is_fitted


_MAX_NEIGHBORS = 8


def profile_measure_coordinates(row_keys, load_profiles):
    """Read projected coordinates in training-row order, including cached tables."""
    coordinates = np.full((len(row_keys), 2), np.nan)
    rows_by_profile = {}
    for row, key in enumerate(row_keys):
        try:
            profile, _, measure = key.split('_')
            profile, measure = int(profile), int(measure)
        except (AttributeError, ValueError):
            continue
        if measure >= 0:
            rows_by_profile.setdefault(profile, []).append((row, measure))

    if not rows_by_profile:
        return coordinates

    for profile, x_json, y_json in load_profiles(rows_by_profile):
        try:
            x = np.asarray(json.loads(x_json), dtype=float)
            y = np.asarray(json.loads(y_json), dtype=float)
        except (TypeError, ValueError):
            continue
        if x.ndim != 1 or y.ndim != 1:
            continue
        rows, measures = np.asarray(rows_by_profile[profile]).T
        valid = measures < min(len(x), len(y))
        coordinates[rows[valid]] = np.column_stack((x[measures[valid]], y[measures[valid]]))
    return coordinates


def _spatial_adjacency(coordinates, radius):
    valid_rows = np.flatnonzero(np.isfinite(coordinates).all(axis=1))
    shape = (len(coordinates), len(coordinates))
    if len(valid_rows) < 2:
        return csr_matrix(shape)

    points = coordinates[valid_rows]
    distances, neighbors = cKDTree(points).query(
        points, k=min(_MAX_NEIGHBORS + 1, len(points)), distance_upper_bound=radius,
    )
    rows = np.broadcast_to(np.arange(len(points))[:, None], neighbors.shape)
    keep = np.isfinite(distances) & (rows != neighbors)
    # A tie between coincident measurements can leave the query point itself
    # out of the results. Apply the same neighbor limit in that case.
    keep &= keep.cumsum(axis=1) <= _MAX_NEIGHBORS
    adjacency = csr_matrix((
        (1 - distances[keep] / radius) / _MAX_NEIGHBORS,
        (valid_rows[rows[keep]], valid_rows[neighbors[keep]]),
    ), shape=shape)
    return adjacency.maximum(adjacency.T)


class _SpatialObjective:
    def __init__(self, adjacency, weight, *, classification=False):
        degree = np.asarray(adjacency.sum(axis=1)).ravel()
        self.laplacian = weight * (diags(degree) - adjacency)
        # 2*diag(degree) bounds the graph Laplacian in the quadratic form.
        # LightGBM can consume one curvature value per training row.
        self.curvature = 2 * weight * degree
        self.classification = classification

    def __call__(self, target, prediction):
        penalty_gradient = self.laplacian @ prediction
        if not self.classification:
            return prediction - target + penalty_gradient, 1 + self.curvature
        if prediction.ndim == 1:
            probability = expit(prediction)
            hessian = np.maximum(probability * (1 - probability), 1e-12)
            return probability - target + penalty_gradient, hessian + self.curvature

        probability = softmax(prediction, axis=1)
        # Twice the covariance diagonal bounds the softmax Hessian.
        hessian = np.maximum(2 * probability * (1 - probability), 1e-12)
        probability[np.arange(len(target)), target.astype(np.intp)] -= 1
        return probability + penalty_gradient, hessian + self.curvature[:, None]

    def release_graph(self):
        # The fitted booster only needs the objective during training.
        self.laplacian = self.curvature = None


def _training_graph(spatial_coordinates, n_samples, weight, radius):
    coordinates = np.asarray(spatial_coordinates, dtype=float)
    if coordinates.shape != (n_samples, 2):
        raise ValueError('Для каждой обучающей строки нужна пара координат X и Y.')
    if not np.isfinite(weight) or weight < 0:
        raise ValueError('Вес spatial loss должен быть конечным и неотрицательным.')
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError('Радиус spatial loss должен быть конечным и положительным.')
    return _spatial_adjacency(coordinates, radius) if weight else csr_matrix((n_samples, n_samples))


class SpatialLGBMRegressor(RegressorMixin, BaseEstimator):
    """Fit a LightGBM regressor with an optional penalty between nearby rows."""

    def __init__(self, model, spatial_loss, spatial_radius):
        self.model = model
        self.spatial_loss = spatial_loss
        self.spatial_radius = spatial_radius

    def fit(self, X, y, *, spatial_coordinates):
        model = clone(self.model)
        adjacency = _training_graph(spatial_coordinates, len(y), self.spatial_loss, self.spatial_radius)
        pair_count = adjacency.nnz // 2
        prediction_offset = 0.0
        if pair_count:
            target = np.asarray(y, dtype=np.float64)
            prediction_offset = float(target.mean())
            objective = _SpatialObjective(adjacency, self.spatial_loss)
            model.set_params(objective=objective)
            model.fit(X, target - prediction_offset)
            objective.release_graph()
        else:
            model.fit(X, y)
        self.model_ = model
        self.spatial_pair_count_ = pair_count
        self.prediction_offset_ = prediction_offset
        self.n_features_in_ = model.n_features_in_
        return self

    @property
    def feature_importances_(self):
        return self.model_.feature_importances_

    def predict(self, X):
        check_is_fitted(self, 'model_')
        return self.model_.predict(X) + self.prediction_offset_


class SpatialLGBMClassifier(ClassifierMixin, BaseEstimator):
    """Fit log loss with a spatial penalty on neighboring class scores."""

    def __init__(self, model, spatial_loss, spatial_radius):
        self.model = model
        self.spatial_loss = spatial_loss
        self.spatial_radius = spatial_radius

    def fit(self, X, y, *, spatial_coordinates):
        model = clone(self.model)
        adjacency = _training_graph(spatial_coordinates, len(y), self.spatial_loss, self.spatial_radius)
        classes, counts = np.unique(y, return_counts=True)
        pair_count = adjacency.nnz // 2 if len(classes) > 1 else 0
        offset = None
        if pair_count:
            # Custom LightGBM objectives start at zero unless given initial scores.
            # Reapply the training priors when converting tree scores to probabilities.
            if len(classes) == 2:
                offset = np.log(counts[1] / counts[0])
                init_score = np.full(len(y), offset)
            else:
                offset = np.log(counts / len(y))
                init_score = np.broadcast_to(offset, (len(y), len(classes))).copy()
            objective = _SpatialObjective(adjacency, self.spatial_loss, classification=True)
            model.set_params(objective=objective)
            model.fit(X, y, init_score=init_score)
            objective.release_graph()
        else:
            model.fit(X, y)
        self.model_ = model
        self.classes_ = model.classes_
        self.n_features_in_ = model.n_features_in_
        self.spatial_pair_count_ = pair_count
        self.prediction_offset_ = offset
        return self

    @property
    def feature_importances_(self):
        return self.model_.feature_importances_

    def predict_proba(self, X):
        check_is_fitted(self, 'model_')
        if self.prediction_offset_ is None:
            probability = self.model_.predict_proba(X)
            if len(self.classes_) == 1:
                return np.ones((len(probability), 1))
            return probability
        scores = self.model_.predict(X, raw_score=True) + self.prediction_offset_
        if len(self.classes_) == 2:
            positive = expit(scores)
            return np.column_stack((1 - positive, positive))
        return softmax(scores, axis=1)

    def predict(self, X):
        probability = self.predict_proba(X)
        return self.classes_[probability.argmax(axis=1)]


def build_lgbm_classifier(form):
    settings = {
        'reg_alpha': form.doubleSpinBox_l1_lgbm.value(),
        'reg_lambda': form.doubleSpinBox_l2_lgbm.value(),
        'num_leaves': form.spinBox_lgbm_num_leaves.value(),
        'colsample_bytree': form.doubleSpinBox_lgbm_feature.value(),
        'subsample': form.doubleSpinBox_lgbm_subsample.value(),
        'subsample_freq': form.spinBox_lgbm_sub_freq.value(),
        'min_child_samples': form.spinBox_lgbm_child.value(),
        'learning_rate': form.doubleSpinBox_lr_lgbm.value(),
        'n_estimators': form.spinBox_estim_lgbm.value(),
    }
    display_names = {'reg_alpha': 'lambda_1', 'reg_lambda': 'lambda_2', 'colsample_bytree': 'feature_fraction'}
    description = '**LGBM**: \n' + ', \n'.join(f'{display_names.get(name, name)}: {value}' for name, value in settings.items())
    return LGBMClassifier(verbosity=-1, boosting_type='gbdt', **settings), description


def classifier_spatial_fit_params(coordinates, training_size=None):
    coordinates = np.asarray(coordinates, dtype=float)
    if training_size is not None and training_size > len(coordinates):
        # SMOTE and ADASYN append synthetic samples after the measured rows.
        coordinates = np.pad(coordinates, ((0, training_size - len(coordinates)), (0, 0)), constant_values=np.nan)
    return {'model__spatial_coordinates': coordinates}


def configure_spatial_lgbm(model, form):
    """Keep the native estimator when spatial regularization is disabled."""
    weight = form.doubleSpinBox_lgbm_spatial_loss.value()
    radius = form.doubleSpinBox_lgbm_spatial_radius.value()
    if weight == 0:
        return model, ''
    estimator = SpatialLGBMClassifier if is_classifier(model) else SpatialLGBMRegressor
    return estimator(model, weight, radius), f'\nspatial_loss: {weight:g}, радиус: {radius:g} м'


def spatial_cv_params(fit_params):
    if not fit_params:
        return {}
    # scikit-learn 1.3 uses fit_params, newer versions use params.
    name = 'params' if 'params' in signature(cross_validate).parameters else 'fit_params'
    return {name: fit_params}


def spatial_training_note(model):
    if not isinstance(model, (SpatialLGBMRegressor, SpatialLGBMClassifier)):
        return ''
    if not model.spatial_pair_count_:
        return '\nSpatial loss пропущен. В заданном радиусе найдено 0 пар с координатами.'
    return f'\nПар для spatial loss: {model.spatial_pair_count_}'
