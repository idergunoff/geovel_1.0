"""Shared preprocessing helpers for geochemical component data."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import Normalizer, PowerTransformer, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted


def transform_components(
    data: pd.DataFrame,
    *,
    logarithm: bool = False,
    power_transform: bool = False,
    normalize_l1: bool = False,
    normalize_l1_all: bool = False,
    selected_columns=None,
    standardize: bool = False,
) -> pd.DataFrame:
    """Transform component data, selecting columns before or after L1.

    ``data`` contains all components, without point names or other metadata.
    Keep the existing log / power / L1 / standardization order. In all-component
    mode, select the output columns after L1 and before standardization.
    """
    if normalize_l1 and normalize_l1_all:
        raise ValueError('Выберите только один вариант norm l1.')
    columns = list(data.columns if selected_columns is None else selected_columns)
    result = data.copy() if normalize_l1_all else data.loc[:, columns].copy()
    if logarithm:
        result = np.log10(result)
    if power_transform:
        result = pd.DataFrame(
            PowerTransformer(method="yeo-johnson", standardize=False).fit_transform(result),
            columns=result.columns,
            index=data.index,
        )
    if normalize_l1 or normalize_l1_all:
        result = pd.DataFrame(
            Normalizer(norm="l1").fit_transform(result),
            columns=result.columns,
            index=data.index,
        )
    result = result.loc[:, columns]
    if standardize:
        result = pd.DataFrame(
            StandardScaler().fit_transform(result),
            columns=result.columns,
            index=data.index,
        )
    return result


class ComponentSelector(TransformerMixin, BaseEstimator):
    """Persist the active component order after normalization of all inputs."""

    def __init__(self, indices):
        self.indices = indices

    def fit(self, X, y=None):
        values = check_array(X)
        self.n_features_in_ = values.shape[1]
        if not self.indices or min(self.indices) < 0 or max(self.indices) >= self.n_features_in_:
            raise ValueError('Некорректный набор выбранных компонентов.')
        return self

    def transform(self, X):
        check_is_fitted(self, 'n_features_in_')
        values = check_array(X)
        if values.shape[1] != self.n_features_in_:
            raise ValueError('Набор компонентов отличается от обучающего.')
        return values[:, self.indices]
