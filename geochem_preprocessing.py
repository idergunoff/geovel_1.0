"""Shared preprocessing helpers for geochemical component data."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import Normalizer, PowerTransformer, StandardScaler


def transform_components(
    data: pd.DataFrame,
    *,
    logarithm: bool = False,
    power_transform: bool = False,
    normalize_l1: bool = False,
    standardize: bool = False,
) -> pd.DataFrame:
    """Transform exactly the component columns supplied in ``data``.

    L1 normalization is row-wise, so callers must select the active components
    before calling this function. Its divisor then follows a changed component
    mask instead of silently including hidden components.
    """
    result = data.copy()
    if logarithm:
        result = np.log10(result)
    if power_transform:
        result = pd.DataFrame(
            PowerTransformer(method="yeo-johnson", standardize=False).fit_transform(result),
            columns=data.columns,
            index=data.index,
        )
    if normalize_l1:
        result = pd.DataFrame(
            Normalizer(norm="l1").fit_transform(result),
            columns=data.columns,
            index=data.index,
        )
    if standardize:
        result = pd.DataFrame(
            StandardScaler().fit_transform(result),
            columns=data.columns,
            index=data.index,
        )
    return result
