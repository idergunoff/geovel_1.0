import numpy as np
import pandas as pd

from geochem_preprocessing import transform_components


def test_l1_normalization_uses_only_supplied_components():
    all_components = pd.DataFrame({"Au": [2.0], "Ag": [3.0], "Cu": [95.0]})

    transformed = transform_components(
        all_components[["Au", "Ag"]],
        normalize_l1=True,
    )

    assert transformed.columns.tolist() == ["Au", "Ag"]
    assert np.allclose(transformed.iloc[0].to_numpy(), [0.4, 0.6])
    assert np.allclose(transformed.abs().sum(axis=1).to_numpy(), [1.0])


def test_l1_normalization_is_recalculated_after_component_change():
    components = pd.DataFrame({"Au": [2.0], "Ag": [3.0], "Cu": [5.0]})

    with_three = transform_components(components, normalize_l1=True)
    with_two = transform_components(components[["Au", "Ag"]], normalize_l1=True)

    assert np.isclose(with_three.loc[0, "Au"], 0.2)
    assert np.isclose(with_two.loc[0, "Au"], 0.4)
