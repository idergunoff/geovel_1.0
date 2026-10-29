import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import Normalizer, PowerTransformer, StandardScaler

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


def test_all_components_are_normalized_before_selection():
    data = pd.DataFrame({'Au': [2., -2., 0.], 'Ag': [3., 3., 0.], 'Cu': [95., 5., 0.]},
                        index=[7, 10, 12])
    result = transform_components(data, selected_columns=['Ag', 'Au'], normalize_l1_all=True)
    np.testing.assert_allclose(result.to_numpy(), [[.03, .02], [.3, -.2], [0., 0.]])
    assert result.columns.tolist() == ['Ag', 'Au']
    assert result.index.tolist() == [7, 10, 12]
    reduced = transform_components(data, selected_columns=['Au'], normalize_l1_all=True)
    np.testing.assert_allclose(reduced['Au'], result['Au'])
    selected = transform_components(data, selected_columns=['Au', 'Ag'], normalize_l1=True)
    np.testing.assert_allclose(selected.iloc[0], [.4, .6])


def test_normalization_modes_cannot_be_combined():
    with pytest.raises(ValueError, match='только один'):
        transform_components(pd.DataFrame({'Au': [1.]}), normalize_l1=True, normalize_l1_all=True)


@pytest.mark.parametrize('all_components', [False, True])
def test_preprocessing_order_with_component_selection(all_components):
    data = pd.DataFrame({'Au': [2., 3., 5., 7.], 'Ag': [3., 8., 6., 4.], 'Cu': [95., 21., 20., 11.]})
    selected = ['Ag', 'Au']
    result = transform_components(data, selected_columns=selected, logarithm=True,
                                  power_transform=True, standardize=True,
                                  normalize_l1=not all_components, normalize_l1_all=all_components)
    source = data if all_components else data[selected]
    expected = PowerTransformer(standardize=False).fit_transform(np.log10(source))
    expected = Normalizer(norm='l1').fit_transform(expected)
    if all_components:
        expected = expected[:, [1, 0]]
    np.testing.assert_allclose(result, StandardScaler().fit_transform(expected))
