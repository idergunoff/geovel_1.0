import numpy as np
import pandas as pd
from scipy.stats import f_oneway

from anova_comparison import compare_anova, transform_features


def fixture_data():
    return pd.DataFrame({'a': [1., 2., 4., 7., 8., 12.],
                         'b': [4., 3., 8., 2., 6., 5.]}), np.array([0, 0, 0, 1, 1, 1])


def test_linear_transforms_preserve_anova_and_input():
    data, labels = fixture_data()
    before = data.copy(deep=True)
    result, frames = compare_anova(data, labels, ['raw', 'zscore', 'minmax', 'l1'])
    for feature in data:
        rows = result[result.feature == feature].set_index('method')
        expected = f_oneway(data[feature][:3], data[feature][3:])
        assert np.allclose(rows.loc[['raw', 'zscore', 'minmax'], 'F'], expected.statistic)
        assert np.allclose(rows.loc[['raw', 'zscore', 'minmax'], 'p'], expected.pvalue)
    pd.testing.assert_frame_equal(data, before)
    assert np.allclose(frames['l1'].abs().sum(axis=1), 1)


def test_row_transforms_and_zero_rows():
    data = pd.DataFrame({'a': [-3., 0.], 'b': [4., 0.]})
    l1, _ = transform_features(data, 'l1')
    l2, _ = transform_features(data, 'l2')
    assert np.allclose(l1.iloc[0], [-3 / 7, 4 / 7])
    assert np.allclose(l2.iloc[0], [-.6, .8])
    assert np.all(l2.iloc[1] == 0)


def test_clr_zero_replacement_and_negative_rejection():
    data = pd.DataFrame({'a': [0., 1.], 'b': [4., 9.]})
    frame, errors = transform_features(data, 'clr', 1.)
    assert not errors
    assert np.allclose(frame.sum(axis=1), 0)
    assert np.allclose(frame.iloc[0], np.log([1., 4.]) - np.log([1., 4.]).mean())
    assert transform_features(data, 'clr', 0)[1]
    data.iloc[0, 0] = -1
    assert transform_features(data, 'clr', 10)[1]


def test_invalid_column_does_not_block_other_branches():
    data, labels = fixture_data()
    data['a'] -= 2
    result, _ = compare_anova(data, labels, ['raw', 'log', 'boxcox', 'yeojohnson'])
    invalid = result[(result.feature == 'a') & result.method.isin(['log', 'boxcox'])]
    assert invalid.p.isna().all()
    assert invalid.status.str.len().gt(0).all()
    assert result[result.method.isin(['raw', 'yeojohnson'])].p.notna().all()


def test_fdr_is_bh_within_each_variant():
    data, labels = fixture_data()
    result, _ = compare_anova(data, labels, ['raw', 'l2'])
    for _, group in result.groupby('method'):
        group = group.sort_values('p')
        expected = np.minimum.accumulate((group.p.to_numpy() * 2 / [1, 2])[::-1])[::-1]
        assert np.allclose(group.FDR, np.minimum(expected, 1))


def test_missing_and_constant_features_are_reported():
    data = pd.DataFrame({'missing': [1., np.nan, 3., 4.], 'constant': [1.] * 4})
    result, _ = compare_anova(data, [0, 0, 1, 1], ['raw', 'l1'])
    assert result.status.str.len().gt(0).all()
    assert result.p.isna().all()
