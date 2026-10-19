"""Independent transformations and one-way ANOVA for classification features."""
import warnings

import numpy as np
import pandas as pd
from scipy import stats

TRANSFORMS = {
    'raw': 'Исходные данные', 'zscore': 'Z-score', 'minmax': 'Min–max',
    'log': 'Log', 'boxcox': 'Box–Cox', 'yeojohnson': 'Yeo–Johnson',
    'l1': 'Normalize L1', 'l2': 'Normalize L2', 'clr': 'CLR',
}


def transform_features(features, method, pseudocount=0.0):
    """Return transformed values and per-feature errors; never mutate input.

    Invalid columns stay NaN. Row transforms require a complete finite matrix.
    CLR replaces zeros only, without shifting positive or negative values.
    """
    if method not in TRANSFORMS:
        raise ValueError(f'Неизвестное преобразование: {method}')
    values = features.to_numpy(dtype=float, copy=True)
    result = np.full(values.shape, np.nan)
    errors = {}
    if method in ('l1', 'l2', 'clr'):
        error = None
        if not np.isfinite(values).all():
            error = 'Для нормировки строк нужны конечные значения во всех признаках'
        elif method == 'clr':
            if np.any(values < 0):
                error = 'CLR не допускает отрицательные значения'
            elif not np.isfinite(pseudocount) or pseudocount < 0:
                error = 'Псевдосчётчик должен быть конечным и неотрицательным'
            elif np.any(values == 0) and pseudocount == 0:
                error = 'Для нулей CLR задайте положительный псевдосчётчик'
            else:
                values[values == 0] = pseudocount
                logs = np.log(values)
                result = logs - logs.mean(axis=1, keepdims=True)
        else:
            # Scaling first prevents overflow when squaring or summing large values.
            scale = np.max(np.abs(values), axis=1, keepdims=True)
            scaled = values / np.where(scale == 0, 1, scale)
            norm = (np.abs(scaled).sum(axis=1, keepdims=True) if method == 'l1'
                    else np.sqrt((scaled ** 2).sum(axis=1, keepdims=True)))
            result = scaled / np.where(norm == 0, 1, norm)
        if error:
            errors = dict.fromkeys(features.columns, error)
    else:
        for index, column in enumerate(features.columns):
            x = values[:, index]
            try:
                if not np.isfinite(x).all():
                    raise ValueError('Есть пропуски или бесконечные значения')
                if method in ('log', 'boxcox') and np.any(x <= 0):
                    raise ValueError('Требуются строго положительные значения')
                constant = np.ptp(x) == 0
                if method == 'raw':
                    y = x
                elif method == 'zscore':
                    y = np.zeros_like(x) if constant else (x - x.mean()) / x.std()
                elif method == 'minmax':
                    y = np.zeros_like(x) if constant else (x - x.min()) / np.ptp(x)
                elif method == 'log':
                    y = np.log(x)
                elif constant:
                    raise ValueError('Степенное преобразование постоянного признака не определено')
                elif method == 'boxcox':
                    y = stats.boxcox(x)[0]
                else:
                    y = stats.yeojohnson(x)[0]
                if not np.isfinite(y).all():
                    raise ValueError('Преобразование дало неконечные значения')
                result[:, index] = y
            except (ValueError, RuntimeError, FloatingPointError, OverflowError) as exc:
                errors[column] = str(exc)
    return pd.DataFrame(result, index=features.index, columns=features.columns), errors


def compare_anova(features, labels, methods, pseudocount=0.0):
    """FDR uses Benjamini–Hochberg across valid features within each variant."""
    labels = np.asarray(labels)
    if len(labels) != len(features):
        raise ValueError('Число меток и объектов не совпадает')
    groups = pd.unique(labels[~pd.isna(labels)])
    rows, transformed = [], {}
    for method in methods:
        frame, errors = transform_features(features, method, pseudocount)
        transformed[method] = frame
        start = len(rows)
        for column in features.columns:
            f, p = np.nan, np.nan
            status = errors.get(column, '')
            if not status:
                samples = [frame[column].to_numpy()[labels == group] for group in groups]
                if len(samples) < 2 or any(len(sample) < 2 for sample in samples):
                    status = 'Нужны минимум два класса и два объекта в каждом'
                else:
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        f, p = stats.f_oneway(*samples)
                    if np.isnan(f) or not np.isfinite(p):
                        status = 'ANOVA не определена (проверьте постоянные признаки)'
            rows.append({'feature': column, 'method': method, 'F': f, 'p': p,
                         'FDR': np.nan, 'status': status})
        valid = [i for i in range(start, len(rows)) if np.isfinite(rows[i]['p'])]
        ordered = sorted(valid, key=lambda i: rows[i]['p'])
        adjusted = 1.0
        for rank in range(len(ordered), 0, -1):
            i = ordered[rank - 1]
            adjusted = min(adjusted, rows[i]['p'] * len(ordered) / rank)
            rows[i]['FDR'] = adjusted
    return pd.DataFrame(rows, columns=['feature', 'method', 'F', 'p', 'FDR', 'status']), transformed
