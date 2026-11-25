"""Store named column sets and report selections for prediction checks."""
import json

from app_settings import settings


PRESETS_KEY = 'prediction_well_check/column_presets'
REPORT_KEY = 'prediction_well_check/report_exclusions'


def saved_presets():
    presets = settings().value(PRESETS_KEY, {})
    return presets if isinstance(presets, dict) else {}


def save_preset(name, preset):
    store = settings()
    presets = saved_presets()
    presets[name] = preset
    store.setValue(PRESETS_KEY, presets)
    store.sync()


def column_config(name, target, raw_text=False, boundary_name=''):
    """Keep the settings that affect the column, using portable canonical names."""
    options = {}
    if target.source == 'well_data' and not raw_text:
        options.update(strict_numeric=target.strict_numeric, allow_explicit_sum=target.allow_explicit_sum)
    elif target.source == 'well_log':
        options.update(aggregation=target.aggregation, operation=target.operation,
                       depth_mode=target.depth_mode, interval=target.interval)
        if target.operation == 'single':
            options['interval_position'] = target.interval_position
        if target.depth_mode == 'fixed':
            options['fixed_depth'] = target.fixed_depth
        elif boundary_name:
            options['boundary_name'] = boundary_name
    return {'source': target.source, 'name': name, 'raw_text': raw_text, 'options': options}


def report_column_key(config):
    return 'analysis:' + json.dumps(config, ensure_ascii=False, sort_keys=True)


def excluded_report_columns(file_format):
    exclusions = settings().value(f'{REPORT_KEY}/{file_format}', [])
    return {key for key in exclusions if isinstance(key, str)} if isinstance(exclusions, list) else set()


def save_report_columns(file_format, available, selected):
    # Retain choices for columns that are absent from this particular report.
    exclusions = excluded_report_columns(file_format).difference(available)
    selected = set(selected)
    exclusions.update(key for index, key in enumerate(available) if index not in selected)
    store = settings()
    store.setValue(f'{REPORT_KEY}/{file_format}', sorted(exclusions))
    store.sync()
