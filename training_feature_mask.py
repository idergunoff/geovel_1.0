"""Validate saved feature masks against the current training table."""

import json


def validate_mask_features(features, available_features, columns, mask_id):
    """Return a copy of a compatible mask, preserving its feature order."""
    if (not isinstance(features, list) or not features
            or any(not isinstance(name, str) or not name for name in features)):
        raise ValueError(f'Маска id{mask_id} пуста или содержит некорректный список признаков. '
                         'Исправьте маску или выберите другую.')
    if len(set(features)) != len(features):
        raise ValueError(f'Маска id{mask_id} содержит повторяющиеся признаки. Исправьте маску.')

    available = set(available_features).intersection(columns)
    missing = [name for name in features if name not in available]
    if missing:
        message = (f'Маска id{mask_id} несовместима с текущей обучающей выборкой.\n\n'
                   f'Недостающие признаки ({len(missing)}):\n' + '\n'.join(missing))
        if any(name.startswith('model_') for name in missing):
            message += '\n\nДля признаков model_* добавьте результат соответствующей модели в параметры анализа.'
        message += ('\n\nДобавьте недостающие признаки в параметры анализа и обновите обучающую выборку, '
                    'затем заново откройте окно обучения. '
                    'Либо выберите совместимую маску или отключите применение маски.\n'
                    'Обучение и валидация с этой маской не запущены.')
        raise ValueError(message)
    return list(features)


class TrainingFeatureMask:
    """Share mask checks between classification and regression dialogs.

    Keep the original feature list so disabling or changing a mask never reuses
    features from an earlier run. UI and database access are injected to avoid
    importing the application's global window/session.
    """

    def __init__(self, features, columns, checkbox, mask_list, load_mask, warn):
        self.features = tuple(features)
        self.columns = tuple(columns)
        self.checkbox = checkbox
        self.mask_list = mask_list
        self.load_mask = load_mask
        self.warn = warn

    def resolve(self):
        """Return selected features, or warn and return None before any work."""
        if not self.checkbox.isChecked():
            return list(self.features)
        item = self.mask_list.currentItem()
        if item is None:
            self.warn('Применение маски включено, но маска не выбрана. '
                      'Выберите маску из списка или отключите применение маски.')
            return None
        mask_id = item.text().rsplit(' id', 1)[-1]
        try:
            features = self.load_mask(mask_id)
        except (AttributeError, TypeError, ValueError):
            self.warn(f'Не удалось прочитать маску id{mask_id}. Возможно, она удалена или повреждена. '
                      'Заново откройте окно обучения и выберите другую маску.')
            return None
        try:
            return validate_mask_features(features, self.features, self.columns, mask_id)
        except ValueError as error:
            self.warn(str(error))
            return None

    def annotate_item(self, item, mask):
        """Show incompatibility in the list while keeping masks available for inspection."""
        try:
            features = json.loads(mask.mask)
        except (TypeError, ValueError):
            features = None
        try:
            validate_mask_features(features, self.features, self.columns, mask.id)
            status = 'Совместима с текущей обучающей выборкой.'
        except ValueError as error:
            item.setText(f'Недоступна: {mask.count_param} id{mask.id}')
            status = str(error)
        item.setToolTip('\n\n'.join(text for text in (mask.mask_info, status) if text))
