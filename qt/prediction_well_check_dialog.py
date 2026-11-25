"""Compare saved predictions and inspect additional well data."""
import csv
import json
from dataclasses import dataclass
from html import escape

from openpyxl import Workbook
from openpyxl.comments import Comment
from openpyxl.styles import Alignment, Font
from openpyxl.utils.exceptions import IllegalCharacterError
from PyQt5 import QtCore, QtGui, QtWidgets

from prediction_comparison_statistics import comparison_statistics
from qt.prediction_well_preferences import (
    column_config, excluded_report_columns, report_column_key, saved_presets,
    save_preset, save_report_columns,
)
from qt.regression_target_wizard import RegressionTargetWizard, WizardCandidate
from regression_target.presentation import AGGREGATION_TITLES, parameter_header_tooltip
from regression_target.service import (
    Resolution, TargetSettings, list_canonical_targets, resolve_target, save_parameter_choice, well_data_rows,
)


def _aggregation_suffix(settings):
    if settings is not None and settings.source == 'well_log':
        return f' ({AGGREGATION_TITLES[settings.aggregation]})'
    return ''


@dataclass
class PredictionCandidate(WizardCandidate):
    prediction_id: int = 0
    trace_index: int = 0
    corrected: bool = False


@dataclass
class AnalysisColumn:
    title: str
    settings: TargetSettings
    tooltip: str
    config: dict
    values: dict[int, Resolution | str | None]

    @property
    def raw_text(self):
        return self.config['raw_text']

    def value(self, well_id):
        result = self.values[well_id]
        if isinstance(result, Resolution):
            return result.value if result.status == 'resolved' else None
        return result


class PredictionWellCheckDialog(RegressionTargetWizard):
    SETTINGS_GROUP = 'prediction_well_check'
    WELL_VALUE_TITLE = 'Значение скважины'
    STATISTICS = (
        ('minimum', 'Мин. |Δ|', 'Минимальная абсолютная ошибка; ниже указаны скважина и профиль.'),
        ('maximum', 'Макс. |Δ|', 'Максимальная абсолютная ошибка; ниже указаны скважина и профиль.'),
        ('mae', 'Средняя |Δ| (MAE)', 'Среднее абсолютных ошибок по всем рассчитанным парам.'),
        ('median', 'Медиана |Δ|', 'Медиана абсолютных ошибок; менее чувствительна к выбросам, чем среднее.'),
        ('rmse', 'RMSE', 'Корень из среднего квадрата ошибок. Сильнее реагирует на большие отклонения.'),
        ('bias', 'Смещение', 'Средняя разница «скважина − прогноз». Плюс: прогноз занижен; минус: завышен.'),
        ('p90', 'P90 |Δ|', '90-й процентиль абсолютных ошибок с линейной интерполяцией между соседними значениями.'),
        ('match_percent', 'В допуске', 'Количество и доля рассчитанных пар с абсолютной ошибкой не больше допуска.'),
    )
    HEADERS = (
        ('Выбор', 'Служебный столбец выбора строк'),
        ('Скважина', 'Название скважины'),
        ('Кривые', 'Количество каротажных кривых скважины'),
        ('Профиль', 'Название профиля, дата исследования и ID профиля'),
        ('Объект', 'Объект исследования'),
        ('Дист., м', 'Расстояние от скважины до ближайшей точки профиля, м'),
        ('ID / точка', 'ID сохранённого прогноза / номер ближайшей точки профиля (с 1)'),
        ('Параметр', 'Выбранный параметр скважины и настройки его расчёта'),
        ('Источник', 'Исходные записи скважины, использованные для расчёта параметра'),
        ('Прогноз', 'Значение сохранённого прогноза в ближайшей к скважине точке'),
        ('Значение', 'Рассчитанное значение выбранного параметра скважины'),
        ('Δ', 'Разница: значение скважины минус прогноз; цвет зависит от абсолютной ошибки'),
        ('Статус', 'Совпадение в пределах заданного абсолютного допуска или состояние данных'),
    )
    REPORT_FIELDS = (
        ('well_id', 'ID скважины'), ('well', 'Скважина'), ('profile_id', 'ID профиля'),
        ('profile', 'Профиль'), ('object', 'Объект'), ('prediction_id', 'ID прогноза'),
        ('trace_number', 'Точка профиля'), ('distance_m', 'Расстояние, м'),
        ('corrected', 'Исправленный прогноз'), ('prediction', 'Прогноз'),
        ('parameter', 'Параметр сравнения'), ('well_value', WELL_VALUE_TITLE),
        ('well_minus_prediction', 'Скважина − прогноз'), ('status', 'Статус'),
        ('settings', 'Настройки сравнения'), ('details', 'Детали источника'),
    )
    STATISTICS_FIELDS = (
        ('count', 'Рассчитанные пары'), ('well_count', 'Уникальные скважины'),
        ('excluded_count', 'Исключённые пары'), ('within_tolerance', 'Пар в допуске'),
        ('tolerance', 'Допуск'),
    )

    def __init__(self, session, samples, description, parent=None, open_well_log=None, *,
                 prediction_type=None):
        self.analysis_columns: list[AnalysisColumn] = []
        self._comparison_config = None
        self._comparison_title = ''
        self._comparison_tooltip = self.HEADERS[7][1]
        self.prediction_title = {'cls': 'Вероятность', 'reg': 'Прогноз регрессии'}.get(
            prediction_type, 'Прогноз')
        candidates = [PredictionCandidate(
            well_id=s.well.id, well_name=s.well.name or f'id{s.well.id}',
            profile_id=s.profile.id,
            profile_name=f'{s.profile.title} / {s.profile.research.date_research} (id{s.profile.id})',
            formation_id=0, distance=s.distance, list_measure=[s.trace_index],
            stored_value=s.value, well_log_count=len(s.well.well_logs),
            object_name=s.profile.research.object.title or '',
            prediction_id=s.prediction_id, trace_index=s.trace_index, corrected=s.corrected)
            for s in samples]
        super().__init__(session, candidates, parent, open_well_log, mode='check')
        self.setWindowTitle('Проверка прогнозов по скважинам объекта')
        screen = self.screen().availableGeometry()
        self.resize(min(1800, int(screen.width() * 0.95)),
                    min(1000, int(screen.height() * 0.95)))
        label = QtWidgets.QLabel(description)
        label.setWordWrap(True)
        self.layout().insertWidget(0, label)
        self._render()

    def _build_ui(self):
        super()._build_ui()
        # Comparison never edits predictions or training markups.
        for widget in (self.replace_button, self.add_resolved_button,
                       self.add_selected_button, self.delete_selected_button,
                       self.select_all_button, self.select_resolved_button,
                       self.select_missing_button, self.select_mismatched_button,
                       self.clear_selection_button):
            widget.hide()
        self.table.setColumnHidden(0, True)
        for label in self.findChildren(QtWidgets.QLabel):
            if label.text() == 'Быстрый выбор:':
                label.hide()
        for column, (title, tooltip) in enumerate(self.HEADERS):
            header = self.table.horizontalHeaderItem(column)
            header.setText(title)
            header.setToolTip(tooltip)
        self.table.horizontalHeaderItem(9).setText(self.prediction_title)
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Interactive)
        self.table.horizontalHeader().setMinimumSectionSize(64)
        self.table.setWordWrap(False)
        self.calculate_button.setText('Рассчитать сравнение')
        self.hide_missing_check.setText('Скрыть строки без значения для сравнения')
        self.absolute_tolerance.setToolTip('Допустимая абсолютная разница в единицах выбранного параметра')
        columns = QtWidgets.QHBoxLayout()
        self.add_column_button = QtWidgets.QPushButton('Добавить расчётный столбец')
        self.add_text_column_button = QtWidgets.QPushButton('Добавить исходный текст')
        self.add_text_column_button.setToolTip(
            'Добавляет отдельный столбец с исходными записями выбранной категории')
        self.remove_column_button = QtWidgets.QPushButton('Удалить последний столбец')
        self.add_column_button.clicked.connect(lambda: self._add_analysis_column(raw_text=False))
        self.add_text_column_button.clicked.connect(lambda: self._add_analysis_column(raw_text=True))
        self.remove_column_button.clicked.connect(self._remove_analysis_column)
        columns.addWidget(self.add_column_button)
        columns.addWidget(self.add_text_column_button)
        columns.addWidget(self.remove_column_button)
        columns.addStretch()
        self.layout().insertLayout(self.layout().indexOf(self.table), columns)
        self._build_presets_ui()
        for buttons in self.findChildren(QtWidgets.QDialogButtonBox):
            buttons.button(QtWidgets.QDialogButtonBox.Cancel).setText('Закрыть')
        legend = QtWidgets.QLabel(
            'При сравнении зелёный цвет означает попадание в допуск. '
            'Цвет меняется от жёлтого к красному по мере роста абсолютной ошибки.')
        legend.setWordWrap(True)
        self.layout().insertWidget(self.layout().count() - 1, legend)
        self._build_statistics_ui()

    def _build_presets_ui(self):
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel('Наборы столбцов'))
        self.presets_combo = QtWidgets.QComboBox()
        self.presets_combo.setToolTip('Выберите набор, чтобы загрузить его в таблицу')
        row.addWidget(self.presets_combo, 1)
        self.save_preset_button = QtWidgets.QPushButton('Сохранить набор…')
        row.addWidget(self.save_preset_button)
        self.layout().insertLayout(self.layout().indexOf(self.table), row)
        self._refresh_presets()
        self.presets_combo.activated[int].connect(self._load_preset)
        self.save_preset_button.clicked.connect(self._save_preset)

    def _refresh_presets(self, selected=None):
        self.presets_combo.clear()
        self.presets_combo.addItem('Выберите набор')
        for name in sorted(saved_presets(), key=str.casefold):
            self.presets_combo.addItem(name, name)
        if selected is not None:
            self.presets_combo.setCurrentIndex(max(0, self.presets_combo.findData(selected)))

    def _save_preset(self):
        name, accepted = QtWidgets.QInputDialog.getText(
            self, 'Сохранение набора', 'Название набора', text=self.presets_combo.currentData() or '')
        name = name.strip()
        if not accepted or not name:
            return
        if name in saved_presets() and QtWidgets.QMessageBox.question(
                self, 'Замена набора', f'Заменить набор «{name}» текущими столбцами?',
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.No) != QtWidgets.QMessageBox.Yes:
            return
        save_preset(name, {'columns': [column.config for column in self.analysis_columns],
                           'comparison': self._comparison_config})
        self._refresh_presets(name)

    def _load_preset(self, index):
        name = self.presets_combo.itemData(index)
        if name is None:
            return
        preset = saved_presets().get(name)
        if preset is None:
            self._refresh_presets()
            return
        sources = {}

        def resolve_config(config):
            source = config['source']
            if config['raw_text'] and source != 'well_data':
                raise ValueError('Исходный текст доступен для информации о скважине.')
            if source not in sources:
                sources[source] = {row.canonical_name.casefold(): row.id
                                   for row in list_canonical_targets(self.session, source)}
            canonical_id = sources[source].get(config['name'].casefold())
            if canonical_id is None:
                raise ValueError(f'Параметр «{config["name"]}» не найден.')
            options = dict(config['options'])
            boundary = options.pop('boundary_name', None)
            if boundary is not None:
                boundary_index = self.boundary_combo.findText(boundary, QtCore.Qt.MatchFixedString)
                if boundary_index < 0:
                    raise ValueError(f'Опорная граница «{boundary}» не найдена.')
                options['boundary_canonical_id'] = self.boundary_combo.itemData(boundary_index)
            return TargetSettings(source, canonical_id, **options)

        try:
            configurations = [(config, resolve_config(config)) for config in preset['columns']]
            comparison = preset['comparison']
            comparison_target = resolve_config(comparison) if comparison is not None else None
        except ValueError as error:
            QtWidgets.QMessageBox.warning(self, 'Загрузка набора', str(error))
            self.presets_combo.setCurrentIndex(0)
            return
        except (KeyError, TypeError, AttributeError):
            QtWidgets.QMessageBox.warning(self, 'Загрузка набора', 'Не удалось прочитать настройки набора.')
            self.presets_combo.setCurrentIndex(0)
            return

        columns = []
        for config, target in configurations:
            columns.append(self._build_analysis_column(config, target, columns))
        self._set_comparison(comparison, comparison_target)
        self.analysis_columns = columns
        if comparison_target is not None:
            self._restore_target_controls(comparison_target)
        self._render()

    def _restore_target_controls(self, target):
        self.source_combo.setCurrentIndex(self.source_combo.findData(target.source))
        if self.canonical_combo.findData(target.canonical_id) < 0:
            self._source_changed()
        self.canonical_combo.setCurrentIndex(self.canonical_combo.findData(target.canonical_id))
        if target.source == 'well_data':
            self.strict_check.setChecked(target.strict_numeric)
            self.sum_check.setChecked(target.allow_explicit_sum)
        elif target.source == 'well_log':
            for combo, value in ((self.aggregation_combo, target.aggregation),
                                 (self.operation_combo, target.operation),
                                 (self.depth_mode_combo, target.depth_mode),
                                 (self.boundary_combo, target.boundary_canonical_id),
                                 (self.position_combo, target.interval_position)):
                combo.setCurrentIndex(combo.findData(value))
            self.fixed_depth.setValue(target.fixed_depth)
            self.interval.setValue(target.interval)

    def _source_changed(self):
        super()._source_changed()
        self.add_text_column_button.setVisible(self.source_combo.currentData() == 'well_data')

    def _hide_candidate(self, status):
        return self._settings is not None and super()._hide_candidate(status)

    def _current_column_config(self, target, raw_text=False):
        index = self.boundary_combo.findData(target.boundary_canonical_id)
        return column_config(self.canonical_combo.currentText(), target, raw_text,
                             self.boundary_combo.itemText(index))

    def _resolve_for_wells(self, settings, *, raw_text=False):
        well_ids = dict.fromkeys(candidate.well_id for candidate in self.candidates)
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            if raw_text:
                return {
                    well_id: '\n'.join(row.value for row in well_data_rows(
                        self.session, well_id, settings.canonical_id) if row.value) or None
                    for well_id in well_ids
                }
            return {well_id: resolve_target(self.session, well_id, settings) for well_id in well_ids}
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

    def calculate(self):
        settings = self.settings()
        if settings is None:
            QtWidgets.QMessageBox.warning(self, 'Нет параметра', 'Выберите канонический параметр.')
            return
        self._set_comparison(self._current_column_config(settings), settings)
        self._render()

    def _set_comparison(self, config, settings):
        values = self._resolve_for_wells(settings) if settings is not None else {}
        self._settings = settings
        self._comparison_config = config
        self._comparison_title = config['name'] + _aggregation_suffix(settings) if config else ''
        self._comparison_tooltip = (parameter_header_tooltip(
            config['name'], settings, config['options'].get('boundary_name', ''))
            if config else self.HEADERS[7][1])
        for candidate in self.candidates:
            candidate.resolution = values.get(candidate.well_id)

    def _add_analysis_column(self, *, raw_text):
        settings = self.settings()
        if settings is None:
            QtWidgets.QMessageBox.warning(self, 'Нет параметра', 'Выберите канонический параметр.')
            return
        if raw_text:
            if settings.source != 'well_data':
                return
            settings = TargetSettings('well_data', settings.canonical_id)
        config = self._current_column_config(settings, raw_text)
        if any(column.config == config for column in self.analysis_columns):
            QtWidgets.QMessageBox.information(self, 'Столбец уже добавлен', config['name'])
            return
        self.analysis_columns.append(self._build_analysis_column(config, settings, self.analysis_columns))
        self._render()
        if self.table.rowCount():
            column = self.table.columnCount() - 1
            self.table.scrollToItem(self.table.item(0, column), QtWidgets.QAbstractItemView.EnsureVisible)

    def _build_analysis_column(self, config, settings, columns):
        name, raw_text = config['name'], config['raw_text']
        title = f'{name} (исходный текст)' if raw_text else name + _aggregation_suffix(settings)
        used_titles = {column.title for column in columns}
        used_titles.update(label for field in self.REPORT_FIELDS for label in field)
        used_titles.add(self.prediction_title)
        used_titles.update(f'{self.WELL_VALUE_TITLE} ({label})' for label in AGGREGATION_TITLES.values())
        suffix = 2
        base_title = title
        while title in used_titles:
            title = f'{base_title} ({suffix})'
            suffix += 1
        tooltip = (f'Категория «{name}». Исходный текст записей скважины, по одной записи на строку.'
                   if raw_text else parameter_header_tooltip(
                       name, settings, config['options'].get('boundary_name', '')))
        values = self._resolve_for_wells(settings, raw_text=raw_text)
        return AnalysisColumn(title, settings, tooltip, config, values)

    def _remove_analysis_column(self):
        if self.analysis_columns:
            self.analysis_columns.pop()
            self._render()

    def _build_statistics_ui(self):
        group = QtWidgets.QGroupBox('Статистика сравнения')
        grid = QtWidgets.QGridLayout(group)
        self.statistics_labels = {}
        for index, (key, title, tooltip) in enumerate(self.STATISTICS):
            label = QtWidgets.QLabel(title)
            label.setToolTip(tooltip)
            value = QtWidgets.QLabel('—')
            value.setWordWrap(True)
            value.setTextFormat(QtCore.Qt.PlainText)
            value.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
            value.setToolTip(tooltip)
            self.statistics_labels[key] = value
            row, col = divmod(index, 4)
            grid.addWidget(label, row * 2, col)
            grid.addWidget(value, row * 2 + 1, col)
            grid.setColumnStretch(col, 1)
        self.statistics_scope = QtWidgets.QLabel()
        self.statistics_scope.setWordWrap(True)
        grid.addWidget(self.statistics_scope, 4, 0, 1, 4)
        tools = QtWidgets.QHBoxLayout()
        self.minimum_button = QtWidgets.QPushButton('К минимальной ошибке')
        self.maximum_button = QtWidgets.QPushButton('К максимальной ошибке')
        self.statistics_export_button = QtWidgets.QPushButton('Экспорт статистики CSV')
        self.excel_export_button = QtWidgets.QPushButton('Экспорт отчёта и статистики Excel')
        self.minimum_button.clicked.connect(lambda: self._show_extreme('minimum_rows'))
        self.maximum_button.clicked.connect(lambda: self._show_extreme('maximum_rows'))
        self.statistics_export_button.clicked.connect(self._export_statistics_csv)
        self.excel_export_button.clicked.connect(self._export_excel)
        tools.addWidget(self.minimum_button)
        tools.addWidget(self.maximum_button)
        tools.addStretch()
        tools.addWidget(self.statistics_export_button)
        tools.addWidget(self.excel_export_button)
        grid.addLayout(tools, 5, 0, 1, 4)
        self.layout().insertWidget(self.layout().count() - 1, group)

    @staticmethod
    def _location(candidate):
        return (f'{candidate.well_name} (id{candidate.well_id}), '
                f'{candidate.profile_name}, прогноз id{candidate.prediction_id}')

    def _render_statistics(self):
        self.statistics = comparison_statistics(self.candidates, self.absolute_tolerance.value())
        stats = self.statistics
        for key, _, tooltip in self.STATISTICS:
            label = self.statistics_labels[key]
            value = getattr(stats, key)
            text = '—' if value is None else f'{value:.6g}'
            if key == 'match_percent' and value is not None:
                text = f'{stats.within_tolerance} из {stats.count} ({value:.1f}%)'
            if key in ('minimum', 'maximum'):
                rows = getattr(stats, key + '_rows')
                if rows:
                    first = rows[0]
                    text += f' — {first.well_name}, {first.profile_name}'
                    if len(rows) > 1:
                        text += f' (+{len(rows) - 1})'
                    tooltip += '\n' + '\n'.join(self._location(c) for c in rows)
            label.setText(text)
            label.setToolTip(tooltip)
        self.statistics_scope.setText(
            (f'Сравнение по параметру «{self._comparison_title}». ' if self._comparison_title else '') +
            f'Рассчитанных пар: {stats.count}; уникальных скважин: {stats.well_count}; '
            f'без результата / исключено: {stats.excluded_count}. '
            f'Допуск: {stats.tolerance:g}. '
            'Все ошибки — в единицах выбранного параметра. '
            'Каждая пара скважина / прогноз имеет одинаковый вес; повторные скважины учитываются отдельно. '
            'Скрытие строк без данных не меняет статистику.')
        self.minimum_button.setEnabled(bool(stats.count))
        self.maximum_button.setEnabled(bool(stats.count))

    def _show_extreme(self, field):
        candidates = getattr(self.statistics, field)
        if not candidates:
            return
        for row in range(self.table.rowCount()):
            if self._candidate_for_row(row) is candidates[0]:
                self.table.selectRow(row)
                self.table.scrollToItem(self.table.item(row, 11))
                return

    def _export_statistics_csv(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, 'Экспорт статистики', 'prediction_well_statistics.csv', 'CSV (*.csv)')
        if not path:
            return
        with open(path, 'w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.writer(stream, delimiter=';')
            writer.writerow(('metric', 'value', 'description', 'well_id', 'well',
                             'profile_id', 'profile', 'prediction_id'))
            writer.writerows(self._statistics_rows())

    def _statistics_rows(self):
        stats = self.statistics
        yield 'parameter', self._comparison_title
        yield 'settings', self._comparison_settings_json()
        for key, _ in self.STATISTICS_FIELDS:
            yield key, getattr(stats, key)
        for key, _, tooltip in self.STATISTICS:
            rows = getattr(stats, key + '_rows', ())
            if rows:
                for c in rows:
                    yield (key, getattr(stats, key), tooltip, c.well_id, c.well_name,
                           c.profile_id, c.profile_name, c.prediction_id)
            else:
                yield key, getattr(stats, key), tooltip

    def _update_parameter_header(self):
        self.table.horizontalHeaderItem(7).setToolTip(self._comparison_tooltip)
        self.table.horizontalHeaderItem(10).setText(self.HEADERS[10][0] + _aggregation_suffix(self._settings))

    def _resolve_ambiguity(self, row, column):
        candidate = self._candidate_for_row(row)
        if candidate is None:
            return
        analysis_index = column - len(self.HEADERS)
        analysis = self.analysis_columns[analysis_index] if analysis_index >= 0 else None
        settings = analysis.settings if analysis else self._settings
        result = analysis.values[candidate.well_id] if analysis else candidate.resolution
        if settings is None or not isinstance(result, Resolution) or not result.candidates:
            return
        if analysis and len(result.candidates) < 2:
            return
        labels = [f'{item.source_name}: {item.raw_value} → {item.value:g} (id{item.source_id})'
                  for item in result.candidates]
        selected, ok = QtWidgets.QInputDialog.getItem(
            self, 'Выбор значения', 'Исходная запись', labels, 0, False)
        if not ok:
            return
        index = labels.index(selected)
        choice = result.candidates[index]
        source, canonical_id = settings.source, settings.canonical_id
        well_id = candidate.well_id
        if source == 'well_log' and result.details.get('pending_selection') == 'boundary_depth':
            source, canonical_id = 'boundary', settings.boundary_canonical_id
            result = resolve_target(self.session, well_id, settings, boundary_candidate=choice)
        else:
            result.select(index)
        save_parameter_choice(self.session, well_id, source, canonical_id, choice.source_id)

        def update(other_settings, previous):
            if other_settings == settings:
                return result
            if ((other_settings.source, other_settings.canonical_id) == (source, canonical_id)
                    or (source == 'boundary' and other_settings.source == 'well_log'
                        and other_settings.depth_mode == 'boundary'
                        and other_settings.boundary_canonical_id == canonical_id)):
                return resolve_target(self.session, well_id, other_settings)
            return previous

        if self._settings is not None:
            primary = update(self._settings, candidate.resolution)
            for peer in self.candidates:
                if peer.well_id == well_id:
                    peer.resolution = primary
        for other in self.analysis_columns:
            if not other.raw_text:
                other.values[well_id] = update(other.settings, other.values[well_id])
        self._render()

    def _open_well_log(self):
        column = self.table.currentColumn() - len(self.HEADERS)
        if column < 0:
            super()._open_well_log()
            return
        candidate = self._candidate_for_row(self.table.currentRow())
        if candidate and self.open_well_log_callback:
            result = self.analysis_columns[column].values[candidate.well_id]
            details = result.details if isinstance(result, Resolution) else {}
            self.open_well_log_callback(candidate.well_id, details)

    @staticmethod
    def _absolute_error(candidate):
        result = candidate.resolution
        if (result is None or result.status != 'resolved' or result.value is None
                or candidate.stored_value is None):
            return None
        return abs(result.value - candidate.stored_value)

    def _error_color(self, error, max_error):
        tolerance = self.absolute_tolerance.value()
        if error <= tolerance:
            return QtGui.QColor('#d9f2df')
        fraction = min(1.0, (error - tolerance) / (max_error - tolerance))
        yellow, red = QtGui.QColor('#fff1b8'), QtGui.QColor('#f4a6a6')
        return QtGui.QColor(*(round(a + (b - a) * fraction) for a, b in zip(
            yellow.getRgb()[:3], red.getRgb()[:3])))

    def _render(self):
        first_analysis_column = len(self.HEADERS)
        old_column_count = self.table.columnCount()
        self.table.setColumnCount(first_analysis_column + len(self.analysis_columns))
        for index, column in enumerate(self.analysis_columns, first_analysis_column):
            header = QtWidgets.QTableWidgetItem(column.title)
            header.setToolTip(column.tooltip)
            self.table.setHorizontalHeaderItem(index, header)
        super()._render()
        errors = [error for c in self.candidates
                  if (error := self._absolute_error(c)) is not None]
        max_error = max(errors, default=0)
        for row in range(self.table.rowCount()):
            candidate = self._candidate_for_row(row)
            self.table.item(row, 6).setText(f'{candidate.prediction_id} / {candidate.trace_index + 1}')
            self.table.item(row, 7).setText(self._comparison_title)
            if candidate.resolution is None:
                self.table.item(row, 8).setText('')
            error = self._absolute_error(candidate)
            color = self._error_color(error, max_error) if error is not None else None
            for index in range(1, first_analysis_column):
                item = self.table.item(row, index)
                item.setToolTip(item.text() + '\n' + item.toolTip())
                if color is not None:
                    item.setBackground(color)
            self.table.item(row, 9).setToolTip('Исправленный прогноз' if candidate.corrected else 'Исходный прогноз')
            for index, column in enumerate(self.analysis_columns, first_analysis_column):
                self.table.setItem(row, index, self._analysis_item(column, candidate.well_id))
        for index in range(1, first_analysis_column):
            self.table.resizeColumnToContents(index)
        for index in range(old_column_count, self.table.columnCount()):
            self.table.resizeColumnToContents(index)
            self.table.setColumnWidth(index, min(350, self.table.columnWidth(index)))
        self.remove_column_button.setEnabled(bool(self.analysis_columns))
        self.save_preset_button.setEnabled(bool(self.analysis_columns) or self._settings is not None)
        self.summary.setText(self.summary.text().split('; уже добавлено:')[0])
        self._render_statistics()

    def _analysis_item(self, column, well_id):
        result = column.values[well_id]
        if column.raw_text:
            item = QtWidgets.QTableWidgetItem(result or '')
            if result:
                item.setToolTip(f'<pre>{escape(result)}</pre>')
            return item
        text = f'{result.value:g}' if result.status == 'resolved' and result.value is not None else ''
        item = QtWidgets.QTableWidgetItem('выберите…' if result.status == 'ambiguous' else text)
        has_alternatives = len(result.candidates) > 1
        item.setBackground(QtGui.QColor('#ffe4a3' if has_alternatives else self.STATUS_COLOR[result.status]))
        tooltip = result.message
        if has_alternatives:
            tooltip += '\nДважды щёлкните, чтобы выбрать исходную запись.'
        item.setToolTip(tooltip)
        return item

    def _report_headers(self):
        labels = {'prediction': self.prediction_title,
                  'well_value': self.WELL_VALUE_TITLE + _aggregation_suffix(self._settings)}
        return ([(key, labels.get(key, title)) for key, title in self.REPORT_FIELDS] +
                [(column.title, column.title) for column in self.analysis_columns])

    def _report_column_keys(self):
        return [key for key, _ in self.REPORT_FIELDS] + [
            report_column_key(column.config) for column in self.analysis_columns]

    def _select_report_columns(self, headers, keys, file_format):
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle('Столбцы отчёта')
        dialog.resize(460, 470)
        layout = QtWidgets.QVBoxLayout(dialog)
        columns = QtWidgets.QListWidget(dialog)
        excluded = excluded_report_columns(file_format)
        for key, (_, title) in zip(keys, headers):
            item = QtWidgets.QListWidgetItem(title, columns)
            item.setCheckState(QtCore.Qt.Unchecked if key in excluded else QtCore.Qt.Checked)
        layout.addWidget(columns)

        actions = QtWidgets.QHBoxLayout()
        select_all = QtWidgets.QPushButton('Отметить все', dialog)
        clear_all = QtWidgets.QPushButton('Снять все', dialog)
        select_all.setAutoDefault(False)
        clear_all.setAutoDefault(False)
        actions.addWidget(select_all)
        actions.addWidget(clear_all)
        actions.addStretch()
        layout.addLayout(actions)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        confirm = buttons.button(QtWidgets.QDialogButtonBox.Ok)
        confirm.setText('Продолжить')
        confirm.setDefault(True)
        buttons.button(QtWidgets.QDialogButtonBox.Cancel).setText('Отмена')
        layout.addWidget(buttons)

        def selected_columns():
            return [index for index in range(columns.count())
                    if columns.item(index).checkState() == QtCore.Qt.Checked]

        def set_all(state):
            with QtCore.QSignalBlocker(columns):
                for index in range(columns.count()):
                    columns.item(index).setCheckState(state)
            confirm.setEnabled(state == QtCore.Qt.Checked)

        select_all.clicked.connect(lambda: set_all(QtCore.Qt.Checked))
        clear_all.clicked.connect(lambda: set_all(QtCore.Qt.Unchecked))
        columns.itemChanged.connect(lambda: confirm.setEnabled(bool(selected_columns())))
        confirm.setEnabled(bool(selected_columns()))
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        selected = selected_columns() if dialog.exec_() == QtWidgets.QDialog.Accepted else []
        dialog.deleteLater()
        return selected

    def _export_csv(self):
        headers = self._report_headers()
        keys = self._report_column_keys()
        selected = self._select_report_columns(headers, keys, 'csv')
        if not selected:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, 'Экспорт сравнения', 'prediction_wells.csv', 'CSV (*.csv)')
        if not path:
            return
        with open(path, 'w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.writer(stream, delimiter=';')
            writer.writerow([headers[index][0] for index in selected])
            writer.writerows(self._report_rows(selected))
        save_report_columns('csv', keys, selected)

    def _comparison_settings_json(self):
        return json.dumps(self._settings.as_dict(), ensure_ascii=False) if self._settings else ''

    def _report_rows(self, selected):
        settings = self._comparison_settings_json()
        for c in self.candidates:
            r = c.resolution
            value = r.value if r else None
            row = (c.well_id, c.well_name, c.profile_id, c.profile_name, c.object_name,
                   c.prediction_id, c.trace_index + 1, c.distance, c.corrected, c.stored_value,
                   self._comparison_title, value,
                   value - c.stored_value if value is not None else None,
                   r.status if r else 'not_calculated', settings,
                   json.dumps(r.details, ensure_ascii=False) if r else '{}',
                   *(column.value(c.well_id) for column in self.analysis_columns))
            yield tuple(row[index] for index in selected)

    def _export_excel(self):
        headers = self._report_headers()
        keys = self._report_column_keys()
        selected = self._select_report_columns(headers, keys, 'xlsx')
        if not selected:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, 'Экспорт отчёта и статистики', 'prediction_wells.xlsx', 'Excel (*.xlsx)')
        if not path:
            return
        if not path.lower().endswith('.xlsx'):
            path += '.xlsx'
        workbook = Workbook()
        try:
            report = workbook.active
            report.title = 'Отчёт'
            self._fill_worksheet(report, [headers[index][1] for index in selected],
                                 self._report_rows(selected))
            for index, source_index in enumerate(selected, 1):
                analysis_index = source_index - len(self.REPORT_FIELDS)
                if analysis_index < 0:
                    continue
                column = self.analysis_columns[analysis_index]
                report.cell(1, index).comment = Comment(column.tooltip, 'GeoVel')
                if column.raw_text:
                    for cells in report.iter_rows(min_row=2, min_col=index, max_col=index):
                        cells[0].alignment = Alignment(wrap_text=True, vertical='top')

            statistics = workbook.create_sheet('Статистика')
            metric_titles = {**dict(self.STATISTICS_FIELDS),
                             **{key: title for key, title, _ in self.STATISTICS},
                             'parameter': 'Параметр сравнения', 'settings': 'Настройки сравнения',
                             'match_percent': 'В допуске, %'}
            rows = ((metric_titles[key], *values) for key, *values in self._statistics_rows())
            self._fill_worksheet(statistics, ('Метрика', 'Значение', 'Описание', 'ID скважины',
                                             'Скважина', 'ID профиля', 'Профиль', 'ID прогноза'), rows)
            workbook.save(path)
        except (OSError, ValueError, IllegalCharacterError) as error:
            QtWidgets.QMessageBox.warning(
                self, 'Ошибка экспорта', f'Не удалось сохранить книгу Excel.\n{error}')
        else:
            save_report_columns('xlsx', keys, selected)
        finally:
            workbook.close()

    @staticmethod
    def _fill_worksheet(sheet, headers, rows):
        sheet.append(headers)
        for cell in sheet[1]:
            cell.data_type = 's'
            cell.font = Font(bold=True)
            sheet.column_dimensions[cell.column_letter].width = min(42, max(16, len(cell.value) + 2))
        for row_number, row in enumerate(rows, 2):
            for column, value in enumerate(row, 1):
                if isinstance(value, str) and len(value) > 32767:
                    raise ValueError('Текст превышает предел ячейки Excel (32 767 символов). '
                                     'Полную запись можно сохранить в CSV.')
                cell = sheet.cell(row_number, column, value)
                # Keep original well text as literal strings, including leading "=".
                if isinstance(value, str):
                    cell.data_type = 's'
        sheet.freeze_panes = 'A2'
        sheet.auto_filter.ref = sheet.dimensions
