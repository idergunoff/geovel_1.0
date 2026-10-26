"""Prediction comparison using the same parameter selector as Add all well."""
import csv
import json
from dataclasses import dataclass

from PyQt5 import QtCore, QtGui, QtWidgets

from prediction_comparison_statistics import comparison_statistics
from qt.regression_target_wizard import RegressionTargetWizard, WizardCandidate


@dataclass
class PredictionCandidate(WizardCandidate):
    prediction_id: int = 0
    trace_index: int = 0
    corrected: bool = False


class PredictionWellCheckDialog(RegressionTargetWizard):
    SETTINGS_GROUP = 'prediction_well_check'
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

    def __init__(self, session, samples, description, parent=None, open_well_log=None):
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
        for widget in (self.source_combo, self.canonical_combo, self.aggregation_combo,
                       self.operation_combo, self.depth_mode_combo, self.boundary_combo,
                       self.position_combo):
            widget.currentIndexChanged.connect(self._invalidate_results)
        for widget in (self.fixed_depth, self.interval):
            widget.valueChanged.connect(self._invalidate_results)
        for widget in (self.strict_check, self.sum_check):
            widget.toggled.connect(self._invalidate_results)
        self._render()

    def _invalidate_results(self):
        self._settings = None
        for candidate in self.candidates:
            candidate.resolution = None
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
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Interactive)
        self.table.horizontalHeader().setMinimumSectionSize(64)
        self.absolute_tolerance.setToolTip('Допустимая абсолютная разница в единицах выбранного параметра')
        for buttons in self.findChildren(QtWidgets.QDialogButtonBox):
            buttons.button(QtWidgets.QDialogButtonBox.Cancel).setText('Закрыть')
        legend = QtWidgets.QLabel(
            'Цвет: зелёный — в пределах допуска; жёлтый → красный — рост абсолютной ошибки; '
            'красный — максимальная ошибка среди всех строк с рассчитанными значениями.')
        legend.setWordWrap(True)
        self.layout().insertWidget(self.layout().count() - 1, legend)
        self._build_statistics_ui()

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
        self.minimum_button.clicked.connect(lambda: self._show_extreme('minimum_rows'))
        self.maximum_button.clicked.connect(lambda: self._show_extreme('maximum_rows'))
        self.statistics_export_button.clicked.connect(self._export_statistics_csv)
        tools.addWidget(self.minimum_button)
        tools.addWidget(self.maximum_button)
        tools.addStretch()
        tools.addWidget(self.statistics_export_button)
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
        stats = self.statistics
        with open(path, 'w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.writer(stream, delimiter=';')
            writer.writerow(('metric', 'value', 'description', 'well_id', 'well',
                             'profile_id', 'profile', 'prediction_id'))
            writer.writerow(('parameter', self.canonical_combo.currentText()))
            writer.writerow(('settings', json.dumps(self._settings.as_dict(), ensure_ascii=False)
                             if self._settings else ''))
            for key in ('count', 'well_count', 'excluded_count', 'within_tolerance', 'tolerance'):
                writer.writerow((key, getattr(stats, key)))
            for key, _, tooltip in self.STATISTICS:
                rows = getattr(stats, key + '_rows', ())
                if rows:
                    for c in rows:
                        writer.writerow((key, getattr(stats, key), tooltip, c.well_id, c.well_name,
                                         c.profile_id, c.profile_name, c.prediction_id))
                else:
                    writer.writerow((key, getattr(stats, key), tooltip))

    def _update_parameter_header(self):
        super()._update_parameter_header()
        self.table.horizontalHeaderItem(7).setText('Параметр')

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
        super()._render()
        errors = [error for c in self.candidates
                  if (error := self._absolute_error(c)) is not None]
        max_error = max(errors, default=0)
        for row in range(self.table.rowCount()):
            candidate = self._candidate_for_row(row)
            self.table.item(row, 6).setText(f'{candidate.prediction_id} / {candidate.trace_index + 1}')
            error = self._absolute_error(candidate)
            if error is not None:
                color = self._error_color(error, max_error)
                for column in range(1, self.table.columnCount()):
                    self.table.item(row, column).setBackground(color)
            for column in range(1, self.table.columnCount()):
                item = self.table.item(row, column)
                item.setToolTip(item.text() + '\n' + item.toolTip())
            self.table.item(row, 9).setToolTip('Исправленный прогноз' if candidate.corrected else 'Исходный прогноз')
        self.table.resizeColumnsToContents()
        self.summary.setText(self.summary.text().split('; уже добавлено:')[0])
        self._render_statistics()

    def _export_csv(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, 'Экспорт сравнения', 'prediction_wells.csv', 'CSV (*.csv)')
        if not path:
            return
        settings = self._settings
        with open(path, 'w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.writer(stream, delimiter=';')
            writer.writerow(('well_id', 'well', 'profile_id', 'profile', 'object', 'prediction_id',
                             'trace_number', 'distance_m', 'corrected', 'prediction', 'parameter',
                             'well_value', 'well_minus_prediction', 'status', 'settings', 'details'))
            for c in self.candidates:
                r = c.resolution
                value = r.value if r else None
                writer.writerow((c.well_id, c.well_name, c.profile_id, c.profile_name, c.object_name,
                                 c.prediction_id, c.trace_index + 1, c.distance, c.corrected, c.stored_value,
                                 self.canonical_combo.currentText(), value,
                                 value - c.stored_value if value is not None else '',
                                 r.status if r else 'not_calculated',
                                 json.dumps(settings.as_dict(), ensure_ascii=False) if settings else '',
                                 json.dumps(r.details, ensure_ascii=False) if r else '{}'))
