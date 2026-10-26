"""Prediction comparison using the same parameter selector as Add all well."""
import csv
import json
from dataclasses import dataclass

from PyQt5 import QtGui, QtWidgets

from qt.regression_target_wizard import RegressionTargetWizard, WizardCandidate


@dataclass
class PredictionCandidate(WizardCandidate):
    prediction_id: int = 0
    trace_index: int = 0
    corrected: bool = False


class PredictionWellCheckDialog(RegressionTargetWizard):
    SETTINGS_GROUP = 'prediction_well_check'
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
