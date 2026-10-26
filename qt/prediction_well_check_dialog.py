"""Prediction comparison using the same parameter selector as Add all well."""
import csv
import json
from dataclasses import dataclass

from PyQt5 import QtWidgets

from qt.regression_target_wizard import RegressionTargetWizard, WizardCandidate


@dataclass
class PredictionCandidate(WizardCandidate):
    prediction_id: int = 0
    trace_index: int = 0
    corrected: bool = False


class PredictionWellCheckDialog(RegressionTargetWizard):
    SETTINGS_GROUP = 'prediction_well_check'

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
        self.table.horizontalHeaderItem(6).setText('Прогноз ID / точка')
        self.table.horizontalHeaderItem(9).setText('Значение прогноза')
        self.table.horizontalHeaderItem(10).setText('Параметр скважины')
        self.table.horizontalHeaderItem(11).setText('Скважина − прогноз')
        self.absolute_tolerance.setToolTip('Допустимая абсолютная разница в единицах выбранного параметра')
        for buttons in self.findChildren(QtWidgets.QDialogButtonBox):
            buttons.button(QtWidgets.QDialogButtonBox.Cancel).setText('Закрыть')

    def _render(self):
        super()._render()
        for row in range(self.table.rowCount()):
            candidate = self._candidate_for_row(row)
            self.table.item(row, 6).setText(f'{candidate.prediction_id} / {candidate.trace_index + 1}')
            self.table.item(row, 9).setToolTip('Исправленный прогноз' if candidate.corrected else 'Исходный прогноз')
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
