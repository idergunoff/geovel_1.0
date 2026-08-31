"""Modal review wizard for automatically populated regression targets."""
from __future__ import annotations

import json
import csv
from dataclasses import dataclass
from typing import Callable

from PyQt5 import QtCore, QtGui, QtWidgets

from models_db.model import CanonicalBoundary
from regression_target.service import (
    Resolution,
    TargetSettings,
    list_canonical_targets,
    resolve_target,
)


@dataclass
class WizardCandidate:
    well_id: int
    well_name: str
    profile_id: int
    profile_name: str
    formation_id: int
    distance: float
    list_measure: list[int]
    already_exists: bool = False
    resolution: Resolution | None = None


class RegressionTargetWizard(QtWidgets.QDialog):
    SOURCE_DATA = (("Глубина границы", "boundary"), ("Информация о скважине", "well_data"),
                   ("Каротажная кривая", "well_log"))
    STATUS_TEXT = {"resolved": "Готово", "ambiguous": "Требуется выбор",
                   "missing": "Нет данных", "invalid": "Ошибка данных"}
    STATUS_COLOR = {"resolved": "#d9f2df", "ambiguous": "#fff1b8",
                    "missing": "#eeeeee", "invalid": "#ffd6d6"}

    def __init__(self, session, candidates: list[WizardCandidate], parent=None,
                 open_well_log: Callable[[int, dict], None] | None = None):
        super().__init__(parent)
        self.session = session
        self.candidates = candidates
        self.open_well_log_callback = open_well_log
        self._settings: TargetSettings | None = None
        self.setWindowTitle("Массовое добавление скважин — целевая переменная")
        self.setModal(True)
        self.resize(1200, 720)
        self._build_ui()
        self._source_changed()

    def _build_ui(self):
        root = QtWidgets.QVBoxLayout(self)
        source_row = QtWidgets.QHBoxLayout()
        self.source_combo = QtWidgets.QComboBox()
        for title, value in self.SOURCE_DATA:
            self.source_combo.addItem(title, value)
        self.canonical_combo = QtWidgets.QComboBox()
        source_row.addWidget(QtWidgets.QLabel("Источник:"))
        source_row.addWidget(self.source_combo)
        source_row.addWidget(QtWidgets.QLabel("Каноническое название:"))
        source_row.addWidget(self.canonical_combo, 1)
        root.addLayout(source_row)

        self.options = QtWidgets.QGroupBox("Настройки источника")
        options = QtWidgets.QGridLayout(self.options)
        self.strict_check = QtWidgets.QCheckBox("Строгий разбор строк")
        self.sum_check = QtWidgets.QCheckBox("Складывать явные выражения через +")
        self.sum_check.setChecked(True)
        self.aggregation_combo = QtWidgets.QComboBox()
        self.aggregation_combo.addItem("Медиана", "median")
        self.aggregation_combo.addItem("Среднее", "mean")
        self.operation_combo = QtWidgets.QComboBox()
        for title, value in (("Значение интервала", "single"), ("Верх / низ", "upper_lower_ratio"),
                             ("Низ / верх", "lower_upper_ratio"), ("Низ − верх", "difference")):
            self.operation_combo.addItem(title, value)
        self.depth_mode_combo = QtWidgets.QComboBox()
        self.depth_mode_combo.addItem("От канонической границы", "boundary")
        self.depth_mode_combo.addItem("Фиксированная глубина", "fixed")
        self.boundary_combo = QtWidgets.QComboBox()
        for row in self.session.query(CanonicalBoundary).order_by(CanonicalBoundary.canonical_name).all():
            self.boundary_combo.addItem(row.canonical_name, row.id)
        self.fixed_depth = QtWidgets.QDoubleSpinBox(); self.fixed_depth.setRange(-100000, 100000); self.fixed_depth.setDecimals(3)
        self.interval = QtWidgets.QDoubleSpinBox(); self.interval.setRange(0.001, 100000); self.interval.setValue(5); self.interval.setDecimals(3)
        self.position_combo = QtWidgets.QComboBox()
        self.position_combo.addItem("Ниже глубины", "below")
        self.position_combo.addItem("Выше глубины", "above")
        self.position_combo.addItem("Симметрично", "centered")
        widgets = (("", self.strict_check), ("", self.sum_check), ("Агрегация", self.aggregation_combo),
                   ("Операция", self.operation_combo), ("Опорная глубина", self.depth_mode_combo),
                   ("Граница", self.boundary_combo), ("Фиксированная глубина", self.fixed_depth),
                   ("Интервал", self.interval), ("Положение", self.position_combo))
        self.log_widgets = []
        for index, (label, widget) in enumerate(widgets):
            col = (index % 3) * 2
            row = index // 3
            if label:
                label_widget = QtWidgets.QLabel(label + ":")
                options.addWidget(label_widget, row, col)
                if index >= 2:
                    self.log_widgets.append(label_widget)
            options.addWidget(widget, row, col + (1 if label else 0), 1, 1 if label else 2)
            if index >= 2:
                self.log_widgets.append(widget)
        root.addWidget(self.options)

        tools = QtWidgets.QHBoxLayout()
        self.calculate_button = QtWidgets.QPushButton("Рассчитать / обновить")
        self.hide_missing_check = QtWidgets.QCheckBox("Скрыть строки без данных")
        self.existing_combo = QtWidgets.QComboBox()
        self.existing_combo.addItem("Существующие: пропускать", "skip")
        self.existing_combo.addItem("Существующие: обновить целевое значение", "target")
        self.existing_combo.addItem("Существующие: пересчитать всё", "all")
        self.export_button = QtWidgets.QPushButton("Экспорт отчёта CSV")
        self.open_log_button = QtWidgets.QPushButton("Открыть каротаж выбранной скважины")
        tools.addWidget(self.calculate_button); tools.addWidget(self.hide_missing_check)
        tools.addWidget(self.existing_combo); tools.addStretch(); tools.addWidget(self.export_button)
        tools.addWidget(self.open_log_button)
        root.addLayout(tools)

        self.table = QtWidgets.QTableWidget(0, 9)
        self.table.setHorizontalHeaderLabels(("Добавить", "Скважина", "Профиль", "Расстояние", "Пласт ID",
                                               "Источник", "Исходные значения", "Целевое значение", "Статус"))
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(6, QtWidgets.QHeaderView.Stretch)
        root.addWidget(self.table, 1)
        self.summary = QtWidgets.QLabel()
        root.addWidget(self.summary)
        buttons = QtWidgets.QDialogButtonBox()
        self.add_resolved_button = buttons.addButton("Добавить однозначные", QtWidgets.QDialogButtonBox.AcceptRole)
        self.add_selected_button = buttons.addButton("Добавить выбранные", QtWidgets.QDialogButtonBox.AcceptRole)
        buttons.addButton(QtWidgets.QDialogButtonBox.Cancel)
        root.addWidget(buttons)

        self.source_combo.currentIndexChanged.connect(self._source_changed)
        self.depth_mode_combo.currentIndexChanged.connect(self._depth_mode_changed)
        self.calculate_button.clicked.connect(self.calculate)
        self.hide_missing_check.toggled.connect(self._render)
        self.existing_combo.currentIndexChanged.connect(self._render)
        self.export_button.clicked.connect(self._export_csv)
        self.table.cellDoubleClicked.connect(self._resolve_ambiguity)
        self.open_log_button.clicked.connect(self._open_well_log)
        self.add_resolved_button.clicked.connect(self._accept_resolved)
        self.add_selected_button.clicked.connect(self._accept_selected)
        buttons.rejected.connect(self.reject)

    def _source_changed(self):
        source = self.source_combo.currentData()
        self.canonical_combo.clear()
        for row in list_canonical_targets(self.session, source):
            self.canonical_combo.addItem(row.canonical_name, row.id)
        is_data = source == "well_data"
        is_log = source == "well_log"
        self.strict_check.setVisible(is_data); self.sum_check.setVisible(is_data)
        for widget in self.log_widgets:
            widget.setVisible(is_log)
        self.open_log_button.setVisible(is_log)
        self._depth_mode_changed()

    def _depth_mode_changed(self):
        boundary = self.depth_mode_combo.currentData() == "boundary"
        self.boundary_combo.setEnabled(boundary)
        self.fixed_depth.setEnabled(not boundary)

    def settings(self) -> TargetSettings | None:
        canonical_id = self.canonical_combo.currentData()
        if canonical_id is None:
            return None
        return TargetSettings(
            source=self.source_combo.currentData(), canonical_id=int(canonical_id),
            strict_numeric=self.strict_check.isChecked(), allow_explicit_sum=self.sum_check.isChecked(),
            aggregation=self.aggregation_combo.currentData(), operation=self.operation_combo.currentData(),
            depth_mode=self.depth_mode_combo.currentData(), fixed_depth=self.fixed_depth.value(),
            boundary_canonical_id=self.boundary_combo.currentData(), interval=self.interval.value(),
            interval_position=self.position_combo.currentData())

    def calculate(self):
        settings = self.settings()
        if settings is None:
            QtWidgets.QMessageBox.warning(self, "Нет показателя", "Выберите каноническое название.")
            return
        self._settings = settings
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            for candidate in self.candidates:
                candidate.resolution = resolve_target(self.session, candidate.well_id, settings)
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        self._render()

    def _render(self):
        self.table.setRowCount(0)
        counts = {key: 0 for key in self.STATUS_TEXT}
        existing = 0
        for index, candidate in enumerate(self.candidates):
            resolution = candidate.resolution
            status = resolution.status if resolution else "missing"
            if candidate.already_exists:
                existing += 1
            if resolution:
                counts[status] += 1
            if self.hide_missing_check.isChecked() and status in ("missing", "invalid"):
                continue
            row = self.table.rowCount(); self.table.insertRow(row)
            check = QtWidgets.QTableWidgetItem()
            check.setFlags(QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsUserCheckable)
            can_update = not candidate.already_exists or self.existing_mode() != "skip"
            check.setCheckState(QtCore.Qt.Checked if resolution and status == "resolved" and can_update else QtCore.Qt.Unchecked)
            check.setData(QtCore.Qt.UserRole, index)
            self.table.setItem(row, 0, check)
            values = (candidate.well_name, candidate.profile_name, f"{candidate.distance:.2f}", str(candidate.formation_id),
                      self.canonical_combo.currentText(), self._candidate_text(resolution),
                      "" if not resolution or resolution.value is None else f"{resolution.value:g}",
                      "Уже добавлена" if candidate.already_exists else self.STATUS_TEXT[status])
            for column, value in enumerate(values, 1):
                item = QtWidgets.QTableWidgetItem(value)
                item.setBackground(QtGui.QColor("#eeeeee" if candidate.already_exists else self.STATUS_COLOR[status]))
                if resolution:
                    item.setToolTip(resolution.message)
                self.table.setItem(row, column, item)
        self.summary.setText(f"Кандидатов: {len(self.candidates)}; готово: {counts['resolved']}; "
                             f"требуют выбора: {counts['ambiguous']}; без данных/ошибки: "
                             f"{counts['missing'] + counts['invalid']}; уже добавлено: {existing}")

    @staticmethod
    def _candidate_text(resolution: Resolution | None) -> str:
        if not resolution:
            return "Нажмите «Рассчитать»"
        if resolution.candidates:
            return "; ".join(f"{row.source_name}={row.raw_value}" for row in resolution.candidates)
        return resolution.message

    def _candidate_for_row(self, row: int) -> WizardCandidate | None:
        item = self.table.item(row, 0)
        return self.candidates[item.data(QtCore.Qt.UserRole)] if item else None

    def _resolve_ambiguity(self, row: int, _column: int):
        candidate = self._candidate_for_row(row)
        if not candidate or not candidate.resolution or len(candidate.resolution.candidates) < 1:
            return
        labels = [f"{item.source_name}: {item.raw_value} → {item.value:g}" for item in candidate.resolution.candidates]
        selected, ok = QtWidgets.QInputDialog.getItem(self, "Выбор значения", "Исходная запись:", labels, 0, False)
        if ok:
            candidate.resolution.select(labels.index(selected))
            self._render()

    def _open_well_log(self):
        row = self.table.currentRow()
        candidate = self._candidate_for_row(row) if row >= 0 else None
        if candidate and self.open_well_log_callback:
            details = candidate.resolution.details if candidate.resolution else {}
            self.open_well_log_callback(candidate.well_id, details)

    def _checked_candidates(self):
        selected = []
        for row in range(self.table.rowCount()):
            if self.table.item(row, 0).checkState() == QtCore.Qt.Checked:
                candidate = self._candidate_for_row(row)
                if candidate:
                    selected.append(candidate)
        return selected

    def _accept_resolved(self):
        for row in range(self.table.rowCount()):
            candidate = self._candidate_for_row(row)
            self.table.item(row, 0).setCheckState(
                QtCore.Qt.Checked if candidate and candidate.resolution and candidate.resolution.status == "resolved"
                and (not candidate.already_exists or self.existing_mode() != "skip") else QtCore.Qt.Unchecked)
        self._accept_selected()

    def _accept_selected(self):
        invalid = [row for row in self._checked_candidates() if (row.already_exists and self.existing_mode() == "skip") or not row.resolution or
                   row.resolution.status != "resolved" or row.resolution.value is None]
        if invalid:
            QtWidgets.QMessageBox.warning(self, "Неразрешённые строки",
                                          "Добавлять можно только строки со статусом «Готово».")
            return
        if not self._checked_candidates():
            QtWidgets.QMessageBox.warning(self, "Нет строк", "Не выбрано ни одной скважины.")
            return
        self.accept()

    def selected_candidates(self) -> list[WizardCandidate]:
        return self._checked_candidates()

    def existing_mode(self) -> str:
        return self.existing_combo.currentData()

    def _export_csv(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Экспорт отчёта", "regression_targets.csv",
                                                       "CSV (*.csv)")
        if not path:
            return
        with open(path, "w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.writer(stream, delimiter=";")
            writer.writerow(("well_id", "well", "profile_id", "profile", "distance", "formation_id",
                             "status", "target_value", "message", "source_details"))
            for row in self.candidates:
                resolution = row.resolution
                writer.writerow((row.well_id, row.well_name, row.profile_id, row.profile_name, row.distance,
                                 row.formation_id, resolution.status if resolution else "not_calculated",
                                 resolution.value if resolution else "", resolution.message if resolution else "",
                                 json.dumps(resolution.details, ensure_ascii=False) if resolution else "{}"))

    def provenance_json(self, candidate: WizardCandidate) -> tuple[str, str, bool]:
        config = json.dumps(self._settings.as_dict(), ensure_ascii=False) if self._settings else "{}"
        resolution = candidate.resolution
        details = json.dumps(resolution.details if resolution else {}, ensure_ascii=False)
        return config, details, bool(resolution and resolution.details.get("manual_override"))
