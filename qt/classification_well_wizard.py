"""Batch review dialog for assigning wells to classification markers."""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from typing import Callable

from PyQt5 import QtCore, QtGui, QtWidgets

from app_settings import restore_form, save_form
from models_db.model import CanonicalBoundary
from regression_target.service import (
    Resolution, TargetSettings, list_canonical_targets, resolve_target,
    save_parameter_choice,
)


@dataclass
class ClassificationWellCandidate:
    well_id: int
    well_name: str
    profile_id: int
    profile_name: str
    formation_id: int
    distance: float
    list_measure: list[int]
    markup_id: int | None = None
    current_marker_id: int | None = None
    values: list[Resolution] = field(default_factory=list)
    object_name: str = ""


class ClassificationWellWizard(QtWidgets.QDialog):
    """Calculate several well attributes and make one exclusive class choice."""

    SETTINGS_GROUP = "classification_well_wizard"
    SETTINGS_WIDGETS = frozenset({
        "comboBox_classification_source",
        "comboBox_classification_canonical",
        "checkBox_classification_strict",
        "checkBox_classification_explicit_sum",
        "comboBox_classification_aggregation",
        "comboBox_classification_operation",
        "comboBox_classification_depth_mode",
        "comboBox_classification_boundary",
        "doubleSpinBox_classification_fixed_depth",
        "doubleSpinBox_classification_interval",
        "comboBox_classification_interval_position",
    })
    SOURCES = (("Глубина границы", "boundary"), ("Информация о скважине", "well_data"),
               ("Каротажная кривая", "well_log"))

    def __init__(self, session, candidates, markers, parent=None, mode="add",
                 open_well_log: Callable[[int, dict], None] | None = None):
        super().__init__(parent)
        self.session = session
        self.candidates = candidates
        self.markers = markers[:2]
        self.mode = mode
        self.open_well_log_callback = open_well_log
        self.parameters: list[tuple[str, TargetSettings]] = []
        # The table is rebuilt whenever a parameter is added, removed or an
        # ambiguous value is resolved.  Keep the user's row decisions outside
        # the disposable cell widgets so those rebuilds do not reset them.
        self._row_choices: dict[int, int] = {}
        self.setWindowTitle("Проверка классов скважин" if mode == "check" else
                            "Массовое добавление скважин в классификацию")
        self.resize(1250, 720)
        self._build_ui()
        self._restore_preferences()
        self.finished.connect(self._save_preferences)
        self._render()

    def _build_ui(self):
        root = QtWidgets.QVBoxLayout(self)
        selector = QtWidgets.QGridLayout()
        self.source = QtWidgets.QComboBox()
        self.source.setObjectName("comboBox_classification_source")
        for title, value in self.SOURCES:
            self.source.addItem(title, value)
        self.canonical = QtWidgets.QComboBox()
        self.canonical.setObjectName("comboBox_classification_canonical")
        self.strict = QtWidgets.QCheckBox("Строгий разбор")
        self.strict.setObjectName("checkBox_classification_strict")
        self.explicit_sum = QtWidgets.QCheckBox("Складывать выражения через +")
        self.explicit_sum.setObjectName("checkBox_classification_explicit_sum")
        self.explicit_sum.setChecked(True)
        self.aggregation = QtWidgets.QComboBox()
        for title, value in (("Медиана", "median"), ("Среднее", "mean"),
                             ("Максимум", "max"), ("Минимум", "min")):
            self.aggregation.addItem(title, value)
        self.aggregation.setObjectName("comboBox_classification_aggregation")
        self.operation = QtWidgets.QComboBox()
        self.operation.setObjectName("comboBox_classification_operation")
        for title, value in (("Значение интервала", "single"), ("Верх / низ", "upper_lower_ratio"),
                             ("Низ / верх", "lower_upper_ratio"), ("Низ − верх", "difference")):
            self.operation.addItem(title, value)
        self.depth_mode = QtWidgets.QComboBox(); self.depth_mode.addItem("От границы", "boundary"); self.depth_mode.addItem("Фиксированная", "fixed")
        self.depth_mode.setObjectName("comboBox_classification_depth_mode")
        self.boundary = QtWidgets.QComboBox()
        self.boundary.setObjectName("comboBox_classification_boundary")
        for row in self.session.query(CanonicalBoundary).order_by(CanonicalBoundary.canonical_name).all():
            self.boundary.addItem(row.canonical_name, row.id)
        self.fixed_depth = QtWidgets.QDoubleSpinBox(); self.fixed_depth.setRange(-100000, 100000); self.fixed_depth.setDecimals(3)
        self.fixed_depth.setObjectName("doubleSpinBox_classification_fixed_depth")
        self.interval = QtWidgets.QDoubleSpinBox(); self.interval.setRange(.001, 100000); self.interval.setValue(5); self.interval.setDecimals(3)
        self.interval.setObjectName("doubleSpinBox_classification_interval")
        self.position = QtWidgets.QComboBox(); self.position.addItem("Ниже", "below"); self.position.addItem("Выше", "above"); self.position.addItem("Симметрично", "centered")
        self.position.setObjectName("comboBox_classification_interval_position")
        controls = (("Источник", self.source), ("Параметр", self.canonical), ("", self.strict),
                    ("", self.explicit_sum), ("Агрегация", self.aggregation), ("Операция", self.operation),
                    ("Глубина", self.depth_mode), ("Граница", self.boundary), ("Фикс. глубина", self.fixed_depth),
                    ("Интервал", self.interval), ("Положение", self.position))
        self.log_controls = []
        for index, (label, widget) in enumerate(controls):
            row, column = divmod(index, 4)
            box = QtWidgets.QHBoxLayout()
            if label:
                label_widget = QtWidgets.QLabel(label + ":"); box.addWidget(label_widget)
            else:
                label_widget = None
            box.addWidget(widget)
            selector.addLayout(box, row, column)
            if index >= 4:
                self.log_controls.extend(item for item in (label_widget, widget) if item)
        self.add_parameter = QtWidgets.QPushButton("Добавить параметр")
        self.remove_parameter = QtWidgets.QPushButton("Удалить последний")
        selector.addWidget(self.add_parameter, 3, 0); selector.addWidget(self.remove_parameter, 3, 1)
        self.parameter_label = QtWidgets.QLabel("Параметры не выбраны")
        selector.addWidget(self.parameter_label, 3, 2, 1, 2)
        root.addLayout(selector)

        self.table = QtWidgets.QTableWidget()
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        root.addWidget(self.table, 1)
        self.summary = QtWidgets.QLabel(); root.addWidget(self.summary)
        buttons = QtWidgets.QDialogButtonBox()
        self.apply_button = buttons.addButton("Применить выбранные классы", QtWidgets.QDialogButtonBox.AcceptRole)
        self.export_button = buttons.addButton("Экспорт CSV", QtWidgets.QDialogButtonBox.ActionRole)
        self.open_log_button = buttons.addButton(
            "Открыть каротаж выбранной скважины", QtWidgets.QDialogButtonBox.ActionRole)
        buttons.addButton(QtWidgets.QDialogButtonBox.Cancel)
        root.addWidget(buttons)

        self.source.currentIndexChanged.connect(self._source_changed)
        self.depth_mode.currentIndexChanged.connect(self._depth_changed)
        self.add_parameter.clicked.connect(self._add_parameter)
        self.remove_parameter.clicked.connect(self._remove_parameter)
        self.apply_button.clicked.connect(self._accept)
        self.export_button.clicked.connect(self._export)
        self.open_log_button.clicked.connect(self._open_well_log)
        self.table.cellDoubleClicked.connect(self._resolve_ambiguity)
        buttons.rejected.connect(self.reject)

    @property
    def _settings_group(self):
        """Keep the add and review workflows from overwriting each other."""
        return f"{self.SETTINGS_GROUP}/{self.mode}"

    @classmethod
    def _is_preference_widget(cls, object_name):
        return object_name in cls.SETTINGS_WIDGETS

    def _restore_preferences(self):
        """Restore source-dependent controls after rebuilding their choices."""
        restore_form(self, self._settings_group, self._is_preference_widget)
        self._source_changed()
        restore_form(self, self._settings_group, self._is_preference_widget)
        self._depth_changed()

    def _save_preferences(self, _result=None):
        save_form(self, self._settings_group, self._is_preference_widget)

    def _source_changed(self):
        self.canonical.clear()
        for row in list_canonical_targets(self.session, self.source.currentData()):
            self.canonical.addItem(row.canonical_name, row.id)
        is_data = self.source.currentData() == "well_data"
        self.strict.setVisible(is_data); self.explicit_sum.setVisible(is_data)
        for widget in self.log_controls:
            widget.setVisible(self.source.currentData() == "well_log")
        self._depth_changed()

    def _depth_changed(self):
        boundary = self.depth_mode.currentData() == "boundary"
        self.boundary.setEnabled(boundary); self.fixed_depth.setEnabled(not boundary)

    def _settings(self):
        if self.canonical.currentData() is None:
            return None
        return TargetSettings(
            source=self.source.currentData(), canonical_id=int(self.canonical.currentData()),
            strict_numeric=self.strict.isChecked(), allow_explicit_sum=self.explicit_sum.isChecked(),
            aggregation=self.aggregation.currentData(), operation=self.operation.currentData(),
            depth_mode=self.depth_mode.currentData(), fixed_depth=self.fixed_depth.value(),
            boundary_canonical_id=self.boundary.currentData(), interval=self.interval.value(),
            interval_position=self.position.currentData())

    def _add_parameter(self):
        settings = self._settings()
        if settings is None:
            QtWidgets.QMessageBox.warning(self, "Нет параметра", "Выберите канонический параметр.")
            return
        title = f"{self.source.currentText()}: {self.canonical.currentText()}"
        if any(old == settings for _, old in self.parameters):
            QtWidgets.QMessageBox.information(self, "Параметр уже добавлен", title)
            return
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            for candidate in self.candidates:
                candidate.values.append(resolve_target(self.session, candidate.well_id, settings))
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        self.parameters.append((title, settings))
        self._render()

    def _remove_parameter(self):
        if not self.parameters:
            return
        self.parameters.pop()
        for candidate in self.candidates:
            if candidate.values:
                candidate.values.pop()
        self._render()

    def _render(self):
        self._remember_row_choices()
        marker_titles = [marker.title for marker in self.markers]
        headers = ["Скважина", "Профиль", "Объект", "Расстояние", "Пласт ID"] + [p[0] for p in self.parameters] + marker_titles + ["Не добавлять"]
        self.table.clear(); self.table.setColumnCount(len(headers)); self.table.setHorizontalHeaderLabels(headers)
        self.table.setRowCount(len(self.candidates))
        for row, candidate in enumerate(self.candidates):
            base = (candidate.well_name, candidate.profile_name, candidate.object_name,
                    f"{candidate.distance:.2f}", str(candidate.formation_id))
            for column, value in enumerate(base):
                self.table.setItem(row, column, QtWidgets.QTableWidgetItem(value))
            for offset, resolution in enumerate(candidate.values, 5):
                value = f"{resolution.value:g}" if resolution.status == "resolved" and resolution.value is not None else "—"
                item = QtWidgets.QTableWidgetItem(value)
                has_alternatives = len(resolution.candidates) > 1
                suffix = ("\nЕсть альтернативные варианты. Дважды щёлкните, чтобы перевыбрать."
                          if has_alternatives else "")
                item.setToolTip(resolution.message + suffix)
                item.setBackground(QtGui.QColor(
                    "#ffe4a3" if has_alternatives else
                    "#d9f2df" if resolution.status == "resolved" else "#ffd6d6"))
                if resolution.status == "ambiguous":
                    item.setText("выберите…")
                self.table.setItem(row, offset, item)
            group = QtWidgets.QButtonGroup(self.table); group.setExclusive(True)
            candidate._button_group = group
            start = 5 + len(self.parameters)
            for choice in range(3):
                radio = QtWidgets.QRadioButton(); group.addButton(radio, choice)
                self.table.setCellWidget(row, start + choice, radio)
                marker_id = self.markers[choice].id if choice < 2 else None
                saved_choice = self._row_choices.get(id(candidate))
                if saved_choice is None:
                    checked = (candidate.current_marker_id == marker_id if marker_id is not None else
                               candidate.current_marker_id is None and choice == 2)
                else:
                    checked = saved_choice == choice
                radio.setChecked(checked)
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        self.parameter_label.setText("Выбрано параметров: " + str(len(self.parameters)))
        self.summary.setText(
            f"Скважин: {len(self.candidates)}. Выбор класса в каждой строке взаимоисключающий. "
            "Выбор исходной записи применяется ко всем столбцам того же параметра.")

    def _remember_row_choices(self):
        """Snapshot radio choices before table cell widgets are destroyed."""
        for candidate in self.candidates:
            group = getattr(candidate, "_button_group", None)
            if group is not None and group.checkedId() >= 0:
                self._row_choices[id(candidate)] = group.checkedId()

    def _candidate_for_row(self, row):
        return self.candidates[row] if 0 <= row < len(self.candidates) else None

    def _resolve_ambiguity(self, row, column):
        """Ask once for a source row and reuse it in matching cells of the well."""
        parameter_index = column - 5
        candidate = self._candidate_for_row(row)
        if (candidate is None or parameter_index < 0 or
                parameter_index >= len(candidate.values)):
            return
        resolution = candidate.values[parameter_index]
        if len(resolution.candidates) < 2:
            return
        labels = [f"{item.source_name}: {item.raw_value} → {item.value:g}"
                  for item in resolution.candidates]
        selected, accepted = QtWidgets.QInputDialog.getItem(
            self, "Выбор значения", "Исходная запись:", labels, 0, False)
        if not accepted:
            return
        selected_index = labels.index(selected)
        selected_candidate = resolution.candidates[selected_index]
        selection_key = self._ambiguity_key(parameter_index, resolution)
        for peer_index, peer_resolution in enumerate(candidate.values):
            if (peer_resolution.status != "ambiguous" or
                    self._ambiguity_key(peer_index, peer_resolution) != selection_key):
                continue
            peer_choice = next((item for item in peer_resolution.candidates
                                if item.source_id == selected_candidate.source_id), None)
            if peer_choice is None:
                continue
            self._select_resolution_candidate(candidate, peer_index, peer_resolution, peer_choice)
        self._render()

    def _ambiguity_key(self, parameter_index, resolution):
        """Identify cells whose ambiguity refers to the same source parameter."""
        if not 0 <= parameter_index < len(self.parameters):
            return None
        settings = self.parameters[parameter_index][1]
        if (settings.source == "well_log" and
                resolution.details.get("pending_selection") == "boundary_depth"):
            return "boundary", settings.boundary_canonical_id
        return settings.source, settings.canonical_id

    def _select_resolution_candidate(self, candidate, parameter_index, resolution, choice):
        """Apply a source choice, recalculating log values that depend on a boundary."""
        settings = self.parameters[parameter_index][1]
        if (settings.source == "well_log" and
                resolution.details.get("pending_selection") == "boundary_depth"):
            candidate.values[parameter_index] = resolve_target(
                self.session, candidate.well_id, settings, boundary_candidate=choice)
            save_parameter_choice(
                self.session, candidate.well_id, "boundary",
                settings.boundary_canonical_id, choice.source_id)
            return
        resolution.select(resolution.candidates.index(choice))
        save_parameter_choice(
            self.session, candidate.well_id, settings.source,
            settings.canonical_id, choice.source_id)

    def _open_well_log(self):
        candidate = self._candidate_for_row(self.table.currentRow())
        if candidate is None or self.open_well_log_callback is None:
            QtWidgets.QMessageBox.information(
                self, "Скважина не выбрана", "Выберите строку скважины в таблице.")
            return
        details = {}
        column = self.table.currentColumn() - 5
        if 0 <= column < len(candidate.values):
            details = candidate.values[column].details
        self.open_well_log_callback(candidate.well_id, details)

    def assignments(self):
        result = []
        for candidate in self.candidates:
            choice = candidate._button_group.checkedId()
            result.append((candidate, self.markers[choice].id if choice in (0, 1) else None))
        return result

    def _accept(self):
        if not self.parameters:
            answer = QtWidgets.QMessageBox.question(self, "Нет параметров", "Продолжить без рассчитанных параметров?",
                                                    QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No)
            if answer != QtWidgets.QMessageBox.Yes:
                return
        if not any(marker_id is not None for _, marker_id in self.assignments()):
            QtWidgets.QMessageBox.warning(self, "Нет выбранных скважин", "Назначьте хотя бы одной скважине класс.")
            return
        self.accept()

    def _export(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Экспорт", "classification_wells.csv", "CSV (*.csv)")
        if not path:
            return
        with open(path, "w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.writer(stream, delimiter=";")
            writer.writerow(["well_id", "well", "profile_id", "profile", "object", "distance", "formation_id"] +
                            [title for title, _ in self.parameters] + ["class"])
            for candidate, marker_id in self.assignments():
                marker = next((m.title for m in self.markers if m.id == marker_id), "не добавлять")
                writer.writerow([candidate.well_id, candidate.well_name, candidate.profile_id, candidate.profile_name,
                                 candidate.object_name, candidate.distance, candidate.formation_id] +
                                [r.value if r.status == "resolved" else "" for r in candidate.values] + [marker])
