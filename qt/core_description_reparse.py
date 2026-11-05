"""Review semantic differences before changing saved descriptions."""

import json

from PyQt5 import QtCore, QtWidgets
from sqlalchemy.exc import SQLAlchemyError

from models_db.model import CoreDescriptionInterval, Well
from core_description.database import load_well_areas

from core_description.reparse import ReparseError, apply_reparse
from core_description.terminology import ROCK_LABELS, RELATION_LABELS, SATURATION_LABELS, display_term
from qt.core_description_import import DescriptionDelegate


class ReparseWellScopeDialog(QtWidgets.QDialog):
    """Explicit well selection, including the entire database; never default to all."""

    def __init__(self, session, current_well_id, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Выбрать скважины для переразбора")
        self.resize(550, 600)
        layout = QtWidgets.QVBoxLayout(self)
        note = QtWidgets.QLabel("Отметьте скважины. Для обработки всей базы используйте «Отметить все».")
        note.setWordWrap(True)
        layout.addWidget(note)
        self.search = QtWidgets.QLineEdit()
        self.search.setPlaceholderText("Поиск скважины")
        self.search.setClearButtonEnabled(True)
        layout.addWidget(self.search)
        self.wells = QtWidgets.QListWidget()
        with session.no_autoflush:
            rows = session.query(Well.id, Well.name).join(
                CoreDescriptionInterval, CoreDescriptionInterval.well_id == Well.id).distinct().order_by(Well.name, Well.id).all()
            areas = load_well_areas(session, [well_id for well_id, _name in rows])
        for well_id, name in rows:
            area = ", ".join(areas.get(well_id, [])) or "площадь не указана"
            item = QtWidgets.QListWidgetItem(f"{name} — {area} (#{well_id})")
            item.setData(QtCore.Qt.UserRole, well_id)
            item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
            item.setCheckState(QtCore.Qt.Checked if well_id == current_well_id else QtCore.Qt.Unchecked)
            self.wells.addItem(item)
        layout.addWidget(self.wells, 1)
        self.count_label = QtWidgets.QLabel()
        layout.addWidget(self.count_label)
        actions = QtWidgets.QHBoxLayout()
        self.select_all = QtWidgets.QPushButton("Отметить все")
        self.clear_all = QtWidgets.QPushButton("Снять все отметки")
        self.select_all.clicked.connect(lambda: self._select(QtCore.Qt.Checked))
        self.clear_all.clicked.connect(lambda: self._select(QtCore.Qt.Unchecked))
        actions.addWidget(self.select_all)
        actions.addWidget(self.clear_all)
        layout.addLayout(actions)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        self.ok_button = buttons.button(QtWidgets.QDialogButtonBox.Ok)
        self.ok_button.setText("Рассчитать и сравнить")
        buttons.button(QtWidgets.QDialogButtonBox.Cancel).setText("Отмена")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.search.textChanged.connect(self._filter)
        self.wells.itemChanged.connect(self._update_count)
        self._update_count()

    def selected_well_ids(self):
        return {self.wells.item(i).data(QtCore.Qt.UserRole) for i in range(self.wells.count())
                if self.wells.item(i).checkState() == QtCore.Qt.Checked}

    def _filter(self, text):
        for index in range(self.wells.count()):
            item = self.wells.item(index)
            item.setHidden(text.casefold() not in item.text().casefold())

    def _select(self, state):
        self.wells.blockSignals(True)
        for i in range(self.wells.count()):
            self.wells.item(i).setCheckState(state)
        self.wells.blockSignals(False)
        self._update_count()

    def _update_count(self, *_args):
        count = len(self.selected_well_ids())
        self.count_label.setText(f"Выбрано скважин: {count} из {self.wells.count()} (включая скрытые поиском).")
        self.ok_button.setEnabled(count > 0)

    def accept(self):
        if self.selected_well_ids():
            super().accept()


def state_text(encoded):
    state = json.loads(encoded)
    rocks = ", ".join(dict.fromkeys(
        f'{display_term(rock["rock_name"], ROCK_LABELS)} ({display_term(rock["role"], RELATION_LABELS)})'
        for rock in state["rocks"])) or "не определены"
    warnings = state["recognition_details"].get("warnings", [])
    result = (f'Породы: {rocks}\nНефтенасыщенность: {display_term(state["oil_saturation"], SATURATION_LABELS)}'
              f'\nУверенность: {state["confidence"]:g}; требует проверки: {"да" if state["needs_review"] else "нет"}')
    if warnings:
        labels = {"locality_marker": "локальный признак", "uncertain_wording": "неопределённая формулировка"}
        result += "\nПредупреждения: " + ", ".join(labels.get(w, w) for w in warnings)
    return result


class ReparsePreviewDialog(QtWidgets.QDialog):
    applied = QtCore.pyqtSignal(object)

    def __init__(self, session, preview, parent=None, *, changed_only=False):
        super().__init__(parent)
        self.session = session
        self.preview = preview
        self.setWindowTitle("Сравнение результатов разбора керна")
        self.resize(1200, 780)
        layout = QtWidgets.QVBoxLayout(self)
        self.summary = QtWidgets.QLabel(
            f"Словарь: {json.loads(preview.dictionary_json)['version']}. "
            f"Интервалов: {len(preview.proposals)}; с изменениями: {sum(p.changed for p in preview.proposals)}; "
            f"с ручными правками: {sum(p.protected for p in preview.proposals)}. "
            "В базу будут записаны только отмеченные видимые строки.")
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)
        options = QtWidgets.QHBoxLayout()
        self.changed_only = QtWidgets.QCheckBox("Только изменившиеся")
        self.changed_only.setChecked(changed_only)
        self.allow_manual = QtWidgets.QCheckBox("Разрешить замену распознавания в строках с ручными правками")
        options.addWidget(self.changed_only)
        options.addWidget(self.allow_manual)
        layout.addLayout(options)
        self.table = QtWidgets.QTableWidget(len(preview.proposals), 6)
        self.table.setHorizontalHeaderLabels(("Применить", "Скважина", "Интервал, м", "Сейчас", "После переразбора", "Статус"))
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        for column, width in ((0, 95), (1, 185), (2, 105), (5, 185)):
            self.table.horizontalHeader().setSectionResizeMode(column, QtWidgets.QHeaderView.Interactive)
            self.table.setColumnWidth(column, width)
        for column in (3, 4):
            self.table.horizontalHeader().setSectionResizeMode(column, QtWidgets.QHeaderView.Stretch)
        with session.no_autoflush:
            areas = load_well_areas(session, {p.well_id for p in preview.proposals})
        for row, proposal in enumerate(preview.proposals):
            check = QtWidgets.QTableWidgetItem()
            check.setFlags(QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsUserCheckable)
            check.setCheckState(QtCore.Qt.Unchecked if proposal.protected else QtCore.Qt.Checked)
            self.table.setItem(row, 0, check)
            area = ", ".join(areas.get(proposal.well_id, [])) or "площадь не указана"
            values = (f"{proposal.well_name} — {area} (#{proposal.well_id})", f"{proposal.top_depth:g}–{proposal.bottom_depth:g}",
                      state_text(proposal.before), state_text(proposal.after),
                      "Ручные правки — защищено" if proposal.protected else
                      "Изменён" if proposal.changed else "Результат прежний; обновится версия")
            for col, value in enumerate(values, 1):
                item = QtWidgets.QTableWidgetItem(value)
                item.setTextAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
                item.setToolTip(value)
                self.table.setItem(row, col, item)
        layout.addWidget(self.table, 3)
        actions = QtWidgets.QHBoxLayout()
        select = QtWidgets.QPushButton("Отметить видимые")
        clear = QtWidgets.QPushButton("Снять отметки")
        select.clicked.connect(lambda: self._select(QtCore.Qt.Checked))
        clear.clicked.connect(lambda: self._select(QtCore.Qt.Unchecked))
        actions.addWidget(select)
        actions.addWidget(clear)
        actions.addStretch()
        layout.addLayout(actions)
        self.text_preview = QtWidgets.QTextBrowser()
        self.text_preview.setOpenExternalLinks(False)
        layout.addWidget(self.text_preview, 1)
        self.error_label = QtWidgets.QLabel()
        self.error_label.setTextFormat(QtCore.Qt.PlainText)
        self.error_label.setWordWrap(True)
        layout.addWidget(self.error_label)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Cancel)
        buttons.button(QtWidgets.QDialogButtonBox.Cancel).setText("Отмена")
        self.apply_button = buttons.addButton("Применить отмеченные", QtWidgets.QDialogButtonBox.AcceptRole)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.changed_only.toggled.connect(self._update_rows)
        self.allow_manual.toggled.connect(self._update_rows)
        self.table.itemChanged.connect(self._update_count)
        self.table.currentCellChanged.connect(self._show_text)
        self._resize_timer = QtCore.QTimer(self)
        self._resize_timer.setSingleShot(True)
        self._resize_timer.timeout.connect(self.table.resizeRowsToContents)
        self.table.horizontalHeader().sectionResized.connect(lambda *_: self._resize_timer.start(50))
        self._update_rows()
        if preview.proposals:
            self.table.setCurrentCell(0, 1)

    def showEvent(self, event):
        super().showEvent(event)
        self.table.resizeRowsToContents()

    def _update_rows(self, *_args):
        self.table.blockSignals(True)
        for row, proposal in enumerate(self.preview.proposals):
            self.table.setRowHidden(row, self.changed_only.isChecked() and not proposal.changed)
            item = self.table.item(row, 0)
            enabled = not proposal.protected or self.allow_manual.isChecked()
            item.setFlags(QtCore.Qt.ItemIsUserCheckable | (QtCore.Qt.ItemIsEnabled if enabled else QtCore.Qt.NoItemFlags))
            if not enabled:
                item.setCheckState(QtCore.Qt.Unchecked)
        self.table.blockSignals(False)
        self.table.resizeRowsToContents()
        self._update_count()

    def selected_ids(self):
        return {p.interval_id for row, p in enumerate(self.preview.proposals)
                if not self.table.isRowHidden(row) and self.table.item(row, 0).checkState() == QtCore.Qt.Checked
                and (not p.protected or self.allow_manual.isChecked())}

    def _update_count(self, *_args):
        count = len(self.selected_ids())
        self.apply_button.setText(f"Применить отмеченные ({count})")
        self.apply_button.setEnabled(count > 0)

    def _select(self, state):
        self.table.blockSignals(True)
        for row in range(self.table.rowCount()):
            item = self.table.item(row, 0)
            if not self.table.isRowHidden(row) and item.flags() & QtCore.Qt.ItemIsEnabled:
                item.setCheckState(state)
        self.table.blockSignals(False)
        self._update_count()

    def _show_text(self, row, *_args):
        if row < 0:
            return
        proposal = self.preview.proposals[row]
        matches = json.loads(proposal.after)["recognition_details"].get("matches", [])
        self.text_preview.setHtml(DescriptionDelegate.highlighted_matches(
            proposal.text, ((m["category"], m["start"], m["end"]) for m in matches)))

    def accept(self):
        selected = self.selected_ids()
        try:
            apply_reparse(self.session, self.preview, selected, allow_manual=self.allow_manual.isChecked())
        except (ReparseError, SQLAlchemyError, ValueError) as error:
            self.error_label.setText(f"Не удалось применить переразбор: {error}")
            return
        self.applied.emit({p.well_id for p in self.preview.proposals if p.interval_id in selected})
        super().accept()
