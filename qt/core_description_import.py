"""Editable preview and fault-isolated batch import of core descriptions."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from PyQt5 import QtCore, QtWidgets

from core_description.database import match_well, save_core_import
from core_description.batch import BatchParseItem, BatchParseResult, parse_batch
from core_description.models import ParsedCoreDocument, ParsedCoreInterval
from core_description.parser import CoreDescriptionParseError, parse_core_document
from models_db.model import Well


SATURATIONS = ("unknown", "none", "weak", "medium", "intense", "present", "uncertain")


class BatchParseWorker(QtCore.QObject):
    """Parse a batch outside the GUI thread and cooperatively support cancel."""

    progress = QtCore.pyqtSignal(int, int, object)
    finished = QtCore.pyqtSignal(object)

    def __init__(self, paths: list[str], recursive: bool = False) -> None:
        super().__init__()
        self.paths = paths
        self.recursive = recursive
        self._cancelled = False

    @QtCore.pyqtSlot()
    def run(self) -> None:
        result = parse_batch(self.paths, recursive=self.recursive,
                             cancelled=lambda: self._cancelled,
                             progress=lambda done, total, item: self.progress.emit(done, total, item))
        self.finished.emit(result)

    @QtCore.pyqtSlot()
    def cancel(self) -> None:
        self._cancelled = True


class CoreDescriptionImportDialog(QtWidgets.QDialog):
    """Preview, correct and selectively persist one parsed document."""

    COLUMNS = (
        "Сохранить", "Верх", "Низ", "Основная порода", "Дополнительные породы",
        "Нефтенасыщенность", "Уверенность", "Исходное описание", "Состояние",
    )

    def __init__(self, session: Any, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.session = session
        self.document: ParsedCoreDocument | None = None
        self.match_method = "manual"
        self.match_confidence = 1.0
        self._automatic: dict[str, dict[str, Any]] = {}
        self.batch_items: list[BatchParseItem] = []
        self._thread: QtCore.QThread | None = None
        self._worker: BatchParseWorker | None = None
        self.setWindowTitle("Импорт описания керна")
        self.resize(1250, 720)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        source = QtWidgets.QHBoxLayout()
        self.open_button = QtWidgets.QPushButton("Добавить файлы…")
        self.open_button.setObjectName("open_core_description")
        self.directory_button = QtWidgets.QPushButton("Добавить каталог…")
        self.directory_button.setObjectName("open_core_description_directory")
        self.recursive_check = QtWidgets.QCheckBox("включая подкаталоги")
        self.path_label = QtWidgets.QLabel("Файл не выбран")
        self.path_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        source.addWidget(self.open_button)
        source.addWidget(self.directory_button)
        source.addWidget(self.recursive_check)
        source.addWidget(self.path_label, 1)
        layout.addLayout(source)

        self.documents_table = QtWidgets.QTableWidget(0, 8)
        self.documents_table.setObjectName("core_description_documents")
        self.documents_table.setHorizontalHeaderLabels((
            "Импорт", "Файл", "Скважина", "Площадь", "Автор", "Скважина БД", "Интервалы", "Статус"))
        self.documents_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.documents_table.setMaximumHeight(180)
        self.documents_table.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)
        layout.addWidget(self.documents_table)

        progress_row = QtWidgets.QHBoxLayout()
        self.progress_bar = QtWidgets.QProgressBar(); self.progress_bar.setVisible(False)
        self.cancel_button = QtWidgets.QPushButton("Отменить обработку"); self.cancel_button.setEnabled(False)
        self.export_button = QtWidgets.QPushButton("Экспорт отчёта…"); self.export_button.setEnabled(False)
        progress_row.addWidget(self.progress_bar, 1); progress_row.addWidget(self.cancel_button); progress_row.addWidget(self.export_button)
        layout.addLayout(progress_row)

        self.metadata = QtWidgets.QLabel()
        self.metadata.setWordWrap(True)
        layout.addWidget(self.metadata)
        well_row = QtWidgets.QHBoxLayout()
        well_row.addWidget(QtWidgets.QLabel("Скважина БД:"))
        self.well_combo = QtWidgets.QComboBox()
        self.well_combo.setMinimumWidth(260)
        well_row.addWidget(self.well_combo)
        self.match_label = QtWidgets.QLabel("Не сопоставлена")
        well_row.addWidget(self.match_label, 1)
        layout.addLayout(well_row)

        filters = QtWidgets.QHBoxLayout()
        self.depth_from = QtWidgets.QDoubleSpinBox(); self.depth_from.setRange(-1e6, 1e6)
        self.depth_to = QtWidgets.QDoubleSpinBox(); self.depth_to.setRange(-1e6, 1e6); self.depth_to.setValue(1e6)
        self.rock_filter = QtWidgets.QLineEdit(); self.rock_filter.setPlaceholderText("любая")
        self.saturation_filter = QtWidgets.QComboBox(); self.saturation_filter.addItems(("любая",) + SATURATIONS)
        self.state_filter = QtWidgets.QComboBox(); self.state_filter.addItems(("любое", "готово", "требует проверки", "ошибка"))
        for label, widget in (("Глубина от", self.depth_from), ("до", self.depth_to),
                              ("Порода", self.rock_filter), ("Насыщенность", self.saturation_filter),
                              ("Состояние", self.state_filter)):
            filters.addWidget(QtWidgets.QLabel(label)); filters.addWidget(widget)
        layout.addLayout(filters)

        actions = QtWidgets.QHBoxLayout()
        for text, slot in (("Выбрать все", lambda: self._select_visible(True)),
                           ("Снять все", lambda: self._select_visible(False)),
                           ("Выбрать уверенные", self._select_confident),
                           ("Выбрать нефтенасыщенные", self._select_oil),
                           ("Вернуть автоматические значения", self._restore_selected),
                           ("Следующая проблемная", self._next_problem)):
            button = QtWidgets.QPushButton(text); button.clicked.connect(slot); actions.addWidget(button)
        layout.addLayout(actions)

        self.table = QtWidgets.QTableWidget(0, len(self.COLUMNS))
        self.table.setHorizontalHeaderLabels(self.COLUMNS)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setAlternatingRowColors(True)
        self.table.horizontalHeader().setSectionResizeMode(7, QtWidgets.QHeaderView.Stretch)
        layout.addWidget(self.table, 1)
        self.summary = QtWidgets.QLabel("Выбрано интервалов: 0")
        layout.addWidget(self.summary)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Save | QtWidgets.QDialogButtonBox.Close)
        self.save_button = buttons.button(QtWidgets.QDialogButtonBox.Save)
        self.save_button.setText("Сохранить выбранные")
        self.save_button.setEnabled(False)
        buttons.rejected.connect(self.reject); buttons.accepted.connect(self.save)
        layout.addWidget(buttons)

        self.open_button.clicked.connect(self.choose_file)
        self.directory_button.clicked.connect(self.choose_directory)
        self.cancel_button.clicked.connect(self.cancel_batch)
        self.export_button.clicked.connect(self.export_report)
        self.documents_table.currentCellChanged.connect(self._document_row_changed)
        self.well_combo.currentIndexChanged.connect(self._well_changed)
        self.table.itemChanged.connect(self._table_changed)
        for widget in (self.depth_from, self.depth_to, self.saturation_filter, self.state_filter):
            widget.valueChanged.connect(self.apply_filters) if hasattr(widget, "valueChanged") else widget.currentIndexChanged.connect(self.apply_filters)
        self.rock_filter.textChanged.connect(self.apply_filters)

    def choose_file(self) -> None:
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(self, "Описания керна", "", "Word (*.doc *.docx)")
        if paths:
            self.start_batch(paths)

    def choose_directory(self) -> None:
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Каталог описаний керна")
        if path:
            self.start_batch([path], recursive=self.recursive_check.isChecked())

    def load_file(self, path: str | Path) -> None:
        try:
            document = parse_core_document(path)
        except CoreDescriptionParseError as error:
            QtWidgets.QMessageBox.critical(self, "Ошибка разбора", str(error)); return
        self.set_document(document)

    def start_batch(self, paths: list[str], recursive: bool = False) -> None:
        """Start parsing without blocking Qt's event loop."""
        if self._thread is not None:
            return
        self.batch_items = []
        self.documents_table.setRowCount(0)
        self.progress_bar.setRange(0, 0); self.progress_bar.setValue(0); self.progress_bar.setVisible(True)
        self.cancel_button.setEnabled(True); self.open_button.setEnabled(False); self.directory_button.setEnabled(False)
        thread = QtCore.QThread(self)
        worker = BatchParseWorker(paths, recursive)
        worker.moveToThread(thread); thread.started.connect(worker.run)
        worker.progress.connect(self._batch_progress); worker.finished.connect(self._batch_finished)
        worker.finished.connect(thread.quit); thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        self._thread, self._worker = thread, worker
        thread.start()

    def cancel_batch(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            self.cancel_button.setEnabled(False)

    def _batch_progress(self, done: int, total: int, _item: BatchParseItem) -> None:
        self.progress_bar.setRange(0, total); self.progress_bar.setValue(done)

    def _batch_finished(self, result: BatchParseResult) -> None:
        self.batch_items = result.items
        self.documents_table.blockSignals(True); self.documents_table.setRowCount(0)
        for item in result.items:
            self._append_document(item)
        self.documents_table.blockSignals(False)
        self.progress_bar.setRange(0, max(1, result.processed)); self.progress_bar.setValue(result.processed)
        self.cancel_button.setEnabled(False); self.open_button.setEnabled(True); self.directory_button.setEnabled(True)
        self.export_button.setEnabled(bool(result.items)); self._worker = None; self._thread = None
        if result.items:
            self.documents_table.setCurrentCell(0, 1)
        suffix = "; отменено" if result.cancelled else ""
        self.summary.setText(f"Обработано: {result.processed}; успешно: {result.succeeded}; ошибок: {result.failed}{suffix}")

    def _append_document(self, item: BatchParseItem) -> None:
        row = self.documents_table.rowCount(); self.documents_table.insertRow(row)
        document = item.document
        values = ("", Path(item.source_path).name, document.well_name_raw if document else "—",
                  document.area_name_raw if document else "—", document.described_by if document else "—",
                  "ожидает выбора" if document else "—", str(len(document.intervals)) if document else "0",
                  "готов" if document else f"ошибка: {item.error}")
        check = QtWidgets.QTableWidgetItem(); check.setFlags(check.flags() | QtCore.Qt.ItemIsUserCheckable)
        check.setCheckState(QtCore.Qt.Checked if document else QtCore.Qt.Unchecked)
        check.setData(QtCore.Qt.UserRole, row); self.documents_table.setItem(row, 0, check)
        for column, value in enumerate(values[1:], 1):
            cell = QtWidgets.QTableWidgetItem(str(value)); cell.setFlags(cell.flags() & ~QtCore.Qt.ItemIsEditable)
            cell.setToolTip(item.error or item.source_path); self.documents_table.setItem(row, column, cell)

    def _document_row_changed(self, row: int, _column: int, *_args: Any) -> None:
        if 0 <= row < len(self.batch_items) and self.batch_items[row].document is not None:
            self.set_document(self.batch_items[row].document)  # type: ignore[arg-type]

    def export_report(self) -> None:
        path, selected = QtWidgets.QFileDialog.getSaveFileName(self, "Сохранить отчёт", "core-import-report.json",
                                                               "JSON (*.json);;CSV (*.csv)")
        if not path:
            return
        rows = [{"source_path": item.source_path, "status": item.status, "error": item.error,
                 "duration_seconds": round(item.duration_seconds, 6),
                 "intervals": len(item.document.intervals) if item.document else 0}
                for item in self.batch_items]
        if path.casefold().endswith(".csv") or selected.startswith("CSV"):
            with open(path, "w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
        else:
            Path(path).write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    def set_document(self, document: ParsedCoreDocument) -> None:
        """Populate the preview; public to allow deterministic UI tests."""
        self.document = document
        self.path_label.setText(document.source_path)
        self.metadata.setText(
            f"Скважина: {document.well_name_raw or '—'}; площадь: {document.area_name_raw or '—'}; "
            f"описал: {document.described_by or '—'}" +
            (f"; предупреждения: {', '.join(document.warnings)}" if document.warnings else ""))
        self._load_wells()
        self.table.blockSignals(True); self.table.setRowCount(0); self._automatic.clear()
        for interval in document.intervals:
            self._append_interval(interval)
        self.table.blockSignals(False)
        self.apply_filters(); self._update_summary()

    def _load_wells(self) -> None:
        self.well_combo.blockSignals(True)
        self.well_combo.clear(); self.well_combo.addItem("— выберите скважину —", None)
        wells = self.session.query(Well).order_by(Well.name, Well.id).all()
        for well in wells:
            self.well_combo.addItem(f"{well.name} (id {well.id})", well.id)
        assert self.document is not None
        result = match_well(self.session, self.document.well_name, self.document.area_name)
        if result.selected:
            index = self.well_combo.findData(result.selected.well_id); self.well_combo.setCurrentIndex(index)
            self.match_method = result.selected.method
            self.match_confidence = result.selected.confidence
            self.match_label.setText(f"{result.selected.method}, {result.selected.confidence:.0%}")
        else:
            self.match_method = "manual"
            self.match_confidence = 1.0
            self.match_label.setText("Требуется ручной выбор" + (f": {', '.join(result.warnings)}" if result.warnings else ""))
        self.well_combo.blockSignals(False)

    def _well_changed(self, *_args: Any) -> None:
        self.match_method = "manual"
        self.match_confidence = 1.0
        self.match_label.setText("Выбрана вручную" if self.well_combo.currentData() is not None
                                 else "Требуется ручной выбор")
        self._update_summary()

    @staticmethod
    def _key(interval: ParsedCoreInterval) -> str:
        return f"{interval.source_table}:{interval.source_row}"

    def _append_interval(self, interval: ParsedCoreInterval) -> None:
        row = self.table.rowCount(); self.table.insertRow(row)
        key = self._key(interval)
        primary = next((r.canonical_value for r in interval.rocks if r.relation == "primary"), "")
        secondary = ", ".join(f"{r.canonical_value} ({r.relation})" for r in interval.rocks if r.relation != "primary")
        state = "ошибка" if interval.errors else "требует проверки" if interval.warnings or interval.confidence < .8 else "готово"
        values = ("", interval.top_depth, interval.bottom_depth, primary, secondary,
                  interval.oil_saturation, interval.confidence, interval.raw_description, state)
        check = QtWidgets.QTableWidgetItem(); check.setData(QtCore.Qt.UserRole, key)
        check.setFlags(check.flags() | QtCore.Qt.ItemIsUserCheckable)
        check.setCheckState(QtCore.Qt.Checked if interval.selected_by_default and not interval.errors and interval.confidence >= .8 else QtCore.Qt.Unchecked)
        self.table.setItem(row, 0, check)
        for column, value in enumerate(values[1:], 1):
            item = QtWidgets.QTableWidgetItem("" if value is None else str(value)); self.table.setItem(row, column, item)
            if column in (7, 8): item.setFlags(item.flags() & ~QtCore.Qt.ItemIsEditable)
        details = "\n".join(f"{m.rule_id}: «{m.matched_text}» [{m.start}:{m.end}]" for m in interval.semantic_matches)
        for column in range(len(self.COLUMNS)): self.table.item(row, column).setToolTip(details)
        self._automatic[key] = {"top_depth": interval.top_depth, "bottom_depth": interval.bottom_depth,
                                "primary": primary, "secondary": secondary,
                                "oil_saturation": interval.oil_saturation, "confidence": interval.confidence}

    def _table_changed(self, _item: QtWidgets.QTableWidgetItem) -> None:
        self._update_summary(); self.apply_filters()

    def _state(self, row: int) -> str:
        return self.table.item(row, 8).text()

    def apply_filters(self) -> None:
        rock = self.rock_filter.text().casefold().strip(); saturation = self.saturation_filter.currentText()
        state = self.state_filter.currentText()
        for row in range(self.table.rowCount()):
            try: top = float(self.table.item(row, 1).text().replace(",", "."))
            except ValueError: top = float("-inf")
            row_rocks = (self.table.item(row, 3).text() + " " + self.table.item(row, 4).text()).casefold()
            visible = self.depth_from.value() <= top <= self.depth_to.value() and (not rock or rock in row_rocks)
            visible &= saturation == "любая" or self.table.item(row, 5).text() == saturation
            visible &= state == "любое" or self._state(row) == state
            self.table.setRowHidden(row, not visible)

    def _select_visible(self, checked: bool) -> None:
        for row in range(self.table.rowCount()):
            if not self.table.isRowHidden(row): self.table.item(row, 0).setCheckState(QtCore.Qt.Checked if checked else QtCore.Qt.Unchecked)

    def _select_confident(self) -> None:
        for row in range(self.table.rowCount()):
            if not self.table.isRowHidden(row):
                ok = self._state(row) == "готово" and float(self.table.item(row, 6).text()) >= .8
                self.table.item(row, 0).setCheckState(QtCore.Qt.Checked if ok else QtCore.Qt.Unchecked)

    def _select_oil(self) -> None:
        oil = {"weak", "medium", "intense", "present", "uncertain"}
        for row in range(self.table.rowCount()):
            if not self.table.isRowHidden(row):
                self.table.item(row, 0).setCheckState(QtCore.Qt.Checked if self.table.item(row, 5).text() in oil else QtCore.Qt.Unchecked)

    def _restore_selected(self) -> None:
        for row in {index.row() for index in self.table.selectedIndexes()}:
            auto = self._automatic[self.table.item(row, 0).data(QtCore.Qt.UserRole)]
            for column, name in ((1, "top_depth"), (2, "bottom_depth"), (3, "primary"), (4, "secondary"), (5, "oil_saturation"), (6, "confidence")):
                self.table.item(row, column).setText(str(auto[name] if auto[name] is not None else ""))

    def _next_problem(self) -> None:
        start = self.table.currentRow() + 1
        for offset in range(self.table.rowCount()):
            row = (start + offset) % self.table.rowCount()
            if not self.table.isRowHidden(row) and self._state(row) != "готово":
                self.table.selectRow(row); self.table.scrollToItem(self.table.item(row, 0)); return

    def _update_summary(self, *_args: Any) -> None:
        count = sum(self.table.item(row, 0).checkState() == QtCore.Qt.Checked for row in range(self.table.rowCount()))
        self.summary.setText(f"Выбрано документов: {1 if count else 0}; интервалов: {count}")
        self.save_button.setEnabled(bool(count and self.well_combo.currentData() is not None))

    def _edits(self, row: int) -> dict[str, Any]:
        key = self.table.item(row, 0).data(QtCore.Qt.UserRole); auto = self._automatic[key]
        values: dict[str, Any] = {
            "top_depth": float(self.table.item(row, 1).text().replace(",", ".")),
            "bottom_depth": float(self.table.item(row, 2).text().replace(",", ".")),
            "oil_saturation": self.table.item(row, 5).text().strip(),
            "confidence": float(self.table.item(row, 6).text().replace(",", ".")),
        }
        primary = self.table.item(row, 3).text().strip()
        secondary = self.table.item(row, 4).text().strip()
        interval = next(i for i in self.document.intervals if self._key(i) == key)  # type: ignore[union-attr]
        if primary != auto["primary"] or secondary != auto["secondary"]:
            rocks = []
            if primary: rocks.insert(0, {"canonical_value": primary, "matched_text": primary, "relation": "primary", "confidence": 1.0, "rule_ids": []})
            for value in filter(None, (part.strip() for part in secondary.split(","))):
                name, separator, relation = value.rpartition(" (")
                if separator and relation.endswith(")"):
                    value, relation = name, relation[:-1]
                else:
                    relation = "secondary"
                rocks.append({"canonical_value": value, "matched_text": value, "relation": relation,
                              "confidence": 1.0, "rule_ids": []})
            values["rocks"] = rocks
        return values

    def save(self) -> None:
        if self.document is None: return
        if self.batch_items:
            self._save_batch()
            return
        selected = {self.table.item(row, 0).data(QtCore.Qt.UserRole) for row in range(self.table.rowCount())
                    if self.table.item(row, 0).checkState() == QtCore.Qt.Checked}
        try:
            edits = {self.table.item(row, 0).data(QtCore.Qt.UserRole): self._edits(row)
                     for row in range(self.table.rowCount()) if self.table.item(row, 0).data(QtCore.Qt.UserRole) in selected}
            result = save_core_import(self.session, self.document, well_id=self.well_combo.currentData(),
                                      selected_interval_ids=selected, edits=edits,
                                      match_method=self.match_method,
                                      match_confidence=self.match_confidence)
        except Exception as error:
            QtWidgets.QMessageBox.critical(self, "Ошибка сохранения", str(error)); return
        QtWidgets.QMessageBox.information(self, "Импорт завершён", f"Сохранено интервалов: {result.intervals_saved}")
        self.accept()

    def _save_batch(self) -> None:
        """Save each checked document in its own service transaction."""
        saved_documents = saved_intervals = skipped = failed = 0
        current_path = self.document.source_path if self.document else None
        for row, item in enumerate(self.batch_items):
            if self.documents_table.item(row, 0).checkState() != QtCore.Qt.Checked:
                skipped += 1
                continue
            document = item.document
            if document is None:
                failed += 1
                continue
            match = match_well(self.session, document.well_name, document.area_name)
            well_id = match.selected.well_id if match.selected else None
            method = match.selected.method if match.selected else "manual"
            confidence = match.selected.confidence if match.selected else 1.0
            edits: dict[str, dict[str, Any]] = {}
            if document.source_path == current_path and self.well_combo.currentData() is not None:
                well_id = self.well_combo.currentData(); method = self.match_method; confidence = self.match_confidence
                selected = {self.table.item(index, 0).data(QtCore.Qt.UserRole)
                            for index in range(self.table.rowCount())
                            if self.table.item(index, 0).checkState() == QtCore.Qt.Checked}
                edits = {self.table.item(index, 0).data(QtCore.Qt.UserRole): self._edits(index)
                         for index in range(self.table.rowCount())
                         if self.table.item(index, 0).data(QtCore.Qt.UserRole) in selected}
            else:
                selected = {self._key(interval) for interval in document.intervals
                            if interval.selected_by_default and not interval.errors and interval.confidence >= .8}
            if well_id is None or not selected:
                item.status = "needs_review"; self.documents_table.item(row, 7).setText("требует проверки")
                skipped += 1
                continue
            try:
                result = save_core_import(self.session, document, well_id=well_id,
                                          selected_interval_ids=selected, edits=edits,
                                          match_method=method, match_confidence=confidence)
            except Exception as error:
                item.status = "save_error"; item.error = str(error)
                self.documents_table.item(row, 7).setText(f"ошибка сохранения: {error}")
                failed += 1
                continue
            item.status = "saved"; self.documents_table.item(row, 7).setText("сохранён")
            saved_documents += 1; saved_intervals += result.intervals_saved
        QtWidgets.QMessageBox.information(
            self, "Пакетный импорт",
            f"Сохранено документов: {saved_documents}; интервалов: {saved_intervals}; "
            f"пропущено: {skipped}; ошибок: {failed}")


def open_core_description_import(session: Any, parent: QtWidgets.QWidget | None = None) -> int:
    """Open the single-document import entry point."""
    return CoreDescriptionImportDialog(session, parent).exec_()
