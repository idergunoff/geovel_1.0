"""Editable preview and fault-isolated batch import of core descriptions."""

from __future__ import annotations

import csv
import json
from html import escape
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

from PyQt5 import QtCore, QtGui, QtWidgets

from core_description.database import WellCatalog, load_well_catalog, match_well, save_core_import
from core_description.areas import area_names_match, normalize_area_name
from core_description.identity import WellMetadata, compare_identity, identity_kind, normalize_elevation, identity_match_count
from core_description.batch import BatchParseItem, BatchParseResult, parse_batch
from core_description.models import ParsedCoreDocument, ParsedCoreInterval
from core_description.parser import CoreDescriptionParseError, parse_core_document
from core_description.terminology import (
    ROCK_LABELS, SATURATION_LABELS, RELATION_LABELS, canonical_term, display_term,
)


SATURATIONS = tuple(SATURATION_LABELS)


class DescriptionDelegate(QtWidgets.QStyledItemDelegate):
    """Render exact classifier spans while keeping ordinary table selection."""

    HTML_ROLE = QtCore.Qt.UserRole + 1
    ROCK_COLOR = "#d5edff"
    OIL_COLOR = "#ffe0a3"

    @classmethod
    def highlighted_text(cls, interval: ParsedCoreInterval) -> str:
        text = interval.raw_description
        spans = []
        for match in interval.semantic_matches:
            color = (cls.ROCK_COLOR if match.category == "rock" else cls.OIL_COLOR
                     if match.category in {"oil_indicator", "oil_saturation", "negation", "uncertainty"}
                     else None)
            if color and 0 <= match.start < match.end <= len(text):
                spans.append((match.start, match.end, color))
        boundaries = sorted({0, len(text)} | {point for start, end, _ in spans for point in (start, end)})
        pieces = []
        for start, end in zip(boundaries, boundaries[1:]):
            part = escape(text[start:end])
            color = next((color for left, right, color in spans if left <= start and end <= right), None)
            pieces.append(f'<span style="background-color:{color};color:#17212b">{part}</span>' if color else part)
        return '<div style="white-space:pre-wrap">' + "".join(pieces) + '</div>'

    def _document(self, option: QtWidgets.QStyleOptionViewItem, index: QtCore.QModelIndex) -> QtGui.QTextDocument:
        document = QtGui.QTextDocument()
        document.setDefaultFont(option.font)
        document.setDocumentMargin(4)
        text_option = document.defaultTextOption()
        text_option.setWrapMode(QtGui.QTextOption.WrapAtWordBoundaryOrAnywhere)
        document.setDefaultTextOption(text_option)
        document.setHtml(index.data(self.HTML_ROLE) or escape(index.data() or ""))
        document.setTextWidth(max(20, option.rect.width()))
        return document

    def paint(self, painter: QtGui.QPainter, option: QtWidgets.QStyleOptionViewItem, index: QtCore.QModelIndex) -> None:
        option = QtWidgets.QStyleOptionViewItem(option)
        self.initStyleOption(option, index)
        document = self._document(option, index)
        option.text = ""
        style = option.widget.style() if option.widget else QtWidgets.QApplication.style()
        style.drawControl(QtWidgets.QStyle.CE_ItemViewItem, option, painter, option.widget)
        context = QtGui.QAbstractTextDocumentLayout.PaintContext()
        context.palette = option.palette
        if option.state & QtWidgets.QStyle.State_Selected:
            context.palette.setColor(QtGui.QPalette.Text, option.palette.color(QtGui.QPalette.HighlightedText))
        painter.save()
        painter.translate(option.rect.topLeft())
        painter.setClipRect(QtCore.QRectF(0, 0, option.rect.width(), option.rect.height()))
        document.documentLayout().draw(painter, context)
        painter.restore()

    def sizeHint(self, option: QtWidgets.QStyleOptionViewItem, index: QtCore.QModelIndex) -> QtCore.QSize:
        option = QtWidgets.QStyleOptionViewItem(option)
        self.initStyleOption(option, index)
        if option.widget is not None:
            option.rect.setWidth(option.widget.columnWidth(index.column()))
        document = self._document(option, index)
        return QtCore.QSize(int(document.idealWidth()), int(document.size().height()) + 2)


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
        "Мощность, м", "Геологический индекс",
    )

    def __init__(self, session: Any, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.session = session
        self.document: ParsedCoreDocument | None = None
        self.match_method = "manual"
        self.match_confidence = 1.0
        self._well_selections: dict[str, tuple[int | None, str, float]] = {}
        self._wells: list[tuple[int, str, tuple[str, ...]]] = []
        self._recommended_ids: set[int] = set()
        self._well_metadata: Mapping[int, WellMetadata] = {}
        self._well_catalog: WellCatalog | None = None
        self._well_altitudes: dict[int, float | None] = {}
        self._automatic: dict[str, dict[str, Any]] = {}
        self.batch_items: list[BatchParseItem] = []
        self._thread: QtCore.QThread | None = None
        self._worker: BatchParseWorker | None = None
        self._close_pending = False
        self.setWindowTitle("Импорт описания керна")
        self.setWindowModality(QtCore.Qt.NonModal)
        self.setWindowFlags(self.windowFlags() | QtCore.Qt.WindowMaximizeButtonHint)
        self.setSizeGripEnabled(True)
        screen = self.screen().availableGeometry()
        self.resize(min(1800, int(screen.width() * .95)), min(1000, int(screen.height() * .9)))
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
        lookup_row = QtWidgets.QHBoxLayout()
        lookup_row.addWidget(QtWidgets.QLabel("Площадь БД:"))
        self.area_combo = QtWidgets.QComboBox()
        self.area_combo.setEditable(True)
        self.area_combo.setInsertPolicy(QtWidgets.QComboBox.NoInsert)
        self.area_combo.setMinimumWidth(260)
        self.area_combo.addItem("Все площади")
        self.area_combo.lineEdit().setClearButtonEnabled(True)
        self.area_combo.setToolTip("Выберите площадь или введите часть названия. «Все площади» снимает фильтр.")
        lookup_row.addWidget(self.area_combo, 1)
        self.well_search = QtWidgets.QLineEdit()
        self.well_search.setPlaceholderText("Поиск по номеру скважины")
        self.well_search.setClearButtonEnabled(True)
        lookup_row.addWidget(self.well_search, 1)
        self.refresh_wells_button = QtWidgets.QPushButton("Обновить список")
        lookup_row.addWidget(self.refresh_wells_button)
        layout.addLayout(lookup_row)
        well_row = QtWidgets.QHBoxLayout()
        well_row.addWidget(QtWidgets.QLabel("Скважина БД:"))
        self.well_combo = QtWidgets.QComboBox()
        self.well_combo.setMinimumWidth(260)
        self.well_combo.setMinimumContentsLength(30)
        self.well_combo.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.well_combo.view().setMinimumWidth(min(1100, self.screen().availableGeometry().width()))
        self.well_combo.setToolTip("★ — подходящий вариант; проверьте номер и площадь перед выбором.")
        well_row.addWidget(self.well_combo, 2)
        self.match_label = QtWidgets.QLabel("Не сопоставлена")
        self.match_label.setWordWrap(True)
        well_row.addWidget(self.match_label, 1)
        self.well_count_label = QtWidgets.QLabel()
        well_row.addWidget(self.well_count_label)
        layout.addLayout(well_row)

        self.comparison_table = QtWidgets.QTableWidget(4, 4)
        self.comparison_table.setObjectName("core_well_comparison")
        self.comparison_table.setHorizontalHeaderLabels(("Признак", "В документе", "В базе данных", "Сравнение"))
        self.comparison_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.comparison_table.verticalHeader().hide()
        self.comparison_table.setMaximumHeight(260)
        self.comparison_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        layout.addWidget(self.comparison_table)

        filters = QtWidgets.QHBoxLayout()
        self.depth_from = QtWidgets.QDoubleSpinBox(); self.depth_from.setRange(-1e6, 1e6)
        self.depth_to = QtWidgets.QDoubleSpinBox(); self.depth_to.setRange(-1e6, 1e6); self.depth_to.setValue(1e6)
        self.rock_filter = QtWidgets.QLineEdit(); self.rock_filter.setPlaceholderText("любая")
        self.saturation_filter = QtWidgets.QComboBox(); self.saturation_filter.addItem("любая", None)
        for code, label in SATURATION_LABELS.items():
            self.saturation_filter.addItem(label, code)
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
        self.table.setWordWrap(True)
        self.table.setTextElideMode(QtCore.Qt.ElideNone)
        self.table.setItemDelegateForColumn(7, DescriptionDelegate(self.table))
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QtWidgets.QHeaderView.Interactive)
        header.setStretchLastSection(False)
        header.moveSection(header.visualIndex(9), 3)
        header.moveSection(header.visualIndex(10), 4)
        self.table.setColumnWidth(7, 560)
        self.table.setColumnWidth(4, 230)
        self.table.setColumnWidth(10, 180)
        self._resize_timer = QtCore.QTimer(self)
        self._resize_timer.setSingleShot(True)
        self._resize_timer.timeout.connect(self.table.resizeRowsToContents)
        header.sectionResized.connect(lambda *_: self._resize_timer.start(50))
        layout.addWidget(self.table, 1)
        legend = QtWidgets.QLabel(
            f'<span style="background-color:{DescriptionDelegate.ROCK_COLOR};color:#17212b"> Породы </span>'
            f' &nbsp; <span style="background-color:{DescriptionDelegate.OIL_COLOR};color:#17212b"> Нефтенасыщенность </span>')
        layout.addWidget(legend)
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
        self.well_search.textChanged.connect(self._filter_wells)
        self.area_combo.currentTextChanged.connect(self._filter_wells)
        self.refresh_wells_button.clicked.connect(lambda: self._load_wells(reset_filters=False, refresh=True))
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
        self._well_selections.clear()
        self.documents_table.setRowCount(0)
        self.progress_bar.setRange(0, 0); self.progress_bar.setValue(0); self.progress_bar.setVisible(True)
        self.save_button.setEnabled(False)
        self.cancel_button.setEnabled(True); self.open_button.setEnabled(False); self.directory_button.setEnabled(False)
        thread = QtCore.QThread(self)
        worker = BatchParseWorker(paths, recursive)
        worker.moveToThread(thread); thread.started.connect(worker.run)
        worker.progress.connect(self._batch_progress); worker.finished.connect(self._batch_finished)
        worker.finished.connect(thread.quit); worker.finished.connect(worker.deleteLater)
        thread.finished.connect(self._batch_thread_finished)
        thread.finished.connect(thread.deleteLater)
        self._thread, self._worker = thread, worker
        thread.start()

    def cancel_batch(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            self.cancel_button.setEnabled(False)

    def reject(self) -> None:
        if self._thread is not None:
            self._close_pending = True
            self.cancel_batch()
            self.summary.setText("Ожидание завершения текущего файла перед закрытием…")
            return
        super().reject()

    def closeEvent(self, event: Any) -> None:
        if self._thread is not None:
            self.reject()
            event.ignore()
            return
        super().closeEvent(event)

    def _batch_thread_finished(self) -> None:
        self._worker = None
        self._thread = None
        if self._close_pending:
            self.reject()
        else:
            self._update_summary()

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
        self.export_button.setEnabled(bool(result.items))
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
        for column in (0, 1, 2, 3, 5, 6, 8, 9):
            self.table.resizeColumnToContents(column)
            self.table.setColumnWidth(column, min(280, self.table.columnWidth(column)))
        self.table.resizeRowsToContents()
        self.apply_filters(); self._update_summary()

    def _load_wells(self, reset_filters: bool = True, *, refresh: bool = False) -> None:
        if self.document is None:
            return
        area_filter = "Все площади" if reset_filters else self.area_combo.currentText()
        if reset_filters:
            self.well_search.blockSignals(True)
            self.well_search.clear()
            self.well_search.blockSignals(False)
        if self._well_catalog is None or refresh:
            self._well_catalog = load_well_catalog(self.session)
        wells = self._well_catalog.wells
        self._well_metadata = self._well_catalog.metadata
        self._well_altitudes = {well.id: well.alt for well in wells}
        areas = {well_id: tuple(dict.fromkeys(value for _, value in metadata.area_entries(self.document.area_name)))
                 for well_id, metadata in self._well_metadata.items()}
        self._wells = [(well.id, well.name, tuple(areas.get(well.id, ()))) for well in wells]
        area_names = sorted({value for values in areas.values() for value in values}, key=lambda value: (
            not area_names_match(self.document.area_name, value), value.casefold()))
        self.area_combo.blockSignals(True)
        self.area_combo.clear()
        self.area_combo.addItem("Все площади")
        self.area_combo.addItems(area_names)
        self.area_combo.setCurrentText(area_filter)
        self.area_combo.blockSignals(False)
        result = match_well(self.session, self.document.well_name, self.document.area_name,
                            document=self.document, catalog=self._well_catalog)
        self._recommended_ids = {candidate.well_id for candidate in result.candidates}
        if result.selected is not None:
            self._recommended_ids.add(result.selected.well_id)
        candidate_ranks = {candidate.well_id: rank for rank, candidate in enumerate(result.candidates)}
        if result.selected is not None:
            candidate_ranks[result.selected.well_id] = 0
        self._wells.sort(key=lambda item: (
            candidate_ranks.get(item[0], len(candidate_ranks)),
            not any(area_names_match(self.document.area_name, area) for area in item[2]),
            item[1].casefold(), item[0]))
        stored = self._well_selections.get(self.document.source_path)
        selected_id = stored[0] if stored else result.selected.well_id if result.selected else None
        self._fill_well_combo(selected_id)
        if stored and self.well_combo.currentData() is not None:
            self.match_method, self.match_confidence = stored[1:]
            self.match_label.setText("Выбрана вручную")
        elif result.selected and self.well_combo.currentData() == result.selected.well_id:
            self.match_method = result.selected.method
            self.match_confidence = result.selected.confidence
            count = identity_match_count(result.selected.comparisons)
            self.match_label.setText(f"Совпадают номер и {count} из 4 признаков" if self.match_method == "exact_name_identity"
                                     else "Точное совпадение" if self.match_confidence == 1.0
                                     else "Совпадает номер; проверьте площадь")
        else:
            self.match_method = "manual"
            self.match_confidence = 1.0
            self.match_label.setText("Варианты отмечены ★; выберите скважину" if result.candidates
                                     else "Выберите подходящую скважину из списка")
        self._update_summary()
        self._update_well_comparison()

    def _fill_well_combo(self, selected_id: int | None) -> None:
        name_filter = self.well_search.text().strip().casefold()
        area_text = self.area_combo.currentText().strip()
        area_filter = normalize_area_name(area_text) if area_text != "Все площади" else ""
        self.well_combo.blockSignals(True)
        self.well_combo.clear()
        self.well_combo.addItem("— выберите скважину —", None)
        for well_id, name, areas in self._wells:
            if name_filter and name_filter not in name.casefold():
                continue
            metadata = self._well_metadata.get(well_id, WellMetadata())
            displayed_areas = areas
            if area_filter:
                found = tuple(value for label, value in metadata.attributes
                              if identity_kind(label) is None and (
                                  normalize_area_name(value) == area_filter
                                  if area_filter.startswith("объект") and area_filter[6:].isdigit()
                                  else area_filter in normalize_area_name(value) or area_names_match(area_text, value)))
                if not found:
                    continue
                displayed_areas = tuple(dict.fromkeys((*areas, *found)))
            prefix = "★ " if well_id in self._recommended_ids else ""
            start = ", ".join(metadata.values("drilling_start")) or "—"
            end = ", ".join(metadata.values("drilling_end")) or "—"
            elevations = metadata.values("wellhead_elevation")
            altitude = self._well_altitudes.get(well_id)
            elevation_values = elevations or ((str(altitude),) if altitude is not None else ())
            elevation = ", ".join(f"{normalized:g} м" if (normalized := normalize_elevation(value)) is not None
                                  else value for value in elevation_values) or "—"
            text = (f"{prefix}{name} — {', '.join(displayed_areas) or 'площадь не указана'}; "
                    f"начало: {start}; окончание: {end}; устье: {elevation} (id {well_id})")
            self.well_combo.addItem(text, well_id)
            self.well_combo.setItemData(self.well_combo.count() - 1, text, QtCore.Qt.ToolTipRole)
        index = self.well_combo.findData(selected_id)
        self.well_combo.setCurrentIndex(max(0, index))
        self.well_combo.blockSignals(False)
        count = self.well_combo.count() - 1
        self.well_count_label.setText(f"Найдено: {count}" if count else "Нет скважин — измените условия поиска")

    def _filter_wells(self, *_args: Any) -> None:
        selected_id = self.well_combo.currentData()
        self._fill_well_combo(selected_id)
        if selected_id != self.well_combo.currentData():
            self._well_changed()

    def _update_well_comparison(self) -> None:
        if self.document is None:
            return
        well_id = self.well_combo.currentData()
        comparisons = compare_identity(self.document, self._well_metadata.get(well_id, WellMetadata()),
                                       self._well_altitudes.get(well_id))
        labels = {"area": "Площадь / залежь", "drilling_start": "Начало бурения",
                  "drilling_end": "Окончание бурения", "wellhead_elevation": "Альтитуда устья, м"}
        statuses = {"match": "Совпадает", "different": "Различается", "missing": "Нет данных для сравнения",
                    "partial": "Совпадает месяц; дата неполная", "conflict": "Противоречивые данные",
                    "invalid_source": "Проверьте значение в документе", "invalid_database": "Проверьте значение в базе"}
        for row, comparison in enumerate(comparisons):
            status = comparison.status
            status_text = statuses[status]
            if comparison.field == "area" and status == "match" and comparison.similarity < 1:
                status_text += f" (сходство {comparison.similarity:.1%})"
            for column, value in enumerate((labels[comparison.field], comparison.source or "—",
                                            comparison.database or "—", status_text)):
                cell = QtWidgets.QTableWidgetItem(value)
                cell.setToolTip("Все значения в базе:\n" + "\n".join(comparison.database_values)
                                if column == 2 and len(comparison.database_values) > 1 else value)
                if status == "match":
                    cell.setBackground(QtGui.QColor("#dcf5df"))
                elif status in {"different", "conflict", "invalid_source", "invalid_database"}:
                    cell.setBackground(QtGui.QColor("#ffe4df"))
                elif status == "partial":
                    cell.setBackground(QtGui.QColor("#fff2ce"))
                self.comparison_table.setItem(row, column, cell)
        self.comparison_table.resizeRowsToContents()
        height = (self.comparison_table.horizontalHeader().height()
                  + sum(self.comparison_table.rowHeight(row) for row in range(self.comparison_table.rowCount()))
                  + 2 * self.comparison_table.frameWidth())
        self.comparison_table.setFixedHeight(min(260, height))

    def _well_changed(self, *_args: Any) -> None:
        self.match_method = "manual"
        self.match_confidence = 1.0
        self.match_label.setText("Выбрана вручную" if self.well_combo.currentData() is not None
                                 else "Требуется ручной выбор")
        if self.document is not None:
            self._well_selections[self.document.source_path] = (
                self.well_combo.currentData(), self.match_method, self.match_confidence)
            for row, item in enumerate(self.batch_items):
                if item.source_path == self.document.source_path:
                    self.documents_table.item(row, 5).setText(self.well_combo.currentText())
        self._update_summary()
        self._update_well_comparison()

    @staticmethod
    def _key(interval: ParsedCoreInterval) -> str:
        return f"{interval.source_table}:{interval.source_row}"

    def _append_interval(self, interval: ParsedCoreInterval) -> None:
        row = self.table.rowCount(); self.table.insertRow(row)
        key = self._key(interval)
        primary = next((display_term(r.canonical_value, ROCK_LABELS)
                        for r in interval.rocks if r.relation == "primary"), "")
        secondary = ", ".join(f"{display_term(r.canonical_value, ROCK_LABELS)} ({display_term(r.relation, RELATION_LABELS)})" for r in interval.rocks if r.relation != "primary")
        state = "ошибка" if interval.errors else "требует проверки" if interval.warnings or interval.confidence < .8 else "готово"
        values = ("", interval.top_depth, interval.bottom_depth, primary, secondary,
                  display_term(interval.oil_saturation, SATURATION_LABELS), interval.confidence, interval.raw_description, state,
                  interval.thickness, interval.geological_index_raw)
        check = QtWidgets.QTableWidgetItem(); check.setData(QtCore.Qt.UserRole, key)
        check.setFlags(check.flags() | QtCore.Qt.ItemIsUserCheckable)
        check.setCheckState(QtCore.Qt.Checked if interval.selected_by_default and not interval.errors and interval.confidence >= .8 else QtCore.Qt.Unchecked)
        self.table.setItem(row, 0, check)
        for column, value in enumerate(values[1:], 1):
            item = QtWidgets.QTableWidgetItem("" if value is None else str(value)); self.table.setItem(row, column, item)
            if column in (7, 8): item.setFlags(item.flags() & ~QtCore.Qt.ItemIsEditable)
        self.table.item(row, 7).setData(DescriptionDelegate.HTML_ROLE, DescriptionDelegate.highlighted_text(interval))
        details = "\n".join(f"{m.rule_id}: «{m.matched_text}» [{m.start}:{m.end}]" for m in interval.semantic_matches)
        for column in range(len(self.COLUMNS)): self.table.item(row, column).setToolTip(details)
        self._automatic[key] = {"top_depth": interval.top_depth, "bottom_depth": interval.bottom_depth,
                                "primary": primary, "secondary": secondary,
                                "oil_saturation": display_term(interval.oil_saturation, SATURATION_LABELS), "confidence": interval.confidence,
                                "thickness": interval.thickness, "geological_index_raw": interval.geological_index_raw}

    def _table_changed(self, _item: QtWidgets.QTableWidgetItem) -> None:
        self._update_summary(); self.apply_filters()
        self._resize_timer.start(50)

    def _state(self, row: int) -> str:
        return self.table.item(row, 8).text()

    def apply_filters(self) -> None:
        rock = self.rock_filter.text().casefold().strip(); saturation = self.saturation_filter.currentData()
        state = self.state_filter.currentText()
        for row in range(self.table.rowCount()):
            try: top = float(self.table.item(row, 1).text().replace(",", "."))
            except ValueError: top = float("-inf")
            row_rocks = (self.table.item(row, 3).text() + " " + self.table.item(row, 4).text()).casefold()
            visible = self.depth_from.value() <= top <= self.depth_to.value() and (not rock or rock in row_rocks)
            visible &= saturation is None or canonical_term(self.table.item(row, 5).text(), SATURATION_LABELS) == saturation
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
        oil = set(SATURATIONS) - {"unknown", "none"}
        for row in range(self.table.rowCount()):
            if not self.table.isRowHidden(row):
                self.table.item(row, 0).setCheckState(QtCore.Qt.Checked if canonical_term(self.table.item(row, 5).text(), SATURATION_LABELS) in oil else QtCore.Qt.Unchecked)

    def _restore_selected(self) -> None:
        for row in {index.row() for index in self.table.selectedIndexes()}:
            auto = self._automatic[self.table.item(row, 0).data(QtCore.Qt.UserRole)]
            for column, name in ((1, "top_depth"), (2, "bottom_depth"), (3, "primary"), (4, "secondary"), (5, "oil_saturation"), (6, "confidence"), (9, "thickness"), (10, "geological_index_raw")):
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
        self.save_button.setEnabled(bool(count and self.well_combo.currentData() is not None
                                         and self._thread is None))

    def _edits(self, row: int) -> dict[str, Any]:
        key = self.table.item(row, 0).data(QtCore.Qt.UserRole); auto = self._automatic[key]
        values: dict[str, Any] = {
            "top_depth": float(self.table.item(row, 1).text().replace(",", ".")),
            "bottom_depth": float(self.table.item(row, 2).text().replace(",", ".")),
            "oil_saturation": canonical_term(self.table.item(row, 5).text(), SATURATION_LABELS),
            "confidence": float(self.table.item(row, 6).text().replace(",", ".")),
            "thickness": float(self.table.item(row, 9).text().replace(",", ".")) if self.table.item(row, 9).text().strip() else None,
            "geological_index_raw": self.table.item(row, 10).text().strip() or None,
        }
        primary = self.table.item(row, 3).text().strip()
        secondary = self.table.item(row, 4).text().strip()
        interval = next(i for i in self.document.intervals if self._key(i) == key)  # type: ignore[union-attr]
        if primary != auto["primary"] or secondary != auto["secondary"]:
            rocks = []
            if primary: rocks.insert(0, {"canonical_value": canonical_term(primary, ROCK_LABELS), "matched_text": primary, "relation": "primary", "confidence": 1.0, "rule_ids": []})
            for value in filter(None, (part.strip() for part in secondary.split(","))):
                name, separator, relation = value.rpartition(" (")
                if separator and relation.endswith(")"):
                    value, relation = name, relation[:-1]
                else:
                    relation = "secondary"
                rocks.append({"canonical_value": canonical_term(value, ROCK_LABELS), "matched_text": value,
                              "relation": canonical_term(relation, RELATION_LABELS),
                              "confidence": 1.0, "rule_ids": []})
            values["rocks"] = rocks
        return values

    def save(self) -> None:
        if self.document is None or self._thread is not None: return
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
            match = match_well(self.session, document.well_name, document.area_name, document=document)
            well_id = match.selected.well_id if match.selected else None
            method = match.selected.method if match.selected else "manual"
            confidence = match.selected.confidence if match.selected else 1.0
            if document.source_path in self._well_selections:
                well_id, method, confidence = self._well_selections[document.source_path]
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


_import_dialog: CoreDescriptionImportDialog | None = None


def open_core_description_import(
    session: Any, parent: QtWidgets.QWidget | None = None,
) -> CoreDescriptionImportDialog:
    """Keep a modeless importer alive and reuse it while it remains open."""
    global _import_dialog
    if _import_dialog is None:
        _import_dialog = CoreDescriptionImportDialog(session, parent)
        _import_dialog.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        _import_dialog.destroyed.connect(_forget_import_dialog)
    _import_dialog.show()
    _import_dialog.raise_()
    _import_dialog.activateWindow()
    return _import_dialog


def _forget_import_dialog() -> None:
    global _import_dialog
    _import_dialog = None
