"""Saved core descriptions, dictionary learning and reviewed semantic reprocessing."""

import json
import time

from PyQt5 import QtCore, QtGui, QtWidgets
from sqlalchemy.exc import SQLAlchemyError

from core_description.presentation import (
    OIL_LAYER_COLOR, has_oil, load_core_intervals, number_text, primary_rock_text,
    secondary_rock_text, saturation_text,
)
from core_description.terminology import ROCK_LABELS, RELATION_LABELS, display_term
from core_description.reparse import (
    ReparseError, interval_token, preview_reparse, processing_metadata, recognition_details, undo_reparse,
)
from core_description.semantics import DictionaryValidationError
from qt.core_description_dictionary import CATEGORY_LABELS, CoreDescriptionDictionaryDialog
from qt.core_description_import import DescriptionDelegate
from qt.core_description_learning import LearnDescriptionDialog
from qt.core_description_reparse import ReparsePreviewDialog, ReparseWellScopeDialog, state_text


def _pretty_json(value):
    try:
        return json.dumps(json.loads(value), ensure_ascii=False, indent=2)
    except (ValueError, TypeError):
        return str(value or "—")


def saved_description_html(interval):
    try:
        details = json.loads(interval.recognition_details)
    except (ValueError, TypeError):
        details = {}
    matches = details.get("matches", []) if isinstance(details, dict) else []
    if not isinstance(matches, list):
        matches = []
    return DescriptionDelegate.highlighted_matches(
        interval.raw_description,
        ((m.get("category"), m.get("start"), m.get("end"))
         for m in matches if isinstance(m, dict)))


class CoreDescriptionViewDialog(QtWidgets.QDialog):
    descriptions_changed = QtCore.pyqtSignal(object)
    COLUMNS = (
        "Индекс", "Кровля, м", "Подошва, м", "Мощность, м", "Основная порода",
        "Второстепенные породы", "Нефтенасыщенность", "Исходное описание",
    )

    def __init__(self, session, well, parent=None):
        super().__init__(parent)
        self.session = session
        self.well_id = well.id
        self.well_name = well.name
        self.setWindowTitle(f"Описание керна — скважина {well.name}")
        self.setWindowModality(QtCore.Qt.NonModal)
        self.setWindowFlag(QtCore.Qt.WindowMaximizeButtonHint)
        self.setSizeGripEnabled(True)
        screen = self.screen().availableGeometry()
        self.resize(min(1500, int(screen.width() * .95)), min(850, int(screen.height() * .9)))
        layout = QtWidgets.QVBoxLayout(self)
        self.intervals = load_core_intervals(session, well.id)
        self.summary = QtWidgets.QLabel(
            f"Скважина {well.name}. Слоёв: {len(self.intervals)}. "
            "Выберите слой для просмотра полного описания и сведений об источнике."
            if self.intervals else f"Для скважины {well.name} описание керна не загружено.")
        self.summary.setTextFormat(QtCore.Qt.PlainText)
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)
        actions = QtWidgets.QHBoxLayout()
        self.reparse_button = QtWidgets.QPushButton("Переразобрать…")
        menu = QtWidgets.QMenu(self.reparse_button)
        menu.addAction("Выбранные интервалы", lambda: self.reparse(selected=True))
        menu.addAction("Всю текущую скважину", self.reparse)
        menu.addAction("Выбрать скважины / вся база…", self.choose_scope)
        self.reparse_button.setMenu(menu)
        self.dictionary_button = QtWidgets.QPushButton("Словарь…")
        self.dictionary_button.clicked.connect(self.edit_dictionary)
        self.learn_button = QtWidgets.QPushButton("Добавить выделенное в словарь…")
        self.learn_button.clicked.connect(lambda: self.learn_selection())
        self.history_button = QtWidgets.QPushButton("История и откат…")
        self.history_button.clicked.connect(self.show_history)
        for button in (self.reparse_button, self.dictionary_button, self.learn_button, self.history_button):
            actions.addWidget(button)
        actions.addStretch()
        layout.addLayout(actions)
        splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.table = QtWidgets.QTableWidget(len(self.intervals), len(self.COLUMNS))
        self.table.setObjectName("saved_core_description_intervals")
        self.table.setHorizontalHeaderLabels(self.COLUMNS)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.table.setWordWrap(True)
        self.table.setTextElideMode(QtCore.Qt.ElideNone)
        self.table.setItemDelegateForColumn(7, DescriptionDelegate(self.table))
        self.table.setAlternatingRowColors(True)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QtWidgets.QHeaderView.Interactive)
        header.setStretchLastSection(True)
        self._resize_timer = QtCore.QTimer(self)
        self._resize_timer.setSingleShot(True)
        self._resize_timer.timeout.connect(self.table.resizeRowsToContents)
        header.sectionResized.connect(lambda *_: self._resize_timer.start(50))
        self._populate_table()
        for column in range(7):
            self.table.resizeColumnToContents(column)
            self.table.setColumnWidth(column, min(200, self.table.columnWidth(column)))
        self.table.setColumnWidth(7, 560)
        self.table.resizeRowsToContents()
        splitter.addWidget(self.table)
        lower = QtWidgets.QTabWidget()
        self.source_text = QtWidgets.QPlainTextEdit()
        self.source_text.setObjectName("core_description_source_text")
        self.source_text.setReadOnly(True)
        self.source_text.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.source_text.customContextMenuRequested.connect(self._source_menu)
        self.source_text.selectionChanged.connect(
            lambda: self.learn_button.setEnabled(bool(self.source_text.textCursor().selectedText().strip())))
        lower.addTab(self.source_text, "Текст — выделите слово или фразу для словаря")
        self.details = QtWidgets.QPlainTextEdit()
        self.details.setReadOnly(True)
        lower.addTab(self.details, "Сведения об интервале")
        splitter.addWidget(lower)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        layout.addWidget(splitter, 1)
        self.status_label = QtWidgets.QLabel()
        self.status_label.setTextFormat(QtCore.Qt.PlainText)
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
        buttons.button(QtWidgets.QDialogButtonBox.Close).setText("Закрыть")
        buttons.rejected.connect(self.close)
        layout.addWidget(buttons)
        self.table.currentCellChanged.connect(self._show_details)
        self.learn_button.setEnabled(False)
        self.history_button.setEnabled(bool(self.intervals))
        self.reparse_button.setEnabled(bool(self.intervals))
        if self.intervals:
            self.table.selectRow(0)

    def _populate_table(self):
        self.table.setRowCount(len(self.intervals))
        for row, interval in enumerate(self.intervals):
            values = (
                interval.geological_index_raw or "—", number_text(interval.top_depth),
                number_text(interval.bottom_depth), number_text(interval.thickness),
                primary_rock_text(interval), secondary_rock_text(interval),
                saturation_text(interval.oil_saturation, omit_unknown=True), interval.raw_description,
            )
            for column, value in enumerate(values):
                item = QtWidgets.QTableWidgetItem(value)
                item.setToolTip(value)
                if has_oil(interval):
                    item.setBackground(QtGui.QColor(OIL_LAYER_COLOR))
                    item.setForeground(QtGui.QColor("#17212b"))
                self.table.setItem(row, column, item)
            self.table.item(row, 7).setData(DescriptionDelegate.HTML_ROLE, saved_description_html(interval))

    def _show_details(self, row, *_args):
        if row < 0:
            self.details.clear()
            self.source_text.clear()
            return
        interval = self.intervals[row]
        self.source_text.setPlainText(interval.raw_description)
        selections = []
        matches = recognition_details(interval).get("matches", [])
        for match in matches if isinstance(matches, list) else []:
            if not isinstance(match, dict):
                continue
            start, end = match.get("start"), match.get("end")
            if not isinstance(start, int) or not isinstance(end, int) or not 0 <= start < end <= len(interval.raw_description):
                continue
            selection = QtWidgets.QTextEdit.ExtraSelection()
            selection.cursor = self.source_text.textCursor()
            selection.cursor.setPosition(len(interval.raw_description[:start].encode("utf-16-le")) // 2)
            selection.cursor.setPosition(len(interval.raw_description[:end].encode("utf-16-le")) // 2,
                                         QtGui.QTextCursor.KeepAnchor)
            selection.format.setBackground(QtGui.QColor(
                DescriptionDelegate.ROCK_COLOR if match.get("category") == "rock" else DescriptionDelegate.OIL_COLOR))
            selections.append(selection)
        self.source_text.setExtraSelections(selections)
        document = interval.document
        parts = [f"Исходное описание:\n{interval.raw_description}",
                 f"Нормализованное описание:\n{interval.normalized_description}",
                 f"Уверенность: {number_text(interval.confidence)}\n"
                 f"Требует проверки: {'Да' if interval.needs_review else 'Нет'}\n"
                 f"Правки вручную: {'Да' if interval.manually_edited else 'Нет'}"]
        for rock in sorted(interval.rocks, key=lambda rock: rock.id or 0):
            parts.append(
                f"Порода: {display_term(rock.rock_name, ROCK_LABELS)}; "
                f"роль: {display_term(rock.role, RELATION_LABELS)}; "
                f"нефтенасыщенность: {saturation_text(rock.oil_saturation)}\n"
                f"Исходный текст: {rock.source_text}\nУверенность: {number_text(rock.confidence)}\n"
                f"Позиция в тексте: {rock.match_start}–{rock.match_end}")
        parts.extend([
            f"Исходная таблица: {interval.source_table_index}; строка: {interval.source_row_index}",
            f"Файл: {document.source_file_name}\nФормат: {document.source_format}\n"
            f"Скважина в документе: {document.well_name_raw or '—'}\n"
            f"Площадь: {document.area_name_raw or '—'}\n"
            f"Автор: {document.described_by or '—'}\nИсходная запись автора: {document.described_by_raw or '—'}\n"
            f"Дата импорта (UTC): {document.imported_at}\nСтатус: {document.status}\n"
            f"Сопоставление: {document.match_method}; уверенность: {number_text(document.match_confidence)}\n"
            f"Версии при импорте — парсер: {document.parser_version}; словарь: {document.dictionary_version or '—'}\n"
            f"Хеш файла: {document.source_file_hash}",
            f"Текущая обработка интервала:\n{json.dumps(processing_metadata(interval), ensure_ascii=False, indent=2)}",
            f"Сведения о распознавании:\n{json.dumps({key: value for key, value in recognition_details(interval).items() if key != 'history'}, ensure_ascii=False, indent=2)}",
            f"Диагностика документа:\n{self._document_diagnostics(document.diagnostics)}",
        ])
        self.details.setPlainText("\n\n".join(parts))

    def reload(self):
        current = self.table.currentRow()
        selected_id = self.intervals[current].id if 0 <= current < len(self.intervals) else None
        self.intervals = load_core_intervals(self.session, self.well_id)
        self.table.blockSignals(True)
        self._populate_table()
        self.table.blockSignals(False)
        self.table.resizeRowsToContents()
        row = next((i for i, item in enumerate(self.intervals) if item.id == selected_id), 0)
        if self.intervals:
            self.table.setCurrentCell(row, 0)
            self._show_details(row)
        else:
            self._show_details(-1)

    def _changed(self, well_ids):
        dialogs = self.parent().findChildren(CoreDescriptionViewDialog) if self.parent() else [self]
        for dialog in dialogs:
            if dialog.well_id in well_ids:
                dialog.reload()
        self.descriptions_changed.emit(well_ids)
        self.status_label.setText("Результаты распознавания обновлены. Предыдущие значения доступны в истории.")

    @staticmethod
    def _document_diagnostics(encoded):
        try:
            diagnostic = json.loads(encoded)
            if isinstance(diagnostic, dict):
                diagnostic.pop("semantic_dictionaries", None)
                return json.dumps(diagnostic, ensure_ascii=False, indent=2)
        except (ValueError, TypeError):
            pass
        return _pretty_json(encoded)

    def choose_scope(self):
        dialog = ReparseWellScopeDialog(self.session, self.well_id, self)
        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            self.reparse(well_ids=dialog.selected_well_ids())

    def reparse(self, checked=False, *, selected=False, changed_only=False, well_ids=None):
        ids = {self.intervals[index.row()].id for index in self.table.selectionModel().selectedRows()} if selected else None
        progress = QtWidgets.QProgressDialog("Подготовка сравнения описаний…", "Отмена", 0, 0, self)
        progress.setWindowModality(QtCore.Qt.WindowModal)
        progress.setMinimumDuration(300)
        progress.setAutoReset(False)
        last_update = [0.0]

        def update_progress(current, total):
            if progress.wasCanceled():
                return False
            now = time.monotonic()
            if now - last_update[0] >= .05 or current == total:
                progress.setMaximum(max(1, total))
                progress.setValue(current)
                QtWidgets.QApplication.processEvents()
                last_update[0] = now
            return not progress.wasCanceled()

        try:
            preview = preview_reparse(self.session, interval_ids=ids,
                                      well_ids=[self.well_id] if well_ids is None else well_ids,
                                      progress=update_progress)
        except (DictionaryValidationError, ReparseError, SQLAlchemyError) as error:
            self.status_label.setText(f"Не удалось подготовить переразбор: {error}")
            return
        finally:
            progress.close()
        dialog = ReparsePreviewDialog(self.session, preview, self, changed_only=changed_only)
        dialog.applied.connect(self._changed)
        dialog.exec_()

    def _source_menu(self, position):
        menu = self.source_text.createStandardContextMenu()
        if self.source_text.textCursor().selectedText().strip():
            add = menu.addMenu("Добавить в словарь")
            for category in ("rock", "oil_indicator", "negation", "oil_saturation"):
                add.addAction(CATEGORY_LABELS[category], lambda _checked=False, code=category: self.learn_selection(code))
        menu.exec_(self.source_text.mapToGlobal(position))

    def learn_selection(self, category="rock"):
        expression = self.source_text.textCursor().selectedText().replace("\u2029", "\n").strip()
        row = self.table.currentRow()
        if not expression or row < 0:
            self.status_label.setText("Выделите слово или фразу в полном тексте описания.")
            return
        interval = self.intervals[row]
        try:
            dialog = LearnDescriptionDialog(expression, interval.raw_description, category=category,
                                            source_label=f"{self.well_name}, {interval.top_depth:g}–{interval.bottom_depth:g} м",
                                            parent=self)
        except DictionaryValidationError as error:
            self.status_label.setText(str(error))
            return
        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            self.status_label.setText("Словарь сохранён. Выберите результаты переразбора для применения.")
            self.reparse(changed_only=True)

    def edit_dictionary(self):
        try:
            dialog = CoreDescriptionDictionaryDialog(self)
        except DictionaryValidationError as error:
            self.status_label.setText(str(error))
            return
        saved = []
        dialog.dictionary_saved.connect(saved.append)
        dialog.exec_()
        if saved:
            self.reparse(changed_only=True)

    def show_history(self):
        row = self.table.currentRow()
        if row < 0:
            return
        interval = self.intervals[row]
        token = interval_token(interval)
        history = recognition_details(interval).get("history", [])
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle(f"История разбора — {interval.top_depth:g}–{interval.bottom_depth:g} м")
        dialog.resize(850, 650)
        layout = QtWidgets.QVBoxLayout(dialog)
        undone = {entry.get("undoes") for entry in history if entry.get("action") == "undo"}
        target = next((entry for entry in reversed(history) if entry.get("action") == "reparse"
                       and entry["id"] not in undone), None)
        text = QtWidgets.QPlainTextEdit()
        text.setReadOnly(True)
        entries = [f'{entry["at"]}: {"Переразбор" if entry["action"] == "reparse" else "Откат"}\n'
                   f'Предыдущий результат:\n{state_text(json.dumps(entry["before"]))}' for entry in reversed(history)]
        text.setPlainText("\n\n".join(entries) or "История переразбора пуста.")
        layout.addWidget(text)
        if target:
            label = QtWidgets.QLabel("При откате будут восстановлены значения:\n" + state_text(json.dumps(target["before"])))
            label.setTextFormat(QtCore.Qt.PlainText)
            label.setWordWrap(True)
            layout.addWidget(label)
        error_label = QtWidgets.QLabel()
        error_label.setWordWrap(True)
        error_label.setTextFormat(QtCore.Qt.PlainText)
        layout.addWidget(error_label)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
        buttons.rejected.connect(dialog.reject)
        undo_button = buttons.addButton("Отменить последний переразбор", QtWidgets.QDialogButtonBox.ActionRole)
        undo_button.setEnabled(target is not None)

        def undo():
            try:
                undo_reparse(self.session, interval.id, expected_token=token)
            except (ReparseError, SQLAlchemyError) as error:
                error_label.setText(str(error))
                return
            self._changed({self.well_id})
            dialog.accept()

        undo_button.clicked.connect(undo)
        layout.addWidget(buttons)
        dialog.exec_()


def open_core_description_view(session, well, parent=None):
    """Retain the window through its parent; each opening reflects the selected well."""
    dialog = CoreDescriptionViewDialog(session, well, parent)
    dialog.setAttribute(QtCore.Qt.WA_DeleteOnClose)
    dialog.show()
    dialog.raise_()
    dialog.activateWindow()
    return dialog
