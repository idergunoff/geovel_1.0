"""Read-only table of all saved core intervals of a selected well."""

import json

from PyQt5 import QtCore, QtGui, QtWidgets

from core_description.presentation import (
    OIL_LAYER_COLOR, has_oil, load_core_intervals, number_text, primary_rock_text,
    secondary_rock_text, saturation_text,
)
from core_description.terminology import ROCK_LABELS, RELATION_LABELS, display_term
from qt.core_description_import import DescriptionDelegate


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
    COLUMNS = (
        "Индекс", "Кровля, м", "Подошва, м", "Мощность, м", "Основная порода",
        "Второстепенные породы", "Нефтенасыщенность", "Исходное описание",
    )

    def __init__(self, session, well, parent=None):
        super().__init__(parent)
        self.well_id = well.id
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
        splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.table = QtWidgets.QTableWidget(len(self.intervals), len(self.COLUMNS))
        self.table.setObjectName("saved_core_description_intervals")
        self.table.setHorizontalHeaderLabels(self.COLUMNS)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
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
        for column in range(7):
            self.table.resizeColumnToContents(column)
            self.table.setColumnWidth(column, min(200, self.table.columnWidth(column)))
        self.table.setColumnWidth(7, 560)
        self.table.resizeRowsToContents()
        splitter.addWidget(self.table)
        self.details = QtWidgets.QPlainTextEdit()
        self.details.setReadOnly(True)
        splitter.addWidget(self.details)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        layout.addWidget(splitter, 1)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
        buttons.button(QtWidgets.QDialogButtonBox.Close).setText("Закрыть")
        buttons.rejected.connect(self.close)
        layout.addWidget(buttons)
        self.table.currentCellChanged.connect(self._show_details)
        if self.intervals:
            self.table.selectRow(0)

    def _show_details(self, row, *_args):
        if row < 0:
            self.details.clear()
            return
        interval = self.intervals[row]
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
            f"Версия парсера: {document.parser_version}; словаря: {document.dictionary_version or '—'}\n"
            f"Хеш файла: {document.source_file_hash}",
            f"Сведения о распознавании:\n{_pretty_json(interval.recognition_details)}",
            f"Диагностика документа:\n{_pretty_json(document.diagnostics)}",
        ])
        self.details.setPlainText("\n\n".join(parts))


def open_core_description_view(session, well, parent=None):
    """Retain the window through its parent; each opening reflects the selected well."""
    dialog = CoreDescriptionViewDialog(session, well, parent)
    dialog.setAttribute(QtCore.Qt.WA_DeleteOnClose)
    dialog.show()
    dialog.raise_()
    dialog.activateWindow()
    return dialog
