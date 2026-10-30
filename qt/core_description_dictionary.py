"""Dictionary editor opened from the core-description importer."""

from __future__ import annotations

from copy import deepcopy
import re
from typing import Any
from uuid import uuid4

from PyQt5 import QtCore, QtWidgets

from core_description.semantics import (
    DictionaryValidationError, analyze_description, load_dictionary_payload,
    save_dictionary, user_dictionary_path, validate_dictionary,
)
from core_description.terminology import (
    RELATION_LABELS, ROCK_LABELS, SATURATION_LABELS, canonical_term, display_term,
)


CATEGORY_LABELS = {
    "rock": "Породы",
    "rock_relation": "Отношения между породами",
    "oil_indicator": "Признаки нефтенасыщенности",
    "oil_saturation": "Степени нефтенасыщенности",
    "negation": "Отрицания признаков нефти",
    "uncertainty": "Неопределённость и локальность",
    "ignore": "Служебные слова",
}
PATTERN_LABELS = {"word": "Слово", "phrase": "Фраза", "regex": "Регулярное выражение"}
VALUE_LABELS = {
    "rock": ROCK_LABELS, "rock_relation": RELATION_LABELS,
    "oil_saturation": SATURATION_LABELS,
    "oil_indicator": {"present": "Признаки нефти"},
    "negation": {"none": "Признаки нефти отсутствуют"},
    "uncertainty": {"uncertain": "Неопределённость", "locality": "Локальность (местами)"},
    "ignore": {"ignore": "Служебное слово"},
}


class DictionaryRuleDialog(QtWidgets.QDialog):
    """Edit a rule and its individual words without losing hidden metadata."""

    def __init__(self, rule: dict[str, Any] | None = None, *, category: str = "rock",
                 parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.rule = deepcopy(rule) if rule is not None else {
            "id": f"user.{uuid4().hex}", "category": category, "canonical_value": "",
            "patterns": [], "pattern_type": "word", "priority": 100, "enabled": True,
        }
        self.setWindowTitle("Изменить слова и правило" if rule is not None else "Добавить слова и правило")
        self.resize(680, 580)
        layout = QtWidgets.QVBoxLayout(self)
        form = QtWidgets.QFormLayout()
        self.category_combo = QtWidgets.QComboBox()
        for code, label in CATEGORY_LABELS.items():
            self.category_combo.addItem(label, code)
        self.category_combo.setCurrentIndex(self.category_combo.findData(self.rule["category"]))
        form.addRow("Раздел:", self.category_combo)
        self.value_combo = QtWidgets.QComboBox()
        self.value_combo.setEditable(True)
        self.value_combo.setInsertPolicy(QtWidgets.QComboBox.NoInsert)
        self.value_combo.setToolTip("Результат распознавания. Можно выбрать значение или ввести своё название.")
        form.addRow("Распознавать как:", self.value_combo)
        self.pattern_type = QtWidgets.QComboBox()
        for code, label in PATTERN_LABELS.items():
            self.pattern_type.addItem(label, code)
        self.pattern_type.setCurrentIndex(self.pattern_type.findData(self.rule["pattern_type"]))
        form.addRow("Тип совпадения:", self.pattern_type)
        layout.addLayout(form)
        layout.addWidget(QtWidgets.QLabel("Слова, словоформы или шаблоны — каждый с новой строки:"))
        self.patterns_edit = QtWidgets.QPlainTextEdit()
        self.patterns_edit.setObjectName("dictionary_patterns")
        self.patterns_edit.setPlainText("\n".join(self.rule["patterns"]))
        self.patterns_edit.setPlaceholderText("Например:\nпесчаник\nпесчаника\nпесчаники")
        layout.addWidget(self.patterns_edit, 1)
        self.pattern_hint = QtWidgets.QLabel()
        self.pattern_hint.setWordWrap(True)
        layout.addWidget(self.pattern_hint)
        self.enabled_check = QtWidgets.QCheckBox("Использовать это правило при распознавании")
        self.enabled_check.setChecked(self.rule["enabled"])
        layout.addWidget(self.enabled_check)
        advanced = QtWidgets.QGroupBox("Дополнительные настройки")
        advanced_form = QtWidgets.QFormLayout(advanced)
        self.priority_spin = QtWidgets.QSpinBox()
        self.priority_spin.setRange(-2147483648, 2147483647)
        self.priority_spin.setValue(self.rule["priority"])
        self.priority_spin.setToolTip("При совпадениях с одной позиции больший приоритет проверяется первым.")
        advanced_form.addRow("Приоритет:", self.priority_spin)
        self.comment_edit = QtWidgets.QLineEdit(self.rule.get("comment", ""))
        advanced_form.addRow("Комментарий:", self.comment_edit)
        layout.addWidget(advanced)
        self.error_label = QtWidgets.QLabel()
        self.error_label.setWordWrap(True)
        layout.addWidget(self.error_label)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.button(QtWidgets.QDialogButtonBox.Ok).setText("Применить")
        buttons.button(QtWidgets.QDialogButtonBox.Cancel).setText("Отмена")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.category_combo.currentIndexChanged.connect(self._category_changed)
        self.pattern_type.currentIndexChanged.connect(self._update_hint)
        self._category_changed()
        self.value_combo.setEditText(display_term(self.rule["canonical_value"], VALUE_LABELS[self.rule["category"]]))
        self._update_hint()

    def _category_changed(self) -> None:
        labels = VALUE_LABELS[self.category_combo.currentData()]
        self.value_combo.clear()
        for code, label in labels.items():
            self.value_combo.addItem(label, code)

    def _update_hint(self) -> None:
        self.pattern_hint.setText(
            "Регулярные выражения проверяются перед сохранением. Учитывайте границы слов в шаблоне."
            if self.pattern_type.currentData() == "regex" else
            "Совпадение целого слова или фразы, без учёта регистра. Добавьте нужные словоформы отдельными строками.")

    def accept(self) -> None:
        rule = deepcopy(self.rule)
        rule.update(
            category=self.category_combo.currentData(),
            canonical_value=canonical_term(self.value_combo.currentText(), VALUE_LABELS[self.category_combo.currentData()]),
            patterns=list(dict.fromkeys(line.strip() for line in self.patterns_edit.toPlainText().splitlines() if line.strip())),
            pattern_type=self.pattern_type.currentData(), priority=self.priority_spin.value(),
            enabled=self.enabled_check.isChecked(),
        )
        if self.comment_edit.text() or "comment" in rule:
            rule["comment"] = self.comment_edit.text()
        if not rule["canonical_value"]:
            self.error_label.setText("Укажите результат распознавания.")
            return
        if not rule["patterns"]:
            self.error_label.setText("Добавьте хотя бы одно слово или шаблон.")
            return
        if rule["pattern_type"] == "regex":
            for pattern in rule["patterns"]:
                try:
                    re.compile(pattern, re.I)
                except re.error as error:
                    self.error_label.setText(f"Неверное регулярное выражение: {error}")
                    return
        try:
            validate_dictionary({"schema_version": 1, "version": "draft", "rules": [rule]})
        except DictionaryValidationError as error:
            self.error_label.setText(f"Проверьте значение и шаблоны: {error}")
            return
        self.rule = rule
        super().accept()


class CoreDescriptionDictionaryDialog(QtWidgets.QDialog):
    """Stage edits in memory and save a complete validated user dictionary."""

    dictionary_saved = QtCore.pyqtSignal(str)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.payload = load_dictionary_payload()
        self._dirty = False
        self.setWindowTitle("Словари парсера описания керна")
        self.setWindowModality(QtCore.Qt.WindowModal)
        self.setWindowFlags(self.windowFlags() | QtCore.Qt.WindowMaximizeButtonHint)
        self.setSizeGripEnabled(True)
        self.resize(1100, 760)
        layout = QtWidgets.QVBoxLayout(self)
        note = QtWidgets.QLabel(
            "Выберите раздел и добавьте или измените слова. Сохранённый словарь применяется при следующей "
            "загрузке документов. Для уже открытых описаний повторите загрузку.")
        note.setWordWrap(True)
        layout.addWidget(note)
        self.version_label = QtWidgets.QLabel()
        layout.addWidget(self.version_label)
        filters = QtWidgets.QHBoxLayout()
        self.category_combo = QtWidgets.QComboBox()
        self.category_combo.addItem("Все разделы", None)
        for code, label in CATEGORY_LABELS.items():
            self.category_combo.addItem(label, code)
        filters.addWidget(self.category_combo)
        self.search_edit = QtWidgets.QLineEdit()
        self.search_edit.setPlaceholderText("Поиск слова или результата распознавания")
        self.search_edit.setClearButtonEnabled(True)
        filters.addWidget(self.search_edit, 1)
        layout.addLayout(filters)
        self.table = QtWidgets.QTableWidget(0, 6)
        self.table.setObjectName("core_dictionary_rules")
        self.table.setHorizontalHeaderLabels(("Раздел", "Распознавать как", "Слова и шаблоны", "Тип", "Включено", "Приоритет"))
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.setWordWrap(False)
        self.table.verticalHeader().hide()
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QtWidgets.QHeaderView.Stretch)
        layout.addWidget(self.table, 1)
        self.count_label = QtWidgets.QLabel()
        layout.addWidget(self.count_label)
        actions = QtWidgets.QHBoxLayout()
        self.add_button = QtWidgets.QPushButton("Добавить…")
        self.edit_button = QtWidgets.QPushButton("Изменить…")
        self.delete_button = QtWidgets.QPushButton("Удалить…")
        for button in (self.add_button, self.edit_button, self.delete_button):
            actions.addWidget(button)
        actions.addStretch()
        layout.addLayout(actions)
        check = QtWidgets.QGroupBox("Пробный разбор с текущими изменениями")
        check_layout = QtWidgets.QVBoxLayout(check)
        self.test_edit = QtWidgets.QPlainTextEdit()
        self.test_edit.setPlaceholderText("Введите фрагмент описания керна для проверки словаря")
        self.test_edit.setMaximumHeight(85)
        check_layout.addWidget(self.test_edit)
        self.test_button = QtWidgets.QPushButton("Проверить текст")
        check_layout.addWidget(self.test_button)
        self.test_result = QtWidgets.QLabel()
        self.test_result.setTextFormat(QtCore.Qt.PlainText)
        self.test_result.setWordWrap(True)
        self.test_result.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        check_layout.addWidget(self.test_result)
        layout.addWidget(check)
        self.status_label = QtWidgets.QLabel()
        self.status_label.setWordWrap(True)
        self.status_label.setTextFormat(QtCore.Qt.PlainText)
        layout.addWidget(self.status_label)
        location = QtWidgets.QLabel(f"Файл пользовательского словаря: {user_dictionary_path()}")
        location.setWordWrap(True)
        location.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        layout.addWidget(location)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
        buttons.button(QtWidgets.QDialogButtonBox.Close).setText("Закрыть")
        self.save_button = buttons.addButton("Сохранить словари", QtWidgets.QDialogButtonBox.ActionRole)
        self.save_button.clicked.connect(self.save)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.category_combo.currentIndexChanged.connect(self._refresh_table)
        self.search_edit.textChanged.connect(self._refresh_table)
        self.table.itemSelectionChanged.connect(self._update_actions)
        self.table.cellDoubleClicked.connect(lambda *_: self.edit_rule())
        self.add_button.clicked.connect(self.add_rule)
        self.edit_button.clicked.connect(self.edit_rule)
        self.delete_button.clicked.connect(self.delete_rule)
        self.test_button.clicked.connect(self.test_text)
        self.test_edit.textChanged.connect(self.test_result.clear)
        self._refresh_table()
        self._update_dirty(False)

    def _selected_id(self) -> str | None:
        row = self.table.currentRow()
        item = self.table.item(row, 0) if row >= 0 else None
        return item.data(QtCore.Qt.UserRole) if item is not None else None

    def _update_actions(self) -> None:
        selected = self._selected_id() is not None
        self.edit_button.setEnabled(selected)
        self.delete_button.setEnabled(selected)

    def _refresh_table(self, *_args: Any, selected_id: str | None = None) -> None:
        selected_id = selected_id or self._selected_id()
        category = self.category_combo.currentData()
        search = self.search_edit.text().strip().casefold()
        self.table.setRowCount(0)
        for rule in self.payload["rules"]:
            value = display_term(rule["canonical_value"], VALUE_LABELS[rule["category"]])
            searchable = " ".join([value, rule["canonical_value"], rule["id"], rule.get("comment", ""), *rule["patterns"]])
            if (category is not None and category != rule["category"]) or (search and search not in searchable.casefold()):
                continue
            row = self.table.rowCount()
            self.table.insertRow(row)
            values = (CATEGORY_LABELS[rule["category"]], value, " · ".join(rule["patterns"]),
                      PATTERN_LABELS[rule["pattern_type"]], "Да" if rule["enabled"] else "Нет", str(rule["priority"]))
            for column, text in enumerate(values):
                item = QtWidgets.QTableWidgetItem(text)
                item.setData(QtCore.Qt.UserRole, rule["id"])
                item.setToolTip("\n".join(rule["patterns"]) if column == 2 else rule.get("comment", ""))
                self.table.setItem(row, column, item)
            if rule["id"] == selected_id:
                self.table.setCurrentCell(row, 0)
        self.count_label.setText(f"Показано правил: {self.table.rowCount()} из {len(self.payload['rules'])}")
        self._update_actions()

    def _update_dirty(self, dirty: bool = True) -> None:
        self._dirty = dirty
        self.save_button.setEnabled(dirty)
        self.version_label.setText(f"Версия словаря: {self.payload['version']}" + (" — есть несохранённые изменения" if dirty else ""))
        self.status_label.clear()
        self.test_result.clear()

    def add_rule(self) -> None:
        editor = DictionaryRuleDialog(category=self.category_combo.currentData() or "rock", parent=self)
        if editor.exec_() == QtWidgets.QDialog.Accepted:
            self.payload["rules"].append(editor.rule)
            self._show_changed_rule(editor.rule)

    def edit_rule(self) -> None:
        selected = self._selected_id()
        for index, rule in enumerate(self.payload["rules"]):
            if rule["id"] == selected:
                editor = DictionaryRuleDialog(rule, parent=self)
                if editor.exec_() == QtWidgets.QDialog.Accepted:
                    self.payload["rules"][index] = editor.rule
                    self._show_changed_rule(editor.rule)
                return

    def _show_changed_rule(self, rule: dict[str, Any]) -> None:
        self._update_dirty()
        self.search_edit.clear()
        if self.category_combo.currentData() is not None:
            self.category_combo.setCurrentIndex(self.category_combo.findData(rule["category"]))
        self._refresh_table(selected_id=rule["id"])

    def delete_rule(self) -> None:
        selected = self._selected_id()
        if selected is None:
            return
        question = QtWidgets.QMessageBox(self)
        question.setWindowTitle("Удалить правило")
        question.setText("Удалить выбранное правило со всеми его словами и шаблонами?")
        question.addButton("Удалить", QtWidgets.QMessageBox.AcceptRole)
        cancel = question.addButton("Отмена", QtWidgets.QMessageBox.RejectRole)
        question.setDefaultButton(cancel)
        question.exec_()
        if question.buttonRole(question.clickedButton()) == QtWidgets.QMessageBox.AcceptRole:
            self.payload["rules"] = [rule for rule in self.payload["rules"] if rule["id"] != selected]
            self._update_dirty()
            self._refresh_table()

    def test_text(self) -> None:
        try:
            dictionary = validate_dictionary(self.payload)
            rocks, oil, _, matches, warnings = analyze_description(self.test_edit.toPlainText(), dictionary)
        except DictionaryValidationError as error:
            self.test_result.setText(f"Ошибка словаря: {error}")
            return
        rock_text = ", ".join(f"{display_term(rock.canonical_value, ROCK_LABELS)} "
                              f"({display_term(rock.relation, RELATION_LABELS)})" for rock in rocks) or "не определены"
        matched = ", ".join(dict.fromkeys(match.matched_text for match in matches)) or "нет"
        warning_labels = {"locality_marker": "локальный признак", "uncertain_wording": "неопределённая формулировка"}
        result = f"Породы: {rock_text}. Нефтенасыщенность: {display_term(oil, SATURATION_LABELS)}.\nСовпадения: {matched}."
        if warnings:
            result += "\nПроверьте: " + ", ".join(warning_labels.get(warning, warning) for warning in warnings)
        self.test_result.setText(result)

    def save(self) -> bool:
        try:
            self.payload = save_dictionary(self.payload)
        except (DictionaryValidationError, OSError) as error:
            self.status_label.setText(f"Не удалось сохранить словари: {error}")
            return False
        self._update_dirty(False)
        self.status_label.setText("Словари сохранены. Новые загрузки документов используют эти изменения.")
        self.dictionary_saved.emit(self.payload["version"])
        return True

    def reject(self) -> None:
        if self._dirty:
            question = QtWidgets.QMessageBox(self)
            question.setWindowTitle("Несохранённые изменения")
            question.setText("Сохранить изменения словарей перед закрытием?")
            save = question.addButton("Сохранить", QtWidgets.QMessageBox.AcceptRole)
            discard = question.addButton("Не сохранять", QtWidgets.QMessageBox.DestructiveRole)
            cancel = question.addButton("Отмена", QtWidgets.QMessageBox.RejectRole)
            question.setDefaultButton(cancel)
            question.exec_()
            clicked = question.clickedButton()
            if clicked is save:
                if not self.save():
                    return
            elif clicked is not discard:
                return
        super().reject()
