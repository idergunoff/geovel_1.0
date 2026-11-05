"""Add words and phrases from a saved description without editing regexes."""

from PyQt5 import QtCore, QtWidgets

from core_description.dictionary_learning import (
    LEARNING_CATEGORIES, build_dictionary_draft, literal_rules,
)
from core_description.semantics import (
    DictionaryValidationError, analyze_description, load_dictionary_payload, save_dictionary,
    validate_dictionary,
)
from core_description.terminology import ROCK_LABELS, SATURATION_LABELS, canonical_term, display_term
from qt.core_description_dictionary import CATEGORY_LABELS, VALUE_LABELS
from qt.core_description_import import DescriptionDelegate


def result_text(text, payload):
    rocks, oil, confidence, matches, warnings = analyze_description(text, validate_dictionary(payload))
    names = ", ".join(dict.fromkeys(display_term(rock.canonical_value, ROCK_LABELS) for rock in rocks)) or "не определены"
    summary = (f"Породы: {names}. Нефтенасыщенность: {display_term(oil, SATURATION_LABELS)}. "
               f"Уверенность: {confidence:g}.")
    if warnings:
        labels = {"locality_marker": "локальный признак", "uncertain_wording": "неопределённая формулировка"}
        summary += " Проверьте: " + ", ".join(labels.get(w, w) for w in warnings)
    return summary, DescriptionDelegate.highlighted_matches(text, ((m.category, m.start, m.end) for m in matches))


class LearnDescriptionDialog(QtWidgets.QDialog):
    def __init__(self, expression, source_text, *, category="rock", source_label="", parent=None):
        super().__init__(parent)
        self.payload = load_dictionary_payload()
        self.original_payload = self.payload
        self.source_text = source_text
        self.rule_id = None
        self.saved_payload = None
        self.draft = None
        self.setWindowTitle("Добавить выражение из описания керна")
        self.resize(850, 720)
        layout = QtWidgets.QVBoxLayout(self)
        self.notice = QtWidgets.QLabel("Выражение будет использоваться для всех новых разборов на этом компьютере.")
        self.notice.setWordWrap(True)
        layout.addWidget(self.notice)
        form = QtWidgets.QFormLayout()
        self.category_combo = QtWidgets.QComboBox()
        for code in LEARNING_CATEGORIES:
            self.category_combo.addItem(CATEGORY_LABELS[code], code)
        self.category_combo.setCurrentIndex(self.category_combo.findData(category))
        form.addRow("Категория:", self.category_combo)
        self.value_combo = QtWidgets.QComboBox()
        form.addRow("Распознавать как:", self.value_combo)
        layout.addLayout(form)
        layout.addWidget(QtWidgets.QLabel("Слово или фраза и варианты написания — каждый с новой строки:"))
        self.patterns_edit = QtWidgets.QPlainTextEdit(expression)
        self.patterns_edit.setMaximumHeight(100)
        layout.addWidget(self.patterns_edit)
        self.before_label = QtWidgets.QLabel()
        self.before_label.setTextFormat(QtCore.Qt.PlainText)
        self.before_label.setWordWrap(True)
        self.before_label.setText("Сейчас: " + result_text(source_text, self.payload)[0])
        layout.addWidget(self.before_label)
        self.after_label = QtWidgets.QLabel()
        self.after_label.setTextFormat(QtCore.Qt.PlainText)
        self.after_label.setWordWrap(True)
        layout.addWidget(self.after_label)
        self.preview = QtWidgets.QTextBrowser()
        self.preview.setOpenExternalLinks(False)
        layout.addWidget(self.preview, 1)
        self.error_label = QtWidgets.QLabel()
        self.error_label.setTextFormat(QtCore.Qt.PlainText)
        self.error_label.setWordWrap(True)
        layout.addWidget(self.error_label)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Cancel)
        buttons.button(QtWidgets.QDialogButtonBox.Cancel).setText("Отмена")
        self.save_button = buttons.addButton("Сохранить и сравнить описания скважины", QtWidgets.QDialogButtonBox.AcceptRole)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.category_combo.currentIndexChanged.connect(self._category_changed)
        self.value_combo.currentTextChanged.connect(self.update_preview)
        self.patterns_edit.textChanged.connect(self.update_preview)
        found = [rule for rule in literal_rules(self.payload, expression) if rule["category"] in LEARNING_CATEGORIES]
        if found:
            rule = found[0]
            self.rule_id = rule["id"]
            self.category_combo.setCurrentIndex(self.category_combo.findData(rule["category"]))
            self.patterns_edit.setPlainText("\n".join(rule["patterns"]))
            self.notice.setText("Выражение уже есть в словаре. Открыта существующая запись; её варианты сохранены.")
            self._category_changed()
            self.value_combo.setCurrentText(display_term(rule["canonical_value"], VALUE_LABELS[rule["category"]]))
        else:
            self._category_changed()
            if category == "rock":
                self.value_combo.setEditText("")
        self.source_label = source_label
        self.update_preview()

    def _category_changed(self):
        category = self.category_combo.currentData()
        self.value_combo.blockSignals(True)
        self.value_combo.clear()
        self.value_combo.setEditable(category == "rock")
        labels = VALUE_LABELS[category]
        for code, label in labels.items():
            if category != "oil_saturation" or code not in {"none", "unknown", "present", "uncertain"}:
                self.value_combo.addItem(label, code)
        if category == "rock":
            for rule in self.payload["rules"]:
                value = rule["canonical_value"]
                if rule["category"] == "rock" and value not in labels and self.value_combo.findText(value) < 0:
                    self.value_combo.addItem(value, value)
        self.value_combo.blockSignals(False)
        self.update_preview()

    def update_preview(self, *_args):
        try:
            category = self.category_combo.currentData()
            value = canonical_term(self.value_combo.currentText(), VALUE_LABELS[category])
            self.draft = build_dictionary_draft(
                self.payload, category=category, canonical_value=value,
                expressions=self.patterns_edit.toPlainText().splitlines(), rule_id=self.rule_id)
            summary, html = result_text(self.source_text, self.draft)
            self.after_label.setText("После сохранения: " + summary)
            self.preview.setHtml(html)
            self.error_label.clear()
            self.save_button.setEnabled(True)
        except DictionaryValidationError as error:
            self.draft = None
            self.after_label.clear()
            self.preview.setPlainText(self.source_text)
            self.error_label.setText(str(error))
            self.save_button.setEnabled(False)

    def accept(self):
        self.update_preview()
        if self.draft is None:
            return
        try:
            if load_dictionary_payload() != self.original_payload:
                raise DictionaryValidationError("Словарь изменён в другом окне. Откройте форму заново.")
            if self.rule_id is None and self.source_label:
                self.draft["rules"][-1]["comment"] = "Добавлено из описания керна: " + self.source_label
            self.saved_payload = save_dictionary(self.draft)
        except (DictionaryValidationError, OSError) as error:
            self.error_label.setText(f"Не удалось сохранить словарь: {error}")
            return
        super().accept()
