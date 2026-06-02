from __future__ import annotations

import copy
import json
from typing import Any, Callable

from PyQt5 import QtCore, QtGui, QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from cluster.well_feature_library import (
    feature_library_entry_mode,
    feature_name_conflict_message,
    filter_feature_library_features,
    normalized_feature_name,
)
from cluster.well_feature_calculator import (
    FEATURE_CALCULATOR_DEFAULT_OUTLIER_POLICY,
    FEATURE_CALCULATOR_DEPTH_GRID_POLICY,
    FEATURE_CALCULATOR_INVALID_MATH_POLICY,
    FEATURE_CALCULATOR_NORMALIZATION_SCOPES,
    FEATURE_CALCULATOR_UNARY_OPERATIONS,
    build_feature_calculator_coverage_report,
    evaluate_feature_calculator_for_well,
    extract_formula_input_names,
    feature_calculator_recommendation,
    parse_feature_calculator_config,
    parse_safe_formula,
    validate_feature_calculator_config,
    validate_feature_calculator_for_dataset,
)
from models_db.model import session
from models_db.model_cluster import (
    CanonicalWellLog,
    ClusterWellLogParameterFromCalculator,
    FeatureCalculator,
    WellLogClusterDatasetData,
)


ROLLING_OPERATIONS = {"rolling_mean", "rolling_median"}
NORMALIZATION_OPERATIONS = {"zscore", "robust", "minmax"}
UNARY_OPERATION_LABELS = [
    "log",
    "abs",
    "sqrt",
    "derivative",
    "rolling_mean",
    "rolling_median",
    "zscore",
    "robust",
    "minmax",
]


class WellLogFeatureConstructorDialog(QtWidgets.QDialog):
    """Well Log calculated-feature constructor for MVP operation-mode features."""

    def __init__(
        self,
        dataset_id: int,
        dataset_name: str,
        wells: list[dict[str, Any]] | None = None,
        canonical_parameters: list[dict[str, Any]] | None = None,
        calculator_parameters: list[dict[str, Any]] | None = None,
        global_features: list[dict[str, Any]] | None = None,
        all_canonical_parameters: list[dict[str, Any]] | None = None,
        refresh_dataset_callback: Callable[[int], None] | None = None,
        collect_state_callback: Callable[[bool], None] | None = None,
        info_callback: Callable[[str, str], None] | None = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.dataset_id = int(dataset_id)
        self.dataset_name = dataset_name
        self.wells = wells or []
        self.canonical_parameters = canonical_parameters or []
        self.calculator_parameters = calculator_parameters or []
        self.global_features = global_features or []
        self.all_canonical_parameters = all_canonical_parameters or self._load_all_canonical_parameters()
        self.refresh_dataset_callback = refresh_dataset_callback
        self.collect_state_callback = collect_state_callback
        self.info_callback = info_callback
        self._selected_feature_id: int | None = None
        self._coverage_rows: list[Any] = []

        self.setObjectName("WellLogFeatureConstructorDialog")
        self.setWindowTitle("Конструктор признаков Well Log")
        self.resize(1280, 860)
        self._setup_ui()
        self._fill_dataset_context()
        self._fill_input_curves()
        self._setup_tooltips()
        self._wire_state_signals()
        self._refresh_global_features()
        self._update_operation_fields()
        self._update_mode_fields()
        self._refresh_action_states()

    def _setup_ui(self) -> None:
        main_layout = QtWidgets.QVBoxLayout(self)

        self.label_dataset_title = QtWidgets.QLabel(self)
        self.label_dataset_title.setObjectName("label_well_log_constructor_dataset_title")
        title_font = self.label_dataset_title.font()
        title_font.setPointSize(title_font.pointSize() + 2)
        title_font.setBold(True)
        self.label_dataset_title.setFont(title_font)
        main_layout.addWidget(self.label_dataset_title)

        self.label_context_summary = QtWidgets.QLabel(self)
        self.label_context_summary.setObjectName("label_well_log_constructor_context_summary")
        self.label_context_summary.setWordWrap(True)
        main_layout.addWidget(self.label_context_summary)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal, self)
        main_layout.addWidget(splitter, stretch=1)

        left_widget = QtWidgets.QWidget(splitter)
        left_layout = QtWidgets.QVBoxLayout(left_widget)

        context_group = QtWidgets.QGroupBox("Контекст набора данных", left_widget)
        context_layout = QtWidgets.QVBoxLayout(context_group)
        self.tree_dataset_context = QtWidgets.QTreeWidget(context_group)
        self.tree_dataset_context.setObjectName("treeWidget_well_log_constructor_dataset_context")
        self.tree_dataset_context.setHeaderLabels(["Контекст набора", "Значение"])
        self.tree_dataset_context.header().setStretchLastSection(True)
        context_layout.addWidget(self.tree_dataset_context)
        left_layout.addWidget(context_group, stretch=1)

        self.create_group = QtWidgets.QGroupBox("Создать расчётный признак", left_widget)
        create_group = self.create_group
        create_layout = QtWidgets.QFormLayout(create_group)

        self.lineEdit_feature_name = QtWidgets.QLineEdit(create_group)
        self.lineEdit_feature_name.setObjectName("lineEdit_well_log_constructor_feature_name")
        create_layout.addRow("Имя признака", self.lineEdit_feature_name)

        self.comboBox_mode = QtWidgets.QComboBox(create_group)
        self.comboBox_mode.setObjectName("comboBox_well_log_constructor_mode")
        self.comboBox_mode.addItem("Operation", "operation")
        self.comboBox_mode.addItem("Formula", "formula")
        self.comboBox_mode.currentIndexChanged.connect(self._update_mode_fields)
        create_layout.addRow("Режим", self.comboBox_mode)

        self.textEdit_formula_expression = QtWidgets.QPlainTextEdit(create_group)
        self.textEdit_formula_expression.setObjectName("plainTextEdit_well_log_constructor_formula_expression")
        self.textEdit_formula_expression.setPlaceholderText("Example: (GR - RHOB) / (GR + RHOB)")
        self.textEdit_formula_expression.setMaximumHeight(74)
        create_layout.addRow("Выражение", self.textEdit_formula_expression)

        self.listWidget_formula_curves = QtWidgets.QListWidget(create_group)
        self.listWidget_formula_curves.setObjectName("listWidget_well_log_constructor_formula_curves")
        self.listWidget_formula_curves.setMaximumHeight(95)
        self.listWidget_formula_curves.itemDoubleClicked.connect(lambda item: self._insert_formula_token(item.text()))
        create_layout.addRow("Доступные кривые", self.listWidget_formula_curves)

        self.widget_formula_tokens = QtWidgets.QWidget(create_group)
        self.widget_formula_tokens.setObjectName("widget_well_log_constructor_formula_tokens")
        formula_buttons_layout = QtWidgets.QGridLayout(self.widget_formula_tokens)
        formula_buttons_layout.setContentsMargins(0, 0, 0, 0)
        self._formula_token_buttons: list[QtWidgets.QPushButton] = []
        for index, token in enumerate(["+", "-", "*", "/", "(", ")", "log", "abs", "sqrt"]):
            button = QtWidgets.QPushButton(token, self.widget_formula_tokens)
            button.setObjectName(f"pushButton_well_log_constructor_formula_token_{token.replace('/', 'div')}")
            button.clicked.connect(lambda _checked=False, value=token: self._insert_formula_token(value))
            self._formula_token_buttons.append(button)
            formula_buttons_layout.addWidget(button, index // 5, index % 5)
        create_layout.addRow("Операторы формулы", self.widget_formula_tokens)

        self.label_formula_inputs = QtWidgets.QLabel(create_group)
        self.label_formula_inputs.setObjectName("label_well_log_constructor_formula_inputs")
        self.label_formula_inputs.setWordWrap(True)
        create_layout.addRow("Использованные кривые", self.label_formula_inputs)

        self.comboBox_input_curve = QtWidgets.QComboBox(create_group)
        self.comboBox_input_curve.setObjectName("comboBox_well_log_constructor_input_curve")
        create_layout.addRow("Входная кривая", self.comboBox_input_curve)

        self.comboBox_operation = QtWidgets.QComboBox(create_group)
        self.comboBox_operation.setObjectName("comboBox_well_log_constructor_operation")
        self.comboBox_operation.addItem("", "")
        for operation in UNARY_OPERATION_LABELS:
            self.comboBox_operation.addItem(operation, operation)
        self.comboBox_operation.currentIndexChanged.connect(self._update_operation_fields)
        create_layout.addRow("Операция", self.comboBox_operation)

        self.comboBox_normalization_scope = QtWidgets.QComboBox(create_group)
        self.comboBox_normalization_scope.setObjectName("comboBox_well_log_constructor_normalization_scope")
        for scope in sorted(FEATURE_CALCULATOR_NORMALIZATION_SCOPES):
            self.comboBox_normalization_scope.addItem(scope, scope)
        default_scope_index = self.comboBox_normalization_scope.findData("whole_well")
        if default_scope_index >= 0:
            self.comboBox_normalization_scope.setCurrentIndex(default_scope_index)
        create_layout.addRow("Область нормировки", self.comboBox_normalization_scope)

        self.spinBox_rolling_window = QtWidgets.QSpinBox(create_group)
        self.spinBox_rolling_window.setObjectName("spinBox_well_log_constructor_rolling_window")
        self.spinBox_rolling_window.setRange(1, 1000000)
        self.spinBox_rolling_window.setValue(3)
        create_layout.addRow("Скользящее окно", self.spinBox_rolling_window)

        self.spinBox_min_periods = QtWidgets.QSpinBox(create_group)
        self.spinBox_min_periods.setObjectName("spinBox_well_log_constructor_min_periods")
        self.spinBox_min_periods.setRange(1, 1000000)
        self.spinBox_min_periods.setValue(1)
        create_layout.addRow("Мин. периодов", self.spinBox_min_periods)

        self.comboBox_outlier_policy = QtWidgets.QComboBox(create_group)
        self.comboBox_outlier_policy.setObjectName("comboBox_well_log_constructor_outlier_policy")
        self.comboBox_outlier_policy.addItem("none", "none")
        create_layout.addRow("Политика выбросов", self.comboBox_outlier_policy)

        validation_buttons_layout = QtWidgets.QHBoxLayout()
        self.pushButton_validate = QtWidgets.QPushButton("Проверить", create_group)
        self.pushButton_validate.setObjectName("pushButton_well_log_constructor_validate")
        self.pushButton_validate.clicked.connect(self._validate_form_clicked)
        validation_buttons_layout.addWidget(self.pushButton_validate)
        self.pushButton_save = QtWidgets.QPushButton("Сохранить", create_group)
        self.pushButton_save.setObjectName("pushButton_well_log_constructor_save")
        self.pushButton_save.clicked.connect(self._save_feature_clicked)
        validation_buttons_layout.addWidget(self.pushButton_save)
        self.pushButton_preview = QtWidgets.QPushButton("Предпросмотр", create_group)
        self.pushButton_preview.setObjectName("pushButton_well_log_constructor_preview")
        self.pushButton_preview.clicked.connect(self._preview_current_definition)
        validation_buttons_layout.addWidget(self.pushButton_preview)
        create_layout.addRow(validation_buttons_layout)

        self.textEdit_validation = QtWidgets.QTextEdit(create_group)
        self.textEdit_validation.setObjectName("textEdit_well_log_constructor_validation")
        self.textEdit_validation.setReadOnly(True)
        self.textEdit_validation.setMaximumHeight(90)
        create_layout.addRow("Ошибки проверки", self.textEdit_validation)
        left_layout.addWidget(create_group)

        right_widget = QtWidgets.QWidget(splitter)
        right_layout = QtWidgets.QVBoxLayout(right_widget)

        features_group = QtWidgets.QGroupBox("Глобальные расчётные признаки", right_widget)
        features_layout = QtWidgets.QVBoxLayout(features_group)

        filters_layout = QtWidgets.QGridLayout()
        filters_layout.addWidget(QtWidgets.QLabel("Поиск", features_group), 0, 0)
        self.lineEdit_library_search = QtWidgets.QLineEdit(features_group)
        self.lineEdit_library_search.setObjectName("lineEdit_well_log_constructor_library_search")
        self.lineEdit_library_search.setPlaceholderText("Feature name or expression")
        self.lineEdit_library_search.textChanged.connect(self._apply_feature_library_filters)
        filters_layout.addWidget(self.lineEdit_library_search, 0, 1)

        filters_layout.addWidget(QtWidgets.QLabel("Тип", features_group), 0, 2)
        self.comboBox_library_type_filter = QtWidgets.QComboBox(features_group)
        self.comboBox_library_type_filter.setObjectName("comboBox_well_log_constructor_library_type_filter")
        self.comboBox_library_type_filter.addItem("All", "all")
        self.comboBox_library_type_filter.addItem("Operation", "operation")
        self.comboBox_library_type_filter.addItem("Formula", "formula")
        self.comboBox_library_type_filter.currentIndexChanged.connect(self._apply_feature_library_filters)
        filters_layout.addWidget(self.comboBox_library_type_filter, 0, 3)

        filters_layout.addWidget(QtWidgets.QLabel("Входная кривая", features_group), 1, 0)
        self.comboBox_library_canonical_filter = QtWidgets.QComboBox(features_group)
        self.comboBox_library_canonical_filter.setObjectName("comboBox_well_log_constructor_library_canonical_filter")
        self.comboBox_library_canonical_filter.currentIndexChanged.connect(self._apply_feature_library_filters)
        filters_layout.addWidget(self.comboBox_library_canonical_filter, 1, 1)

        self.checkBox_library_used_current_dataset = QtWidgets.QCheckBox("Используется в текущем наборе", features_group)
        self.checkBox_library_used_current_dataset.setObjectName("checkBox_well_log_constructor_library_used_current_dataset")
        self.checkBox_library_used_current_dataset.stateChanged.connect(self._apply_feature_library_filters)
        filters_layout.addWidget(self.checkBox_library_used_current_dataset, 1, 2, 1, 2)
        features_layout.addLayout(filters_layout)

        self.table_global_features = QtWidgets.QTableWidget(features_group)
        self.table_global_features.setObjectName("tableWidget_well_log_constructor_global_features")
        self.table_global_features.setColumnCount(7)
        self.table_global_features.setHorizontalHeaderLabels(["ID", "Feature", "Type", "Expression", "Inputs", "In dataset", "Used count"])
        self.table_global_features.horizontalHeader().setStretchLastSection(True)
        self.table_global_features.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table_global_features.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table_global_features.itemSelectionChanged.connect(self._feature_selection_changed)
        features_layout.addWidget(self.table_global_features)
        right_layout.addWidget(features_group, stretch=2)

        definition_group = QtWidgets.QGroupBox("Определение выбранного признака", right_widget)
        definition_layout = QtWidgets.QVBoxLayout(definition_group)
        self.textEdit_definition = QtWidgets.QTextEdit(definition_group)
        self.textEdit_definition.setObjectName("textEdit_well_log_constructor_definition")
        self.textEdit_definition.setReadOnly(True)
        definition_layout.addWidget(self.textEdit_definition)
        actions_layout = QtWidgets.QHBoxLayout()
        self.pushButton_add_to_dataset = QtWidgets.QPushButton("ADD TO DATASET", definition_group)
        self.pushButton_add_to_dataset.setObjectName("pushButton_well_log_constructor_add_to_dataset")
        self.pushButton_add_to_dataset.clicked.connect(self._add_selected_feature_to_dataset)
        actions_layout.addWidget(self.pushButton_add_to_dataset)
        self.pushButton_remove_from_dataset = QtWidgets.QPushButton("REMOVE FROM DATASET", definition_group)
        self.pushButton_remove_from_dataset.setObjectName("pushButton_well_log_constructor_remove_from_dataset")
        self.pushButton_remove_from_dataset.clicked.connect(self._remove_selected_feature_from_dataset)
        actions_layout.addWidget(self.pushButton_remove_from_dataset)
        self.pushButton_duplicate_as_new = QtWidgets.QPushButton("DUPLICATE AS NEW", definition_group)
        self.pushButton_duplicate_as_new.setObjectName("pushButton_well_log_constructor_duplicate_as_new")
        self.pushButton_duplicate_as_new.clicked.connect(self._duplicate_selected_feature_as_new)
        actions_layout.addWidget(self.pushButton_duplicate_as_new)
        self.pushButton_delete_global = QtWidgets.QPushButton("DELETE GLOBAL FEATURE", definition_group)
        self.pushButton_delete_global.setObjectName("pushButton_well_log_constructor_delete_global_feature")
        self.pushButton_delete_global.clicked.connect(self._delete_selected_global_feature)
        actions_layout.addWidget(self.pushButton_delete_global)
        definition_layout.addLayout(actions_layout)
        right_layout.addWidget(definition_group, stretch=1)

        diagnostics_group = QtWidgets.QGroupBox("Покрытие, диагностика и предпросмотр", right_widget)
        diagnostics_layout = QtWidgets.QVBoxLayout(diagnostics_group)
        self.label_coverage_summary = QtWidgets.QLabel(diagnostics_group)
        self.label_coverage_summary.setObjectName("label_well_log_constructor_coverage_summary")
        self.label_coverage_summary.setWordWrap(True)
        diagnostics_layout.addWidget(self.label_coverage_summary)

        self.table_coverage = QtWidgets.QTableWidget(diagnostics_group)
        self.table_coverage.setObjectName("tableWidget_well_log_constructor_coverage")
        self.table_coverage.setColumnCount(4)
        self.table_coverage.setHorizontalHeaderLabels(["Well", "Status", "Points", "Errors"])
        self.table_coverage.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table_coverage.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table_coverage.itemSelectionChanged.connect(self._coverage_selection_changed)
        diagnostics_layout.addWidget(self.table_coverage, stretch=1)

        self.textEdit_error_detail = QtWidgets.QTextEdit(diagnostics_group)
        self.textEdit_error_detail.setObjectName("textEdit_well_log_constructor_error_detail")
        self.textEdit_error_detail.setReadOnly(True)
        self.textEdit_error_detail.setMaximumHeight(115)
        diagnostics_layout.addWidget(self.textEdit_error_detail)

        preview_controls = QtWidgets.QHBoxLayout()
        preview_controls.addWidget(QtWidgets.QLabel("Скважина для предпросмотра", diagnostics_group))
        self.comboBox_preview_well = QtWidgets.QComboBox(diagnostics_group)
        self.comboBox_preview_well.setObjectName("comboBox_well_log_constructor_preview_well")
        self.comboBox_preview_well.currentIndexChanged.connect(self._preview_selected_feature)
        preview_controls.addWidget(self.comboBox_preview_well, stretch=1)
        self.pushButton_preview_selected = QtWidgets.QPushButton("Предпросмотр выбранного", diagnostics_group)
        self.pushButton_preview_selected.setObjectName("pushButton_well_log_constructor_preview_selected")
        self.pushButton_preview_selected.clicked.connect(self._preview_selected_feature)
        preview_controls.addWidget(self.pushButton_preview_selected)
        diagnostics_layout.addLayout(preview_controls)

        self.preview_figure = Figure(figsize=(4.8, 3.2))
        self.preview_canvas = FigureCanvas(self.preview_figure)
        self.preview_canvas.setObjectName("canvas_well_log_constructor_preview")
        diagnostics_layout.addWidget(self.preview_canvas, stretch=1)
        right_layout.addWidget(diagnostics_group, stretch=3)

        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setSizes([470, 630])

        buttons_layout = QtWidgets.QHBoxLayout()
        buttons_layout.addStretch(1)
        self.pushButton_close = QtWidgets.QPushButton("Закрыть", self)
        self.pushButton_close.setObjectName("pushButton_well_log_constructor_close")
        self.pushButton_close.clicked.connect(self.close)
        buttons_layout.addWidget(self.pushButton_close)
        main_layout.addLayout(buttons_layout)

    def _setup_tooltips(self) -> None:
        constructor_tip = (
            "CONSTR открывает конструктор расчётных признаков Well Log для текущего набора. "
            "Глобальные признаки можно переиспользовать в разных наборах; добавление или удаление признака сбрасывает данные COLLECT."
        )
        self.setToolTip(constructor_tip)
        tooltip_by_widget = {
            self.label_dataset_title: "Текущий набор данных, скважины и параметры которого используются для проверки покрытия.",
            self.tree_dataset_context: "Контекст текущего набора: скважины, канонические кривые и расчётные признаки, уже привязанные к набору.",
            self.lineEdit_feature_name: "Имя глобального признака. Оно не должно совпадать с канонической кривой, другим расчётным признаком или системным столбцом данных.",
            self.comboBox_mode: "Выберите операцию для одной канонической кривой или формулу для нескольких канонических кривых.",
            self.textEdit_formula_expression: "Режим формулы поддерживает имена канонических кривых, числа, + - * /, скобки и функции log/abs/sqrt. eval, импорты и обращение к атрибутам запрещены.",
            self.listWidget_formula_curves: "Имена канонических кривых, доступные в формулах. Дважды щёлкните по кривой, чтобы вставить её в выражение.",
            self.comboBox_input_curve: "Каноническая кривая, которая используется как вход для расчётного признака в режиме операции.",
            self.comboBox_operation: "Операция расчёта. Интерполяция не используется: входные кривые должны сохранять исходную сетку измеренных глубин.",
            self.comboBox_normalization_scope: "Область нормировки: whole_well использует всю кривую скважины, а interval — только интервал текущего набора для этой скважины.",
            self.spinBox_rolling_window: "Размер скользящего окна в отсчётах по глубине для операций rolling_mean и rolling_median.",
            self.spinBox_min_periods: "Минимальное число доступных отсчётов, необходимое внутри скользящего окна.",
            self.comboBox_outlier_policy: "Политика выбросов зарезервирована для будущих расширений; в MVP используется none, значения не обрезаются.",
            self.pushButton_validate: "VALIDATE проверяет текущее несохранённое определение расчётного признака перед сохранением или предпросмотром.",
            self.pushButton_save: "SAVE создаёт новый глобальный признак в feature_calculator. Существующие глобальные признаки в MVP не редактируются.",
            self.pushButton_preview: "PREVIEW рассчитывает несохранённое определение для выбранной скважины без сохранения и без изменения набора данных.",
            self.lineEdit_library_search: "Фильтрует глобальные признаки по имени или выражению.",
            self.comboBox_library_type_filter: "Фильтрует библиотеку глобальных признаков по режиму: операция или формула.",
            self.comboBox_library_canonical_filter: "Фильтрует глобальные признаки по канонической кривой, использованной как вход.",
            self.checkBox_library_used_current_dataset: "Показывает только глобальные признаки, уже привязанные к текущему набору данных.",
            self.table_global_features: "Библиотека глобальных признаков. Один глобальный признак можно переиспользовать в нескольких наборах Well Log.",
            self.textEdit_definition: "Определение выбранного глобального расчётного признака и состояние его использования в наборах данных.",
            self.pushButton_add_to_dataset: "ADD TO DATASET проверяет глубинные сетки и математику для текущего набора. Если проверка успешна, признак появляется в списке параметров как [calc], а данные COLLECT сбрасываются.",
            self.pushButton_remove_from_dataset: "REMOVE FROM DATASET отвязывает выбранный расчётный признак от текущего набора и требует повторно выполнить COLLECT.",
            self.pushButton_duplicate_as_new: "Копирует выбранный неизменяемый глобальный признак в форму создания, чтобы сохранить его под новым именем.",
            self.pushButton_delete_global: "DELETE GLOBAL FEATURE удаляет глобальный признак только если он не используется ни в одном наборе данных.",
            self.label_coverage_summary: "Отчёт покрытия показывает, можно ли добавить выбранный расчётный признак без блокировки COLLECT.",
            self.table_coverage: "Применимость по скважинам: несовместимая глубинная сетка или некорректная математика блокируют ADD TO DATASET и COLLECT.",
            self.textEdit_error_detail: "Подробности выбранной ошибки покрытия или предпросмотра, включая рекомендацию.",
            self.comboBox_preview_well: "Скважина для предпросмотра. Предпросмотр не сохраняет признаки и не имеет побочных эффектов.",
            self.pushButton_preview_selected: "Строит предпросмотр выбранного сохранённого глобального признака для одной скважины.",
        }
        for widget, tooltip in tooltip_by_widget.items():
            widget.setToolTip(tooltip)
        self.pushButton_close.setToolTip("Закрывает конструктор без побочных эффектов; несохранённые правки формы отбрасываются.")

    def _wire_state_signals(self) -> None:
        self.lineEdit_feature_name.textChanged.connect(self._refresh_action_states)
        self.textEdit_formula_expression.textChanged.connect(self._refresh_action_states)
        self.comboBox_mode.currentIndexChanged.connect(self._refresh_action_states)
        self.comboBox_input_curve.currentIndexChanged.connect(self._refresh_action_states)
        self.comboBox_operation.currentIndexChanged.connect(self._refresh_action_states)
        self.comboBox_normalization_scope.currentIndexChanged.connect(self._refresh_action_states)
        self.spinBox_rolling_window.valueChanged.connect(self._refresh_action_states)
        self.spinBox_min_periods.valueChanged.connect(self._refresh_action_states)

    def _form_has_minimum_definition(self) -> bool:
        if not self.lineEdit_feature_name.text().strip():
            return False
        if self._current_mode() == "formula":
            return bool(self.textEdit_formula_expression.toPlainText().strip())
        return bool(self.comboBox_input_curve.currentData()) and bool(self.comboBox_operation.currentData())

    def _selected_feature_linked_to_dataset(self) -> bool:
        if self._selected_feature_id is None:
            return False
        return (
            session.query(ClusterWellLogParameterFromCalculator.id)
            .filter(
                ClusterWellLogParameterFromCalculator.dataset_id == self.dataset_id,
                ClusterWellLogParameterFromCalculator.calculator_id == int(self._selected_feature_id),
            )
            .first()
            is not None
        )

    def _refresh_action_states(self, *_args: Any) -> None:
        has_minimum_definition = self._form_has_minimum_definition()
        has_preview_well = self.comboBox_preview_well.count() > 0 if hasattr(self, "comboBox_preview_well") else bool(self.wells)
        self.pushButton_validate.setEnabled(has_minimum_definition)
        self.pushButton_save.setEnabled(has_minimum_definition)
        self.pushButton_preview.setEnabled(has_minimum_definition and has_preview_well)

        selected_feature = self._selected_feature()
        has_selected_feature = selected_feature is not None
        linked = self._selected_feature_linked_to_dataset() if has_selected_feature else False
        usage_count = self._selected_feature_usage_count(int(selected_feature.id)) if selected_feature is not None else 0
        if hasattr(self, "pushButton_add_to_dataset") and not self.pushButton_add_to_dataset.isEnabled():
            # Coverage diagnostics may keep this disabled for an incompatible selected feature.
            pass
        elif hasattr(self, "pushButton_add_to_dataset"):
            self.pushButton_add_to_dataset.setEnabled(has_selected_feature and not linked)
        if hasattr(self, "pushButton_remove_from_dataset"):
            self.pushButton_remove_from_dataset.setEnabled(has_selected_feature and linked)
        if hasattr(self, "pushButton_delete_global"):
            self.pushButton_delete_global.setEnabled(has_selected_feature and usage_count == 0)
        if hasattr(self, "pushButton_duplicate_as_new"):
            self.pushButton_duplicate_as_new.setEnabled(has_selected_feature)
        if hasattr(self, "pushButton_preview_selected"):
            self.pushButton_preview_selected.setEnabled(has_selected_feature and has_preview_well)

    def _load_all_canonical_parameters(self) -> list[dict[str, Any]]:
        rows = session.query(CanonicalWellLog).order_by(CanonicalWellLog.canonical_name, CanonicalWellLog.id).all()
        return [
            {"canonical_id": int(row.id), "canonical_name": row.canonical_name or f"canonical_id={row.id}"}
            for row in rows
        ]

    def _fill_dataset_context(self) -> None:
        self.label_dataset_title.setText(f"Dataset: {self.dataset_name} (id={self.dataset_id})")
        self.label_context_summary.setText(
            f"Wells: {len(self.wells)}; dataset canonical curves: {len(self.canonical_parameters)}; "
            f"calculator features in dataset: {len(self.calculator_parameters)}."
        )
        self.tree_dataset_context.clear()
        wells_item = QtWidgets.QTreeWidgetItem(["Wells", str(len(self.wells))])
        for well in self.wells:
            QtWidgets.QTreeWidgetItem(
                wells_item,
                [str(well.get("name", well.get("id"))), f"{well.get('top_md', '')} - {well.get('bottom_md', '')}"],
            )
        self.tree_dataset_context.addTopLevelItem(wells_item)

        canonical_item = QtWidgets.QTreeWidgetItem(["Canonical curves in dataset", str(len(self.canonical_parameters))])
        for curve in self.canonical_parameters:
            QtWidgets.QTreeWidgetItem(canonical_item, [str(curve.get("canonical_name")), str(curve.get("canonical_id"))])
        self.tree_dataset_context.addTopLevelItem(canonical_item)

        calculators_item = QtWidgets.QTreeWidgetItem(["Calculator features in dataset", str(len(self.calculator_parameters))])
        for feature in self.calculator_parameters:
            QtWidgets.QTreeWidgetItem(calculators_item, [str(feature.get("feature_name")), str(feature.get("calculator_id"))])
        self.tree_dataset_context.addTopLevelItem(calculators_item)
        self.tree_dataset_context.expandAll()
        if hasattr(self, "comboBox_preview_well"):
            current_well_id = self.comboBox_preview_well.currentData()
            self.comboBox_preview_well.blockSignals(True)
            self.comboBox_preview_well.clear()
            for well in self.wells:
                label = f"{well.get('name', well.get('id'))} [{well.get('top_md', '')} - {well.get('bottom_md', '')}]"
                self.comboBox_preview_well.addItem(str(label), int(well.get("id")))
            if current_well_id is not None:
                index = self.comboBox_preview_well.findData(current_well_id)
                if index >= 0:
                    self.comboBox_preview_well.setCurrentIndex(index)
            self.comboBox_preview_well.blockSignals(False)

    def _fill_input_curves(self) -> None:
        self.comboBox_input_curve.clear()
        self.comboBox_input_curve.addItem("", None)
        self.listWidget_formula_curves.clear()
        if hasattr(self, "comboBox_library_canonical_filter"):
            self.comboBox_library_canonical_filter.blockSignals(True)
            current_filter = self.comboBox_library_canonical_filter.currentData()
            self.comboBox_library_canonical_filter.clear()
            self.comboBox_library_canonical_filter.addItem("All", "all")
        for curve in self.all_canonical_parameters:
            canonical_id = curve.get("canonical_id")
            canonical_name = curve.get("canonical_name") or f"canonical_id={canonical_id}"
            curve_payload = {"canonical_id": canonical_id, "canonical_name": canonical_name}
            self.comboBox_input_curve.addItem(str(canonical_name), curve_payload)
            item = QtWidgets.QListWidgetItem(str(canonical_name))
            item.setData(QtCore.Qt.UserRole, curve_payload)
            self.listWidget_formula_curves.addItem(item)
            if hasattr(self, "comboBox_library_canonical_filter"):
                self.comboBox_library_canonical_filter.addItem(str(canonical_name), str(canonical_name))
        if hasattr(self, "comboBox_library_canonical_filter"):
            if current_filter is not None:
                index = self.comboBox_library_canonical_filter.findData(current_filter)
                if index >= 0:
                    self.comboBox_library_canonical_filter.setCurrentIndex(index)
            self.comboBox_library_canonical_filter.blockSignals(False)

    def _refresh_global_features(self, select_feature_id: int | None = None) -> None:
        rows = session.query(FeatureCalculator).order_by(FeatureCalculator.feature_name, FeatureCalculator.id).all()
        links = session.query(ClusterWellLogParameterFromCalculator).all()
        linked_ids = {
            int(row.calculator_id)
            for row in links
            if int(row.dataset_id) == self.dataset_id
        }
        usage_counts: dict[int, int] = {}
        for link in links:
            usage_counts[int(link.calculator_id)] = usage_counts.get(int(link.calculator_id), 0) + 1

        self.global_features = []
        self.calculator_parameters = []
        for row in rows:
            inputs_text = self._format_inputs(row.used_canonical_well_log)
            feature = {
                "id": int(row.id),
                "feature_name": row.feature_name,
                "transform_type": row.transform_type,
                "expression": row.expression,
                "used_canonical_well_log": row.used_canonical_well_log,
                "inputs_text": inputs_text,
                "params_json": row.params_json,
                "created_at": row.created_at,
                "in_dataset": int(row.id) in linked_ids,
                "usage_count": usage_counts.get(int(row.id), 0),
            }
            self.global_features.append(feature)
            if int(row.id) in linked_ids:
                self.calculator_parameters.append({"calculator_id": int(row.id), "feature_name": row.feature_name})

        self._apply_feature_library_filters(select_feature_id=select_feature_id)
        self._fill_dataset_context()

    def _apply_feature_library_filters(self, *_args: Any, select_feature_id: int | None = None) -> None:
        if not hasattr(self, "table_global_features"):
            return
        mode_filter = self.comboBox_library_type_filter.currentData() if hasattr(self, "comboBox_library_type_filter") else "all"
        canonical_filter = self.comboBox_library_canonical_filter.currentData() if hasattr(self, "comboBox_library_canonical_filter") else "all"
        used_current = bool(self.checkBox_library_used_current_dataset.isChecked()) if hasattr(self, "checkBox_library_used_current_dataset") else False
        search_text = self.lineEdit_library_search.text() if hasattr(self, "lineEdit_library_search") else ""
        filtered_features = filter_feature_library_features(
            self.global_features,
            search_text=search_text,
            mode_filter=str(mode_filter or "all"),
            canonical_filter=str(canonical_filter or "all"),
            used_in_current_dataset=used_current,
        )
        self.table_global_features.setRowCount(len(filtered_features))
        selected_row = -1
        for row_index, feature in enumerate(filtered_features):
            feature_mode = feature_library_entry_mode(feature)
            values = [
                feature.get("id"),
                feature.get("feature_name"),
                feature_mode,
                feature.get("expression"),
                feature.get("inputs_text") or self._format_inputs(feature.get("used_canonical_well_log")),
                "yes" if feature.get("in_dataset") else "no",
                int(feature.get("usage_count") or 0),
            ]
            for column, value in enumerate(values):
                item = QtWidgets.QTableWidgetItem("" if value is None else str(value))
                if column == 0:
                    item.setData(QtCore.Qt.UserRole, feature.get("id"))
                self.table_global_features.setItem(row_index, column, item)
            if select_feature_id is not None and int(feature["id"]) == int(select_feature_id):
                selected_row = row_index
            elif select_feature_id is None and self._selected_feature_id is not None and int(feature["id"]) == int(self._selected_feature_id):
                selected_row = row_index
        self.table_global_features.resizeColumnsToContents()
        if selected_row >= 0:
            self.table_global_features.selectRow(selected_row)
        elif filtered_features:
            self.table_global_features.selectRow(0)
        else:
            self._selected_feature_id = None
            self.textEdit_definition.clear()
            self._coverage_rows = []
            if hasattr(self, "table_coverage"):
                self.table_coverage.setRowCount(0)
                self.label_coverage_summary.setText(self._coverage_summary_text(None))
                self.textEdit_error_detail.clear()
            self._refresh_action_states()

    def _format_inputs(self, raw_inputs: Any) -> str:
        try:
            inputs = json.loads(raw_inputs) if isinstance(raw_inputs, str) else raw_inputs
        except Exception:
            return str(raw_inputs or "")
        if isinstance(inputs, list):
            names = []
            for entry in inputs:
                if isinstance(entry, dict):
                    names.append(str(entry.get("canonical_name") or entry.get("canonical_id") or ""))
                else:
                    names.append(str(entry))
            return ", ".join(name for name in names if name)
        return str(raw_inputs or "")

    def _current_mode(self) -> str:
        return str(self.comboBox_mode.currentData() or "operation")

    def _set_form_row_visible(self, widget: QtWidgets.QWidget, visible: bool) -> None:
        label = self.create_group.layout().labelForField(widget)
        if label is not None:
            label.setVisible(visible)
        widget.setVisible(visible)

    def _update_mode_fields(self) -> None:
        is_formula = self._current_mode() == "formula"
        self.create_group.setTitle("Create formula feature" if is_formula else "Create operation feature")
        for widget in (self.textEdit_formula_expression, self.listWidget_formula_curves, self.label_formula_inputs):
            self._set_form_row_visible(widget, is_formula)
        self._set_form_row_visible(self.widget_formula_tokens, is_formula)
        for button in self._formula_token_buttons:
            button.setVisible(is_formula)
        for widget in (
            self.comboBox_input_curve,
            self.comboBox_operation,
            self.spinBox_rolling_window,
            self.spinBox_min_periods,
            self.comboBox_normalization_scope,
        ):
            self._set_form_row_visible(widget, not is_formula)
        self._update_operation_fields()

    def _insert_formula_token(self, token: str) -> None:
        cursor = self.textEdit_formula_expression.textCursor()
        if token in {"log", "abs", "sqrt"}:
            cursor.insertText(f"{token}()")
            cursor.movePosition(QtGui.QTextCursor.Left)
        elif token in {"+", "-", "*", "/"}:
            cursor.insertText(f" {token} ")
        else:
            cursor.insertText(token)
        self.textEdit_formula_expression.setTextCursor(cursor)
        self.textEdit_formula_expression.setFocus()

    def _canonical_by_formula_name(self) -> dict[str, dict[str, Any]]:
        mapping: dict[str, dict[str, Any]] = {}
        for curve in self.all_canonical_parameters:
            name = str(curve.get("canonical_name") or "").strip()
            if name:
                mapping[name] = curve
        return mapping

    def _resolve_formula_inputs(self, expression: str) -> tuple[list[dict[str, Any]], list[str], list[str]]:
        names, parse_errors = extract_formula_input_names(expression)
        errors = [error.message for error in parse_errors]
        canonical_by_name = self._canonical_by_formula_name()
        inputs: list[dict[str, Any]] = []
        for name in names:
            canonical = canonical_by_name.get(name)
            if canonical is None:
                errors.append(f"Unknown canonical curve '{name}'. Use names from Available curves.")
                continue
            inputs.append(
                {
                    "canonical_id": int(canonical["canonical_id"]) if canonical.get("canonical_id") is not None else None,
                    "canonical_name": canonical.get("canonical_name"),
                }
            )
        return inputs, names, errors

    def _build_formula_config(self) -> tuple[dict[str, Any] | None, list[str]]:
        errors: list[str] = []
        feature_name = self.lineEdit_feature_name.text().strip()
        conflict_message = feature_name_conflict_message(
            feature_name,
            existing_feature_names=self._existing_feature_names(),
            canonical_names=self._canonical_names(),
        )
        if conflict_message:
            errors.append(conflict_message)

        expression = self.textEdit_formula_expression.toPlainText().strip()
        if not expression:
            errors.append("Expression is required for formula mode.")

        inputs, names, input_errors = self._resolve_formula_inputs(expression)
        errors.extend(input_errors)
        if expression and not names:
            errors.append("Formula must reference at least one canonical curve.")
        if names and len(inputs) == len(names):
            formula, formula_errors = parse_safe_formula(expression, set(names))
            if formula_errors or formula is None:
                errors.extend(error.message for error in formula_errors)

        outlier_policy = self.comboBox_outlier_policy.currentData() or FEATURE_CALCULATOR_DEFAULT_OUTLIER_POLICY
        if outlier_policy != FEATURE_CALCULATOR_DEFAULT_OUTLIER_POLICY:
            errors.append("Only outlier_policy=none is supported in MVP.")

        self.label_formula_inputs.setText(", ".join(entry.get("canonical_name", "") for entry in inputs) or "—")
        if errors:
            return None, errors

        config: dict[str, Any] = {
            "version": 1,
            "mode": "formula",
            "expression": expression,
            "inputs": inputs,
            "depth_grid_policy": FEATURE_CALCULATOR_DEPTH_GRID_POLICY,
            "invalid_math_policy": FEATURE_CALCULATOR_INVALID_MATH_POLICY,
            "outlier_policy": outlier_policy,
        }
        parsed_config, parse_errors = parse_feature_calculator_config({"feature_name": feature_name, "params_json": config})
        if parse_errors:
            errors.extend(error.message for error in parse_errors)
        if parsed_config is not None:
            errors.extend(error.message for error in validate_feature_calculator_config(parsed_config))
        return (None, errors) if errors else (config, [])

    def _build_current_config(self) -> tuple[dict[str, Any] | None, list[str]]:
        if self._current_mode() == "formula":
            return self._build_formula_config()
        return self._build_operation_config()

    def _update_operation_fields(self) -> None:
        operation = self.comboBox_operation.currentData()
        is_rolling = operation in ROLLING_OPERATIONS
        is_normalization = operation in NORMALIZATION_OPERATIONS
        operation_mode = self._current_mode() == "operation"
        self.spinBox_rolling_window.setEnabled(operation_mode and is_rolling)
        self.spinBox_min_periods.setEnabled(operation_mode and is_rolling)
        self.comboBox_normalization_scope.setEnabled(operation_mode and is_normalization)

    def _existing_feature_names(self) -> set[str]:
        return {str(name).strip().casefold() for (name,) in session.query(FeatureCalculator.feature_name).all() if name}

    def _canonical_names(self) -> set[str]:
        return {str(curve.get("canonical_name", "")).strip().casefold() for curve in self.all_canonical_parameters if curve.get("canonical_name")}

    def _build_operation_config(self) -> tuple[dict[str, Any] | None, list[str]]:
        errors: list[str] = []
        feature_name = self.lineEdit_feature_name.text().strip()
        conflict_message = feature_name_conflict_message(
            feature_name,
            existing_feature_names=self._existing_feature_names(),
            canonical_names=self._canonical_names(),
        )
        if conflict_message:
            errors.append(conflict_message)

        input_curve = self.comboBox_input_curve.currentData()
        if not input_curve:
            errors.append("Input curve is required.")
            input_curve = None

        operation = self.comboBox_operation.currentData()
        if not operation:
            errors.append("Operation is required.")
        elif operation not in FEATURE_CALCULATOR_UNARY_OPERATIONS:
            errors.append(f"Operation '{operation}' is not supported.")

        normalization_scope = self.comboBox_normalization_scope.currentData() or "whole_well"
        if operation in NORMALIZATION_OPERATIONS and normalization_scope not in FEATURE_CALCULATOR_NORMALIZATION_SCOPES:
            errors.append("Normalization scope must be whole_well or interval.")

        window = int(self.spinBox_rolling_window.value())
        min_periods = int(self.spinBox_min_periods.value())
        if operation in ROLLING_OPERATIONS:
            if window <= 0:
                errors.append("Rolling window must be greater than 0.")
            if min_periods <= 0:
                errors.append("Min periods must be greater than 0.")
            if min_periods > window:
                errors.append("Min periods must be less than or equal to rolling window.")

        outlier_policy = self.comboBox_outlier_policy.currentData() or FEATURE_CALCULATOR_DEFAULT_OUTLIER_POLICY
        if outlier_policy != FEATURE_CALCULATOR_DEFAULT_OUTLIER_POLICY:
            errors.append("Only outlier_policy=none is supported in MVP.")

        if errors or not input_curve or not operation:
            return None, errors

        config: dict[str, Any] = {
            "version": 1,
            "mode": "operation",
            "operation": operation,
            "inputs": [
                {
                    "canonical_id": int(input_curve["canonical_id"]) if input_curve.get("canonical_id") is not None else None,
                    "canonical_name": input_curve.get("canonical_name"),
                }
            ],
            "normalization_scope": normalization_scope,
            "depth_grid_policy": FEATURE_CALCULATOR_DEPTH_GRID_POLICY,
            "invalid_math_policy": FEATURE_CALCULATOR_INVALID_MATH_POLICY,
            "outlier_policy": outlier_policy,
        }
        if operation in ROLLING_OPERATIONS:
            config["window"] = window
            config["min_periods"] = min_periods

        parsed_config, parse_errors = parse_feature_calculator_config({"feature_name": feature_name, "params_json": config})
        if parse_errors:
            errors.extend(error.message for error in parse_errors)
        if parsed_config is not None:
            errors.extend(error.message for error in validate_feature_calculator_config(parsed_config))
        return config, errors

    def _validate_form_clicked(self) -> None:
        _config, errors = self._build_current_config()
        if errors:
            self.textEdit_validation.setPlainText("\n".join(errors))
            QtWidgets.QMessageBox.warning(self, "Validate calculated feature", "Исправьте ошибки формы перед сохранением.")
            return
        self.textEdit_validation.setPlainText("Validation OK.")
        QtWidgets.QMessageBox.information(self, "Validate calculated feature", "Calculated feature definition is valid.")

    def _expression_for_config(self, config: dict[str, Any]) -> str:
        operation = str(config.get("operation") or "")
        input_name = str(config.get("inputs", [{}])[0].get("canonical_name") or "X")
        if operation in ROLLING_OPERATIONS:
            return f"{operation}({input_name}, {config.get('window')})"
        return f"{operation}({input_name})"

    def _save_feature_clicked(self) -> None:
        config, errors = self._build_current_config()
        if errors or config is None:
            self.textEdit_validation.setPlainText("\n".join(errors))
            QtWidgets.QMessageBox.warning(self, "SAVE feature", "Невозможно сохранить признак: форма содержит ошибки.")
            return
        feature_name = self.lineEdit_feature_name.text().strip()
        inputs_json = json.dumps(config["inputs"], ensure_ascii=False)
        params_json = json.dumps(config, ensure_ascii=False, sort_keys=True)
        expression = str(config.get("expression") or self._expression_for_config(config))
        feature = FeatureCalculator(
            feature_name=feature_name,
            expression=expression,
            used_canonical_well_log=inputs_json,
            transform_type=str(config.get("operation") or config.get("mode")),
            params_json=params_json,
        )
        try:
            session.add(feature)
            session.commit()
        except Exception as exc:
            session.rollback()
            QtWidgets.QMessageBox.critical(self, "SAVE feature", f"Не удалось сохранить FeatureCalculator: {exc}")
            return
        self.lineEdit_feature_name.clear()
        if self._current_mode() == "formula":
            self.textEdit_formula_expression.clear()
            self.label_formula_inputs.setText("—")
        self.textEdit_validation.setPlainText(f"Saved feature '{feature_name}'.")
        self._refresh_global_features(select_feature_id=int(feature.id))
        self._set_info(f"WELL LOG CONSTR: сохранён {config.get('mode')}-признак {feature_name}.", "green")
        QtWidgets.QMessageBox.information(self, "SAVE feature", f"Feature '{feature_name}' saved.")

    def _feature_selection_changed(self) -> None:
        selected = self.table_global_features.selectedItems()
        if not selected:
            self._selected_feature_id = None
            self.textEdit_definition.clear()
            self._coverage_rows = []
            if hasattr(self, "table_coverage"):
                self.table_coverage.setRowCount(0)
                self.label_coverage_summary.setText(self._coverage_summary_text(None))
                self.textEdit_error_detail.clear()
            self._refresh_action_states()
            return
        row = selected[0].row()
        id_item = self.table_global_features.item(row, 0)
        feature_id = id_item.data(QtCore.Qt.UserRole) if id_item else None
        try:
            self._selected_feature_id = int(feature_id)
        except (TypeError, ValueError):
            self._selected_feature_id = None
        self._show_selected_definition()

    def _selected_feature(self) -> FeatureCalculator | None:
        if self._selected_feature_id is None:
            return None
        return session.query(FeatureCalculator).filter(FeatureCalculator.id == int(self._selected_feature_id)).first()

    def _show_selected_definition(self) -> None:
        feature = self._selected_feature()
        if feature is None:
            self.textEdit_definition.clear()
            self.pushButton_add_to_dataset.setEnabled(False)
            self.pushButton_remove_from_dataset.setEnabled(False)
            self.pushButton_delete_global.setEnabled(False)
            self.pushButton_duplicate_as_new.setEnabled(False)
            self.pushButton_preview_selected.setEnabled(False)
            return
        config, parse_errors = parse_feature_calculator_config(feature)
        linked = (
            session.query(ClusterWellLogParameterFromCalculator.id)
            .filter(
                ClusterWellLogParameterFromCalculator.dataset_id == self.dataset_id,
                ClusterWellLogParameterFromCalculator.calculator_id == int(feature.id),
            )
            .first()
            is not None
        )
        usage_count = self._selected_feature_usage_count(int(feature.id))
        lines = [
            f"ID: {feature.id}",
            f"Name: {feature.feature_name}",
            f"Type: {feature.transform_type}",
            f"Expression: {feature.expression or ''}",
            f"Inputs: {self._format_inputs(feature.used_canonical_well_log)}",
            f"Created at: {feature.created_at}",
            f"Used in current dataset: {'yes' if linked else 'no'}",
            f"Used in any dataset links: {usage_count}",
        ]
        if config is not None:
            lines.extend(
                [
                    f"Mode: {config.mode}",
                    f"Operation: {config.operation or ''}",
                    f"Normalization scope: {config.normalization_scope}",
                    f"Outlier policy: {config.outlier_policy}",
                    "Params JSON:",
                    json.dumps(config.raw_params, ensure_ascii=False, indent=2, sort_keys=True),
                ]
            )
        if parse_errors:
            lines.append("Errors:")
            lines.extend(f"- [{error.code}] {error.message}" for error in parse_errors)
        self.textEdit_definition.setPlainText("\n".join(lines))
        self.pushButton_remove_from_dataset.setEnabled(linked)
        self.pushButton_delete_global.setEnabled(usage_count == 0)
        self.pushButton_duplicate_as_new.setEnabled(True)
        self._refresh_selected_feature_diagnostics(feature, config, parse_errors)

    def _status_error_count_text(self, errors: list[Any]) -> str:
        if not errors:
            return ""
        return "; ".join(f"[{getattr(error, 'code', 'error')}] {getattr(error, 'message', str(error))}" for error in errors[:2])

    def _coverage_summary_text(self, report: Any | None) -> str:
        if report is None:
            return "Выберите сохранённый признак, чтобы рассчитать отчёт покрытия. Предпросмотр также можно построить для несохранённого определения из формы."
        summary = report.summary
        points = "—" if summary.min_points is None else f"{summary.min_points} / {summary.max_points}"
        inputs = ", ".join(summary.input_curves) or "—"
        can_add = "ДА" if summary.can_be_added else "НЕТ"
        return (
            f"Скважин: {summary.wells_total}; OK: {summary.wells_ok}; Ошибок: {summary.error_count}; "
            f"Мин./макс. точек: {points}; Входы: {inputs}; Режим: {summary.mode}; "
            f"Нормировка: {summary.normalization_scope}; Можно добавить в набор: {can_add}"
        )

    def _refresh_selected_feature_diagnostics(self, feature: FeatureCalculator, config: Any | None, parse_errors: list[Any]) -> None:
        self._coverage_rows = []
        self.table_coverage.setRowCount(0)
        self.textEdit_error_detail.clear()
        if parse_errors or config is None:
            self.label_coverage_summary.setText("Покрытие недоступно: в определении выбранного признака есть ошибки разбора.")
            self.pushButton_add_to_dataset.setEnabled(False)
            return
        use_busy_cursor = len(self.wells) >= 50
        if use_busy_cursor:
            self.label_coverage_summary.setText(
                f"Рассчитывается покрытие для {len(self.wells)} скважин. Большие наборы могут обрабатываться несколько секунд..."
            )
            QtWidgets.QApplication.processEvents()
            QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            report = build_feature_calculator_coverage_report(self.dataset_id, config)
        finally:
            if use_busy_cursor:
                QtWidgets.QApplication.restoreOverrideCursor()
        self._coverage_rows = report.rows
        self.label_coverage_summary.setText(self._coverage_summary_text(report))
        self.pushButton_add_to_dataset.setEnabled(report.summary.can_be_added)
        self.table_coverage.setRowCount(len(report.rows))
        for row_index, row in enumerate(report.rows):
            values = [
                row.well_name or f"well_id={row.well_id}",
                row.status,
                row.points,
                self._status_error_count_text(row.errors),
            ]
            for column, value in enumerate(values):
                item = QtWidgets.QTableWidgetItem(str(value))
                item.setData(QtCore.Qt.UserRole, row_index)
                if row.errors:
                    item.setForeground(QtGui.QBrush(QtGui.QColor("#b00020")))
                elif column == 1:
                    item.setForeground(QtGui.QBrush(QtGui.QColor("#0a7f22")))
                self.table_coverage.setItem(row_index, column, item)
        linked = (
            session.query(ClusterWellLogParameterFromCalculator.id)
            .filter(
                ClusterWellLogParameterFromCalculator.dataset_id == self.dataset_id,
                ClusterWellLogParameterFromCalculator.calculator_id == int(feature.id),
            )
            .first()
            is not None
        )
        self.pushButton_add_to_dataset.setEnabled(report.summary.can_be_added and not linked)
        self.table_coverage.resizeColumnsToContents()
        if report.rows:
            self.table_coverage.selectRow(0)
        self._preview_selected_feature()

    def _coverage_selection_changed(self) -> None:
        selected = self.table_coverage.selectedItems()
        if not selected:
            self.textEdit_error_detail.clear()
            return
        row_index = selected[0].data(QtCore.Qt.UserRole)
        try:
            row = self._coverage_rows[int(row_index)]
        except (TypeError, ValueError, IndexError):
            self.textEdit_error_detail.clear()
            return
        lines = [
            f"Скважина: {row.well_name or '—'} (id={row.well_id})",
            f"Интервал глубин: {row.top_md:g} - {row.bottom_md:g}",
            f"Статус: {row.status}",
            f"Точек: {row.points}",
        ]
        if row.errors:
            error = row.errors[0]
            lines.extend([
                f"Признак: {getattr(error, 'feature_name', None) or 'calculator'}",
                f"Входная кривая: {getattr(error, 'canonical_name', None) or '—'}",
                f"Код ошибки: {getattr(error, 'code', 'error')}",
                f"Сообщение: {getattr(error, 'message', str(error))}",
                f"Рекомендация: {feature_calculator_recommendation(error)}",
            ])
        else:
            lines.append(f"Рекомендация: {feature_calculator_recommendation(None)}")
        self.textEdit_error_detail.setPlainText("\n".join(lines))

    def _well_context_by_id(self, well_id: int | None) -> dict[str, Any] | None:
        if well_id is None:
            return None
        for well in self.wells:
            if int(well.get("id")) == int(well_id):
                return well
        return None

    def _preview_config(self, config: Any, *, title: str) -> None:
        well_id = self.comboBox_preview_well.currentData()
        well = self._well_context_by_id(well_id)
        self.preview_figure.clear()
        ax = self.preview_figure.add_subplot(111)
        if well is None:
            ax.text(0.5, 0.5, "Выберите скважину для предпросмотра", ha="center", va="center")
            self.preview_canvas.draw_idle()
            return
        result = evaluate_feature_calculator_for_well(
            config,
            well_id=int(well["id"]),
            top_md=float(well.get("top_md")),
            bottom_md=float(well.get("bottom_md")),
        )
        if result.errors:
            error = result.errors[0]
            ax.text(0.5, 0.5, f"Ошибка предпросмотра:\n[{error.code}] {error.message}", ha="center", va="center", wrap=True)
            ax.set_axis_off()
            self.textEdit_error_detail.setPlainText(
                f"Ошибка предпросмотра для {well.get('name', well.get('id'))}:\n"
                f"Признак: {config.feature_name or title}\n"
                f"Операция/формула: {config.operation or config.expression}\n"
                f"Интервал глубин: {float(well.get('top_md')):g} - {float(well.get('bottom_md')):g}\n"
                f"Код ошибки: {error.code}\n"
                f"Сообщение: {error.message}\n"
                f"Рекомендация: {feature_calculator_recommendation(error)}"
            )
        else:
            points = [(value, depth) for value, depth in zip(result.values, result.depths) if value is not None]
            if points:
                x_values, y_depths = zip(*points)
                ax.plot(x_values, y_depths, linewidth=1.1, marker=".", markersize=3, color="#1f77b4")
                ax.invert_yaxis()
                ax.grid(True, alpha=0.25)
                ax.set_xlabel(config.feature_name or title)
                ax.set_ylabel("Глубина MD")
                ax.set_title(f"{title}: {well.get('name', well.get('id'))}")
            else:
                ax.text(0.5, 0.5, "Предпросмотр не вернул точек", ha="center", va="center")
        self.preview_figure.tight_layout()
        self.preview_canvas.draw_idle()

    def _preview_selected_feature(self, *_args: Any) -> None:
        feature = self._selected_feature()
        if feature is None or not hasattr(self, "preview_canvas"):
            return
        config, parse_errors = parse_feature_calculator_config(feature)
        if parse_errors or config is None:
            self.preview_figure.clear()
            ax = self.preview_figure.add_subplot(111)
            ax.text(0.5, 0.5, "В выбранном признаке есть ошибки разбора", ha="center", va="center")
            ax.set_axis_off()
            self.preview_canvas.draw_idle()
            return
        self._preview_config(config, title=str(feature.feature_name))

    def _preview_current_definition(self) -> None:
        config_dict, errors = self._build_current_config()
        if errors or config_dict is None:
            self.textEdit_validation.setPlainText("\n".join(errors))
            QtWidgets.QMessageBox.warning(self, "Preview calculated feature", "Исправьте ошибки формы перед предпросмотром.")
            return
        feature_name = self.lineEdit_feature_name.text().strip() or "Признак предпросмотра"
        config, parse_errors = parse_feature_calculator_config({"feature_name": feature_name, "params_json": config_dict})
        if parse_errors or config is None:
            self.textEdit_validation.setPlainText("\n".join(error.message for error in parse_errors))
            return
        self.textEdit_validation.setPlainText("Предпросмотр рассчитан без сохранения определения признака.")
        self._preview_config(config, title=feature_name)

    def _format_errors(self, errors: list[Any], limit: int = 10) -> str:
        lines = []
        for error in errors[:limit]:
            feature_name = getattr(error, "feature_name", None) or "calculator"
            well_name = getattr(error, "well_name", None) or (f"well_id={getattr(error, 'well_id', None)}" if getattr(error, "well_id", None) else "dataset")
            code = getattr(error, "code", "error")
            message = getattr(error, "message", str(error))
            lines.append(f"{feature_name} / {well_name}: [{code}] {message}")
        if len(errors) > limit:
            lines.append(f"… ещё ошибок: {len(errors) - limit}.")
        return "\n".join(lines)

    def _add_selected_feature_to_dataset(self) -> None:
        feature = self._selected_feature()
        if feature is None:
            QtWidgets.QMessageBox.warning(self, "ADD TO DATASET", "Выберите calculator-признак.")
            return
        existing = (
            session.query(ClusterWellLogParameterFromCalculator)
            .filter(
                ClusterWellLogParameterFromCalculator.dataset_id == self.dataset_id,
                ClusterWellLogParameterFromCalculator.calculator_id == int(feature.id),
            )
            .first()
        )
        if existing is not None:
            QtWidgets.QMessageBox.information(self, "ADD TO DATASET", "Этот признак уже добавлен в текущий dataset.")
            return
        errors = validate_feature_calculator_for_dataset(self.dataset_id, int(feature.id))
        if errors:
            message = self._format_errors(errors)
            self.textEdit_validation.setPlainText(message)
            QtWidgets.QMessageBox.critical(self, "ADD TO DATASET", "Признак нельзя добавить в dataset:\n" + message)
            return
        try:
            session.add(ClusterWellLogParameterFromCalculator(dataset_id=self.dataset_id, calculator_id=int(feature.id)))
            removed_rows = self._invalidate_dataset_data()
            session.commit()
        except Exception as exc:
            session.rollback()
            QtWidgets.QMessageBox.critical(self, "ADD TO DATASET", f"Не удалось добавить связь с dataset: {exc}")
            return
        self._refresh_after_dataset_change(select_feature_id=int(feature.id))
        self._set_info(
            f"WELL LOG CONSTR: признак {feature.feature_name} добавлен в dataset id={self.dataset_id}; удалено строк data {removed_rows}.",
            "green",
        )
        QtWidgets.QMessageBox.information(self, "ADD TO DATASET", f"Feature '{feature.feature_name}' added to dataset.")

    def _remove_selected_feature_from_dataset(self) -> None:
        feature = self._selected_feature()
        if feature is None:
            QtWidgets.QMessageBox.warning(self, "REMOVE FROM DATASET", "Выберите calculator-признак.")
            return
        removed_links = (
            session.query(ClusterWellLogParameterFromCalculator)
            .filter(
                ClusterWellLogParameterFromCalculator.dataset_id == self.dataset_id,
                ClusterWellLogParameterFromCalculator.calculator_id == int(feature.id),
            )
            .delete(synchronize_session=False)
        )
        if not removed_links:
            session.rollback()
            QtWidgets.QMessageBox.information(self, "REMOVE FROM DATASET", "Этот признак не привязан к текущему dataset.")
            return
        removed_rows = self._invalidate_dataset_data()
        session.commit()
        self._refresh_after_dataset_change(select_feature_id=int(feature.id))
        self._set_info(
            f"WELL LOG CONSTR: признак {feature.feature_name} удалён из dataset id={self.dataset_id}; удалено строк data {removed_rows}.",
            "green",
        )
        QtWidgets.QMessageBox.information(self, "REMOVE FROM DATASET", f"Feature '{feature.feature_name}' removed from dataset.")

    def _selected_feature_usage_count(self, feature_id: int) -> int:
        return int(
            session.query(ClusterWellLogParameterFromCalculator)
            .filter(ClusterWellLogParameterFromCalculator.calculator_id == int(feature_id))
            .count()
            or 0
        )

    def _delete_selected_global_feature(self) -> None:
        feature = self._selected_feature()
        if feature is None:
            QtWidgets.QMessageBox.warning(self, "DELETE GLOBAL FEATURE", "Выберите глобальный calculator-признак.")
            return
        usage_count = self._selected_feature_usage_count(int(feature.id))
        if usage_count:
            QtWidgets.QMessageBox.critical(
                self,
                "DELETE GLOBAL FEATURE",
                "Глобальный признак нельзя удалить, потому что он используется "
                f"в {usage_count} dataset connection(s). Сначала удалите привязки из datasets.",
            )
            return
        confirmation_text = (
            "Удаление глобального признака необратимо.\n\n"
            f"Name: {feature.feature_name}\n"
            f"Expression: {feature.expression or ''}\n"
            f"Created at: {feature.created_at}\n\n"
            "Удалить этот FeatureCalculator?"
        )
        answer = QtWidgets.QMessageBox.question(
            self,
            "DELETE GLOBAL FEATURE",
            confirmation_text,
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No,
        )
        if answer != QtWidgets.QMessageBox.Yes:
            return
        feature_id = int(feature.id)
        feature_name = str(feature.feature_name)
        try:
            session.delete(feature)
            session.commit()
        except Exception as exc:
            session.rollback()
            QtWidgets.QMessageBox.critical(self, "DELETE GLOBAL FEATURE", f"Не удалось удалить FeatureCalculator: {exc}")
            return
        self._selected_feature_id = None
        self._refresh_global_features()
        self._set_info(f"WELL LOG CONSTR: удалён глобальный признак {feature_name} (id={feature_id}).", "green")
        QtWidgets.QMessageBox.information(self, "DELETE GLOBAL FEATURE", f"Feature '{feature_name}' deleted permanently.")

    def _duplicate_selected_feature_as_new(self) -> None:
        feature = self._selected_feature()
        if feature is None:
            QtWidgets.QMessageBox.warning(self, "DUPLICATE AS NEW", "Выберите глобальный calculator-признак.")
            return
        config, parse_errors = parse_feature_calculator_config(feature)
        if parse_errors or config is None:
            self.textEdit_validation.setPlainText("\n".join(error.message for error in parse_errors))
            QtWidgets.QMessageBox.warning(self, "DUPLICATE AS NEW", "Нельзя дублировать признак с ошибками params_json.")
            return

        existing_names = self._existing_feature_names()
        base_name = f"{feature.feature_name}_copy"
        new_name = base_name
        suffix = 2
        while normalized_feature_name(new_name) in existing_names:
            new_name = f"{base_name}_{suffix}"
            suffix += 1

        self.lineEdit_feature_name.setText(new_name)
        if config.mode == "formula":
            index = self.comboBox_mode.findData("formula")
            if index >= 0:
                self.comboBox_mode.setCurrentIndex(index)
            self.textEdit_formula_expression.setPlainText(config.expression or str(feature.expression or ""))
            self.label_formula_inputs.setText(", ".join(inp.canonical_name or str(inp.canonical_id) for inp in config.inputs) or "—")
        else:
            index = self.comboBox_mode.findData("operation")
            if index >= 0:
                self.comboBox_mode.setCurrentIndex(index)
            input_name = config.inputs[0].canonical_name if config.inputs else None
            for combo_index in range(self.comboBox_input_curve.count()):
                payload = self.comboBox_input_curve.itemData(combo_index)
                if isinstance(payload, dict) and str(payload.get("canonical_name")) == str(input_name):
                    self.comboBox_input_curve.setCurrentIndex(combo_index)
                    break
            operation_index = self.comboBox_operation.findData(config.operation)
            if operation_index >= 0:
                self.comboBox_operation.setCurrentIndex(operation_index)
            scope_index = self.comboBox_normalization_scope.findData(config.normalization_scope)
            if scope_index >= 0:
                self.comboBox_normalization_scope.setCurrentIndex(scope_index)
            raw_params = copy.deepcopy(config.raw_params) if isinstance(config.raw_params, dict) else {}
            if "window" in raw_params:
                self.spinBox_rolling_window.setValue(int(raw_params.get("window") or self.spinBox_rolling_window.value()))
            if "min_periods" in raw_params:
                self.spinBox_min_periods.setValue(int(raw_params.get("min_periods") or self.spinBox_min_periods.value()))
            self._update_operation_fields()
        self.textEdit_validation.setPlainText(
            f"Признак '{feature.feature_name}' скопирован в форму как '{new_name}'. "
            "Проверьте параметры и нажмите Save, чтобы создать новый глобальный признак. Исходный признак и наборы данных не изменены."
        )
        self.lineEdit_feature_name.setFocus()

    def _invalidate_dataset_data(self) -> int:
        removed_rows = (
            session.query(WellLogClusterDatasetData)
            .filter(WellLogClusterDatasetData.dataset_id == self.dataset_id)
            .delete(synchronize_session=False)
        )
        if callable(self.collect_state_callback):
            self.collect_state_callback(False)
        return int(removed_rows or 0)

    def _refresh_after_dataset_change(self, select_feature_id: int | None = None) -> None:
        if callable(self.refresh_dataset_callback):
            self.refresh_dataset_callback(self.dataset_id)
        self._refresh_global_features(select_feature_id=select_feature_id)
        self._show_selected_definition()

    def _set_info(self, message: str, color: str) -> None:
        if callable(self.info_callback):
            self.info_callback(message, color)
