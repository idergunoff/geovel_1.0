from __future__ import annotations

import json
from typing import Any, Callable

from PyQt5 import QtCore, QtGui, QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

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
        self.setWindowTitle("Well Log Feature Constructor")
        self.resize(1280, 860)
        self._setup_ui()
        self._fill_dataset_context()
        self._fill_input_curves()
        self._refresh_global_features()
        self._update_operation_fields()
        self._update_mode_fields()

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

        context_group = QtWidgets.QGroupBox("Dataset context", left_widget)
        context_layout = QtWidgets.QVBoxLayout(context_group)
        self.tree_dataset_context = QtWidgets.QTreeWidget(context_group)
        self.tree_dataset_context.setObjectName("treeWidget_well_log_constructor_dataset_context")
        self.tree_dataset_context.setHeaderLabels(["Dataset context", "Value"])
        self.tree_dataset_context.header().setStretchLastSection(True)
        context_layout.addWidget(self.tree_dataset_context)
        left_layout.addWidget(context_group, stretch=1)

        self.create_group = QtWidgets.QGroupBox("Create calculated feature", left_widget)
        create_group = self.create_group
        create_layout = QtWidgets.QFormLayout(create_group)

        self.lineEdit_feature_name = QtWidgets.QLineEdit(create_group)
        self.lineEdit_feature_name.setObjectName("lineEdit_well_log_constructor_feature_name")
        create_layout.addRow("Feature name", self.lineEdit_feature_name)

        self.comboBox_mode = QtWidgets.QComboBox(create_group)
        self.comboBox_mode.setObjectName("comboBox_well_log_constructor_mode")
        self.comboBox_mode.addItem("Operation", "operation")
        self.comboBox_mode.addItem("Formula", "formula")
        self.comboBox_mode.currentIndexChanged.connect(self._update_mode_fields)
        create_layout.addRow("Mode", self.comboBox_mode)

        self.textEdit_formula_expression = QtWidgets.QPlainTextEdit(create_group)
        self.textEdit_formula_expression.setObjectName("plainTextEdit_well_log_constructor_formula_expression")
        self.textEdit_formula_expression.setPlaceholderText("Example: (GR - RHOB) / (GR + RHOB)")
        self.textEdit_formula_expression.setMaximumHeight(74)
        create_layout.addRow("Expression", self.textEdit_formula_expression)

        self.listWidget_formula_curves = QtWidgets.QListWidget(create_group)
        self.listWidget_formula_curves.setObjectName("listWidget_well_log_constructor_formula_curves")
        self.listWidget_formula_curves.setMaximumHeight(95)
        self.listWidget_formula_curves.itemDoubleClicked.connect(lambda item: self._insert_formula_token(item.text()))
        create_layout.addRow("Available curves", self.listWidget_formula_curves)

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
        create_layout.addRow("Formula tokens", self.widget_formula_tokens)

        self.label_formula_inputs = QtWidgets.QLabel(create_group)
        self.label_formula_inputs.setObjectName("label_well_log_constructor_formula_inputs")
        self.label_formula_inputs.setWordWrap(True)
        create_layout.addRow("Used curves", self.label_formula_inputs)

        self.comboBox_input_curve = QtWidgets.QComboBox(create_group)
        self.comboBox_input_curve.setObjectName("comboBox_well_log_constructor_input_curve")
        create_layout.addRow("Input curve", self.comboBox_input_curve)

        self.comboBox_operation = QtWidgets.QComboBox(create_group)
        self.comboBox_operation.setObjectName("comboBox_well_log_constructor_operation")
        self.comboBox_operation.addItem("", "")
        for operation in UNARY_OPERATION_LABELS:
            self.comboBox_operation.addItem(operation, operation)
        self.comboBox_operation.currentIndexChanged.connect(self._update_operation_fields)
        create_layout.addRow("Operation", self.comboBox_operation)

        self.comboBox_normalization_scope = QtWidgets.QComboBox(create_group)
        self.comboBox_normalization_scope.setObjectName("comboBox_well_log_constructor_normalization_scope")
        for scope in sorted(FEATURE_CALCULATOR_NORMALIZATION_SCOPES):
            self.comboBox_normalization_scope.addItem(scope, scope)
        default_scope_index = self.comboBox_normalization_scope.findData("whole_well")
        if default_scope_index >= 0:
            self.comboBox_normalization_scope.setCurrentIndex(default_scope_index)
        create_layout.addRow("Normalization scope", self.comboBox_normalization_scope)

        self.spinBox_rolling_window = QtWidgets.QSpinBox(create_group)
        self.spinBox_rolling_window.setObjectName("spinBox_well_log_constructor_rolling_window")
        self.spinBox_rolling_window.setRange(1, 1000000)
        self.spinBox_rolling_window.setValue(3)
        create_layout.addRow("Rolling window", self.spinBox_rolling_window)

        self.spinBox_min_periods = QtWidgets.QSpinBox(create_group)
        self.spinBox_min_periods.setObjectName("spinBox_well_log_constructor_min_periods")
        self.spinBox_min_periods.setRange(1, 1000000)
        self.spinBox_min_periods.setValue(1)
        create_layout.addRow("Min periods", self.spinBox_min_periods)

        self.comboBox_outlier_policy = QtWidgets.QComboBox(create_group)
        self.comboBox_outlier_policy.setObjectName("comboBox_well_log_constructor_outlier_policy")
        self.comboBox_outlier_policy.addItem("none", "none")
        create_layout.addRow("Outlier policy", self.comboBox_outlier_policy)

        validation_buttons_layout = QtWidgets.QHBoxLayout()
        self.pushButton_validate = QtWidgets.QPushButton("Validate", create_group)
        self.pushButton_validate.setObjectName("pushButton_well_log_constructor_validate")
        self.pushButton_validate.clicked.connect(self._validate_form_clicked)
        validation_buttons_layout.addWidget(self.pushButton_validate)
        self.pushButton_save = QtWidgets.QPushButton("Save", create_group)
        self.pushButton_save.setObjectName("pushButton_well_log_constructor_save")
        self.pushButton_save.clicked.connect(self._save_feature_clicked)
        validation_buttons_layout.addWidget(self.pushButton_save)
        self.pushButton_preview = QtWidgets.QPushButton("Preview", create_group)
        self.pushButton_preview.setObjectName("pushButton_well_log_constructor_preview")
        self.pushButton_preview.clicked.connect(self._preview_current_definition)
        validation_buttons_layout.addWidget(self.pushButton_preview)
        create_layout.addRow(validation_buttons_layout)

        self.textEdit_validation = QtWidgets.QTextEdit(create_group)
        self.textEdit_validation.setObjectName("textEdit_well_log_constructor_validation")
        self.textEdit_validation.setReadOnly(True)
        self.textEdit_validation.setMaximumHeight(90)
        create_layout.addRow("Validation errors", self.textEdit_validation)
        left_layout.addWidget(create_group)

        right_widget = QtWidgets.QWidget(splitter)
        right_layout = QtWidgets.QVBoxLayout(right_widget)

        features_group = QtWidgets.QGroupBox("Global calculator features", right_widget)
        features_layout = QtWidgets.QVBoxLayout(features_group)
        self.table_global_features = QtWidgets.QTableWidget(features_group)
        self.table_global_features.setObjectName("tableWidget_well_log_constructor_global_features")
        self.table_global_features.setColumnCount(6)
        self.table_global_features.setHorizontalHeaderLabels(["ID", "Feature", "Transform", "Expression", "Inputs", "In dataset"])
        self.table_global_features.horizontalHeader().setStretchLastSection(True)
        self.table_global_features.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table_global_features.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table_global_features.itemSelectionChanged.connect(self._feature_selection_changed)
        features_layout.addWidget(self.table_global_features)
        right_layout.addWidget(features_group, stretch=2)

        definition_group = QtWidgets.QGroupBox("Selected feature definition", right_widget)
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
        definition_layout.addLayout(actions_layout)
        right_layout.addWidget(definition_group, stretch=1)

        diagnostics_group = QtWidgets.QGroupBox("Coverage, diagnostics and preview", right_widget)
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
        preview_controls.addWidget(QtWidgets.QLabel("Preview well", diagnostics_group))
        self.comboBox_preview_well = QtWidgets.QComboBox(diagnostics_group)
        self.comboBox_preview_well.setObjectName("comboBox_well_log_constructor_preview_well")
        self.comboBox_preview_well.currentIndexChanged.connect(self._preview_selected_feature)
        preview_controls.addWidget(self.comboBox_preview_well, stretch=1)
        self.pushButton_preview_selected = QtWidgets.QPushButton("Preview selected", diagnostics_group)
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
        self.pushButton_close = QtWidgets.QPushButton("Close", self)
        self.pushButton_close.setObjectName("pushButton_well_log_constructor_close")
        self.pushButton_close.clicked.connect(self.close)
        buttons_layout.addWidget(self.pushButton_close)
        main_layout.addLayout(buttons_layout)

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
        for curve in self.all_canonical_parameters:
            canonical_id = curve.get("canonical_id")
            canonical_name = curve.get("canonical_name") or f"canonical_id={canonical_id}"
            curve_payload = {"canonical_id": canonical_id, "canonical_name": canonical_name}
            self.comboBox_input_curve.addItem(str(canonical_name), curve_payload)
            item = QtWidgets.QListWidgetItem(str(canonical_name))
            item.setData(QtCore.Qt.UserRole, curve_payload)
            self.listWidget_formula_curves.addItem(item)

    def _refresh_global_features(self, select_feature_id: int | None = None) -> None:
        rows = session.query(FeatureCalculator).order_by(FeatureCalculator.feature_name, FeatureCalculator.id).all()
        linked_ids = {
            int(row.calculator_id)
            for row in session.query(ClusterWellLogParameterFromCalculator)
            .filter(ClusterWellLogParameterFromCalculator.dataset_id == self.dataset_id)
            .all()
        }
        self.global_features = []
        self.calculator_parameters = []
        for row in rows:
            feature = {
                "id": int(row.id),
                "feature_name": row.feature_name,
                "transform_type": row.transform_type,
                "expression": row.expression,
                "used_canonical_well_log": row.used_canonical_well_log,
                "params_json": row.params_json,
                "created_at": row.created_at,
                "in_dataset": int(row.id) in linked_ids,
            }
            self.global_features.append(feature)
            if int(row.id) in linked_ids:
                self.calculator_parameters.append({"calculator_id": int(row.id), "feature_name": row.feature_name})

        self.table_global_features.setRowCount(len(self.global_features))
        selected_row = -1
        for row_index, feature in enumerate(self.global_features):
            values = [
                feature.get("id"),
                feature.get("feature_name"),
                feature.get("transform_type"),
                feature.get("expression"),
                self._format_inputs(feature.get("used_canonical_well_log")),
                "yes" if feature.get("in_dataset") else "no",
            ]
            for column, value in enumerate(values):
                item = QtWidgets.QTableWidgetItem("" if value is None else str(value))
                if column == 0:
                    item.setData(QtCore.Qt.UserRole, feature.get("id"))
                self.table_global_features.setItem(row_index, column, item)
            if select_feature_id is not None and int(feature["id"]) == int(select_feature_id):
                selected_row = row_index
        self.table_global_features.resizeColumnsToContents()
        if selected_row >= 0:
            self.table_global_features.selectRow(selected_row)
        elif self.global_features:
            self.table_global_features.selectRow(0)
        else:
            self._selected_feature_id = None
            self.textEdit_definition.clear()
        self._fill_dataset_context()

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
        if not feature_name:
            errors.append("Feature name is required.")
        elif feature_name.casefold() in self._existing_feature_names():
            errors.append(f"Feature name '{feature_name}' already exists in feature_calculator.")
        elif feature_name.casefold() in self._canonical_names():
            errors.append(f"Feature name '{feature_name}' matches a canonical curve name.")

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
        if not feature_name:
            errors.append("Feature name is required.")
        elif feature_name.casefold() in self._existing_feature_names():
            errors.append(f"Feature name '{feature_name}' already exists in feature_calculator.")
        elif feature_name.casefold() in self._canonical_names():
            errors.append(f"Feature name '{feature_name}' matches a canonical curve name.")

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
        lines = [
            f"ID: {feature.id}",
            f"Name: {feature.feature_name}",
            f"Type: {feature.transform_type}",
            f"Expression: {feature.expression or ''}",
            f"Inputs: {self._format_inputs(feature.used_canonical_well_log)}",
            f"Created at: {feature.created_at}",
            f"Used in current dataset: {'yes' if linked else 'no'}",
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
        self._refresh_selected_feature_diagnostics(feature, config, parse_errors)

    def _status_error_count_text(self, errors: list[Any]) -> str:
        if not errors:
            return ""
        return "; ".join(f"[{getattr(error, 'code', 'error')}] {getattr(error, 'message', str(error))}" for error in errors[:2])

    def _coverage_summary_text(self, report: Any | None) -> str:
        if report is None:
            return "Select a saved feature to calculate coverage report. Preview can also be built for the unsaved form definition."
        summary = report.summary
        points = "—" if summary.min_points is None else f"{summary.min_points} / {summary.max_points}"
        inputs = ", ".join(summary.input_curves) or "—"
        can_add = "YES" if summary.can_be_added else "NO"
        return (
            f"Wells: {summary.wells_total}; OK: {summary.wells_ok}; Errors: {summary.error_count}; "
            f"Min/Max points: {points}; Inputs: {inputs}; Mode: {summary.mode}; "
            f"Normalization: {summary.normalization_scope}; Can be added to dataset: {can_add}"
        )

    def _refresh_selected_feature_diagnostics(self, feature: FeatureCalculator, config: Any | None, parse_errors: list[Any]) -> None:
        self._coverage_rows = []
        self.table_coverage.setRowCount(0)
        self.textEdit_error_detail.clear()
        if parse_errors or config is None:
            self.label_coverage_summary.setText("Coverage unavailable: selected feature definition has parse errors.")
            self.pushButton_add_to_dataset.setEnabled(False)
            return
        report = build_feature_calculator_coverage_report(self.dataset_id, config)
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
            f"Well: {row.well_name or '—'} (id={row.well_id})",
            f"Depth interval: {row.top_md:g} - {row.bottom_md:g}",
            f"Status: {row.status}",
            f"Points: {row.points}",
        ]
        if row.errors:
            error = row.errors[0]
            lines.extend([
                f"Feature: {getattr(error, 'feature_name', None) or 'calculator'}",
                f"Input curve: {getattr(error, 'canonical_name', None) or '—'}",
                f"Error code: {getattr(error, 'code', 'error')}",
                f"Message: {getattr(error, 'message', str(error))}",
                f"Recommendation: {feature_calculator_recommendation(error)}",
            ])
        else:
            lines.append(f"Recommendation: {feature_calculator_recommendation(None)}")
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
            ax.text(0.5, 0.5, "Select well for preview", ha="center", va="center")
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
            ax.text(0.5, 0.5, f"Preview error:\n[{error.code}] {error.message}", ha="center", va="center", wrap=True)
            ax.set_axis_off()
            self.textEdit_error_detail.setPlainText(
                f"Preview error for {well.get('name', well.get('id'))}:\n"
                f"Feature: {config.feature_name or title}\n"
                f"Operation/formula: {config.operation or config.expression}\n"
                f"Depth interval: {float(well.get('top_md')):g} - {float(well.get('bottom_md')):g}\n"
                f"Error code: {error.code}\n"
                f"Message: {error.message}\n"
                f"Recommendation: {feature_calculator_recommendation(error)}"
            )
        else:
            points = [(value, depth) for value, depth in zip(result.values, result.depths) if value is not None]
            if points:
                x_values, y_depths = zip(*points)
                ax.plot(x_values, y_depths, linewidth=1.1, marker=".", markersize=3, color="#1f77b4")
                ax.invert_yaxis()
                ax.grid(True, alpha=0.25)
                ax.set_xlabel(config.feature_name or title)
                ax.set_ylabel("Depth MD")
                ax.set_title(f"{title}: {well.get('name', well.get('id'))}")
            else:
                ax.text(0.5, 0.5, "Preview returned no points", ha="center", va="center")
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
            ax.text(0.5, 0.5, "Selected feature has parse errors", ha="center", va="center")
            ax.set_axis_off()
            self.preview_canvas.draw_idle()
            return
        self._preview_config(config, title=str(feature.feature_name))

    def _preview_current_definition(self) -> None:
        config_dict, errors = self._build_current_config()
        if errors or config_dict is None:
            self.textEdit_validation.setPlainText("\n".join(errors))
            QtWidgets.QMessageBox.warning(self, "Preview calculated feature", "Исправьте ошибки формы перед preview.")
            return
        feature_name = self.lineEdit_feature_name.text().strip() or "Preview feature"
        config, parse_errors = parse_feature_calculator_config({"feature_name": feature_name, "params_json": config_dict})
        if parse_errors or config is None:
            self.textEdit_validation.setPlainText("\n".join(error.message for error in parse_errors))
            return
        self.textEdit_validation.setPlainText("Preview calculated without saving feature definition.")
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
            lines.append(f"… plus {len(errors) - limit} more errors.")
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
