from __future__ import annotations

import json
from typing import Any, Callable

from PyQt5 import QtCore, QtWidgets

from cluster.well_feature_calculator import (
    FEATURE_CALCULATOR_DEFAULT_OUTLIER_POLICY,
    FEATURE_CALCULATOR_DEPTH_GRID_POLICY,
    FEATURE_CALCULATOR_INVALID_MATH_POLICY,
    FEATURE_CALCULATOR_NORMALIZATION_SCOPES,
    FEATURE_CALCULATOR_UNARY_OPERATIONS,
    parse_feature_calculator_config,
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

        self.setObjectName("WellLogFeatureConstructorDialog")
        self.setWindowTitle("Well Log Feature Constructor")
        self.resize(1100, 720)
        self._setup_ui()
        self._fill_dataset_context()
        self._fill_input_curves()
        self._refresh_global_features()
        self._update_operation_fields()

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

        create_group = QtWidgets.QGroupBox("Create operation feature", left_widget)
        create_layout = QtWidgets.QFormLayout(create_group)

        self.lineEdit_feature_name = QtWidgets.QLineEdit(create_group)
        self.lineEdit_feature_name.setObjectName("lineEdit_well_log_constructor_feature_name")
        create_layout.addRow("Feature name", self.lineEdit_feature_name)

        self.comboBox_mode = QtWidgets.QComboBox(create_group)
        self.comboBox_mode.setObjectName("comboBox_well_log_constructor_mode")
        self.comboBox_mode.addItem("Operation", "operation")
        self.comboBox_mode.setEnabled(False)
        create_layout.addRow("Mode", self.comboBox_mode)

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

    def _fill_input_curves(self) -> None:
        self.comboBox_input_curve.clear()
        self.comboBox_input_curve.addItem("", None)
        for curve in self.all_canonical_parameters:
            canonical_id = curve.get("canonical_id")
            canonical_name = curve.get("canonical_name") or f"canonical_id={canonical_id}"
            self.comboBox_input_curve.addItem(str(canonical_name), {"canonical_id": canonical_id, "canonical_name": canonical_name})

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

    def _update_operation_fields(self) -> None:
        operation = self.comboBox_operation.currentData()
        is_rolling = operation in ROLLING_OPERATIONS
        is_normalization = operation in NORMALIZATION_OPERATIONS
        self.spinBox_rolling_window.setEnabled(is_rolling)
        self.spinBox_min_periods.setEnabled(is_rolling)
        self.comboBox_normalization_scope.setEnabled(is_normalization)

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
        _config, errors = self._build_operation_config()
        if errors:
            self.textEdit_validation.setPlainText("\n".join(errors))
            QtWidgets.QMessageBox.warning(self, "Validate operation feature", "Исправьте ошибки формы перед сохранением.")
            return
        self.textEdit_validation.setPlainText("Validation OK.")
        QtWidgets.QMessageBox.information(self, "Validate operation feature", "Operation feature definition is valid.")

    def _expression_for_config(self, config: dict[str, Any]) -> str:
        operation = str(config.get("operation") or "")
        input_name = str(config.get("inputs", [{}])[0].get("canonical_name") or "X")
        if operation in ROLLING_OPERATIONS:
            return f"{operation}({input_name}, {config.get('window')})"
        return f"{operation}({input_name})"

    def _save_feature_clicked(self) -> None:
        config, errors = self._build_operation_config()
        if errors or config is None:
            self.textEdit_validation.setPlainText("\n".join(errors))
            QtWidgets.QMessageBox.warning(self, "SAVE feature", "Невозможно сохранить признак: форма содержит ошибки.")
            return
        feature_name = self.lineEdit_feature_name.text().strip()
        inputs_json = json.dumps(config["inputs"], ensure_ascii=False)
        params_json = json.dumps(config, ensure_ascii=False, sort_keys=True)
        expression = self._expression_for_config(config)
        feature = FeatureCalculator(
            feature_name=feature_name,
            expression=expression,
            used_canonical_well_log=inputs_json,
            transform_type=str(config["operation"]),
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
        self.textEdit_validation.setPlainText(f"Saved feature '{feature_name}'.")
        self._refresh_global_features(select_feature_id=int(feature.id))
        self._set_info(f"WELL LOG CONSTR: сохранён operation-признак {feature_name}.", "green")
        QtWidgets.QMessageBox.information(self, "SAVE feature", f"Feature '{feature_name}' saved.")

    def _feature_selection_changed(self) -> None:
        selected = self.table_global_features.selectedItems()
        if not selected:
            self._selected_feature_id = None
            self.textEdit_definition.clear()
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
