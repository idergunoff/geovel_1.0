from __future__ import annotations

import json
from typing import Any

from PyQt5 import QtCore, QtWidgets


class WellLogFeatureConstructorDialog(QtWidgets.QDialog):
    """Stage-1 placeholder dialog for the Well Log calculated-feature constructor."""

    def __init__(
        self,
        dataset_id: int,
        dataset_name: str,
        wells: list[dict[str, Any]] | None = None,
        canonical_parameters: list[dict[str, Any]] | None = None,
        calculator_parameters: list[dict[str, Any]] | None = None,
        global_features: list[dict[str, Any]] | None = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.dataset_id = int(dataset_id)
        self.dataset_name = dataset_name
        self.wells = wells or []
        self.canonical_parameters = canonical_parameters or []
        self.calculator_parameters = calculator_parameters or []
        self.global_features = global_features or []

        self.setObjectName("WellLogFeatureConstructorDialog")
        self.setWindowTitle("Well Log Feature Constructor")
        self.resize(820, 560)
        self._setup_ui()
        self._fill_dataset_context()
        self._fill_global_features()

    def _setup_ui(self) -> None:
        main_layout = QtWidgets.QVBoxLayout(self)

        self.label_dataset_title = QtWidgets.QLabel(self)
        self.label_dataset_title.setObjectName("label_well_log_constructor_dataset_title")
        title_font = self.label_dataset_title.font()
        title_font.setPointSize(title_font.pointSize() + 2)
        title_font.setBold(True)
        self.label_dataset_title.setFont(title_font)
        main_layout.addWidget(self.label_dataset_title)

        self.label_description = QtWidgets.QLabel(self)
        self.label_description.setObjectName("label_well_log_constructor_description")
        self.label_description.setWordWrap(True)
        self.label_description.setText(
            "Конструктор расчётных признаков Well Log находится в разработке. "
            "На этом этапе доступен только просмотр контекста dataset и глобальной библиотеки "
            "feature_calculator; Validate, Save и Add to dataset будут включены на следующих этапах."
        )
        main_layout.addWidget(self.label_description)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal, self)
        main_layout.addWidget(splitter, stretch=1)

        context_widget = QtWidgets.QWidget(splitter)
        context_layout = QtWidgets.QVBoxLayout(context_widget)
        self.label_context_summary = QtWidgets.QLabel(context_widget)
        self.label_context_summary.setWordWrap(True)
        context_layout.addWidget(self.label_context_summary)

        self.tree_dataset_context = QtWidgets.QTreeWidget(context_widget)
        self.tree_dataset_context.setObjectName("treeWidget_well_log_constructor_dataset_context")
        self.tree_dataset_context.setHeaderLabels(["Dataset context", "Value"])
        self.tree_dataset_context.header().setStretchLastSection(True)
        context_layout.addWidget(self.tree_dataset_context)

        features_widget = QtWidgets.QWidget(splitter)
        features_layout = QtWidgets.QVBoxLayout(features_widget)
        features_label = QtWidgets.QLabel("Глобальные расчётные признаки (feature_calculator)", features_widget)
        features_layout.addWidget(features_label)

        self.table_global_features = QtWidgets.QTableWidget(features_widget)
        self.table_global_features.setObjectName("tableWidget_well_log_constructor_global_features")
        self.table_global_features.setColumnCount(5)
        self.table_global_features.setHorizontalHeaderLabels(["ID", "Feature", "Mode/Transform", "Inputs", "Params"])
        self.table_global_features.horizontalHeader().setStretchLastSection(True)
        self.table_global_features.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table_global_features.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        features_layout.addWidget(self.table_global_features)

        splitter.addWidget(context_widget)
        splitter.addWidget(features_widget)
        splitter.setSizes([360, 460])

        buttons_layout = QtWidgets.QHBoxLayout()
        buttons_layout.addStretch(1)
        self.pushButton_validate = QtWidgets.QPushButton("Validate", self)
        self.pushButton_validate.setObjectName("pushButton_well_log_constructor_validate")
        self.pushButton_validate.setEnabled(False)
        buttons_layout.addWidget(self.pushButton_validate)

        self.pushButton_save = QtWidgets.QPushButton("Save", self)
        self.pushButton_save.setObjectName("pushButton_well_log_constructor_save")
        self.pushButton_save.setEnabled(False)
        buttons_layout.addWidget(self.pushButton_save)

        self.pushButton_add_to_dataset = QtWidgets.QPushButton("Add to dataset", self)
        self.pushButton_add_to_dataset.setObjectName("pushButton_well_log_constructor_add_to_dataset")
        self.pushButton_add_to_dataset.setEnabled(False)
        buttons_layout.addWidget(self.pushButton_add_to_dataset)

        self.pushButton_close = QtWidgets.QPushButton("Close", self)
        self.pushButton_close.setObjectName("pushButton_well_log_constructor_close")
        self.pushButton_close.clicked.connect(self.close)
        buttons_layout.addWidget(self.pushButton_close)
        main_layout.addLayout(buttons_layout)

    def _fill_dataset_context(self) -> None:
        self.label_dataset_title.setText(f"Dataset: {self.dataset_name} (id={self.dataset_id})")
        self.label_context_summary.setText(
            f"Скважин: {len(self.wells)}; canonical-параметров: {len(self.canonical_parameters)}; "
            f"уже привязанных calculator-параметров: {len(self.calculator_parameters)}."
        )
        self.tree_dataset_context.clear()
        wells_root = QtWidgets.QTreeWidgetItem(["Wells", str(len(self.wells))])
        for well in self.wells:
            interval = f"{well.get('top_md', '—')} - {well.get('bottom_md', '—')}"
            wells_root.addChild(QtWidgets.QTreeWidgetItem([str(well.get("name") or f"well_id={well.get('id')}"), interval]))
        self.tree_dataset_context.addTopLevelItem(wells_root)

        canonical_root = QtWidgets.QTreeWidgetItem(["Canonical parameters", str(len(self.canonical_parameters))])
        for canonical in self.canonical_parameters:
            canonical_root.addChild(
                QtWidgets.QTreeWidgetItem([str(canonical.get("canonical_name") or "—"), f"id={canonical.get('canonical_id')}"])
            )
        self.tree_dataset_context.addTopLevelItem(canonical_root)

        calculator_root = QtWidgets.QTreeWidgetItem(["Calculator parameters in dataset", str(len(self.calculator_parameters))])
        for calculator in self.calculator_parameters:
            calculator_root.addChild(
                QtWidgets.QTreeWidgetItem([str(calculator.get("feature_name") or "—"), f"id={calculator.get('calculator_id')}"])
            )
        self.tree_dataset_context.addTopLevelItem(calculator_root)
        self.tree_dataset_context.expandAll()

    def _fill_global_features(self) -> None:
        self.table_global_features.setRowCount(len(self.global_features))
        for row_index, feature in enumerate(self.global_features):
            params = feature.get("params_json") or ""
            params_preview = params
            try:
                params_preview = json.dumps(json.loads(params), ensure_ascii=False, separators=(",", ":"))
            except Exception:
                pass
            values = [
                feature.get("id"),
                feature.get("feature_name"),
                feature.get("transform_type") or feature.get("mode"),
                feature.get("used_canonical_well_log"),
                params_preview,
            ]
            for column, value in enumerate(values):
                item = QtWidgets.QTableWidgetItem("" if value is None else str(value))
                if column == 0:
                    item.setTextAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
                self.table_global_features.setItem(row_index, column, item)
        self.table_global_features.resizeColumnsToContents()
