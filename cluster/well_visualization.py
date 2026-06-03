from __future__ import annotations

import re

from .common import *

def _build_well_log_cluster_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Агрегирует строки визуализации Well Log по кластерам для summary/stub-окна.
    """
    grouped: dict[int, dict[str, Any]] = {}
    for row in rows:
        label = int(row["cluster_label"])
        item = grouped.setdefault(
            label,
            {
                "cluster_label": label,
                "row_count": 0,
                "well_ids": set(),
                "depth_min": None,
                "depth_max": None,
            },
        )
        item["row_count"] += 1
        item["well_ids"].add(int(row["well_id"]))
        depth = float(row["depth_md"])
        item["depth_min"] = depth if item["depth_min"] is None else min(float(item["depth_min"]), depth)
        item["depth_max"] = depth if item["depth_max"] is None else max(float(item["depth_max"]), depth)

    summary_rows = []
    for label, item in sorted(grouped.items(), key=lambda pair: pair[0]):
        summary_rows.append(
            {
                "cluster_label": int(label),
                "row_count": int(item["row_count"]),
                "well_count": len(item["well_ids"]),
                "well_ids": sorted(int(well_id) for well_id in item["well_ids"]),
                "depth_min": item["depth_min"],
                "depth_max": item["depth_max"],
            }
        )
    return summary_rows


def build_well_log_cluster_visualization_data(
        *,
        run_context: ClusterRunContext,
        labels: list[int],
        kept_row_indices: list[int],
        metrics: dict[str, Any],
        config: dict[str, Any],
        pca_info: dict[str, Any],
        cluster_info: dict[str, Any],
        smoothing_changes: int
) -> WellLogClusterVisualizationData:
    """
    Формирует runtime-структуру WellLogClusterVisualizationData после ручного CALC.
    """
    meta_rows = run_context.get("meta", [])
    feature_names = [str(name) for name in run_context.get("feature_names", [])]
    rows: list[dict[str, Any]] = []
    for clean_row_idx, label in enumerate(labels):
        if clean_row_idx >= len(kept_row_indices):
            break
        source_row_idx = int(kept_row_indices[clean_row_idx])
        if source_row_idx < 0 or source_row_idx >= len(meta_rows):
            continue
        meta = dict(meta_rows[source_row_idx])
        source_values = meta.get("source_curve_values", {}) or {}
        source_curve_names = meta.get("source_curve_names", {}) or {}
        rows.append(
            {
                "cluster_label": int(label),
                "well_id": int(meta["well_id"]),
                "well_name": str(meta.get("well_name", f"well_id={meta['well_id']}")),
                "depth_md": float(meta["depth_md"]),
                "row_index_in_well": int(meta.get("row_index_in_well", 0)),
                "source_row_index": int(meta.get("source_row_index", source_row_idx)),
                "features": {name: source_values.get(name) for name in feature_names},
                "source_curve_names": {name: source_curve_names.get(name, name) for name in feature_names},
            }
        )

    cluster_summary = _build_well_log_cluster_summary(rows)
    created_at = dt.datetime.now(dt.timezone.utc).isoformat()
    run_id_payload = f"well_log:{run_context['dataset_id']}:{run_context.get('data_hash', '')}:{created_at}"
    run_id = hashlib.sha256(run_id_payload.encode("utf-8")).hexdigest()[:16]
    summary = {
        "row_count": len(rows),
        "label_count": len(labels),
        "cluster_count": len([item for item in cluster_summary if item["cluster_label"] != -1]),
        "noise_count": sum(1 for label in labels if int(label) == -1),
        "well_count": len({int(row["well_id"]) for row in rows}),
        "smoothing_changes": int(smoothing_changes),
    }
    diagnostics = dict(run_context.get("diagnostics", {}) or {})
    diagnostics.update(
        {
            "kept_row_count_after_clean": len(kept_row_indices),
            "dropped_row_count_after_clean": max(0, int(run_context["row_count"]) - len(kept_row_indices)),
            "pca_info": pca_info,
            "cluster_info": cluster_info,
        }
    )

    return WellLogClusterVisualizationData(
        run_id=run_id,
        source_type="well_log",
        dataset_id=int(run_context["dataset_id"]),
        dataset_title=str(run_context["dataset_title"]),
        created_at=created_at,
        data_hash=str(run_context.get("data_hash", "")),
        feature_names=feature_names,
        labels=[int(label) for label in labels],
        rows=rows,
        cluster_summary=cluster_summary,
        metrics=metrics,
        config=config,
        diagnostics=diagnostics,
        summary=summary,
    )


def _cluster_label_sort_key(label: Any) -> tuple[int, int]:
    """Стабильная сортировка labels: noise (-1) в конце списка."""
    try:
        label_int = int(label)
    except (TypeError, ValueError):
        label_int = 0
    return (1, label_int) if label_int == -1 else (0, label_int)


def _well_log_cluster_color(label: int) -> str:
    """Возвращает стабильный цвет кластера для всех MVP-контролов визуализации."""
    if int(label) == -1:
        return "#9E9E9E"
    palette = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
        "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#aec7e8", "#ffbb78",
    ]
    return palette[int(label) % len(palette)]


def _format_well_log_visual_metric(value: Any, precision: int = 4) -> str:
    return _safe_num(value, precision, fallback="—")


class WellLogClusterVisualizationWindow(QtWidgets.QDialog):
    """
    MVP-окно визуализации результата Well Log clustering.

    Окно хранит подготовленные runtime-структуры: список скважин/кривых, легенду,
    фильтр кластеров, кэш интервалов и статистические профили кластеров.
    Режим «1 скважина → все кривые» рисует выбранную скважину без повторного CALC.
    """

    MODE_TITLES = (
        "Summary",
        "1 скважина → все кривые",
        "1 кривая → разные скважины",
        "Средний портрет кластера",
    )

    _TRACK_WIDTH_PX = 190
    _CLUSTER_TRACK_WIDTH_PX = 125
    _GRAPH_HEIGHT_PX = 600

    def __init__(self, parent=None):
        super().__init__(parent)
        self.visualization_data: WellLogClusterVisualizationData | None = None
        self.interval_cache: dict[int, list[dict[str, Any]]] = {}
        self.profile_cache: dict[int, dict[str, Any]] = {}
        self._cluster_checkboxes: dict[int, Any] = {}
        self._highlight_interval: dict[str, Any] | None = None
        self._build_ui()

    def _prepare_scrollable_canvas(
            self,
            *,
            figure: Figure,
            canvas: FigureCanvas,
            scroll_area: QtWidgets.QScrollArea,
            width_px: int,
            height_px: int | None = None,
    ) -> None:
        """Resizes a Matplotlib canvas to exact content size and clears stale Qt paint artefacts."""
        height_px = int(height_px or self._GRAPH_HEIGHT_PX)
        width_px = max(320, int(width_px))
        figure.clear()
        figure.set_size_inches(width_px / float(figure.dpi), height_px / float(figure.dpi), forward=True)
        canvas.setMinimumSize(width_px, height_px)
        canvas.setMaximumSize(width_px, height_px)
        canvas.resize(width_px, height_px)
        canvas.updateGeometry()
        if scroll_area.widget() is canvas:
            scroll_area.horizontalScrollBar().setValue(0)
            scroll_area.verticalScrollBar().setValue(0)
            scroll_area.viewport().update()

    @staticmethod
    def _draw_empty_graph(figure: Figure, canvas: FigureCanvas, message: str) -> None:
        ax = figure.add_subplot(111)
        ax.text(0.5, 0.5, message, ha="center", va="center")
        ax.set_axis_off()
        canvas.draw()

    def _build_ui(self) -> None:
        self.setWindowTitle("Well Log Cluster Visualization")
        self.resize(1500, 950)
        self.setMinimumSize(1180, 760)
        screen = QtWidgets.QApplication.primaryScreen()
        if screen is not None:
            available = screen.availableGeometry()
            self.resize(int(available.width() * 0.92), int(available.height() * 0.90))
            self.move(available.center() - self.rect().center())
        root_layout = QtWidgets.QVBoxLayout(self)
        root_layout.setContentsMargins(8, 8, 8, 8)
        root_layout.setSpacing(6)

        self.title_label = QtWidgets.QLabel("Well Log Cluster Visualization")
        self.title_label.setWordWrap(True)
        self.title_label.setStyleSheet("font-weight: 600; font-size: 14px;")
        root_layout.addWidget(self.title_label)

        controls_group = QtWidgets.QGroupBox("Controls")
        controls_layout = QtWidgets.QGridLayout(controls_group)
        controls_layout.setContentsMargins(8, 8, 8, 8)
        controls_layout.setHorizontalSpacing(8)
        controls_layout.setVerticalSpacing(4)
        controls_layout.addWidget(QtWidgets.QLabel("Mode:"), 0, 0)
        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItems(self.MODE_TITLES)
        self.mode_combo.setToolTip("Переключает режим визуализации без повторного расчета кластеров.")
        self.mode_combo.currentIndexChanged.connect(self._sync_mode_tab)
        controls_layout.addWidget(self.mode_combo, 0, 1)

        controls_layout.addWidget(QtWidgets.QLabel("Well:"), 0, 2)
        self.well_combo = QtWidgets.QComboBox()
        self.well_combo.setToolTip("Скважина для режима «1 скважина → все кривые» и таблицы интервалов.")
        self.well_combo.currentIndexChanged.connect(self._handle_well_changed)
        controls_layout.addWidget(self.well_combo, 0, 3)

        controls_layout.addWidget(QtWidgets.QLabel("Curve:"), 0, 4)
        self.curve_combo = QtWidgets.QComboBox()
        self.curve_combo.setToolTip("Кривая для межскважинного сравнения small multiples.")
        self.curve_combo.currentIndexChanged.connect(self._render_curve_across_wells)
        controls_layout.addWidget(self.curve_combo, 0, 5)

        self.show_noise_checkbox = QtWidgets.QCheckBox("Show noise")
        self.show_noise_checkbox.setToolTip("Показывать строки HDBSCAN noise (cluster -1) на графиках и в таблицах.")
        self.show_noise_checkbox.setChecked(True)
        self.show_noise_checkbox.stateChanged.connect(self._apply_cluster_filter_to_tables)
        controls_layout.addWidget(self.show_noise_checkbox, 1, 0)

        controls_layout.addWidget(QtWidgets.QLabel("Cluster opacity:"), 1, 1)
        self.opacity_spin = QtWidgets.QDoubleSpinBox()
        self.opacity_spin.setToolTip("Прозрачность цветной подложки кластерных интервалов на графиках.")
        self.opacity_spin.setRange(0.05, 1.0)
        self.opacity_spin.setSingleStep(0.05)
        self.opacity_spin.setValue(0.35)
        self.opacity_spin.valueChanged.connect(self._render_one_well_curves)
        self.opacity_spin.valueChanged.connect(self._render_curve_across_wells)
        controls_layout.addWidget(self.opacity_spin, 1, 2)

        export_layout = QtWidgets.QHBoxLayout()
        self.export_intervals_button = QtWidgets.QPushButton("Export intervals")
        self.export_intervals_button.setToolTip("Сохранить все видимые интервалы кластеров в CSV или XLSX.")
        self.export_intervals_button.clicked.connect(self.export_intervals)
        export_layout.addWidget(self.export_intervals_button)

        self.export_profiles_button = QtWidgets.QPushButton("Export profiles")
        self.export_profiles_button.setToolTip("Сохранить статистические портреты видимых кластеров в CSV или XLSX.")
        self.export_profiles_button.clicked.connect(self.export_cluster_profiles)
        export_layout.addWidget(self.export_profiles_button)

        self.save_screenshot_button = QtWidgets.QPushButton("Save graph")
        self.save_screenshot_button.setToolTip("Сохранить текущий график активного режима в PNG/SVG/PDF.")
        self.save_screenshot_button.clicked.connect(self.save_current_graph_screenshot)
        export_layout.addWidget(self.save_screenshot_button)
        controls_layout.addLayout(export_layout, 1, 3, 1, 3)

        self.status_label = QtWidgets.QLabel("Готово")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("color: #2e7d32;")
        controls_layout.addWidget(self.status_label, 2, 0, 1, 6)
        root_layout.addWidget(controls_group)

        splitter = QtWidgets.QSplitter(Qt.Horizontal)
        left_widget = QtWidgets.QWidget()
        left_widget.setMaximumWidth(300)
        left_widget.setMinimumWidth(210)
        left_layout = QtWidgets.QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 6, 0)
        left_layout.addWidget(QtWidgets.QLabel("Cluster legend / filter"))
        self.legend_scroll = QtWidgets.QScrollArea()
        self.legend_scroll.setWidgetResizable(True)
        self.legend_container = QtWidgets.QWidget()
        self.legend_layout = QtWidgets.QVBoxLayout(self.legend_container)
        self.legend_layout.addStretch(1)
        self.legend_scroll.setWidget(self.legend_container)
        left_layout.addWidget(self.legend_scroll)
        splitter.addWidget(left_widget)

        center_widget = QtWidgets.QWidget()
        center_layout = QtWidgets.QVBoxLayout(center_widget)
        center_layout.setContentsMargins(0, 0, 0, 0)
        self.mode_tabs = QtWidgets.QTabWidget()
        self.mode_tabs.currentChanged.connect(self._sync_mode_combo)
        self.summary_text = QtWidgets.QPlainTextEdit()
        self.summary_text.setReadOnly(True)
        self.mode_tabs.addTab(self.summary_text, "Summary")

        self.one_well_tab = QtWidgets.QWidget()
        one_well_layout = QtWidgets.QVBoxLayout(self.one_well_tab)
        self.one_well_hint = QtWidgets.QLabel(
            "Выберите скважину: будут показаны все кривые выбранной скважины и cluster track."
        )
        self.one_well_hint.setWordWrap(True)
        one_well_layout.addWidget(self.one_well_hint)
        self.one_well_figure = Figure(figsize=(12, 6.5), dpi=110)
        self.one_well_canvas = FigureCanvas(self.one_well_figure)
        self.one_well_canvas.setMinimumSize(900, 520)
        self.one_well_toolbar = NavigationToolbar(self.one_well_canvas, self.one_well_tab)
        self.one_well_scroll = QtWidgets.QScrollArea()
        self.one_well_scroll.setWidgetResizable(False)
        self.one_well_scroll.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.one_well_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.one_well_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.one_well_scroll.setWidget(self.one_well_canvas)
        one_well_layout.addWidget(self.one_well_toolbar)
        one_well_layout.addWidget(self.one_well_scroll, stretch=1)
        self.mode_tabs.addTab(self.one_well_tab, self.MODE_TITLES[1])

        self.curve_wells_tab = QtWidgets.QWidget()
        curve_wells_layout = QtWidgets.QVBoxLayout(self.curve_wells_tab)
        curve_wells_layout.setContentsMargins(4, 4, 4, 4)
        self.curve_wells_splitter = QtWidgets.QSplitter(Qt.Horizontal)

        curve_wells_side_panel = QtWidgets.QWidget()
        curve_wells_side_panel.setMinimumWidth(260)
        curve_wells_side_panel.setMaximumWidth(380)
        curve_wells_side_layout = QtWidgets.QVBoxLayout(curve_wells_side_panel)
        curve_wells_side_layout.setContentsMargins(0, 0, 8, 0)
        self.curve_wells_hint = QtWidgets.QLabel(
            "Выберите кривую и список скважин: графики будут показаны вертикальными треками слева направо, "
            "с одинаковой палитрой кластеров."
        )
        self.curve_wells_hint.setWordWrap(True)
        curve_wells_side_layout.addWidget(self.curve_wells_hint)

        curve_wells_side_layout.addWidget(QtWidgets.QLabel("Wells:"))
        self.curve_wells_list = QtWidgets.QListWidget()
        self.curve_wells_list.setToolTip("Выберите одну или несколько скважин для сравнения выбранной кривой.")
        self.curve_wells_list.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        self.curve_wells_list.itemSelectionChanged.connect(self._render_curve_across_wells)
        curve_wells_side_layout.addWidget(self.curve_wells_list, stretch=1)

        curve_wells_options = QtWidgets.QFormLayout()
        curve_wells_options.setContentsMargins(0, 0, 0, 0)
        self.curve_wells_sort_combo = QtWidgets.QComboBox()
        self.curve_wells_sort_combo.setToolTip("Сортировка small multiples по имени, доле кластера или похожести кластерной последовательности.")
        self.curve_wells_sort_combo.addItems([
            "по имени",
            "по доле выбранного кластера",
            "по похожести последовательности кластеров",
        ])
        self.curve_wells_sort_combo.currentIndexChanged.connect(self._render_curve_across_wells)
        curve_wells_options.addRow("Sort:", self.curve_wells_sort_combo)

        self.curve_wells_cluster_combo = QtWidgets.QComboBox()
        self.curve_wells_cluster_combo.setToolTip("Кластер для расчета доли и сортировки скважин в режиме сравнения кривой.")
        self.curve_wells_cluster_combo.currentIndexChanged.connect(self._render_curve_across_wells)
        curve_wells_options.addRow("Cluster:", self.curve_wells_cluster_combo)

        self.curve_wells_limit_spin = QtWidgets.QSpinBox()
        self.curve_wells_limit_spin.setToolTip("Максимальное число скважин, выводимых на график small multiples.")
        self.curve_wells_limit_spin.setRange(1, 50)
        self.curve_wells_limit_spin.setValue(12)
        self.curve_wells_limit_spin.valueChanged.connect(self._render_curve_across_wells)
        curve_wells_options.addRow("Top wells:", self.curve_wells_limit_spin)

        self.curve_wells_shared_x = QtWidgets.QCheckBox("общая шкала X")
        self.curve_wells_shared_x.setToolTip("Использовать одну шкалу значений кривой для всех скважин.")
        self.curve_wells_shared_x.setChecked(True)
        self.curve_wells_shared_x.stateChanged.connect(self._render_curve_across_wells)
        curve_wells_options.addRow("Scale:", self.curve_wells_shared_x)

        self.curve_wells_shared_depth = QtWidgets.QCheckBox("общая глубина")
        self.curve_wells_shared_depth.setToolTip(
            "Включено: все скважины показаны в общей глубинной шкале для корреляции. "
            "Выключено: каждый трек растянут по своей глубине."
        )
        self.curve_wells_shared_depth.setChecked(False)
        self.curve_wells_shared_depth.stateChanged.connect(self._render_curve_across_wells)
        curve_wells_options.addRow("Depth:", self.curve_wells_shared_depth)
        curve_wells_side_layout.addLayout(curve_wells_options)

        self.curve_wells_summary_table = QtWidgets.QTableWidget(0, 5)
        self.curve_wells_summary_table.setHorizontalHeaderLabels(["Well", "Rows", "Dominant cluster", "Selected cluster %", "Clusters summary"])
        self.curve_wells_summary_table.setSortingEnabled(True)
        self.curve_wells_summary_table.setMinimumHeight(120)
        curve_wells_side_layout.addWidget(self.curve_wells_summary_table, stretch=1)
        self.curve_wells_splitter.addWidget(curve_wells_side_panel)

        curve_wells_graph_panel = QtWidgets.QWidget()
        curve_wells_graph_layout = QtWidgets.QVBoxLayout(curve_wells_graph_panel)
        curve_wells_graph_layout.setContentsMargins(0, 0, 0, 0)
        self.curve_wells_figure = Figure(figsize=(12, 7), dpi=110)
        self.curve_wells_canvas = FigureCanvas(self.curve_wells_figure)
        self.curve_wells_canvas.setMinimumSize(900, 560)
        self.curve_wells_toolbar = NavigationToolbar(self.curve_wells_canvas, self.curve_wells_tab)
        self.curve_wells_scroll = QtWidgets.QScrollArea()
        self.curve_wells_scroll.setWidgetResizable(False)
        self.curve_wells_scroll.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.curve_wells_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.curve_wells_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.curve_wells_scroll.setWidget(self.curve_wells_canvas)
        curve_wells_graph_layout.addWidget(self.curve_wells_toolbar)
        curve_wells_graph_layout.addWidget(self.curve_wells_scroll, stretch=1)
        self.curve_wells_splitter.addWidget(curve_wells_graph_panel)
        self.curve_wells_splitter.setStretchFactor(0, 0)
        self.curve_wells_splitter.setStretchFactor(1, 1)
        self.curve_wells_splitter.setSizes([300, 1200])
        curve_wells_layout.addWidget(self.curve_wells_splitter, stretch=1)
        self.mode_tabs.addTab(self.curve_wells_tab, self.MODE_TITLES[2])

        self.cluster_profile_tab = QtWidgets.QWidget()
        cluster_profile_layout = QtWidgets.QVBoxLayout(self.cluster_profile_tab)
        self.cluster_profile_hint = QtWidgets.QLabel(
            "Средний портрет кластера: heatmap показывает стандартизованные средние "
            "по всем кластерам и кривым, bar chart детализирует выбранный кластер."
        )
        self.cluster_profile_hint.setWordWrap(True)
        cluster_profile_layout.addWidget(self.cluster_profile_hint)

        cluster_profile_controls = QtWidgets.QHBoxLayout()
        cluster_profile_controls.addWidget(QtWidgets.QLabel("Cluster:"))
        self.profile_cluster_combo = QtWidgets.QComboBox()
        self.profile_cluster_combo.setToolTip("Кластер, для которого строится bar chart наиболее отличающихся кривых.")
        self.profile_cluster_combo.currentIndexChanged.connect(self._render_cluster_profile_view)
        cluster_profile_controls.addWidget(self.profile_cluster_combo)
        cluster_profile_controls.addWidget(QtWidgets.QLabel("Top features:"))
        self.profile_top_features_spin = QtWidgets.QSpinBox()
        self.profile_top_features_spin.setToolTip("Количество кривых с максимальным отклонением среднего значения от общего среднего.")
        self.profile_top_features_spin.setRange(3, 50)
        self.profile_top_features_spin.setValue(15)
        self.profile_top_features_spin.valueChanged.connect(self._render_cluster_profile_view)
        cluster_profile_controls.addWidget(self.profile_top_features_spin)
        cluster_profile_controls.addStretch(1)
        cluster_profile_layout.addLayout(cluster_profile_controls)

        self.cluster_profile_description = QtWidgets.QPlainTextEdit()
        self.cluster_profile_description.setReadOnly(True)
        self.cluster_profile_description.setMaximumHeight(115)
        cluster_profile_layout.addWidget(self.cluster_profile_description)

        self.cluster_profile_figure = Figure(figsize=(12, 6.5), dpi=110)
        self.cluster_profile_canvas = FigureCanvas(self.cluster_profile_figure)
        self.cluster_profile_canvas.setMinimumSize(900, 520)
        self.cluster_profile_toolbar = NavigationToolbar(self.cluster_profile_canvas, self.cluster_profile_tab)
        self.cluster_profile_scroll = QtWidgets.QScrollArea()
        self.cluster_profile_scroll.setWidgetResizable(True)
        self.cluster_profile_scroll.setWidget(self.cluster_profile_canvas)
        cluster_profile_layout.addWidget(self.cluster_profile_toolbar)
        cluster_profile_layout.addWidget(self.cluster_profile_scroll, stretch=1)
        self.mode_tabs.addTab(self.cluster_profile_tab, self.MODE_TITLES[3])

        self.interval_table = QtWidgets.QTableWidget(0, 6)
        self.interval_table.setToolTip("Непрерывные глубинные интервалы кластеров для выбранной скважины; клик подсвечивает интервал на графике.")
        self.interval_table.setHorizontalHeaderLabels(["Well", "From MD", "To MD", "Thickness", "Cluster", "Rows"])
        self.interval_table.setSortingEnabled(True)
        self.interval_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.interval_table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.interval_table.itemSelectionChanged.connect(self._handle_interval_selection_changed)
        self.mode_tabs.addTab(self.interval_table, "Cluster intervals")

        self.profile_table = QtWidgets.QTableWidget(0, 8)
        self.profile_table.setToolTip("Табличный средний портрет видимых кластеров по выбранным каротажным кривым.")
        self.profile_table.setHorizontalHeaderLabels(["Cluster", "Feature", "Count", "Mean", "Z-mean", "Median", "Std", "P10–P90"])
        self.profile_table.setSortingEnabled(True)
        self.mode_tabs.addTab(self.profile_table, "Cluster profiles")

        center_layout.addWidget(self.mode_tabs)
        splitter.addWidget(center_widget)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([240, 1200])
        root_layout.addWidget(splitter, stretch=1)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if not getattr(self, "_initial_maximize_done", False):
            self._initial_maximize_done = True
            self.showMaximized()

    def _set_status(self, message: str, level: str = "info") -> None:
        """Показывает UX-status в окне и дублирует сообщение в общий статус приложения."""
        palette = {
            "info": "#1565c0",
            "success": "#2e7d32",
            "warning": "#8a5a00",
            "error": "#c62828",
        }
        if hasattr(self, "status_label"):
            self.status_label.setText(message)
            self.status_label.setStyleSheet(f"color: {palette.get(level, palette['info'])};")
        try:
            set_info(message, {"success": "green", "warning": "brown", "error": "red"}.get(level, "blue"))
        except Exception:
            pass

    def _default_export_basename(self, suffix: str) -> str:
        data = self.visualization_data or {}
        dataset_title = re.sub(r"[^0-9A-Za-zА-Яа-я_.-]+", "_", str(data.get("dataset_title", "well_log_cluster"))).strip("_")
        if not dataset_title:
            dataset_title = "well_log_cluster"
        run_id = str(data.get("run_id", "run"))[:8] or "run"
        return f"{dataset_title}_{run_id}_{suffix}"

    def _save_dataframe(self, dataframe: pd.DataFrame, caption: str, suffix: str) -> bool:
        if dataframe.empty:
            self._set_status(f"{caption}: нет данных для экспорта.", "warning")
            QMessageBox.information(self, "Well Log Cluster Export", "Нет данных для экспорта.")
            return False
        default_name = f"{self._default_export_basename(suffix)}.xlsx"
        file_path, selected_filter = QFileDialog.getSaveFileName(
            self,
            caption,
            default_name,
            "Excel (*.xlsx);;CSV (*.csv)",
        )
        if not file_path:
            self._set_status(f"{caption}: экспорт отменен пользователем.", "warning")
            return False
        try:
            lower_path = file_path.lower()
            if lower_path.endswith(".csv") or "CSV" in selected_filter:
                if not lower_path.endswith(".csv"):
                    file_path = f"{file_path}.csv"
                dataframe.to_csv(file_path, index=False, encoding="utf-8-sig")
            else:
                if not lower_path.endswith(".xlsx"):
                    file_path = f"{file_path}.xlsx"
                dataframe.to_excel(file_path, index=False)
            self._set_status(f"Экспорт завершен: {file_path}", "success")
            return True
        except Exception as exc:
            self._set_status(f"Ошибка экспорта: {exc}", "error")
            QMessageBox.critical(self, "Well Log Cluster Export", f"Не удалось сохранить файл:\n{exc}")
            return False

    def _intervals_dataframe(self) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for well_id, intervals in sorted(self.interval_cache.items(), key=lambda pair: pair[0]):
            for interval in intervals:
                label = int(interval.get("cluster_label", 0))
                if not self._is_cluster_visible(label):
                    continue
                rows.append(
                    {
                        "well_id": int(well_id),
                        "well_name": interval.get("well_name", ""),
                        "from_md": interval.get("from_md"),
                        "to_md": interval.get("to_md"),
                        "thickness": interval.get("thickness"),
                        "cluster_id": label,
                        "row_count": interval.get("row_count", 0),
                        "row_index_start": interval.get("row_index_start"),
                        "row_index_end": interval.get("row_index_end"),
                    }
                )
        return pd.DataFrame(rows)

    def _cluster_profiles_dataframe(self) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for label in sorted(self.profile_cache.keys(), key=_cluster_label_sort_key):
            if not self._is_cluster_visible(int(label)):
                continue
            profile = self.profile_cache.get(int(label), {}) or {}
            for feature_name, stats in sorted((profile.get("features", {}) or {}).items(), key=lambda pair: str(pair[0])):
                stats = stats or {}
                rows.append(
                    {
                        "cluster_id": int(label),
                        "feature": str(feature_name),
                        "row_count": profile.get("row_count", 0),
                        "well_count": profile.get("well_count", 0),
                        "depth_min": profile.get("depth_min"),
                        "depth_max": profile.get("depth_max"),
                        "count": stats.get("count"),
                        "mean": stats.get("mean"),
                        "standardized_mean": stats.get("standardized_mean"),
                        "median": stats.get("median"),
                        "std": stats.get("std"),
                        "p10": stats.get("p10"),
                        "p90": stats.get("p90"),
                        "global_mean": stats.get("global_mean"),
                        "global_std": stats.get("global_std"),
                        "valid_fraction_in_cluster": stats.get("valid_fraction_in_cluster"),
                        "description": profile.get("description", ""),
                    }
                )
        return pd.DataFrame(rows)

    def export_intervals(self) -> None:
        """Экспортирует интервалы всех видимых кластеров по всем скважинам."""
        try:
            self._save_dataframe(
                self._intervals_dataframe(),
                "Сохранить интервалы кластеров Well Log",
                "intervals",
            )
        except Exception as exc:
            self._set_status(f"Ошибка подготовки таблицы интервалов: {exc}", "error")
            QMessageBox.critical(self, "Well Log Cluster Export", f"Ошибка подготовки интервалов:\n{exc}")

    def export_cluster_profiles(self) -> None:
        """Экспортирует средние портреты всех видимых кластеров."""
        try:
            self._save_dataframe(
                self._cluster_profiles_dataframe(),
                "Сохранить профили кластеров Well Log",
                "cluster_profiles",
            )
        except Exception as exc:
            self._set_status(f"Ошибка подготовки профилей кластеров: {exc}", "error")
            QMessageBox.critical(self, "Well Log Cluster Export", f"Ошибка подготовки профилей:\n{exc}")

    def _current_graph_figure(self) -> tuple[Figure | None, str]:
        index = self.mode_tabs.currentIndex() if hasattr(self, "mode_tabs") else 0
        if index == 1:
            return self.one_well_figure, "one_well"
        if index == 2:
            return self.curve_wells_figure, "curve_across_wells"
        if index == 3:
            return self.cluster_profile_figure, "cluster_profile"
        return None, "summary"

    def save_current_graph_screenshot(self) -> None:
        """Сохраняет текущий matplotlib-график активного режима в файл отчета."""
        figure, suffix = self._current_graph_figure()
        if figure is None:
            self._set_status("В режиме Summary нет графика для сохранения.", "warning")
            QMessageBox.information(self, "Well Log Cluster Screenshot", "Переключитесь на режим с графиком.")
            return
        default_name = f"{self._default_export_basename(suffix)}.png"
        file_path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Сохранить скриншот графика Well Log",
            default_name,
            "PNG (*.png);;SVG (*.svg);;PDF (*.pdf)",
        )
        if not file_path:
            self._set_status("Сохранение графика отменено пользователем.", "warning")
            return
        try:
            lower_path = file_path.lower()
            if not lower_path.endswith((".png", ".svg", ".pdf")):
                if "SVG" in selected_filter:
                    file_path = f"{file_path}.svg"
                elif "PDF" in selected_filter:
                    file_path = f"{file_path}.pdf"
                else:
                    file_path = f"{file_path}.png"
            figure.savefig(file_path, dpi=200, bbox_inches="tight")
            self._set_status(f"График сохранен: {file_path}", "success")
        except Exception as exc:
            self._set_status(f"Ошибка сохранения графика: {exc}", "error")
            QMessageBox.critical(self, "Well Log Cluster Screenshot", f"Не удалось сохранить график:\n{exc}")

    def load_visualization_data(self, visualization_data: WellLogClusterVisualizationData) -> None:
        self.visualization_data = visualization_data
        self.interval_cache = self._build_interval_cache(visualization_data.get("rows", []))
        self.profile_cache = self._build_profile_cache(visualization_data.get("rows", []), visualization_data.get("feature_names", []))
        self._populate_controls()
        self._populate_legend()
        self._populate_summary()
        self._populate_interval_table()
        self._populate_profile_table()
        self._render_one_well_curves()
        self._render_curve_across_wells()
        self._render_cluster_profile_view()
        self._set_status(
            f"Загружен результат Well Log: {len(visualization_data.get('rows', []))} строк, "
            f"{len(self.interval_cache)} скважин, {len(self.profile_cache)} кластеров.",
            "success",
        )

    def _populate_controls(self) -> None:
        data = self.visualization_data or {}
        rows = data.get("rows", [])
        wells = OrderedDict()
        for row in sorted(rows, key=lambda item: (str(item.get("well_name", "")), float(item.get("depth_md", 0.0)))):
            wells[int(row["well_id"])] = str(row.get("well_name", f"well_id={row['well_id']}"))
        self.well_combo.blockSignals(True)
        self.well_combo.clear()
        for well_id, well_name in wells.items():
            self.well_combo.addItem(f"{well_name} (id={well_id})", int(well_id))
        self.well_combo.blockSignals(False)

        self.curve_wells_list.blockSignals(True)
        self.curve_wells_list.clear()
        for row_idx, (well_id, well_name) in enumerate(wells.items()):
            item = QtWidgets.QListWidgetItem(f"{well_name} (id={well_id})")
            item.setData(Qt.UserRole, int(well_id))
            self.curve_wells_list.addItem(item)
            if row_idx < int(self.curve_wells_limit_spin.value()):
                item.setSelected(True)
        self.curve_wells_list.blockSignals(False)

        self.curve_combo.blockSignals(True)
        self.curve_combo.clear()
        self.curve_combo.addItems([str(name) for name in data.get("feature_names", [])])
        self.curve_combo.blockSignals(False)

        labels = sorted({int(row.get("cluster_label", 0)) for row in rows}, key=_cluster_label_sort_key)
        self.curve_wells_cluster_combo.blockSignals(True)
        self.curve_wells_cluster_combo.clear()
        for label in labels:
            self.curve_wells_cluster_combo.addItem(f"cluster {label}", int(label))
        self.curve_wells_cluster_combo.blockSignals(False)

        if hasattr(self, "profile_cluster_combo"):
            self.profile_cluster_combo.blockSignals(True)
            self.profile_cluster_combo.clear()
            for label in labels:
                self.profile_cluster_combo.addItem(f"cluster {label}", int(label))
            self.profile_cluster_combo.blockSignals(False)
        self._highlight_interval = None

        self.title_label.setText(
            f"Dataset: {data.get('dataset_title', '—')} (id={data.get('dataset_id', '—')}) | "
            f"run={data.get('run_id', '—')} | hash={str(data.get('data_hash', ''))[:12]}"
        )

    def _populate_legend(self) -> None:
        while self.legend_layout.count() > 1:
            item = self.legend_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._cluster_checkboxes.clear()
        labels = sorted(self.profile_cache.keys(), key=_cluster_label_sort_key)
        rows_by_label = Counter(int(row.get("cluster_label", 0)) for row in (self.visualization_data or {}).get("rows", []))
        for label in labels:
            checkbox = QtWidgets.QCheckBox(f"cluster {label} — {rows_by_label.get(label, 0)} rows")
            checkbox.setChecked(label != -1 or self.show_noise_checkbox.isChecked())
            checkbox.stateChanged.connect(self._apply_cluster_filter_to_tables)
            checkbox.setStyleSheet(f"QCheckBox {{ color: {_well_log_cluster_color(label)}; font-weight: 600; }}")
            self.legend_layout.insertWidget(self.legend_layout.count() - 1, checkbox)
            self._cluster_checkboxes[int(label)] = checkbox

    def _populate_summary(self) -> None:
        data = self.visualization_data or {}
        summary = data.get("summary", {})
        metrics = data.get("metrics", {}).get("metrics", {}) if isinstance(data.get("metrics"), dict) else {}
        diagnostics = data.get("diagnostics", {})
        lines = [
            "Well Log clustering result summary",
            "==================================",
            f"Dataset: {data.get('dataset_title', '—')} (id={data.get('dataset_id', '—')})",
            f"Run id: {data.get('run_id', '—')}",
            f"Created at: {data.get('created_at', '—')}",
            f"Rows with labels: {summary.get('row_count', 0)}",
            f"Wells: {summary.get('well_count', 0)}",
            f"Clusters (without noise): {summary.get('cluster_count', 0)}",
            f"Noise rows: {summary.get('noise_count', 0)}",
            f"Smoothing changes: {summary.get('smoothing_changes', 0)}",
            "",
            "Available modes:",
        ]
        lines.extend(f"  - {title}" for title in self.MODE_TITLES[1:])
        lines.extend(["", "Metrics:"])
        if metrics:
            lines.extend(f"  {name}: {_format_well_log_visual_metric(value)}" for name, value in metrics.items())
        else:
            lines.append("  metrics were not requested or unavailable")
        lines.extend(["", "Prepared caches:"])
        lines.append(f"  interval_cache wells: {len(self.interval_cache)}")
        lines.append(f"  interval_cache intervals: {sum(len(v) for v in self.interval_cache.values())}")
        lines.append(f"  profile_cache clusters: {len(self.profile_cache)}")
        lines.extend(["", "Diagnostics:"])
        for key in ("valid_well_count", "valid_row_count", "kept_row_count_after_clean", "dropped_row_count_after_clean"):
            if key in diagnostics:
                lines.append(f"  {key}: {diagnostics.get(key)}")
        self.summary_text.setPlainText("\n".join(lines))

    def _is_cluster_visible(self, label: int) -> bool:
        if int(label) == -1 and not self.show_noise_checkbox.isChecked():
            return False
        checkbox = self._cluster_checkboxes.get(int(label))
        return True if checkbox is None else bool(checkbox.isChecked())

    def _apply_cluster_filter_to_tables(self) -> None:
        if -1 in self._cluster_checkboxes:
            self._cluster_checkboxes[-1].blockSignals(True)
            self._cluster_checkboxes[-1].setChecked(self.show_noise_checkbox.isChecked() and self._cluster_checkboxes[-1].isChecked())
            self._cluster_checkboxes[-1].blockSignals(False)
        self._populate_interval_table()
        self._populate_profile_table()
        self._render_one_well_curves()
        self._render_curve_across_wells()
        self._render_cluster_profile_view()

    def _populate_interval_table(self) -> None:
        well_id = self._current_well_id()
        if well_id is None:
            intervals = []
        else:
            intervals = [
                item for item in self.interval_cache.get(int(well_id), [])
                if self._is_cluster_visible(int(item["cluster_label"]))
            ]
        self.interval_table.blockSignals(True)
        self.interval_table.setSortingEnabled(False)
        self.interval_table.setRowCount(len(intervals))
        for row_idx, interval in enumerate(intervals):
            values = [
                interval.get("well_name", "—"),
                _format_well_log_visual_metric(interval.get("from_md"), 2),
                _format_well_log_visual_metric(interval.get("to_md"), 2),
                _format_well_log_visual_metric(interval.get("thickness"), 2),
                str(interval.get("cluster_label", "—")),
                str(interval.get("row_count", 0)),
            ]
            for col_idx, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                item.setData(Qt.UserRole, interval)
                if col_idx == 4:
                    item.setForeground(QBrush(QColor(_well_log_cluster_color(int(interval.get("cluster_label", 0))))))
                self.interval_table.setItem(row_idx, col_idx, item)
        self.interval_table.setSortingEnabled(True)
        self.interval_table.blockSignals(False)
        self.interval_table.resizeColumnsToContents()

    def _current_well_id(self) -> int | None:
        value = self.well_combo.currentData()
        try:
            return int(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    def _handle_well_changed(self) -> None:
        self._highlight_interval = None
        self._populate_interval_table()
        self._render_one_well_curves()

    def _handle_interval_selection_changed(self) -> None:
        selected_items = self.interval_table.selectedItems()
        self._highlight_interval = None
        if selected_items:
            interval = selected_items[0].data(Qt.UserRole)
            if isinstance(interval, dict):
                self._highlight_interval = interval
        self._render_one_well_curves()

    def _rows_for_current_well(self) -> list[dict[str, Any]]:
        well_id = self._current_well_id()
        if well_id is None:
            return []
        rows = [
            row for row in (self.visualization_data or {}).get("rows", [])
            if int(row.get("well_id", -1)) == int(well_id)
            and self._is_cluster_visible(int(row.get("cluster_label", 0)))
        ]
        return sorted(rows, key=lambda item: (float(item.get("depth_md", 0.0)), int(item.get("row_index_in_well", 0))))

    def _render_one_well_curves(self) -> None:
        if not hasattr(self, "one_well_figure"):
            return
        rows = self._rows_for_current_well()
        feature_names = [str(name) for name in (self.visualization_data or {}).get("feature_names", [])]
        if not rows:
            self._prepare_scrollable_canvas(
                figure=self.one_well_figure,
                canvas=self.one_well_canvas,
                scroll_area=self.one_well_scroll,
                width_px=self._TRACK_WIDTH_PX * 3,
                height_px=560,
            )
            self._draw_empty_graph(self.one_well_figure, self.one_well_canvas, "Нет строк для выбранной скважины/фильтра кластеров")
            return

        finite_features: list[str] = []
        for feature_name in feature_names:
            if any(_to_finite_float((row.get("features", {}) or {}).get(feature_name)) is not None for row in rows):
                finite_features.append(feature_name)
        if not finite_features:
            self._prepare_scrollable_canvas(
                figure=self.one_well_figure,
                canvas=self.one_well_canvas,
                scroll_area=self.one_well_scroll,
                width_px=self._TRACK_WIDTH_PX * 3,
                height_px=560,
            )
            self._draw_empty_graph(self.one_well_figure, self.one_well_canvas, "У выбранной скважины нет числовых значений кривых")
            return

        depths = [float(row.get("depth_md", 0.0)) for row in rows]
        depth_min = min(depths)
        depth_max = max(depths)
        if depth_min == depth_max:
            depth_min -= 0.5
            depth_max += 0.5
        track_count = len(finite_features) + 1
        canvas_width = int(len(finite_features) * self._TRACK_WIDTH_PX + self._CLUSTER_TRACK_WIDTH_PX)
        self._prepare_scrollable_canvas(
            figure=self.one_well_figure,
            canvas=self.one_well_canvas,
            scroll_area=self.one_well_scroll,
            width_px=canvas_width,
            height_px=560,
        )
        axes = self.one_well_figure.subplots(1, track_count, sharey=True)
        if track_count == 1:
            axes = [axes]
        else:
            axes = list(np.ravel(axes))
        opacity = float(self.opacity_spin.value()) if hasattr(self, "opacity_spin") else 0.35
        well_id = int(rows[0].get("well_id", 0))
        well_name = str(rows[0].get("well_name", "—"))
        well_title = f"{well_name} (id={well_id})"
        intervals = [
            item for item in self.interval_cache.get(int(rows[0].get("well_id")), [])
            if self._is_cluster_visible(int(item.get("cluster_label", 0)))
        ]

        for feature_idx, feature_name in enumerate(finite_features):
            ax = axes[feature_idx]
            for interval in intervals:
                ax.axhspan(
                    float(interval.get("from_md", depth_min)),
                    float(interval.get("to_md", depth_max)),
                    color=_well_log_cluster_color(int(interval.get("cluster_label", 0))),
                    alpha=max(0.03, opacity * 0.18),
                    linewidth=0,
                )
            values = [_to_finite_float((row.get("features", {}) or {}).get(feature_name)) for row in rows]
            plot_x = [value for value in values if value is not None]
            plot_y = [depth for value, depth in zip(values, depths) if value is not None]
            if plot_x:
                ax.plot(plot_x, plot_y, linewidth=1.1, marker=".", markersize=3)
            ax.set_title(feature_name, fontsize=8, pad=3)
            ax.set_xlabel("")
            ax.tick_params(axis="x", labelsize=7, pad=1)
            ax.grid(True, alpha=0.25)
            if feature_idx == 0:
                ax.set_ylabel("Depth MD")
            else:
                ax.tick_params(labelleft=False)
            if self._highlight_interval is not None:
                ax.axhspan(
                    float(self._highlight_interval.get("from_md", depth_min)),
                    float(self._highlight_interval.get("to_md", depth_max)),
                    color="#ffe680",
                    alpha=0.35,
                    linewidth=0,
                )

        cluster_ax = axes[-1]
        cluster_ax.set_title("Cl", fontsize=8, pad=3)
        cluster_ax.set_xlim(0.0, 1.0)
        cluster_ax.set_xticks([])
        cluster_ax.grid(False)
        for interval in intervals:
            label = int(interval.get("cluster_label", 0))
            from_md = float(interval.get("from_md", depth_min))
            to_md = float(interval.get("to_md", depth_max))
            height = max(to_md - from_md, 1e-9)
            rect = Rectangle(
                (0.08, from_md),
                0.84,
                height,
                facecolor=_well_log_cluster_color(label),
                alpha=opacity,
                edgecolor="black" if self._intervals_match(interval, self._highlight_interval) else "none",
                linewidth=2.0 if self._intervals_match(interval, self._highlight_interval) else 0.0,
            )
            cluster_ax.add_patch(rect)
            if height > (depth_max - depth_min) * 0.025:
                cluster_ax.text(0.5, from_md + height / 2.0, str(label), ha="center", va="center", fontsize=8)
        cluster_ax.tick_params(labelleft=False)
        self.one_well_figure.suptitle(well_title, fontsize=11)
        for ax in axes:
            ax.set_ylim(depth_max, depth_min)
        self.one_well_figure.tight_layout(rect=(0, 0, 1, 0.94), pad=0.35, w_pad=0.35)
        self.one_well_canvas.draw()

    def _current_curve_name(self) -> str | None:
        value = self.curve_combo.currentText() if hasattr(self, "curve_combo") else ""
        value = str(value).strip()
        return value or None

    def _selected_curve_well_ids(self) -> list[int]:
        if not hasattr(self, "curve_wells_list"):
            return []
        selected_ids: list[int] = []
        for item in self.curve_wells_list.selectedItems():
            try:
                selected_ids.append(int(item.data(Qt.UserRole)))
            except (TypeError, ValueError):
                continue
        if selected_ids:
            return selected_ids
        fallback_ids: list[int] = []
        for row_idx in range(self.curve_wells_list.count()):
            item = self.curve_wells_list.item(row_idx)
            try:
                fallback_ids.append(int(item.data(Qt.UserRole)))
            except (TypeError, ValueError):
                continue
        return fallback_ids

    def _selected_curve_cluster_label(self) -> int | None:
        if not hasattr(self, "curve_wells_cluster_combo"):
            return None
        value = self.curve_wells_cluster_combo.currentData()
        try:
            return int(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    def _curve_well_payloads(self, curve_name: str) -> list[dict[str, Any]]:
        selected_well_ids = set(self._selected_curve_well_ids())
        grouped: dict[int, list[dict[str, Any]]] = {}
        for row in (self.visualization_data or {}).get("rows", []):
            well_id = int(row.get("well_id", -1))
            if selected_well_ids and well_id not in selected_well_ids:
                continue
            if not self._is_cluster_visible(int(row.get("cluster_label", 0))):
                continue
            value = _to_finite_float((row.get("features", {}) or {}).get(curve_name))
            if value is None:
                continue
            grouped.setdefault(well_id, []).append(row)

        selected_cluster = self._selected_curve_cluster_label()
        payloads: list[dict[str, Any]] = []
        for well_id, well_rows in grouped.items():
            sorted_rows = sorted(well_rows, key=lambda item: (float(item.get("depth_md", 0.0)), int(item.get("row_index_in_well", 0))))
            labels = [int(row.get("cluster_label", 0)) for row in sorted_rows]
            label_counts = Counter(labels)
            dominant_label, dominant_count = label_counts.most_common(1)[0] if label_counts else (None, 0)
            selected_count = label_counts.get(selected_cluster, 0) if selected_cluster is not None else dominant_count
            payloads.append(
                {
                    "well_id": int(well_id),
                    "well_name": str(sorted_rows[0].get("well_name", f"well_id={well_id}")),
                    "rows": sorted_rows,
                    "labels": labels,
                    "label_counts": label_counts,
                    "dominant_label": dominant_label,
                    "dominant_fraction": float(dominant_count / len(sorted_rows)) if sorted_rows else 0.0,
                    "selected_cluster_fraction": float(selected_count / len(sorted_rows)) if sorted_rows else 0.0,
                }
            )
        return payloads

    @staticmethod
    def _cluster_sequence_distance(labels: list[int], reference_labels: list[int]) -> float:
        if not labels or not reference_labels:
            return 1.0
        sample_count = min(64, len(labels), len(reference_labels))
        if sample_count <= 0:
            return 1.0
        mismatches = 0
        for idx in range(sample_count):
            left_idx = int(round(idx * (len(labels) - 1) / max(1, sample_count - 1)))
            right_idx = int(round(idx * (len(reference_labels) - 1) / max(1, sample_count - 1)))
            if int(labels[left_idx]) != int(reference_labels[right_idx]):
                mismatches += 1
        return float(mismatches / sample_count)

    def _sort_curve_well_payloads(self, payloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
        sort_idx = self.curve_wells_sort_combo.currentIndex() if hasattr(self, "curve_wells_sort_combo") else 0
        if sort_idx == 1:
            return sorted(payloads, key=lambda item: (-float(item.get("selected_cluster_fraction", 0.0)), str(item.get("well_name", ""))))
        if sort_idx == 2:
            reference = min(payloads, key=lambda item: str(item.get("well_name", "")))["labels"] if payloads else []
            return sorted(
                payloads,
                key=lambda item: (self._cluster_sequence_distance(item.get("labels", []), reference), str(item.get("well_name", ""))),
            )
        return sorted(payloads, key=lambda item: str(item.get("well_name", "")))

    def _populate_curve_wells_summary_table(self, payloads: list[dict[str, Any]]) -> None:
        if not hasattr(self, "curve_wells_summary_table"):
            return
        self.curve_wells_summary_table.setSortingEnabled(False)
        self.curve_wells_summary_table.setRowCount(len(payloads))
        for row_idx, payload in enumerate(payloads):
            label_counts: Counter = payload.get("label_counts", Counter())
            cluster_summary = ", ".join(
                f"{label}:{count} ({count / max(1, len(payload.get('rows', []))):.0%})"
                for label, count in sorted(label_counts.items(), key=lambda pair: _cluster_label_sort_key(pair[0]))
            )
            values = [
                str(payload.get("well_name", "—")),
                str(len(payload.get("rows", []))),
                str(payload.get("dominant_label", "—")),
                f"{float(payload.get('selected_cluster_fraction', 0.0)) * 100:.1f}%",
                cluster_summary or "—",
            ]
            for col_idx, value in enumerate(values):
                item = QTableWidgetItem(value)
                if col_idx == 2 and payload.get("dominant_label") is not None:
                    item.setForeground(QBrush(QColor(_well_log_cluster_color(int(payload["dominant_label"])))))
                self.curve_wells_summary_table.setItem(row_idx, col_idx, item)
        self.curve_wells_summary_table.setSortingEnabled(True)
        self.curve_wells_summary_table.resizeColumnsToContents()

    def _render_curve_across_wells(self) -> None:
        if not hasattr(self, "curve_wells_figure"):
            return
        curve_name = self._current_curve_name()
        if not curve_name:
            self._prepare_scrollable_canvas(
                figure=self.curve_wells_figure,
                canvas=self.curve_wells_canvas,
                scroll_area=self.curve_wells_scroll,
                width_px=self._TRACK_WIDTH_PX * 3,
                height_px=600,
            )
            self._draw_empty_graph(self.curve_wells_figure, self.curve_wells_canvas, "Выберите кривую для сравнения")
            self._populate_curve_wells_summary_table([])
            return

        payloads = self._sort_curve_well_payloads(self._curve_well_payloads(curve_name))
        limit = int(self.curve_wells_limit_spin.value()) if hasattr(self, "curve_wells_limit_spin") else 12
        payloads = payloads[:max(1, limit)]
        self._populate_curve_wells_summary_table(payloads)
        if not payloads:
            self._prepare_scrollable_canvas(
                figure=self.curve_wells_figure,
                canvas=self.curve_wells_canvas,
                scroll_area=self.curve_wells_scroll,
                width_px=self._TRACK_WIDTH_PX * 3,
                height_px=600,
            )
            self._draw_empty_graph(
                self.curve_wells_figure,
                self.curve_wells_canvas,
                "Нет числовых значений выбранной кривой для выбранных скважин/кластеров",
            )
            return

        shared_x = bool(self.curve_wells_shared_x.isChecked()) if hasattr(self, "curve_wells_shared_x") else True
        shared_depth = bool(self.curve_wells_shared_depth.isChecked()) if hasattr(self, "curve_wells_shared_depth") else False
        all_values = []
        all_depths = []
        for payload in payloads:
            payload_depths = [float(row.get("depth_md", 0.0)) for row in payload["rows"]]
            payload["depth_min"] = min(payload_depths) if payload_depths else None
            payload["depth_max"] = max(payload_depths) if payload_depths else None
            all_depths.extend(payload_depths)
            if shared_x:
                for row in payload["rows"]:
                    value = _to_finite_float((row.get("features", {}) or {}).get(curve_name))
                    if value is not None:
                        all_values.append(value)
        global_x_min = min(all_values) if all_values else None
        global_x_max = max(all_values) if all_values else None
        global_depth_min = min(all_depths) if all_depths else None
        global_depth_max = max(all_depths) if all_depths else None
        if global_depth_min is not None and global_depth_max is not None and global_depth_min == global_depth_max:
            global_depth_min -= 0.5
            global_depth_max += 0.5
        opacity = float(self.opacity_spin.value()) if hasattr(self, "opacity_spin") else 0.35

        canvas_width = int(len(payloads) * self._TRACK_WIDTH_PX)
        self._prepare_scrollable_canvas(
            figure=self.curve_wells_figure,
            canvas=self.curve_wells_canvas,
            scroll_area=self.curve_wells_scroll,
            width_px=canvas_width,
            height_px=600,
        )
        axes = self.curve_wells_figure.subplots(1, len(payloads), sharex=shared_x, sharey=shared_depth)
        axes = [axes] if len(payloads) == 1 else list(np.ravel(axes))
        for ax, payload in zip(axes, payloads):
            rows = payload["rows"]
            depths = [float(row.get("depth_md", 0.0)) for row in rows]
            values = [_to_finite_float((row.get("features", {}) or {}).get(curve_name)) for row in rows]
            plot_points = [(value, depth) for value, depth in zip(values, depths) if value is not None]
            if not plot_points:
                continue
            plot_x, plot_y = zip(*plot_points)
            depth_min = float(payload.get("depth_min") if payload.get("depth_min") is not None else min(depths))
            depth_max = float(payload.get("depth_max") if payload.get("depth_max") is not None else max(depths))
            if shared_depth and global_depth_min is not None and global_depth_max is not None:
                depth_min = float(global_depth_min)
                depth_max = float(global_depth_max)
            elif depth_min == depth_max:
                depth_min -= 0.5
                depth_max += 0.5
            ax.set_ylim(depth_max, depth_min)
            for interval in self.interval_cache.get(int(payload["well_id"]), []):
                label = int(interval.get("cluster_label", 0))
                if not self._is_cluster_visible(label):
                    continue
                ax.axhspan(
                    float(interval.get("from_md", depth_min)),
                    float(interval.get("to_md", depth_max)),
                    color=_well_log_cluster_color(label),
                    alpha=max(0.03, opacity * 0.22),
                    linewidth=0,
                )
            ax.plot(plot_x, plot_y, linewidth=1.15, marker=".", markersize=3, color="#1f2933")
            if shared_x and global_x_min is not None and global_x_max is not None and global_x_min != global_x_max:
                ax.set_xlim(global_x_min, global_x_max)
            ax.grid(True, alpha=0.25)
            ax.set_xlabel("")
            ax.set_ylabel("Depth MD" if ax is axes[0] else "", fontsize=8)
            ax.tick_params(axis="both", labelsize=7, pad=1)
            if ax is not axes[0]:
                ax.tick_params(labelleft=False)
            ax.set_title(f"{payload['well_name']} (id={payload['well_id']})", fontsize=8, pad=3)
        depth_mode = "общая глубина" if shared_depth else "своя глубина"
        self.curve_wells_figure.suptitle(f"{curve_name} · {depth_mode}", fontsize=11)
        self.curve_wells_figure.tight_layout(rect=(0, 0, 1, 0.94), pad=0.35, w_pad=0.35)
        self.curve_wells_canvas.draw()

    def _current_profile_cluster_label(self) -> int | None:
        if not hasattr(self, "profile_cluster_combo"):
            return None
        value = self.profile_cluster_combo.currentData()
        try:
            return int(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    def _render_cluster_profile_view(self) -> None:
        if not hasattr(self, "cluster_profile_figure"):
            return
        self.cluster_profile_figure.clear()
        labels = [
            int(label) for label in sorted(self.profile_cache.keys(), key=_cluster_label_sort_key)
            if self._is_cluster_visible(int(label))
        ]
        feature_names = [str(name) for name in (self.visualization_data or {}).get("feature_names", [])]
        if not labels or not feature_names:
            ax = self.cluster_profile_figure.add_subplot(111)
            ax.text(0.5, 0.5, "Нет профилей для выбранных кластеров/кривых", ha="center", va="center")
            ax.set_axis_off()
            self.cluster_profile_canvas.draw_idle()
            if hasattr(self, "cluster_profile_description"):
                self.cluster_profile_description.setPlainText("Нет данных для среднего портрета кластера.")
            return

        selected_label = self._current_profile_cluster_label()
        if selected_label not in self.profile_cache or not self._is_cluster_visible(int(selected_label)):
            selected_label = labels[0]

        matrix = np.full((len(labels), len(feature_names)), np.nan, dtype=float)
        for row_idx, label in enumerate(labels):
            features = (self.profile_cache.get(int(label), {}) or {}).get("features", {}) or {}
            for col_idx, feature_name in enumerate(feature_names):
                value = _to_finite_float((features.get(feature_name, {}) or {}).get("standardized_mean"))
                if value is not None:
                    matrix[row_idx, col_idx] = float(value)

        canvas_width = max(900, int(len(feature_names) * 85))
        self.cluster_profile_canvas.setMinimumSize(canvas_width, 560)
        self.cluster_profile_figure.set_size_inches(max(12.0, len(feature_names) * 0.75), 6.7, forward=True)
        axes = self.cluster_profile_figure.subplots(1, 2, gridspec_kw={"width_ratios": [1.45, 1.0]})
        heatmap_ax, bar_ax = axes
        finite_values = matrix[np.isfinite(matrix)]
        value_limit = max(1.0, float(np.nanmax(np.abs(finite_values)))) if finite_values.size else 1.0
        image = heatmap_ax.imshow(matrix, aspect="auto", cmap="coolwarm", vmin=-value_limit, vmax=value_limit)
        heatmap_ax.set_yticks(range(len(labels)))
        heatmap_ax.set_yticklabels([f"cluster {label}" for label in labels])
        heatmap_ax.set_xticks(range(len(feature_names)))
        heatmap_ax.set_xticklabels(feature_names, rotation=55, ha="right", fontsize=8)
        heatmap_ax.set_title("Heatmap: cluster × feature (z-mean)")
        heatmap_ax.set_xlabel("Feature")
        heatmap_ax.set_ylabel("Cluster")
        if selected_label in labels:
            selected_row = labels.index(int(selected_label))
            heatmap_ax.add_patch(Rectangle((-0.5, selected_row - 0.5), len(feature_names), 1, fill=False, edgecolor="black", linewidth=1.8))
        self.cluster_profile_figure.colorbar(image, ax=heatmap_ax, fraction=0.046, pad=0.04, label="standardized mean")

        selected_profile = self.profile_cache.get(int(selected_label), {}) if selected_label is not None else {}
        selected_features = []
        for feature_name, stats in (selected_profile.get("features", {}) or {}).items():
            z_mean = _to_finite_float((stats or {}).get("standardized_mean"))
            if z_mean is not None:
                selected_features.append((str(feature_name), float(z_mean)))
        selected_features.sort(key=lambda item: abs(item[1]), reverse=True)
        top_n = int(self.profile_top_features_spin.value()) if hasattr(self, "profile_top_features_spin") else 15
        selected_features = selected_features[:max(1, top_n)]
        if selected_features:
            names = [item[0] for item in selected_features][::-1]
            values = [item[1] for item in selected_features][::-1]
            colors = ["#d62728" if value > 0 else "#1f77b4" for value in values]
            bar_ax.barh(names, values, color=colors, alpha=0.85)
            bar_ax.axvline(0.0, color="black", linewidth=0.8)
            bar_ax.set_xlabel("standardized mean")
            bar_ax.set_title(f"Cluster {selected_label}: strongest deviations")
            bar_ax.grid(True, axis="x", alpha=0.25)
        else:
            bar_ax.text(0.5, 0.5, "Нет числовой статистики для выбранного кластера", ha="center", va="center")
            bar_ax.set_axis_off()

        self.cluster_profile_figure.tight_layout()
        self.cluster_profile_canvas.draw_idle()
        if hasattr(self, "cluster_profile_description"):
            self.cluster_profile_description.setPlainText(str(selected_profile.get("description", "Нет описания кластера.")))

    @staticmethod
    def _build_cluster_profile_description(label: int, profile: dict[str, Any]) -> str:
        row_count = int(profile.get("row_count", 0))
        well_count = int(profile.get("well_count", 0))
        depth_min = profile.get("depth_min")
        depth_max = profile.get("depth_max")
        features = profile.get("features", {}) or {}
        ranked = []
        for feature_name, stats in features.items():
            z_mean = _to_finite_float((stats or {}).get("standardized_mean"))
            mean_value = _to_finite_float((stats or {}).get("mean"))
            if z_mean is not None:
                ranked.append((str(feature_name), float(z_mean), mean_value))
        ranked.sort(key=lambda item: abs(item[1]), reverse=True)
        high = [item for item in ranked if item[1] > 0][:3]
        low = [item for item in ranked if item[1] < 0][:3]

        def format_feature(items: list[tuple[str, float, float | None]]) -> str:
            if not items:
                return "нет выраженных отклонений"
            chunks = []
            for feature_name, z_mean, mean_value in items:
                mean_text = _format_well_log_visual_metric(mean_value, 4)
                chunks.append(f"{feature_name} (z={z_mean:+.2f}, mean={mean_text})")
            return "; ".join(chunks)

        depth_text = "—"
        if depth_min is not None and depth_max is not None:
            depth_text = f"{_format_well_log_visual_metric(depth_min, 2)}–{_format_well_log_visual_metric(depth_max, 2)} MD"
        return "\n".join([
            f"Cluster {label}: {row_count} rows, {well_count} wells, depth range {depth_text}.",
            f"Повышенные средние относительно всего dataset: {format_feature(high)}.",
            f"Пониженные средние относительно всего dataset: {format_feature(low)}.",
            "Интерпретация: признаки с |z|≈1 и выше сильнее всего отличают кластер от общего среднего; "
            "знак z показывает направление отклонения.",
        ])

    @staticmethod
    def _intervals_match(left: dict[str, Any] | None, right: dict[str, Any] | None) -> bool:
        if not left or not right:
            return False
        return (
            int(left.get("well_id", -1)) == int(right.get("well_id", -2))
            and int(left.get("cluster_label", -999999)) == int(right.get("cluster_label", -999998))
            and abs(float(left.get("from_md", 0.0)) - float(right.get("from_md", 0.0))) < 1e-9
            and abs(float(left.get("to_md", 0.0)) - float(right.get("to_md", 0.0))) < 1e-9
        )

    def _populate_profile_table(self) -> None:
        profile_rows = []
        for label, profile in sorted(self.profile_cache.items(), key=lambda pair: _cluster_label_sort_key(pair[0])):
            if not self._is_cluster_visible(int(label)):
                continue
            for feature_name, stats in sorted((profile.get("features", {}) or {}).items()):
                profile_rows.append((label, feature_name, stats))
        self.profile_table.setSortingEnabled(False)
        self.profile_table.setRowCount(len(profile_rows))
        for row_idx, (label, feature_name, stats) in enumerate(profile_rows):
            values = [
                str(label),
                str(feature_name),
                str(stats.get("count", 0)),
                _format_well_log_visual_metric(stats.get("mean"), 4),
                _format_well_log_visual_metric(stats.get("standardized_mean"), 4),
                _format_well_log_visual_metric(stats.get("median"), 4),
                _format_well_log_visual_metric(stats.get("std"), 4),
                f"{_format_well_log_visual_metric(stats.get('p10'), 4)}–{_format_well_log_visual_metric(stats.get('p90'), 4)}",
            ]
            for col_idx, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                if col_idx == 0:
                    item.setForeground(QBrush(QColor(_well_log_cluster_color(int(label)))))
                self.profile_table.setItem(row_idx, col_idx, item)
        self.profile_table.setSortingEnabled(True)
        self.profile_table.resizeColumnsToContents()

    def _sync_mode_tab(self, index: int) -> None:
        if 0 <= int(index) < self.mode_tabs.count() and self.mode_tabs.currentIndex() != int(index):
            self.mode_tabs.setCurrentIndex(int(index))

    def _sync_mode_combo(self, index: int) -> None:
        if 0 <= int(index) < self.mode_combo.count() and self.mode_combo.currentIndex() != int(index):
            self.mode_combo.setCurrentIndex(int(index))
        if int(index) == 1:
            self._render_one_well_curves()
        elif int(index) == 2:
            self._render_curve_across_wells()
        elif int(index) == 3:
            self._render_cluster_profile_view()

    @staticmethod
    def _build_interval_cache(rows: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
        grouped: dict[int, list[dict[str, Any]]] = {}
        for row in rows:
            grouped.setdefault(int(row["well_id"]), []).append(row)
        interval_cache: dict[int, list[dict[str, Any]]] = {}
        for well_id, well_rows in grouped.items():
            sorted_rows = sorted(well_rows, key=lambda item: (float(item.get("depth_md", 0.0)), int(item.get("row_index_in_well", 0))))
            depths = [float(row.get("depth_md", 0.0)) for row in sorted_rows]
            diffs = [next_depth - depth for depth, next_depth in zip(depths, depths[1:]) if next_depth > depth]
            default_step = float(np.median(diffs)) if diffs else 0.0
            intervals: list[dict[str, Any]] = []
            current: dict[str, Any] | None = None

            def close_current(next_depth: float | None = None) -> None:
                nonlocal current
                if current is None:
                    return
                last_depth = float(current.pop("last_depth", current["from_md"]))
                current["to_md"] = float(next_depth) if next_depth is not None and next_depth > last_depth else last_depth + default_step
                current["thickness"] = max(0.0, float(current["to_md"]) - float(current["from_md"]))
                intervals.append(current)
                current = None

            for row_idx, row in enumerate(sorted_rows):
                label = int(row.get("cluster_label", 0))
                depth = float(row.get("depth_md", 0.0))
                if current is None or int(current["cluster_label"]) != label:
                    close_current(depth)
                    current = {
                        "well_id": int(well_id),
                        "well_name": str(row.get("well_name", f"well_id={well_id}")),
                        "from_md": depth,
                        "to_md": depth,
                        "last_depth": depth,
                        "cluster_label": label,
                        "row_count": 1,
                        "row_index_start": int(row.get("row_index_in_well", row_idx)),
                        "row_index_end": int(row.get("row_index_in_well", row_idx)),
                    }
                else:
                    current["last_depth"] = depth
                    current["row_count"] = int(current.get("row_count", 0)) + 1
                    current["row_index_end"] = int(row.get("row_index_in_well", row_idx))
            close_current(None)
            interval_cache[int(well_id)] = intervals
        return interval_cache

    @staticmethod
    def _build_profile_cache(rows: list[dict[str, Any]], feature_names: list[str]) -> dict[int, dict[str, Any]]:
        global_feature_stats: dict[str, dict[str, float]] = {}
        for feature_name in feature_names:
            values = []
            for row in rows:
                value = (row.get("features", {}) or {}).get(feature_name)
                value_float = _to_finite_float(value)
                if value_float is not None:
                    values.append(value_float)
            if values:
                arr = np.asarray(values, dtype=float)
                global_feature_stats[str(feature_name)] = {
                    "mean": float(np.mean(arr)),
                    "std": float(np.std(arr)),
                    "count": int(arr.size),
                }

        grouped: dict[int, list[dict[str, Any]]] = {}
        for row in rows:
            grouped.setdefault(int(row.get("cluster_label", 0)), []).append(row)
        profile_cache: dict[int, dict[str, Any]] = {}
        for label, cluster_rows in grouped.items():
            features: dict[str, Any] = {}
            depths = []
            for row in cluster_rows:
                depth_value = _to_finite_float(row.get("depth_md"))
                if depth_value is not None:
                    depths.append(depth_value)
            for feature_name in feature_names:
                values = []
                for row in cluster_rows:
                    value = (row.get("features", {}) or {}).get(feature_name)
                    value_float = _to_finite_float(value)
                    if value_float is not None:
                        values.append(value_float)
                if not values:
                    continue
                arr = np.asarray(values, dtype=float)
                global_stats = global_feature_stats.get(str(feature_name), {})
                global_std = float(global_stats.get("std", 0.0) or 0.0)
                mean_value = float(np.mean(arr))
                standardized_mean = None
                if global_std > 1e-12:
                    standardized_mean = float((mean_value - float(global_stats.get("mean", 0.0))) / global_std)
                features[str(feature_name)] = {
                    "count": int(arr.size),
                    "mean": mean_value,
                    "median": float(np.median(arr)),
                    "std": float(np.std(arr)),
                    "p10": float(np.percentile(arr, 10)),
                    "p90": float(np.percentile(arr, 90)),
                    "global_mean": global_stats.get("mean"),
                    "global_std": global_stats.get("std"),
                    "standardized_mean": standardized_mean,
                    "valid_fraction_in_cluster": float(arr.size / max(1, len(cluster_rows))),
                }
            profile = {
                "cluster_label": int(label),
                "row_count": len(cluster_rows),
                "well_count": len({int(row["well_id"]) for row in cluster_rows}),
                "depth_min": float(min(depths)) if depths else None,
                "depth_max": float(max(depths)) if depths else None,
                "features": features,
            }
            profile["description"] = WellLogClusterVisualizationWindow._build_cluster_profile_description(int(label), profile)
            profile_cache[int(label)] = profile
        return profile_cache


def show_well_log_cluster_visualization(visualization_data: WellLogClusterVisualizationData) -> None:
    """
    Открывает или обновляет MVP-окно визуализации Well Log с summary, контролами,
    легендой, фильтром кластеров и подготовленными кэшами интервалов/профилей.
    """
    global well_log_cluster_visualization_window
    try:
        dialog = well_log_cluster_visualization_window
        if dialog is None or not isinstance(dialog, WellLogClusterVisualizationWindow):
            dialog = WellLogClusterVisualizationWindow(MainWindow)
            well_log_cluster_visualization_window = dialog
        dialog.load_visualization_data(visualization_data)
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()
    except Exception as exc:
        set_info(f"Не удалось открыть окно визуализации Well Log: {exc}", "brown")


def show_well_log_cluster_visualization_stub(visualization_data: WellLogClusterVisualizationData) -> None:
    """Backward-compatible alias для старого имени MVP-окна визуализации."""
    show_well_log_cluster_visualization(visualization_data)

