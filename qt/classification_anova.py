"""Classification-only controls layered on the shared ANOVA form."""
import math

from PyQt5 import QtCore, QtWidgets
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
import seaborn as sns

from anova_comparison import TRANSFORMS, compare_anova
from qt.anova_form import Ui_Anova


class ClassificationAnovaDialog(QtWidgets.QDialog):
    def __init__(self, data, features, palette, parent=None):
        super().__init__(parent)
        self.ui = Ui_Anova()
        self.ui.setupUi(self)
        self.data = data.copy()
        self.features = list(features)
        self.palette = palette
        self.frames = {}
        self.ui.listWidget_checkbox_well.hide()
        controls = QtWidgets.QGroupBox('Преобразования перед ANOVA (независимые варианты)')
        layout = QtWidgets.QGridLayout(controls)
        self.checks = {}
        for index, (method, title) in enumerate(TRANSFORMS.items()):
            check = QtWidgets.QCheckBox(title)
            check.setObjectName('anova_' + method)
            check.setChecked(method == 'raw')
            check.setToolTip('По строкам, на всём наборе признаков' if method in ('l1', 'l2', 'clr')
                             else 'Отдельно для каждого признака')
            layout.addWidget(check, index // 5, index % 5)
            self.checks[method] = check
        self.pseudocount = QtWidgets.QDoubleSpinBox()
        self.pseudocount.setDecimals(10)
        self.pseudocount.setRange(0, 1e10)
        self.pseudocount.setValue(1e-6)
        self.pseudocount.setToolTip('Заменяет только нули в CLR; отрицательные значения запрещены')
        layout.addWidget(QtWidgets.QLabel('CLR: замена нулей'), 2, 0)
        layout.addWidget(self.pseudocount, 2, 1)
        self.apply_button = QtWidgets.QPushButton('Рассчитать сравнение')
        layout.addWidget(self.apply_button, 2, 2)
        self.summary = QtWidgets.QLabel()
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary, 3, 0, 1, 5)
        self.ui.gridLayout.addWidget(controls, 2, 0)
        self.table = QtWidgets.QTableWidget()
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.ui.gridLayout.addWidget(self.table, 3, 0)
        self.ui.gridLayout.setRowStretch(3, 2)
        self.figure = Figure()
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.ui.horizontalLayout.addWidget(self.canvas)
        self.ui.listWidget.addItems(sorted(self.features))
        self.ui.listWidget.currentItemChanged.connect(self.draw_graph)
        for radio in (self.ui.radioButton_box, self.ui.radioButton_boxen,
                      self.ui.radioButton_strip, self.ui.radioButton_violin):
            radio.clicked.connect(self.draw_graph)
        self.apply_button.clicked.connect(self.recalculate)
        self.ui.listWidget.setCurrentRow(0)
        self.recalculate()

    def recalculate(self):
        methods = [method for method, check in self.checks.items() if check.isChecked()]
        if not methods:
            self.summary.setText('Выберите хотя бы один вариант и нажмите «Рассчитать сравнение».')
            self.frames = {}
            self.table.clear()
            self.table.setRowCount(0)
            self.draw_graph()
            return
        results, self.frames = compare_anova(
            self.data[self.features], self.data['mark'], methods, self.pseudocount.value())
        self.table.setSortingEnabled(False)
        self.table.clear()
        self.table.setColumnCount(1 + 4 * len(methods))
        self.table.setRowCount(len(self.features))
        headers = ['Признак']
        summaries = []
        for method in methods:
            headers.extend(f'{TRANSFORMS[method]}: {metric}' for metric in ('F', 'p', 'FDR', 'Статус'))
            subset = results[results.method == method]
            summaries.append(f'{TRANSFORMS[method]}: {(subset.FDR < 0.05).sum()} значимых, '
                             f'{(subset.status != "").sum()} недоступных')
        self.table.setHorizontalHeaderLabels(headers)
        for row, feature in enumerate(self.features):
            self.table.setItem(row, 0, QtWidgets.QTableWidgetItem(str(feature)))
            for index, method in enumerate(methods):
                record = results[(results.feature == feature) & (results.method == method)].iloc[0]
                for offset, key in enumerate(('F', 'p', 'FDR', 'status')):
                    value = record[key]
                    item = QtWidgets.QTableWidgetItem()
                    if key == 'status':
                        item.setText(value or 'OK')
                        item.setToolTip(value)
                    elif not math.isnan(value):
                        item.setData(QtCore.Qt.DisplayRole, float(value))
                    else:
                        item.setText('—')
                    self.table.setItem(row, 1 + 4 * index + offset, item)
        self.table.setSortingEnabled(True)
        self.table.resizeColumnsToContents()
        self.summary.setText('FDR (Бенджамини–Хохберг) < 0.05, поправка отдельно по каждому варианту. '
                             + '; '.join(summaries))
        self.draw_graph()

    def draw_graph(self, *_):
        self.figure.clear()
        current = self.ui.listWidget.currentItem()
        if current and self.frames:
            feature = current.text()
            columns = min(3, len(self.frames))
            rows = math.ceil(len(self.frames) / columns)
            for index, (method, frame) in enumerate(self.frames.items()):
                ax = self.figure.add_subplot(rows, columns, index + 1)
                plot = frame[[feature]].copy()
                plot['mark'] = self.data['mark']
                if plot[feature].notna().any():
                    kind = (sns.boxplot if self.ui.radioButton_box.isChecked() else
                            sns.violinplot if self.ui.radioButton_violin.isChecked() else
                            sns.stripplot if self.ui.radioButton_strip.isChecked() else sns.boxenplot)
                    kind(data=plot, x='mark', y=feature, palette=self.palette, ax=ax)
                else:
                    ax.text(0.5, 0.5, 'Преобразование недоступно', ha='center', transform=ax.transAxes)
                ax.set_title(TRANSFORMS[method])
            self.figure.tight_layout()
        self.canvas.draw()
