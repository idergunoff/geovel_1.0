"""Terrain catalog and object assignments, independent of the global main window."""
import json

import numpy as np
from PyQt5 import QtCore, QtWidgets
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT

from models_db.model import CommonGrid, GeoradarObject
from terrain_service import (TerrainError, add_grid, bind_grid, delete_grid, surface,
                             validate_object_grid, object_profiles, profile_points, invalidate_profile)


class TerrainManager(QtWidgets.QDialog):
    def __init__(self, db, parent=None, object_id=None):
        super().__init__(parent)
        self.db = db
        self.setWindowTitle('Сетки рельефа')
        self.resize(1150, 780)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(QtWidgets.QLabel(
            'Сетки XYZ в системе координат профилей (Пулково), высоты в метрах. '
            'Привязанная сетка должна полностью покрывать все профили объекта.'))
        self.table = QtWidgets.QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(['Название', 'Файл', 'Шаг X / Y, м', 'Узлов', 'Загружена', 'Область применения', 'Объекты'])
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.table, 2)
        actions = QtWidgets.QHBoxLayout()
        self.add_button = QtWidgets.QPushButton('Загрузить сетку…')
        self.delete_button = QtWidgets.QPushButton('Удалить')
        actions.addWidget(self.add_button)
        actions.addWidget(self.delete_button)
        actions.addStretch()
        layout.addLayout(actions)
        splitter = QtWidgets.QSplitter()
        left = QtWidgets.QWidget()
        fields = QtWidgets.QVBoxLayout(left)
        self.title = QtWidgets.QLineEdit()
        self.comment = QtWidgets.QPlainTextEdit()
        self.comment.setMaximumHeight(75)
        self.common = QtWidgets.QCheckBox('Общая сетка для объектов без привязки')
        self.common.setToolTip('Без привязки выбирается первая общая сетка с полным покрытием. '
                               'Локальные сетки используются только при явной привязке.')
        form = QtWidgets.QFormLayout()
        form.addRow('Название', self.title)
        form.addRow('Комментарий', self.comment)
        fields.addLayout(form)
        fields.addWidget(self.common)
        self.save_button = QtWidgets.QPushButton('Сохранить изменения')
        fields.addWidget(self.save_button)
        fields.addWidget(QtWidgets.QLabel('Объекты (можно выбрать несколько):'))
        self.objects = QtWidgets.QListWidget()
        self.objects.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        fields.addWidget(self.objects)
        self.bind_button = QtWidgets.QPushButton('Привязать сетку к выбранным объектам')
        self.unbind_button = QtWidgets.QPushButton('Вернуть выбранным объектам общую сетку')
        fields.addWidget(self.bind_button)
        fields.addWidget(self.unbind_button)
        splitter.addWidget(left)
        right = QtWidgets.QWidget()
        preview = QtWidgets.QVBoxLayout(right)
        self.figure = Figure()
        self.canvas = FigureCanvasQTAgg(self.figure)
        preview.addWidget(NavigationToolbar2QT(self.canvas, self))
        preview.addWidget(self.canvas)
        self.bounds = QtWidgets.QLabel()
        self.bounds.setWordWrap(True)
        preview.addWidget(self.bounds)
        splitter.addWidget(right)
        splitter.setSizes([420, 650])
        layout.addWidget(splitter, 3)
        close = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
        close.rejected.connect(self.reject)
        layout.addWidget(close)
        self.add_button.clicked.connect(self.upload)
        self.delete_button.clicked.connect(self.remove)
        self.save_button.clicked.connect(self.save)
        self.bind_button.clicked.connect(lambda: self.assign(False))
        self.unbind_button.clicked.connect(lambda: self.assign(True))
        self.table.itemSelectionChanged.connect(self.show_grid)
        self.objects.itemSelectionChanged.connect(self.preview)
        self.refresh(object_id=object_id)

    def grid(self):
        row = self.table.currentRow()
        return self.db.get(CommonGrid, self.table.item(row, 0).data(QtCore.Qt.UserRole)) if row >= 0 else None

    def refresh(self, selected_id=None, object_id=None):
        selected_objects = {item.data(QtCore.Qt.UserRole) for item in self.objects.selectedItems()}
        if object_id:
            selected_objects.add(object_id)
        self.objects.blockSignals(True)
        self.objects.clear()
        objects = self.db.query(GeoradarObject).order_by(GeoradarObject.title).all()
        assignments = {}
        for obj in objects:
            grid = self.db.get(CommonGrid, obj.relief_grid_id) if obj.relief_grid_id else None
            item = QtWidgets.QListWidgetItem(f'{obj.title} — {grid.title if grid else "общая сетка"}')
            item.setData(QtCore.Qt.UserRole, obj.id)
            self.objects.addItem(item)
            item.setSelected(obj.id in selected_objects)
            assignments.setdefault(obj.relief_grid_id, []).append(obj.title)
        self.objects.blockSignals(False)
        self.table.blockSignals(True)
        grids = self.db.query(CommonGrid).filter_by(type='r').order_by(CommonGrid.id).all()
        self.table.setRowCount(len(grids))
        selected_row = 0
        for row, grid in enumerate(grids):
            try:
                meta = surface(grid).metadata()
                step = ' / '.join(f'{value:g}' for value in meta['step'])
                nodes = str(meta['nodes'])
            except TerrainError:
                step, nodes = 'Ошибка сетки', '—'
            values = [grid.title, grid.source_name or '—', step, nodes,
                      grid.created_at.strftime('%d.%m.%Y %H:%M') if grid.created_at else '—',
                      'Локальная' if grid.is_local else 'Общая', ', '.join(assignments.get(grid.id, []))]
            for column, value in enumerate(values):
                item = QtWidgets.QTableWidgetItem(value)
                item.setToolTip(grid.comment or '')
                self.table.setItem(row, column, item)
            self.table.item(row, 0).setData(QtCore.Qt.UserRole, grid.id)
            if grid.id == selected_id:
                selected_row = row
        self.table.blockSignals(False)
        if grids:
            self.table.selectRow(selected_row)
        self.show_grid()

    def show_grid(self):
        grid = self.grid()
        self.title.setText(grid.title if grid else '')
        self.comment.setPlainText(grid.comment or '' if grid else '')
        self.common.setChecked(bool(grid and not grid.is_local))
        for widget in (self.save_button, self.bind_button, self.delete_button):
            widget.setEnabled(grid is not None)
        self.preview()

    def preview(self):
        self.figure.clear()
        axis = self.figure.add_subplot(111)
        grid = self.grid()
        self.bounds.clear()
        if grid:
            try:
                terrain = surface(grid)
                # Bound preview work independently of the size of the source file.
                data = terrain.data[::max(1, len(terrain.data) // 30000)]
                plot = axis.scatter(data[:, 0], data[:, 1], c=data[:, 2], s=4, cmap='terrain', rasterized=True)
                self.figure.colorbar(plot, ax=axis, label='Высота, м')
                meta = terrain.metadata()
                self.bounds.setText('Границы X/Y: ' + ', '.join(f'{value:g}' for value in meta['bounds']) +
                                    '\nПропуски высот не входят в область покрытия.')
                for item in self.objects.selectedItems():
                    for profile in object_profiles(self.db, item.data(QtCore.Qt.UserRole)):
                        try:
                            points = profile_points(profile)
                        except TerrainError:
                            continue
                        axis.plot(points[:, 0], points[:, 1], linewidth=1, label=profile.title)
            except TerrainError as exc:
                self.bounds.setText(str(exc))
        axis.set_xlabel('X, м')
        axis.set_ylabel('Y, м')
        axis.set_aspect('equal', adjustable='datalim')
        self.figure.tight_layout()
        self.canvas.draw_idle()

    def error(self, exc):
        self.db.rollback()
        QtWidgets.QMessageBox.warning(self, 'Сетки рельефа', str(exc))

    def upload(self):
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Загрузить сетку XYZ', '', 'Сетки XYZ (*.dat *.txt *.xyz);;Все файлы (*)')
        if not filename:
            return
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            grid = add_grid(self.db, filename)
            self.db.commit()
            grid_id = grid.id
        except (TerrainError, OSError, UnicodeError) as exc:
            self.error(exc)
            return
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        self.refresh(selected_id=grid_id)

    def save(self):
        grid = self.grid()
        if grid is None:
            return
        if not self.title.text().strip():
            self.error(TerrainError('Введите название сетки.'))
            return
        grid.title = self.title.text().strip()
        grid.comment = self.comment.toPlainText()
        local = not self.common.isChecked()
        if grid.is_local != local:
            grid.is_local = local
            for obj in self.db.query(GeoradarObject).filter(GeoradarObject.relief_grid_id.is_(None)):
                for profile in object_profiles(self.db, obj.id):
                    invalidate_profile(self.db, profile)
        self.db.commit()
        self.refresh(selected_id=grid.id)

    def assign(self, unbind):
        objects = self.objects.selectedItems()
        grid = self.grid()
        if not objects or (grid is None and not unbind):
            self.error(TerrainError('Выберите объекты и сетку.'))
            return
        object_ids = [item.data(QtCore.Qt.UserRole) for item in objects]
        total_profiles = sum(len(object_profiles(self.db, object_id)) for object_id in object_ids)
        progress = QtWidgets.QProgressDialog('Проверка покрытия…', 'Отмена', 0, max(1, total_profiles), self)
        progress.setWindowTitle('Пересчёт рельефа')
        progress.setWindowModality(QtCore.Qt.ApplicationModal)
        progress.setMinimumDuration(0)
        progress.setAutoClose(False)
        progress.setAutoReset(False)
        progress.setValue(0)
        completed = 0

        def advance(done, total, profile):
            if profile:
                progress.setLabelText(f'Пересчёт: {profile.research.object.title} / {profile.title}')
            else:
                progress.setLabelText('Сохранение результатов…')
            progress.setValue(completed + done)
            QtWidgets.QApplication.processEvents()
            if progress.wasCanceled():
                raise TerrainError('Пересчёт отменён. Привязки и прежние данные сохранены.')

        missing_speed_cover = []
        try:
            # Validate the entire batch before any assignment is changed.
            if not unbind:
                for object_id in object_ids:
                    validate_object_grid(self.db, object_id, grid)
            for object_id in object_ids:
                missing_speed_cover.extend(bind_grid(
                    self.db, object_id, None if unbind else grid.id, progress=advance))
                completed += len(object_profiles(self.db, object_id))
            self.db.commit()
        except TerrainError as exc:
            progress.close()
            self.error(exc)
            return
        finally:
            progress.close()
        self.refresh(selected_id=grid.id if grid else None)
        message = ('Привязки сохранены. Рельеф, land и доступные speed_cover пересчитаны '
                   'для всех исследований выбранных объектов. Сохранённые обучающие наборы не изменены.')
        if missing_speed_cover:
            message += ('\n\nНе удалось рассчитать speed_cover: отсутствуют или не совпадают по длине '
                        'данные кровли и границ пласта.\n' + '\n'.join(missing_speed_cover))
        QtWidgets.QMessageBox.information(self, 'Сетки рельефа', message)

    def remove(self):
        grid = self.grid()
        if grid is None:
            return
        if QtWidgets.QMessageBox.question(self, 'Удаление сетки', f'Удалить сетку «{grid.title}»?') != QtWidgets.QMessageBox.Yes:
            return
        try:
            delete_grid(self.db, grid.id)
            self.db.commit()
        except TerrainError as exc:
            self.error(exc)
            return
        self.refresh()
