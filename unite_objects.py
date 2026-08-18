"""Build a single prediction map from an explicitly collected set of profiles."""

import json

import pandas as pd
from PyQt5 import QtCore, QtWidgets

from func import *
from krige import draw_map


class UniteObjectsDialog(QtWidgets.QDialog):
    """A small, persistent basket of profiles used for a combined map/export."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Общее построение по модели")
        self.resize(650, 380)
        self._profile_ids = []

        layout = QtWidgets.QVBoxLayout(self)
        self.model_label = QtWidgets.QLabel()
        self.model_label.setWordWrap(True)
        layout.addWidget(self.model_label)
        self.profile_list = QtWidgets.QListWidget()
        self.profile_list.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        layout.addWidget(self.profile_list)

        add_buttons = QtWidgets.QHBoxLayout()
        self.add_object_button = QtWidgets.QPushButton("Добавить текущий объект")
        self.add_profile_button = QtWidgets.QPushButton("Добавить текущий профиль")
        add_buttons.addWidget(self.add_object_button)
        add_buttons.addWidget(self.add_profile_button)
        layout.addLayout(add_buttons)

        action_buttons = QtWidgets.QHBoxLayout()
        self.remove_button = QtWidgets.QPushButton("Удалить")
        self.clear_button = QtWidgets.QPushButton("Очистить")
        self.draw_button = QtWidgets.QPushButton("DRAW")
        self.excel_button = QtWidgets.QPushButton("EXCEL")
        self.remove_button.setStyleSheet("background-color: rgb(255, 153, 153);")
        self.draw_button.setStyleSheet("background-color: rgb(255, 204, 121);")
        self.excel_button.setStyleSheet("background-color: rgb(153, 193, 241);")
        for button in (self.remove_button, self.clear_button, self.draw_button, self.excel_button):
            action_buttons.addWidget(button)
        layout.addLayout(action_buttons)

        self.add_object_button.clicked.connect(self.add_current_object)
        self.add_profile_button.clicked.connect(self.add_current_profile)
        self.remove_button.clicked.connect(self.remove_selected)
        self.clear_button.clicked.connect(self.clear_profiles)
        self.draw_button.clicked.connect(self.draw)
        self.excel_button.clicked.connect(self.export_excel)

    def refresh_model_label(self):
        item = ui.listWidget_model_pred.currentItem()
        text = item.text().rsplit(" id", 1)[0] if item else "не выбрана"
        self.model_label.setText(f"Выбранная модель: <b>{text}</b>")

    def _add_profiles(self, profiles):
        added = 0
        for profile in profiles:
            if profile.id in self._profile_ids:
                continue
            research = profile.research
            object_title = research.object.title
            date = research.date_research or "без даты"
            item = QtWidgets.QListWidgetItem(f"{object_title} / {date} / {profile.title}")
            item.setData(QtCore.Qt.UserRole, profile.id)
            self.profile_list.addItem(item)
            self._profile_ids.append(profile.id)
            added += 1
        set_info(f"Добавлено профилей в общее построение: {added}", "green")

    def add_current_object(self):
        object_id = get_object_id()
        if object_id is None:
            self._warning("Сначала выберите объект.")
            return
        profiles = (session.query(Profile).join(Research)
                    .filter(Research.object_id == object_id)
                    .order_by(Research.date_research, Profile.title).all())
        self._add_profiles(profiles)

    def add_current_profile(self):
        profile = session.query(Profile).filter_by(id=get_profile_id()).first()
        if profile is None:
            self._warning("Сначала выберите профиль.")
            return
        self._add_profiles([profile])

    def remove_selected(self):
        for item in self.profile_list.selectedItems():
            self._profile_ids.remove(item.data(QtCore.Qt.UserRole))
            self.profile_list.takeItem(self.profile_list.row(item))

    def clear_profiles(self):
        self._profile_ids.clear()
        self.profile_list.clear()

    def _selected_model(self):
        item = ui.listWidget_model_pred.currentItem()
        if item is None:
            self._warning("Выберите модель в блоке Model Prediction.")
            return None
        prediction = session.query(ProfileModelPrediction).filter_by(
            id=item.text().rsplit(" id", 1)[-1]).first()
        if prediction is None:
            self._warning("Выбранный прогноз не найден.")
        return prediction

    def build_table(self):
        selected = self._selected_model()
        if selected is None:
            return None
        if not self._profile_ids:
            self._warning("Добавьте хотя бы один объект или профиль.")
            return None

        frames = []
        for profile_id in self._profile_ids:
            profile = session.query(Profile).filter_by(id=profile_id).first()
            prediction = session.query(ProfileModelPrediction).filter_by(
                profile_id=profile_id,
                model_id=selected.model_id,
                type_model=selected.type_model,
            ).first()
            if prediction is None:
                self._warning(f"Модель не рассчитана для профиля {profile.title}.")
                return None
            values = (json.loads(prediction.corrected[0].correct)
                      if ui.checkBox_corr_pred.isChecked() and prediction.corrected
                      else json.loads(prediction.prediction))
            data = {
                "object": profile.research.object.title,
                "research": str(profile.research.date_research or ""),
                "profile": profile.title,
                "x_pulc": json.loads(profile.x_pulc),
                "y_pulc": json.loads(profile.y_pulc),
                "prediction": values,
            }
            if ui.checkBox_use_land.isChecked():
                if not profile.formations:
                    self._warning(f"Для профиля {profile.title} не рассчитан рельеф.")
                    return None
                land = json.loads(profile.formations[0].land)
                data["land"] = land
                data["abs_uf"] = [height - value for height, value in zip(land, values)]
            frames.append(pd.DataFrame(data))
        return pd.concat(frames, ignore_index=True)

    def draw(self):
        table = self.build_table()
        if table is None or table.empty:
            return
        value_column = "abs_uf" if ui.checkBox_use_land.isChecked() else "prediction"
        model_title = ui.listWidget_model_pred.currentItem().text().rsplit(" id", 1)[0]
        draw_map(table["x_pulc"], table["y_pulc"], table[value_column], model_title)

    def export_excel(self):
        table = self.build_table()
        if table is None or table.empty:
            return
        model_title = ui.listWidget_model_pred.currentItem().text().rsplit(" id", 1)[0]
        filename = QtWidgets.QFileDialog.getSaveFileName(
            self, "Сохранить общее построение", f"united_{model_title}.xlsx",
            "Excel files (*.xlsx)")[0]
        if not filename:
            return
        table.to_excel(filename, index=False)
        set_info(f"Файл {filename} сохранен", "green")

    def _warning(self, message):
        set_info(message, "red")
        QtWidgets.QMessageBox.warning(self, "Общее построение", message)


_unite_dialog = None


def open_unite_objects_dialog():
    global _unite_dialog
    if _unite_dialog is None:
        _unite_dialog = UniteObjectsDialog(MainWindow)
    _unite_dialog.refresh_model_label()
    _unite_dialog.show()
    _unite_dialog.raise_()
    _unite_dialog.activateWindow()
