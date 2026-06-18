import json

import numpy as np
from PyQt5 import QtWidgets

from models_db.model import Profile, CurrentProfile, session
from qt.ml_clutter_experiment_form import Ui_MLClutterExperiment


class MLClutterExperimentWindow(QtWidgets.QDialog):
    """Stage-1 control window for the experimental ML air-clutter workflow."""

    PROFILE_ID_ROLE = 256

    def __init__(self, parent=None, profile_id_getter=None, profile_name_getter=None, info_callback=None):
        super().__init__(parent)
        self.ui = Ui_MLClutterExperiment()
        self.ui.setupUi(self)
        self.profile_id_getter = profile_id_getter
        self.profile_name_getter = profile_name_getter
        self.info_callback = info_callback
        self.clean_profile = None
        self.real_noisy_profile = None
        self.experiment_profiles = {}

        self.ui.pushButton_refresh_profiles.clicked.connect(self.add_current_selected_profile)
        self.ui.pushButton_use_current_profile.clicked.connect(self.use_drawn_current_profile)
        self.ui.pushButton_load_clean.clicked.connect(lambda: self.load_selected_profile("clean"))
        self.ui.pushButton_load_real_noisy.clicked.connect(lambda: self.load_selected_profile("real_noisy"))
        self.ui.listWidget_profiles.currentItemChanged.connect(lambda *_: self.show_selected_profile_stats())
        self._show_stats(
            "Experiment profile list is empty. "
            "Click 'Add Current Profile' to add the profile selected in the main window."
        )

    def add_current_selected_profile(self):
        profile_id = self.profile_id_getter() if self.profile_id_getter else None
        if not profile_id:
            self._show_stats("Select a profile in the main window before adding it to the experiment.")
            self._log("ML Clutter: no profile is selected in the main window", "red")
            return
        profile = session.query(Profile).filter_by(id=profile_id).first()
        if profile is None:
            self._show_stats(f"Profile id{profile_id} was not found in the database.")
            self._log(f"ML Clutter: profile id{profile_id} was not found", "red")
            return
        data = np.array(json.loads(profile.signal), dtype=float)
        self.experiment_profiles[profile.id] = {"profile": profile, "data": data}

        existing_item = self._find_experiment_item(profile.id)
        if existing_item is None:
            item = QtWidgets.QListWidgetItem(f"{profile.title} ({data.shape[0]} measurements) id{profile.id}")
            item.setData(self.PROFILE_ID_ROLE, profile.id)
            self.ui.listWidget_profiles.addItem(item)
            self.ui.listWidget_profiles.setCurrentItem(item)
        else:
            self.ui.listWidget_profiles.setCurrentItem(existing_item)

        self._show_stats(self._format_validation_report(profile.title, data, source="main-window selected profile"))
        self._log(f"Profile '{profile.title}' added to ML Clutter experiment")

    def use_drawn_current_profile(self):
        current = session.query(CurrentProfile).first()
        if current is None:
            self._show_stats("CurrentProfile is empty. Draw or load a profile first.")
            self._log("ML Clutter: CurrentProfile is empty", "red")
            return
        data = np.array(json.loads(current.signal), dtype=float)
        self.clean_profile = data
        profile_name = self.profile_name_getter() if self.profile_name_getter else f"id{current.profile_id}"
        self._show_stats(self._format_validation_report(profile_name, data, source="drawn CurrentProfile -> clean"))
        self._log(f"Drawn CurrentProfile loaded as clean data for ML Clutter: {profile_name}")

    def load_selected_profile(self, role):
        item = self.ui.listWidget_profiles.currentItem()
        if item is None:
            self._show_stats("Add and select an experiment profile first.")
            return
        profile_id = item.data(self.PROFILE_ID_ROLE)
        profile_entry = self.experiment_profiles.get(profile_id)
        if profile_entry is None:
            self._show_stats("Selected experiment profile is not available. Add it again from the main window.")
            return
        profile = profile_entry["profile"]
        data = profile_entry["data"]
        if role == "clean":
            self.clean_profile = data
            source = "experiment profile -> clean"
        else:
            self.real_noisy_profile = data
            source = "experiment profile -> real noisy"
        self._show_stats(self._format_validation_report(profile.title, data, source=source))
        self._log(f"Experiment profile '{profile.title}' loaded as {role.replace('_', ' ')}")

    def show_selected_profile_stats(self):
        item = self.ui.listWidget_profiles.currentItem()
        if item is None:
            return
        profile_entry = self.experiment_profiles.get(item.data(self.PROFILE_ID_ROLE))
        if profile_entry is None:
            return
        profile = profile_entry["profile"]
        data = profile_entry["data"]
        self._show_stats(self._format_validation_report(profile.title, data, source="experiment profile"))


    def _find_experiment_item(self, profile_id):
        for row in range(self.ui.listWidget_profiles.count()):
            item = self.ui.listWidget_profiles.item(row)
            if item.data(self.PROFILE_ID_ROLE) == profile_id:
                return item
        return None

    @staticmethod
    def _format_validation_report(name, data, source):
        lines = [f"Name: {name}", f"Source: {source}", f"Shape: {data.shape}"]
        lines.append(f"2D array: {'OK' if data.ndim == 2 else 'ERROR'}")
        lines.append(f"Expected [num_traces, 512]: {'OK' if data.ndim == 2 and data.shape[1] == 512 else 'WARNING'}")
        lines.append(f"Finite amplitudes: {'OK' if np.isfinite(data).all() else 'ERROR'}")
        if data.size:
            lines.append(f"Min / Max: {np.nanmin(data):.6g} / {np.nanmax(data):.6g}")
            lines.append(f"Mean / Std: {np.nanmean(data):.6g} / {np.nanstd(data):.6g}")
        lines.append("Axis convention: rows are traces, columns are samples/time/depth.")
        return "\n".join(lines)

    def _show_stats(self, text):
        self.ui.textEdit_profile_stats.setPlainText(text)
        self.ui.textEdit_results_log.append(text)

    def _log(self, text, color="green"):
        self.ui.textEdit_results_log.append(text)
        if self.info_callback:
            self.info_callback(text, color)
