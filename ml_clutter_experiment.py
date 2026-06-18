import json

import numpy as np
from PyQt5 import QtWidgets

from models_db.model import Profile, CurrentProfile, session
from qt.ml_clutter_experiment_form import Ui_MLClutterExperiment


class MLClutterExperimentWindow(QtWidgets.QDialog):
    """Stage-1 control window for the experimental ML air-clutter workflow."""

    def __init__(self, parent=None, profile_id_getter=None, profile_name_getter=None, info_callback=None):
        super().__init__(parent)
        self.ui = Ui_MLClutterExperiment()
        self.ui.setupUi(self)
        self.profile_id_getter = profile_id_getter
        self.profile_name_getter = profile_name_getter
        self.info_callback = info_callback
        self.clean_profile = None
        self.real_noisy_profile = None

        self.ui.pushButton_refresh_profiles.clicked.connect(self.refresh_profiles)
        self.ui.pushButton_use_current_profile.clicked.connect(self.use_current_profile)
        self.ui.pushButton_load_clean.clicked.connect(lambda: self.load_selected_profile("clean"))
        self.ui.pushButton_load_real_noisy.clicked.connect(lambda: self.load_selected_profile("real_noisy"))
        self.ui.listWidget_profiles.currentItemChanged.connect(lambda *_: self.show_selected_profile_stats())
        self.refresh_profiles()

    def refresh_profiles(self):
        self.ui.listWidget_profiles.clear()
        for profile in session.query(Profile).order_by(Profile.id).all():
            count_measure = self._safe_profile_trace_count(profile)
            item = QtWidgets.QListWidgetItem(f"{profile.title} ({count_measure} measurements) id{profile.id}")
            item.setData(256, profile.id)
            self.ui.listWidget_profiles.addItem(item)
        self._log("Profile list refreshed")

    def use_current_profile(self):
        current = session.query(CurrentProfile).first()
        if current is None:
            self._show_stats("Current profile is empty. Draw or load a profile first.")
            self._log("Current profile is empty", "red")
            return
        data = np.array(json.loads(current.signal), dtype=float)
        self.clean_profile = data
        profile_name = self.profile_name_getter() if self.profile_name_getter else f"id{current.profile_id}"
        self._show_stats(self._format_validation_report(profile_name, data, source="CurrentProfile -> clean"))
        self._log(f"Current profile loaded as clean data for ML Clutter: {profile_name}")

    def load_selected_profile(self, role):
        item = self.ui.listWidget_profiles.currentItem()
        if item is None:
            self._show_stats("Select a profile in the list first.")
            return
        profile = session.query(Profile).filter_by(id=item.data(256)).first()
        if profile is None:
            self._show_stats("Selected profile was not found in the database.")
            return
        data = np.array(json.loads(profile.signal), dtype=float)
        if role == "clean":
            self.clean_profile = data
            source = "database profile -> clean"
        else:
            self.real_noisy_profile = data
            source = "database profile -> real noisy"
        self._show_stats(self._format_validation_report(profile.title, data, source=source))
        self._log(f"Profile '{profile.title}' loaded as {role.replace('_', ' ')}")

    def show_selected_profile_stats(self):
        item = self.ui.listWidget_profiles.currentItem()
        if item is None:
            return
        profile = session.query(Profile).filter_by(id=item.data(256)).first()
        if profile is None:
            return
        data = np.array(json.loads(profile.signal), dtype=float)
        self._show_stats(self._format_validation_report(profile.title, data, source="database profile"))

    @staticmethod
    def _safe_profile_trace_count(profile):
        try:
            return len(json.loads(profile.signal))
        except (TypeError, json.JSONDecodeError):
            return 0

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
