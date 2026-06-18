import json

import numpy as np
from PyQt5 import QtWidgets

from ml_air_clutter.config import NormalizationConfig
from ml_air_clutter.preprocessing import Normalizer, build_preprocessing_report
from models_db.model import Profile, CurrentProfile, session
from qt.ml_clutter_experiment_form import Ui_MLClutterExperiment


class MLClutterExperimentWindow(QtWidgets.QDialog):
    """Control window for the experimental ML air-clutter workflow."""

    PROFILE_ID_ROLE = 256
    MIN_NUM_TRACES = 1

    def __init__(
        self,
        parent=None,
        profile_id_getter=None,
        profile_name_getter=None,
        info_callback=None,
        visualization_callback=None,
    ):
        super().__init__(parent)
        self.ui = Ui_MLClutterExperiment()
        self.ui.setupUi(self)
        self.profile_id_getter = profile_id_getter
        self.profile_name_getter = profile_name_getter
        self.info_callback = info_callback
        self.visualization_callback = visualization_callback
        self.clean_profile = None
        self.real_noisy_profile = None
        self.normalized_clean_profile = None
        self.normalization_result = None
        self.experiment_config = {"normalization": None}
        self.experiment_profiles = {}

        self.ui.pushButton_refresh_profiles.clicked.connect(self.add_current_selected_profile)
        self.ui.pushButton_use_current_profile.clicked.connect(self.use_drawn_current_profile)
        self.ui.pushButton_load_clean.clicked.connect(lambda: self.load_selected_profile("clean"))
        self.ui.pushButton_load_real_noisy.clicked.connect(lambda: self.load_selected_profile("real_noisy"))
        self.ui.pushButton_apply_preprocessing.clicked.connect(self.normalize_clean_profile)
        self.ui.pushButton_inverse_preprocessing.clicked.connect(self.preview_inverse_normalization)
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
        data = self._profile_signal_to_array(profile.signal)
        self.experiment_profiles[profile.id] = {"profile": profile, "data": data}

        existing_item = self._find_experiment_item(profile.id)
        if existing_item is None:
            item = QtWidgets.QListWidgetItem(f"{profile.title} ({data.shape[0]} traces) id{profile.id}")
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
        data = self._profile_signal_to_array(current.signal)
        profile_name = self.profile_name_getter() if self.profile_name_getter else f"id{current.profile_id}"
        self._load_profile_array(data, "clean", profile_name, "drawn CurrentProfile -> clean")

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
        source = "experiment profile -> clean" if role == "clean" else "experiment profile -> real noisy"
        self._load_profile_array(data, role, profile.title, source)

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

    def _load_profile_array(self, data, role, name, source):
        prepared, validation = self._prepare_profile_for_role(data, name)
        self._show_stats(self._format_validation_report(name, prepared, source=source, validation=validation))
        if not validation["valid"]:
            self._log(f"ML Clutter: profile '{name}' rejected as {role.replace('_', ' ')}: {validation['reason']}", "red")
            return
        if role == "clean":
            self.clean_profile = prepared
            self.normalized_clean_profile = None
            self.normalization_result = None
        else:
            self.real_noisy_profile = prepared
        self._display_profile(prepared, f"ML Clutter {role.replace('_', ' ')}: {name}")
        self._log(f"Profile '{name}' loaded as {role.replace('_', ' ')} and displayed in MainWindow")

    def normalize_clean_profile(self):
        if self.clean_profile is None:
            self._show_preprocessing_stats("Load a clean profile before normalization.")
            self._log("ML Clutter: clean profile is not loaded for normalization", "red")
            return
        if not self.ui.checkBox_enable_preprocessing.isChecked():
            self.normalized_clean_profile = self.clean_profile.copy()
            self.normalization_result = None
            self.experiment_config["normalization"] = {"enabled": False}
            self._show_preprocessing_stats("Preprocessing is disabled. Clean profile remains in the original scale.")
            self._display_profile(self.normalized_clean_profile, "ML Clutter clean profile without preprocessing")
            return

        config = self._current_normalization_config()
        try:
            result = Normalizer.fit_transform(self.clean_profile, config)
        except ValueError as exc:
            self._show_preprocessing_stats(str(exc))
            self._log(f"ML Clutter: normalization failed: {exc}", "red")
            return

        self.normalized_clean_profile = result.data
        self.normalization_result = result
        self.experiment_config["normalization"] = {
            "enabled": True,
            "config": config.to_dict(),
            "params": result.params,
        }
        report = build_preprocessing_report(
            self.profile_statistics(self.clean_profile),
            self.profile_statistics(result.data),
            result,
        )
        self._show_preprocessing_stats(report)
        self._display_profile(result.data, f"ML Clutter normalized clean ({config.mode})")
        self._log(f"Clean profile normalized with mode '{config.mode}'")

    def preview_inverse_normalization(self):
        if self.normalization_result is None:
            self._show_preprocessing_stats("Normalize a clean profile first; no inverse parameters are available.")
            return
        restored = self.normalization_result.inverse_transform()
        max_abs_error = float(np.max(np.abs(restored - self.clean_profile)))
        text = "\n".join([
            "Inverse normalization preview",
            f"Max absolute reconstruction error: {max_abs_error:.6g}",
            self._format_validation_report("inverse-normalized clean", restored, "normalized clean -> original scale"),
        ])
        self._show_preprocessing_stats(text)
        self._display_profile(restored, "ML Clutter inverse-normalized clean preview")
        self._log(f"Inverse normalization preview prepared; max abs error={max_abs_error:.6g}")

    def _current_normalization_config(self):
        mode = self.ui.comboBox_normalization_mode.currentData() or "standard"
        return NormalizationConfig(mode=mode)

    def _prepare_profile_for_role(self, data, name):
        validation = self.validate_profile(data)
        if validation["needs_transpose"]:
            answer = QtWidgets.QMessageBox.question(
                self,
                "Transpose profile?",
                (
                    f"Profile '{name}' has shape {data.shape}. It looks like [512, num_traces].\n"
                    "Transpose it to [num_traces, 512]?"
                ),
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.Yes,
            )
            if answer == QtWidgets.QMessageBox.Yes:
                data = data.T
                validation = self.validate_profile(data)
                validation["transposed"] = True
        return data, validation

    @classmethod
    def validate_profile(cls, data):
        result = {"valid": False, "reason": "", "needs_transpose": False, "transposed": False}
        if data.ndim != 2:
            result["reason"] = f"Profile must be a 2D array, got ndim={data.ndim}."
            return result
        if data.shape[1] != 512:
            if data.shape[0] == 512:
                result["needs_transpose"] = True
                result["reason"] = "Profile has [512, num_traces] shape and must be transposed."
            else:
                result["reason"] = f"Profile must have 512 samples per trace, got shape={data.shape}."
            return result
        if data.shape[0] < cls.MIN_NUM_TRACES:
            result["reason"] = f"Profile must contain at least {cls.MIN_NUM_TRACES} traces, got {data.shape[0]}."
            return result
        if np.isnan(data).any():
            result["reason"] = "Profile contains NaN values."
            return result
        if np.isinf(data).any():
            result["reason"] = "Profile contains infinite values."
            return result
        if not np.isfinite(data).all():
            result["reason"] = "Profile contains non-finite amplitudes."
            return result
        result["valid"] = True
        result["reason"] = "OK"
        return result

    def _display_profile(self, data, title):
        if self.visualization_callback:
            self.visualization_callback(data.tolist(), title)

    @staticmethod
    def _profile_signal_to_array(signal):
        return np.array(json.loads(signal), dtype=float)

    def _find_experiment_item(self, profile_id):
        for row in range(self.ui.listWidget_profiles.count()):
            item = self.ui.listWidget_profiles.item(row)
            if item.data(self.PROFILE_ID_ROLE) == profile_id:
                return item
        return None

    @classmethod
    def _format_validation_report(cls, name, data, source, validation=None):
        validation = validation or cls.validate_profile(data)
        stats = cls.profile_statistics(data)
        lines = [f"Name: {name}", f"Source: {source}", f"Shape: {data.shape}"]
        lines.append(f"Validation: {'OK' if validation['valid'] else 'ERROR'}")
        lines.append(f"Reason: {validation['reason']}")
        if validation.get("needs_transpose"):
            lines.append("Suggestion: transpose profile from [512, num_traces] to [num_traces, 512].")
        if validation.get("transposed"):
            lines.append("Action: profile was transposed to [num_traces, 512].")
        lines.extend([
            f"Traces: {stats['num_traces']}",
            f"Samples per trace: {stats['samples_per_trace']}",
            f"Min / Max: {stats['min']:.6g} / {stats['max']:.6g}",
            f"Mean / Std: {stats['mean']:.6g} / {stats['std']:.6g}",
            f"Median: {stats['median']:.6g}",
            f"Percentile 1 / 99: {stats['p01']:.6g} / {stats['p99']:.6g}",
            f"Has NaN: {'yes' if stats['has_nan'] else 'no'}",
            f"Has inf: {'yes' if stats['has_inf'] else 'no'}",
            "Axis convention: rows are traces, columns are samples/time/depth.",
        ])
        return "\n".join(lines)

    @staticmethod
    def profile_statistics(data):
        finite = data[np.isfinite(data)]
        if finite.size == 0:
            finite = np.array([np.nan])
        return {
            "num_traces": data.shape[0] if data.ndim >= 1 else 0,
            "samples_per_trace": data.shape[1] if data.ndim == 2 else 0,
            "min": float(np.nanmin(finite)),
            "max": float(np.nanmax(finite)),
            "mean": float(np.nanmean(finite)),
            "std": float(np.nanstd(finite)),
            "median": float(np.nanmedian(finite)),
            "p01": float(np.nanpercentile(finite, 1)),
            "p99": float(np.nanpercentile(finite, 99)),
            "has_nan": bool(np.isnan(data).any()),
            "has_inf": bool(np.isinf(data).any()),
        }

    def _show_stats(self, text):
        self.ui.textEdit_profile_stats.setPlainText(text)
        self.ui.textEdit_results_log.append(text)

    def _show_preprocessing_stats(self, text):
        self.ui.textEdit_preprocessing_stats.setPlainText(text)
        self.ui.textEdit_results_log.append(text)

    def _log(self, text, color="green"):
        self.ui.textEdit_results_log.append(text)
        if self.info_callback:
            self.info_callback(text, color)
