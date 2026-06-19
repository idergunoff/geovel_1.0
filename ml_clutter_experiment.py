import json
from pathlib import Path

import numpy as np
from PyQt5 import QtWidgets

from ml_air_clutter.config import NormalizationConfig, SyntheticClutterConfig
from ml_air_clutter.noise_patterns import PatternExtractionConfig, extract_energy_patterns, extract_pattern_from_bbox
from ml_air_clutter.pattern_generator import PatternClutterConfig, generate_pattern_clutter
from ml_air_clutter.pattern_library import NoisePattern, PatternLibrary
from ml_air_clutter.preprocessing import Normalizer, build_preprocessing_report
from ml_air_clutter.synthetic_clutter import generate_synthetic_clutter
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
        self.real_noisy_profiles = []
        self.pattern_library = PatternLibrary()
        self.normalized_clean_profile = None
        self.normalization_result = None
        self.experiment_config = {"normalization": None, "synthetic_clutter": None, "pattern_library": None}
        self.experiment_profiles = {}
        self.synthetic_clutter_result = None
        self.pattern_clutter_result = None

        self.ui.pushButton_refresh_profiles.clicked.connect(self.add_current_selected_profile)
        self.ui.pushButton_use_current_profile.clicked.connect(self.use_drawn_current_profile)
        self.ui.pushButton_load_clean.clicked.connect(lambda: self.load_selected_profile("clean"))
        self.ui.pushButton_load_real_noisy.clicked.connect(lambda: self.load_selected_profile("real_noisy"))
        self.ui.pushButton_apply_preprocessing.clicked.connect(self.normalize_clean_profile)
        self.ui.pushButton_inverse_preprocessing.clicked.connect(self.preview_inverse_normalization)
        self.ui.pushButton_generate_synthetic_clutter.clicked.connect(self.generate_synthetic_clutter_preview)
        self.ui.pushButton_add_noisy_to_library.clicked.connect(self.add_loaded_real_noisy_to_pattern_sources)
        self.ui.pushButton_extract_manual_pattern.clicked.connect(self.extract_manual_pattern)
        self.ui.pushButton_extract_energy_patterns.clicked.connect(self.extract_energy_patterns)
        self.ui.pushButton_preview_pattern.clicked.connect(self.preview_selected_pattern)
        self.ui.pushButton_save_pattern_library.clicked.connect(self.save_pattern_library)
        self.ui.pushButton_load_pattern_library.clicked.connect(self.load_pattern_library)
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
            self._register_real_noisy_profile(name, prepared, source)
        self._display_profile(prepared, f"ML Clutter {role.replace('_', ' ')}: {name}")
        self._log(f"Profile '{name}' loaded as {role.replace('_', ' ')} and displayed in MainWindow")

    def add_loaded_real_noisy_to_pattern_sources(self):
        if self.real_noisy_profile is None:
            self._show_pattern_stats("Load a real noisy profile before adding it to the pattern source list.")
            self._log("ML Clutter: no real noisy profile loaded for pattern extraction", "red")
            return
        name = f"real_noisy_{len(self.real_noisy_profiles) + 1}"
        self._register_real_noisy_profile(name, self.real_noisy_profile, "loaded real noisy profile")

    def extract_manual_pattern(self):
        source = self._selected_noisy_source()
        if source is None:
            self._show_pattern_stats("Add and select a real noisy radarogram source first.")
            return
        bbox = [
            self.ui.spinBox_pattern_x_start.value(),
            self.ui.spinBox_pattern_x_end.value(),
            self.ui.spinBox_pattern_z_start.value(),
            self.ui.spinBox_pattern_z_end.value(),
        ]
        try:
            extracted = extract_pattern_from_bbox(source["data"], bbox)
            pattern = NoisePattern.create(
                source_profile=source["name"],
                array=extracted["array"],
                mask=extracted["mask"],
                bbox=bbox,
                normalization=extracted["normalization"],
                tags=[self.ui.comboBox_pattern_tag.currentData() or "unknown"],
                comment="manual bbox extraction",
            )
            self.pattern_library.add_pattern(pattern)
        except ValueError as exc:
            self._show_pattern_stats(str(exc))
            self._log(f"ML Clutter: manual pattern extraction failed: {exc}", "red")
            return
        self._refresh_pattern_library_ui()
        self._display_profile(pattern.array, f"ML Clutter extracted pattern {pattern.pattern_id}")
        self._log(f"Manual real-noise pattern extracted from '{source['name']}': {pattern.pattern_id}")

    def extract_energy_patterns(self):
        source = self._selected_noisy_source()
        if source is None:
            self._show_pattern_stats("Add and select a real noisy radarogram source first.")
            return
        config = PatternExtractionConfig()
        try:
            extracted_patterns = extract_energy_patterns(source["data"], config)
            for extracted in extracted_patterns:
                self.pattern_library.add_pattern(NoisePattern.create(
                    source_profile=source["name"],
                    array=extracted["array"],
                    mask=extracted["mask"],
                    bbox=extracted["bbox"],
                    normalization=extracted["normalization"],
                    tags=["unknown"],
                    comment=f"auto energy extraction; score={extracted['energy_score']:.6g}",
                ))
        except ValueError as exc:
            self._show_pattern_stats(str(exc))
            self._log(f"ML Clutter: energy pattern extraction failed: {exc}", "red")
            return
        self._refresh_pattern_library_ui()
        self._log(f"Auto-extracted {len(extracted_patterns)} high-energy real-noise patterns from '{source['name']}'")

    def preview_selected_pattern(self):
        pattern = self._selected_pattern()
        if pattern is None:
            self._show_pattern_stats("Select a pattern from the library before preview.")
            return
        self._display_profile(pattern.array, f"ML Clutter pattern {pattern.pattern_id}")
        self._display_profile(pattern.mask, f"ML Clutter pattern mask {pattern.pattern_id}")
        self._show_pattern_stats(self._format_pattern_report(pattern))

    def save_pattern_library(self):
        if not self.pattern_library.patterns:
            self._show_pattern_stats("Pattern library is empty; extract at least one pattern before saving.")
            return
        directory = QtWidgets.QFileDialog.getExistingDirectory(self, "Save pattern library directory")
        if not directory:
            return
        index_path = self.pattern_library.save(directory)
        self.experiment_config["pattern_library"] = {"path": str(index_path), "summary": self.pattern_library.summary()}
        self._show_pattern_stats(f"Pattern library saved to {index_path}\n{json.dumps(self.pattern_library.summary(), indent=2)}")
        self._log(f"Pattern library saved: {index_path}")

    def load_pattern_library(self):
        directory = QtWidgets.QFileDialog.getExistingDirectory(self, "Load pattern library directory")
        if not directory:
            return
        try:
            self.pattern_library = PatternLibrary.load(directory)
        except (OSError, ValueError, KeyError) as exc:
            self._show_pattern_stats(f"Failed to load pattern library: {exc}")
            self._log(f"ML Clutter: failed to load pattern library: {exc}", "red")
            return
        self._refresh_pattern_library_ui()
        self.experiment_config["pattern_library"] = {"path": str(Path(directory) / "pattern_library_index.json"), "summary": self.pattern_library.summary()}
        self._log(f"Pattern library loaded from {directory}")


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

    def generate_synthetic_clutter_preview(self):
        if self.clean_profile is None:
            self._show_generator_stats("Load a clean profile before synthetic clutter generation.")
            self._log("ML Clutter: clean profile is not loaded for synthetic clutter generation", "red")
            return

        source_profile = self.normalized_clean_profile if self.normalized_clean_profile is not None else self.clean_profile
        mode = self.ui.comboBox_gen_mode.currentData() or "synthetic"
        config = self._current_synthetic_clutter_config()
        try:
            if mode == "synthetic":
                noisy, clutter, mask, meta = generate_synthetic_clutter(source_profile, config)
            else:
                pattern_config = self._current_pattern_clutter_config(mode)
                noisy, clutter, mask, meta = generate_pattern_clutter(source_profile, self.pattern_library, pattern_config, config)
        except ValueError as exc:
            self._show_generator_stats(str(exc))
            self._log(f"ML Clutter: synthetic clutter generation failed: {exc}", "red")
            return

        result = {
            "noisy": noisy,
            "clutter": clutter,
            "mask": mask,
            "meta": meta,
        }
        if mode == "synthetic":
            self.synthetic_clutter_result = result
            self.experiment_config["synthetic_clutter"] = meta
        else:
            self.pattern_clutter_result = result
            self.experiment_config["pattern_clutter"] = meta
        report = self._format_generator_report(meta, clutter, mask, noisy)
        self._show_generator_stats(report)
        self._display_profile(source_profile, "ML Clutter synthetic source clean")
        self._display_profile(clutter, "ML Clutter synthetic clutter")
        self._display_profile(mask, "ML Clutter synthetic clutter mask")
        self._display_profile(noisy, "ML Clutter synthetic noisy profile")
        if mode == "synthetic":
            self._log(f"Synthetic clutter generated: {len(meta['objects'])} objects, SNR={meta['actual_snr_db']:.3g} dB")
        else:
            self._log(f"{mode.title()} clutter generated: {len(meta['placements'])} real pattern placements, SNR={meta['actual_snr_db']:.3g} dB")

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

    def _current_synthetic_clutter_config(self):
        return SyntheticClutterConfig(
            seed=int(self.ui.spinBox_gen_seed.value()),
            target_snr_db=float(self.ui.doubleSpinBox_gen_target_snr.value()),
            hyperbolas=self.ui.checkBox_gen_hyperbolas.isChecked(),
            sloped_events=self.ui.checkBox_gen_sloped_events.isChecked(),
            ringing=self.ui.checkBox_gen_ringing.isChecked(),
            vertical_spikes=self.ui.checkBox_gen_vertical_spikes.isChecked(),
            noise_zones=self.ui.checkBox_gen_noise_zones.isChecked(),
        )

    def _current_pattern_clutter_config(self, mode):
        pattern = self._selected_pattern()
        pattern_ids = [pattern.pattern_id] if pattern is not None else None
        return PatternClutterConfig(
            seed=int(self.ui.spinBox_gen_seed.value()),
            mode=mode,
            pattern_ids=pattern_ids,
            num_patterns=int(self.ui.spinBox_pattern_num.value()),
            pattern_strength=float(self.ui.doubleSpinBox_pattern_strength.value()),
            synthetic_strength=float(self.ui.doubleSpinBox_synthetic_strength.value()),
            target_snr_db=float(self.ui.doubleSpinBox_gen_target_snr.value()),
        )

    @staticmethod
    def _format_generator_report(meta, clutter, mask, noisy):
        if "objects" in meta:
            object_counts = {}
            for obj in meta["objects"]:
                object_counts[obj["type"]] = object_counts.get(obj["type"], 0) + 1
            title = "Synthetic clutter generation report"
            generated = f"Generated objects: {len(meta['objects'])} ({object_counts})"
        else:
            title = "Real-pattern clutter generation report"
            generated = f"Pattern placements: {len(meta.get('placements', []))}"
        lines = [
            title,
            f"Config: {meta['config']}",
            generated,
            f"Target SNR scale: {meta['target_snr_scale']:.6g}",
            f"Actual SNR: {meta['actual_snr_db']:.6g} dB",
            f"Clutter min/max: {float(np.min(clutter)):.6g} / {float(np.max(clutter)):.6g}",
            f"Clutter RMS: {float(np.sqrt(np.mean(clutter ** 2))):.6g}",
            f"Mask coverage: {float(np.mean(mask > 0)) * 100.0:.3g}%",
            f"Noisy min/max: {float(np.min(noisy)):.6g} / {float(np.max(noisy)):.6g}",
            "Meta preview:",
            json.dumps(meta, indent=2)[:6000],
        ]
        return "\n".join(lines)

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

    def _register_real_noisy_profile(self, name, data, source):
        entry = {"name": name, "data": np.asarray(data, dtype=float), "source": source}
        self.real_noisy_profiles.append(entry)
        item = QtWidgets.QListWidgetItem(f"{name} ({entry['data'].shape[0]} traces)")
        item.setData(self.PROFILE_ID_ROLE, len(self.real_noisy_profiles) - 1)
        self.ui.listWidget_noisy_profiles.addItem(item)
        self.ui.listWidget_noisy_profiles.setCurrentItem(item)
        self.ui.spinBox_pattern_x_end.setMaximum(entry["data"].shape[0])
        self.ui.spinBox_pattern_x_end.setValue(min(entry["data"].shape[0], max(1, self.ui.spinBox_pattern_x_end.value())))
        self.ui.spinBox_pattern_z_end.setValue(min(entry["data"].shape[1], max(1, self.ui.spinBox_pattern_z_end.value())))
        self._show_pattern_stats(
            f"Real noisy source registered: {name}\nShape: {entry['data'].shape}\nSource: {source}"
        )

    def _selected_noisy_source(self):
        item = self.ui.listWidget_noisy_profiles.currentItem()
        if item is None:
            return None
        index = item.data(self.PROFILE_ID_ROLE)
        if index is None or index < 0 or index >= len(self.real_noisy_profiles):
            return None
        return self.real_noisy_profiles[index]

    def _selected_pattern(self):
        item = self.ui.listWidget_pattern_library.currentItem()
        if item is None:
            return None
        return self.pattern_library.get(item.data(self.PROFILE_ID_ROLE))

    def _refresh_pattern_library_ui(self):
        self.ui.listWidget_pattern_library.clear()
        for pattern in self.pattern_library.patterns:
            item = QtWidgets.QListWidgetItem(f"{pattern.pattern_id[:8]} {pattern.array.shape} tags={','.join(pattern.tags)}")
            item.setData(self.PROFILE_ID_ROLE, pattern.pattern_id)
            self.ui.listWidget_pattern_library.addItem(item)
        self._show_pattern_stats(json.dumps(self.pattern_library.summary(), indent=2))

    @staticmethod
    def _format_pattern_report(pattern):
        return "\n".join([
            "Noise pattern report",
            f"Pattern id: {pattern.pattern_id}",
            f"Source profile: {pattern.source_profile}",
            f"BBox: {pattern.bbox}",
            f"Tags: {', '.join(pattern.tags)}",
            f"Created at: {pattern.created_at}",
            f"Comment: {pattern.comment}",
            f"Stats: {json.dumps(pattern.stats, indent=2)}",
        ])


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

    def _show_generator_stats(self, text):
        self.ui.textEdit_generator_meta.setPlainText(text)
        self.ui.textEdit_results_log.append(text)

    def _show_pattern_stats(self, text):
        self.ui.textEdit_pattern_library.setPlainText(text)
        self.ui.textEdit_results_log.append(text)

    def _log(self, text, color="green"):
        self.ui.textEdit_results_log.append(text)
        if self.info_callback:
            self.info_callback(text, color)
