# -*- coding: utf-8 -*-

from PyQt5 import QtCore, QtWidgets


class Ui_MLClutterExperiment(object):
    def setupUi(self, MLClutterExperiment):
        MLClutterExperiment.setObjectName("MLClutterExperiment")
        MLClutterExperiment.resize(980, 720)
        self.verticalLayout = QtWidgets.QVBoxLayout(MLClutterExperiment)
        self.verticalLayout.setObjectName("verticalLayout")
        self.headerLabel = QtWidgets.QLabel(MLClutterExperiment)
        self.headerLabel.setObjectName("headerLabel")
        self.verticalLayout.addWidget(self.headerLabel)
        self.tabWidget = QtWidgets.QTabWidget(MLClutterExperiment)
        self.tabWidget.setObjectName("tabWidget")

        self.tab_profiles = self._add_profiles_tab()
        self.tab_noise_patterns = self._add_placeholder_tab(
            "tab_noise_patterns",
            "Add noisy radarograms and configure extraction of real air-clutter patterns.",
        )
        self.tab_generator = self._add_generator_tab()
        self.tab_dataset = self._add_placeholder_tab(
            "tab_dataset",
            "Generate synthetic noisy/clean training pairs and patch datasets.",
        )
        self.tab_model = self._add_placeholder_tab(
            "tab_model",
            "Choose baseline CNN, DnCNN, or U-Net model settings.",
        )
        self.tab_training = self._add_placeholder_tab(
            "tab_training",
            "Run training, validation, checkpoints, and training metrics.",
        )
        self.tab_inference = self._add_placeholder_tab(
            "tab_inference",
            "Apply trained clutter-removal models to synthetic or real profiles.",
        )
        self.tab_results = self._add_results_tab()

        self.verticalLayout.addWidget(self.tabWidget)
        self.retranslateUi(MLClutterExperiment)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MLClutterExperiment)

    def _add_profiles_tab(self):
        tab = QtWidgets.QWidget()
        tab.setObjectName("tab_profiles")
        layout = QtWidgets.QGridLayout(tab)
        layout.setObjectName("gridLayout_profiles")
        self.listWidget_profiles = QtWidgets.QListWidget(tab)
        self.listWidget_profiles.setObjectName("listWidget_profiles")
        layout.addWidget(self.listWidget_profiles, 1, 0, 6, 1)
        self.pushButton_refresh_profiles = QtWidgets.QPushButton(tab)
        self.pushButton_refresh_profiles.setObjectName("pushButton_refresh_profiles")
        layout.addWidget(self.pushButton_refresh_profiles, 1, 1)
        self.pushButton_use_current_profile = QtWidgets.QPushButton(tab)
        self.pushButton_use_current_profile.setObjectName("pushButton_use_current_profile")
        layout.addWidget(self.pushButton_use_current_profile, 2, 1)
        self.pushButton_load_clean = QtWidgets.QPushButton(tab)
        self.pushButton_load_clean.setObjectName("pushButton_load_clean")
        layout.addWidget(self.pushButton_load_clean, 3, 1)
        self.pushButton_load_real_noisy = QtWidgets.QPushButton(tab)
        self.pushButton_load_real_noisy.setObjectName("pushButton_load_real_noisy")
        layout.addWidget(self.pushButton_load_real_noisy, 4, 1)
        self.groupBox_profile_stats = QtWidgets.QGroupBox(tab)
        self.groupBox_profile_stats.setObjectName("groupBox_profile_stats")
        stats_layout = QtWidgets.QVBoxLayout(self.groupBox_profile_stats)
        self.textEdit_profile_stats = QtWidgets.QTextEdit(self.groupBox_profile_stats)
        self.textEdit_profile_stats.setReadOnly(True)
        self.textEdit_profile_stats.setObjectName("textEdit_profile_stats")
        stats_layout.addWidget(self.textEdit_profile_stats)
        layout.addWidget(self.groupBox_profile_stats, 7, 0, 1, 2)
        self.groupBox_preprocessing = QtWidgets.QGroupBox(tab)
        self.groupBox_preprocessing.setObjectName("groupBox_preprocessing")
        preprocessing_layout = QtWidgets.QGridLayout(self.groupBox_preprocessing)
        self.comboBox_normalization_mode = QtWidgets.QComboBox(self.groupBox_preprocessing)
        self.comboBox_normalization_mode.setObjectName("comboBox_normalization_mode")
        self.comboBox_normalization_mode.addItem("Standard", "standard")
        self.comboBox_normalization_mode.addItem("Robust", "robust")
        self.comboBox_normalization_mode.addItem("Percentile clipping + standardization", "percentile_standard")
        preprocessing_layout.addWidget(self.comboBox_normalization_mode, 0, 0, 1, 2)
        self.checkBox_enable_preprocessing = QtWidgets.QCheckBox(self.groupBox_preprocessing)
        self.checkBox_enable_preprocessing.setObjectName("checkBox_enable_preprocessing")
        self.checkBox_enable_preprocessing.setChecked(True)
        preprocessing_layout.addWidget(self.checkBox_enable_preprocessing, 1, 0, 1, 2)
        self.pushButton_apply_preprocessing = QtWidgets.QPushButton(self.groupBox_preprocessing)
        self.pushButton_apply_preprocessing.setObjectName("pushButton_apply_preprocessing")
        preprocessing_layout.addWidget(self.pushButton_apply_preprocessing, 2, 0)
        self.pushButton_inverse_preprocessing = QtWidgets.QPushButton(self.groupBox_preprocessing)
        self.pushButton_inverse_preprocessing.setObjectName("pushButton_inverse_preprocessing")
        preprocessing_layout.addWidget(self.pushButton_inverse_preprocessing, 2, 1)
        self.textEdit_preprocessing_stats = QtWidgets.QTextEdit(self.groupBox_preprocessing)
        self.textEdit_preprocessing_stats.setReadOnly(True)
        self.textEdit_preprocessing_stats.setObjectName("textEdit_preprocessing_stats")
        preprocessing_layout.addWidget(self.textEdit_preprocessing_stats, 3, 0, 1, 2)
        layout.addWidget(self.groupBox_preprocessing, 8, 0, 1, 2)
        self.tabWidget.addTab(tab, "")
        return tab


    def _add_generator_tab(self):
        tab = QtWidgets.QWidget()
        tab.setObjectName("tab_generator")
        layout = QtWidgets.QGridLayout(tab)
        intro = QtWidgets.QLabel(tab)
        intro.setWordWrap(True)
        intro.setText("Configure analytical synthetic air-clutter generation for the loaded clean profile.")
        layout.addWidget(intro, 0, 0, 1, 3)
        self.checkBox_gen_hyperbolas = QtWidgets.QCheckBox(tab)
        self.checkBox_gen_hyperbolas.setObjectName("checkBox_gen_hyperbolas")
        self.checkBox_gen_hyperbolas.setChecked(True)
        layout.addWidget(self.checkBox_gen_hyperbolas, 1, 0)
        self.checkBox_gen_sloped_events = QtWidgets.QCheckBox(tab)
        self.checkBox_gen_sloped_events.setObjectName("checkBox_gen_sloped_events")
        self.checkBox_gen_sloped_events.setChecked(True)
        layout.addWidget(self.checkBox_gen_sloped_events, 1, 1)
        self.checkBox_gen_ringing = QtWidgets.QCheckBox(tab)
        self.checkBox_gen_ringing.setObjectName("checkBox_gen_ringing")
        self.checkBox_gen_ringing.setChecked(True)
        layout.addWidget(self.checkBox_gen_ringing, 1, 2)
        self.checkBox_gen_vertical_spikes = QtWidgets.QCheckBox(tab)
        self.checkBox_gen_vertical_spikes.setObjectName("checkBox_gen_vertical_spikes")
        self.checkBox_gen_vertical_spikes.setChecked(True)
        layout.addWidget(self.checkBox_gen_vertical_spikes, 2, 0)
        self.checkBox_gen_noise_zones = QtWidgets.QCheckBox(tab)
        self.checkBox_gen_noise_zones.setObjectName("checkBox_gen_noise_zones")
        self.checkBox_gen_noise_zones.setChecked(True)
        layout.addWidget(self.checkBox_gen_noise_zones, 2, 1)
        self.spinBox_gen_seed = QtWidgets.QSpinBox(tab)
        self.spinBox_gen_seed.setObjectName("spinBox_gen_seed")
        self.spinBox_gen_seed.setRange(0, 2147483647)
        self.spinBox_gen_seed.setValue(42)
        layout.addWidget(QtWidgets.QLabel("Seed", tab), 3, 0)
        layout.addWidget(self.spinBox_gen_seed, 3, 1)
        self.doubleSpinBox_gen_target_snr = QtWidgets.QDoubleSpinBox(tab)
        self.doubleSpinBox_gen_target_snr.setObjectName("doubleSpinBox_gen_target_snr")
        self.doubleSpinBox_gen_target_snr.setRange(-30.0, 60.0)
        self.doubleSpinBox_gen_target_snr.setValue(6.0)
        self.doubleSpinBox_gen_target_snr.setSuffix(" dB")
        layout.addWidget(QtWidgets.QLabel("Target SNR", tab), 4, 0)
        layout.addWidget(self.doubleSpinBox_gen_target_snr, 4, 1)
        self.pushButton_generate_synthetic_clutter = QtWidgets.QPushButton(tab)
        self.pushButton_generate_synthetic_clutter.setObjectName("pushButton_generate_synthetic_clutter")
        layout.addWidget(self.pushButton_generate_synthetic_clutter, 5, 0, 1, 2)
        self.textEdit_generator_meta = QtWidgets.QTextEdit(tab)
        self.textEdit_generator_meta.setReadOnly(True)
        self.textEdit_generator_meta.setObjectName("textEdit_generator_meta")
        layout.addWidget(self.textEdit_generator_meta, 6, 0, 1, 3)
        layout.setRowStretch(6, 1)
        self.tabWidget.addTab(tab, "")
        return tab

    def _add_placeholder_tab(self, object_name, message):
        tab = QtWidgets.QWidget()
        tab.setObjectName(object_name)
        layout = QtWidgets.QVBoxLayout(tab)
        label = QtWidgets.QLabel(tab)
        label.setWordWrap(True)
        label.setText(message)
        layout.addWidget(label)
        layout.addStretch()
        self.tabWidget.addTab(tab, "")
        return tab

    def _add_results_tab(self):
        tab = QtWidgets.QWidget()
        tab.setObjectName("tab_results")
        layout = QtWidgets.QVBoxLayout(tab)
        self.textEdit_results_log = QtWidgets.QTextEdit(tab)
        self.textEdit_results_log.setReadOnly(True)
        self.textEdit_results_log.setObjectName("textEdit_results_log")
        layout.addWidget(self.textEdit_results_log)
        self.tabWidget.addTab(tab, "")
        return tab

    def retranslateUi(self, MLClutterExperiment):
        _translate = QtCore.QCoreApplication.translate
        MLClutterExperiment.setWindowTitle(_translate("MLClutterExperiment", "ML Clutter Experiment"))
        self.headerLabel.setText(_translate("MLClutterExperiment", "Experimental ML Clutter block: data, generator, dataset, model, training, inference and results."))
        self.pushButton_refresh_profiles.setText(_translate("MLClutterExperiment", "Add Current Profile"))
        self.pushButton_use_current_profile.setText(_translate("MLClutterExperiment", "Use Drawn CurrentProfile"))
        self.pushButton_load_clean.setText(_translate("MLClutterExperiment", "Load as Clean"))
        self.pushButton_load_real_noisy.setText(_translate("MLClutterExperiment", "Load as Real Noisy"))
        self.groupBox_profile_stats.setTitle(_translate("MLClutterExperiment", "Profile statistics and validation"))
        self.groupBox_preprocessing.setTitle(_translate("MLClutterExperiment", "Preprocessing and normalization"))
        self.checkBox_enable_preprocessing.setText(_translate("MLClutterExperiment", "Enable preprocessing"))
        self.pushButton_apply_preprocessing.setText(_translate("MLClutterExperiment", "Normalize Clean Profile"))
        self.pushButton_inverse_preprocessing.setText(_translate("MLClutterExperiment", "Preview Inverse Transform"))
        self.checkBox_gen_hyperbolas.setText(_translate("MLClutterExperiment", "Hyperbolas"))
        self.checkBox_gen_sloped_events.setText(_translate("MLClutterExperiment", "Sloped events"))
        self.checkBox_gen_ringing.setText(_translate("MLClutterExperiment", "Ringing"))
        self.checkBox_gen_vertical_spikes.setText(_translate("MLClutterExperiment", "Vertical spikes"))
        self.checkBox_gen_noise_zones.setText(_translate("MLClutterExperiment", "Wide noise zones"))
        self.pushButton_generate_synthetic_clutter.setText(_translate("MLClutterExperiment", "Generate Synthetic Clutter"))
        tab_names = ["Profiles", "Noise Patterns", "Generator", "Dataset", "Model", "Training", "Inference", "Results"]
        for index, name in enumerate(tab_names):
            self.tabWidget.setTabText(index, _translate("MLClutterExperiment", name))
