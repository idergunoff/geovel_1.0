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
        self.tab_generator = self._add_placeholder_tab(
            "tab_generator",
            "Configure analytical, real-pattern, or mixed clutter generation.",
        )
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
        self.pushButton_refresh_profiles.setText(_translate("MLClutterExperiment", "Refresh"))
        self.pushButton_use_current_profile.setText(_translate("MLClutterExperiment", "Use Current Profile"))
        self.pushButton_load_clean.setText(_translate("MLClutterExperiment", "Load as Clean"))
        self.pushButton_load_real_noisy.setText(_translate("MLClutterExperiment", "Load as Real Noisy"))
        self.groupBox_profile_stats.setTitle(_translate("MLClutterExperiment", "Profile statistics and validation"))
        tab_names = ["Profiles", "Noise Patterns", "Generator", "Dataset", "Model", "Training", "Inference", "Results"]
        for index, name in enumerate(tab_names):
            self.tabWidget.setTabText(index, _translate("MLClutterExperiment", name))
