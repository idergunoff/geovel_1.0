# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'feature_selection.ui'
#
# Created by: PyQt5 UI code generator 5.15.10
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_FeatureSelection(object):
    def setupUi(self, FeatureSelection):
        FeatureSelection.setObjectName("FeatureSelection")
        FeatureSelection.resize(565, 410)
        self.gridLayout_2 = QtWidgets.QGridLayout(FeatureSelection)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.label_params = QtWidgets.QLabel(FeatureSelection)
        self.label_params.setObjectName("label_params")
        self.gridLayout_2.addWidget(self.label_params, 0, 0, 1, 1)
        self.label_info = QtWidgets.QLabel(FeatureSelection)
        self.label_info.setObjectName("label_info")
        self.gridLayout_2.addWidget(self.label_info, 0, 1, 1, 1)
        self.listWidget_features = QtWidgets.QListWidget(FeatureSelection)
        self.listWidget_features.setObjectName("listWidget_features")
        self.gridLayout_2.addWidget(self.listWidget_features, 1, 0, 1, 1)
        self.plainTextEdit_results = QtWidgets.QPlainTextEdit(FeatureSelection)
        self.plainTextEdit_results.setObjectName("plainTextEdit_results")
        self.gridLayout_2.addWidget(self.plainTextEdit_results, 1, 1, 1, 1)
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.label = QtWidgets.QLabel(FeatureSelection)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 0, 0, 1, 1)
        self.label_2 = QtWidgets.QLabel(FeatureSelection)
        self.label_2.setWhatsThis("")
        self.label_2.setAccessibleDescription("")
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 0, 1, 1, 1)
        self.label_3 = QtWidgets.QLabel(FeatureSelection)
        self.label_3.setObjectName("label_3")
        self.gridLayout.addWidget(self.label_3, 0, 2, 1, 1)
        self.comboBox_method = QtWidgets.QComboBox(FeatureSelection)
        self.comboBox_method.setObjectName("comboBox_method")
        self.comboBox_method.addItem("")
        self.comboBox_method.addItem("")
        self.comboBox_method.addItem("")
        self.comboBox_method.addItem("")
        self.comboBox_method.addItem("")
        self.comboBox_method.addItem("")
        self.comboBox_method.addItem("")
        self.comboBox_method.addItem("")
        self.gridLayout.addWidget(self.comboBox_method, 1, 0, 1, 1)
        self.doubleSpinBox_threshold = QtWidgets.QDoubleSpinBox(FeatureSelection)
        self.doubleSpinBox_threshold.setMaximum(1.0)
        self.doubleSpinBox_threshold.setSingleStep(0.01)
        self.doubleSpinBox_threshold.setProperty("value", 0.01)
        self.doubleSpinBox_threshold.setObjectName("doubleSpinBox_threshold")
        self.gridLayout.addWidget(self.doubleSpinBox_threshold, 1, 1, 1, 1)
        self.spinBox_num_param = QtWidgets.QSpinBox(FeatureSelection)
        self.spinBox_num_param.setMaximum(1000)
        self.spinBox_num_param.setProperty("value", 100)
        self.spinBox_num_param.setObjectName("spinBox_num_param")
        self.gridLayout.addWidget(self.spinBox_num_param, 1, 2, 1, 1)
        self.pushButton_select_features = QtWidgets.QPushButton(FeatureSelection)
        self.pushButton_select_features.setStyleSheet("background-color: rgb(255, 255, 191);")
        self.pushButton_select_features.setObjectName("pushButton_select_features")
        self.gridLayout.addWidget(self.pushButton_select_features, 1, 3, 1, 1)
        self.pushButton_import_param = QtWidgets.QPushButton(FeatureSelection)
        self.pushButton_import_param.setStyleSheet("background-color: rgb(227, 200, 255);")
        self.pushButton_import_param.setObjectName("pushButton_import_param")
        self.gridLayout.addWidget(self.pushButton_import_param, 0, 3, 1, 1)
        self.gridLayout_2.addLayout(self.gridLayout, 2, 0, 1, 2)

        self.retranslateUi(FeatureSelection)
        QtCore.QMetaObject.connectSlotsByName(FeatureSelection)

    def retranslateUi(self, FeatureSelection):
        _translate = QtCore.QCoreApplication.translate
        FeatureSelection.setWindowTitle(_translate("FeatureSelection", "Feature Selection"))
        self.label_params.setText(_translate("FeatureSelection", "Параметры:"))
        self.label_info.setText(_translate("FeatureSelection", "Отчет:"))
        self.label.setText(_translate("FeatureSelection", "method"))
        self.label_2.setToolTip(_translate("FeatureSelection", "<html><head/><body><p>Для Quasi-constant выбирайте маленький порог </p><p>Для Correlation выбирайте порог &gt; 0.75</p></body></html>"))
        self.label_2.setText(_translate("FeatureSelection", "threshold"))
        self.label_3.setToolTip(_translate("FeatureSelection", "<html><head/><body><p>Количество параметров для отбора<br/>Для Boruta -- количество итераций</p><p><br/></p></body></html>"))
        self.label_3.setText(_translate("FeatureSelection", "params"))
        self.comboBox_method.setItemText(0, _translate("FeatureSelection", "Quasi-constant"))
        self.comboBox_method.setItemText(1, _translate("FeatureSelection", "SelectKBest"))
        self.comboBox_method.setItemText(2, _translate("FeatureSelection", "Correlation"))
        self.comboBox_method.setItemText(3, _translate("FeatureSelection", "Forward Selection"))
        self.comboBox_method.setItemText(4, _translate("FeatureSelection", "Backward Selection"))
        self.comboBox_method.setItemText(5, _translate("FeatureSelection", "LASSO"))
        self.comboBox_method.setItemText(6, _translate("FeatureSelection", "Random Forest"))
        self.comboBox_method.setItemText(7, _translate("FeatureSelection", "Boruta"))
        self.pushButton_select_features.setText(_translate("FeatureSelection", "calc"))
        self.pushButton_import_param.setText(_translate("FeatureSelection", "import param"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    FeatureSelection = QtWidgets.QDialog()
    ui = Ui_FeatureSelection()
    ui.setupUi(FeatureSelection)
    FeatureSelection.show()
    sys.exit(app.exec_())
