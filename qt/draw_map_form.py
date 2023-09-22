# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'qt/draw_map_form.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_DrawMapForm(object):
    def setupUi(self, DrawMapForm):
        DrawMapForm.setObjectName("DrawMapForm")
        DrawMapForm.resize(390, 172)
        self.gridLayout_2 = QtWidgets.QGridLayout(DrawMapForm)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.label_3 = QtWidgets.QLabel(DrawMapForm)
        self.label_3.setObjectName("label_3")
        self.gridLayout.addWidget(self.label_3, 0, 0, 1, 1)
        self.comboBox_estimator = QtWidgets.QComboBox(DrawMapForm)
        self.comboBox_estimator.setObjectName("comboBox_estimator")
        self.comboBox_estimator.addItem("")
        self.comboBox_estimator.addItem("")
        self.comboBox_estimator.addItem("")
        self.comboBox_estimator.addItem("")
        self.comboBox_estimator.addItem("")
        self.gridLayout.addWidget(self.comboBox_estimator, 0, 1, 1, 1)
        self.label_27 = QtWidgets.QLabel(DrawMapForm)
        self.label_27.setObjectName("label_27")
        self.gridLayout.addWidget(self.label_27, 0, 2, 1, 1)
        self.comboBox_var_model = QtWidgets.QComboBox(DrawMapForm)
        self.comboBox_var_model.setObjectName("comboBox_var_model")
        self.comboBox_var_model.addItem("")
        self.comboBox_var_model.addItem("")
        self.comboBox_var_model.addItem("")
        self.comboBox_var_model.addItem("")
        self.comboBox_var_model.addItem("")
        self.comboBox_var_model.addItem("")
        self.comboBox_var_model.addItem("")
        self.gridLayout.addWidget(self.comboBox_var_model, 0, 3, 1, 1)
        self.label_4 = QtWidgets.QLabel(DrawMapForm)
        self.label_4.setObjectName("label_4")
        self.gridLayout.addWidget(self.label_4, 1, 0, 1, 1)
        self.comboBox_dist_func = QtWidgets.QComboBox(DrawMapForm)
        self.comboBox_dist_func.setObjectName("comboBox_dist_func")
        self.gridLayout.addWidget(self.comboBox_dist_func, 1, 1, 1, 1)
        self.label_5 = QtWidgets.QLabel(DrawMapForm)
        self.label_5.setObjectName("label_5")
        self.gridLayout.addWidget(self.label_5, 1, 2, 1, 1)
        self.comboBox_bin_func = QtWidgets.QComboBox(DrawMapForm)
        self.comboBox_bin_func.setObjectName("comboBox_bin_func")
        self.comboBox_bin_func.addItem("")
        self.comboBox_bin_func.addItem("")
        self.comboBox_bin_func.addItem("")
        self.comboBox_bin_func.addItem("")
        self.comboBox_bin_func.addItem("")
        self.comboBox_bin_func.addItem("")
        self.comboBox_bin_func.addItem("")
        self.comboBox_bin_func.addItem("")
        self.comboBox_bin_func.addItem("")
        self.gridLayout.addWidget(self.comboBox_bin_func, 1, 3, 1, 1)
        self.label_28 = QtWidgets.QLabel(DrawMapForm)
        self.label_28.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_28.setObjectName("label_28")
        self.gridLayout.addWidget(self.label_28, 2, 0, 1, 1)
        self.spinBox_nlags = QtWidgets.QSpinBox(DrawMapForm)
        self.spinBox_nlags.setMinimum(1)
        self.spinBox_nlags.setMaximum(1000)
        self.spinBox_nlags.setProperty("value", 6)
        self.spinBox_nlags.setObjectName("spinBox_nlags")
        self.gridLayout.addWidget(self.spinBox_nlags, 2, 1, 1, 1)
        self.label_6 = QtWidgets.QLabel(DrawMapForm)
        self.label_6.setObjectName("label_6")
        self.gridLayout.addWidget(self.label_6, 2, 2, 1, 1)
        self.spinBox_sparse = QtWidgets.QSpinBox(DrawMapForm)
        self.spinBox_sparse.setMinimum(1)
        self.spinBox_sparse.setMaximum(50)
        self.spinBox_sparse.setProperty("value", 2)
        self.spinBox_sparse.setObjectName("spinBox_sparse")
        self.gridLayout.addWidget(self.spinBox_sparse, 2, 3, 1, 1)
        self.label_2 = QtWidgets.QLabel(DrawMapForm)
        self.label_2.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label_2.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 3, 0, 1, 1)
        self.spinBox_grid = QtWidgets.QSpinBox(DrawMapForm)
        self.spinBox_grid.setMinimum(20)
        self.spinBox_grid.setMaximum(250)
        self.spinBox_grid.setProperty("value", 75)
        self.spinBox_grid.setObjectName("spinBox_grid")
        self.gridLayout.addWidget(self.spinBox_grid, 3, 1, 1, 1)
        self.label = QtWidgets.QLabel(DrawMapForm)
        self.label.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 3, 2, 1, 1)
        self.comboBox_cmap = QtWidgets.QComboBox(DrawMapForm)
        self.comboBox_cmap.setObjectName("comboBox_cmap")
        self.gridLayout.addWidget(self.comboBox_cmap, 3, 3, 1, 1)
        self.checkBox_filt = QtWidgets.QCheckBox(DrawMapForm)
        self.checkBox_filt.setContextMenuPolicy(QtCore.Qt.PreventContextMenu)
        self.checkBox_filt.setLayoutDirection(QtCore.Qt.RightToLeft)
        self.checkBox_filt.setChecked(True)
        self.checkBox_filt.setObjectName("checkBox_filt")
        self.gridLayout.addWidget(self.checkBox_filt, 4, 0, 1, 1)
        self.spinBox_filt = QtWidgets.QSpinBox(DrawMapForm)
        self.spinBox_filt.setMinimum(5)
        self.spinBox_filt.setSingleStep(2)
        self.spinBox_filt.setProperty("value", 13)
        self.spinBox_filt.setObjectName("spinBox_filt")
        self.gridLayout.addWidget(self.spinBox_filt, 4, 1, 1, 1)
        self.pushButton_map = QtWidgets.QPushButton(DrawMapForm)
        self.pushButton_map.setStyleSheet("background-color: rgb(255, 204, 121);")
        self.pushButton_map.setObjectName("pushButton_map")
        self.gridLayout.addWidget(self.pushButton_map, 4, 2, 1, 2)
        self.gridLayout_2.addLayout(self.gridLayout, 0, 0, 1, 1)

        self.retranslateUi(DrawMapForm)
        QtCore.QMetaObject.connectSlotsByName(DrawMapForm)

    def retranslateUi(self, DrawMapForm):
        _translate = QtCore.QCoreApplication.translate
        DrawMapForm.setWindowTitle(_translate("DrawMapForm", "Draw Map Form"))
        self.label_3.setText(_translate("DrawMapForm", "estimator:"))
        self.comboBox_estimator.setItemText(0, _translate("DrawMapForm", "matheron"))
        self.comboBox_estimator.setItemText(1, _translate("DrawMapForm", "cressie"))
        self.comboBox_estimator.setItemText(2, _translate("DrawMapForm", "dowd"))
        self.comboBox_estimator.setItemText(3, _translate("DrawMapForm", "minmax"))
        self.comboBox_estimator.setItemText(4, _translate("DrawMapForm", "entropy"))
        self.label_27.setText(_translate("DrawMapForm", "var_model:"))
        self.comboBox_var_model.setItemText(0, _translate("DrawMapForm", "spherical"))
        self.comboBox_var_model.setItemText(1, _translate("DrawMapForm", "exponential"))
        self.comboBox_var_model.setItemText(2, _translate("DrawMapForm", "gaussian"))
        self.comboBox_var_model.setItemText(3, _translate("DrawMapForm", "cubic"))
        self.comboBox_var_model.setItemText(4, _translate("DrawMapForm", "stable"))
        self.comboBox_var_model.setItemText(5, _translate("DrawMapForm", "matern"))
        self.comboBox_var_model.setItemText(6, _translate("DrawMapForm", "nugget"))
        self.label_4.setText(_translate("DrawMapForm", "dist func:"))
        self.label_5.setText(_translate("DrawMapForm", "bin_func:"))
        self.comboBox_bin_func.setItemText(0, _translate("DrawMapForm", "even"))
        self.comboBox_bin_func.setItemText(1, _translate("DrawMapForm", "uniform"))
        self.comboBox_bin_func.setItemText(2, _translate("DrawMapForm", "fd"))
        self.comboBox_bin_func.setItemText(3, _translate("DrawMapForm", "sturges"))
        self.comboBox_bin_func.setItemText(4, _translate("DrawMapForm", "scott"))
        self.comboBox_bin_func.setItemText(5, _translate("DrawMapForm", "doane"))
        self.comboBox_bin_func.setItemText(6, _translate("DrawMapForm", "sqrt"))
        self.comboBox_bin_func.setItemText(7, _translate("DrawMapForm", "kmeans"))
        self.comboBox_bin_func.setItemText(8, _translate("DrawMapForm", "ward"))
        self.label_28.setToolTip(_translate("DrawMapForm", "Количество ячеек усреднения для вариограммы"))
        self.label_28.setText(_translate("DrawMapForm", "nlags:"))
        self.label_6.setText(_translate("DrawMapForm", "sparse list:"))
        self.label_2.setText(_translate("DrawMapForm", "grid:"))
        self.spinBox_grid.setToolTip(_translate("DrawMapForm", "Размер сетки"))
        self.label.setText(_translate("DrawMapForm", "cmap:"))
        self.checkBox_filt.setText(_translate("DrawMapForm", "filter"))
        self.pushButton_map.setText(_translate("DrawMapForm", "DRAW"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    DrawMapForm = QtWidgets.QWidget()
    ui = Ui_DrawMapForm()
    ui.setupUi(DrawMapForm)
    DrawMapForm.show()
    sys.exit(app.exec_())
