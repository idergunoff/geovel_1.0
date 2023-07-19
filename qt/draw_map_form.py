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
        DrawMapForm.resize(318, 190)
        self.gridLayout_2 = QtWidgets.QGridLayout(DrawMapForm)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.label_27 = QtWidgets.QLabel(DrawMapForm)
        self.label_27.setObjectName("label_27")
        self.gridLayout.addWidget(self.label_27, 0, 0, 1, 1)
        self.comboBox_var_model = QtWidgets.QComboBox(DrawMapForm)
        self.comboBox_var_model.setObjectName("comboBox_var_model")
        self.comboBox_var_model.addItem("")
        self.comboBox_var_model.addItem("")
        self.comboBox_var_model.addItem("")
        self.comboBox_var_model.addItem("")
        self.comboBox_var_model.addItem("")
        self.gridLayout.addWidget(self.comboBox_var_model, 0, 1, 1, 1)
        self.checkBox_weight = QtWidgets.QCheckBox(DrawMapForm)
        self.checkBox_weight.setObjectName("checkBox_weight")
        self.gridLayout.addWidget(self.checkBox_weight, 0, 2, 1, 1)
        self.label_28 = QtWidgets.QLabel(DrawMapForm)
        self.label_28.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_28.setObjectName("label_28")
        self.gridLayout.addWidget(self.label_28, 1, 0, 1, 1)
        self.spinBox_nlags = QtWidgets.QSpinBox(DrawMapForm)
        self.spinBox_nlags.setMinimum(1)
        self.spinBox_nlags.setMaximum(1000)
        self.spinBox_nlags.setProperty("value", 6)
        self.spinBox_nlags.setObjectName("spinBox_nlags")
        self.gridLayout.addWidget(self.spinBox_nlags, 1, 1, 1, 1)
        self.checkBox_vector = QtWidgets.QCheckBox(DrawMapForm)
        self.checkBox_vector.setChecked(True)
        self.checkBox_vector.setObjectName("checkBox_vector")
        self.gridLayout.addWidget(self.checkBox_vector, 1, 2, 1, 1)
        self.comboBox_cmap = QtWidgets.QComboBox(DrawMapForm)
        self.comboBox_cmap.setObjectName("comboBox_cmap")
        self.gridLayout.addWidget(self.comboBox_cmap, 2, 1, 1, 1)
        self.spinBox_grid = QtWidgets.QSpinBox(DrawMapForm)
        self.spinBox_grid.setMinimum(20)
        self.spinBox_grid.setMaximum(250)
        self.spinBox_grid.setProperty("value", 75)
        self.spinBox_grid.setObjectName("spinBox_grid")
        self.gridLayout.addWidget(self.spinBox_grid, 3, 1, 1, 1)
        self.label = QtWidgets.QLabel(DrawMapForm)
        self.label.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 2, 0, 1, 1)
        self.label_2 = QtWidgets.QLabel(DrawMapForm)
        self.label_2.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label_2.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 3, 0, 1, 1)
        self.gridLayout_2.addLayout(self.gridLayout, 0, 0, 1, 1)
        self.pushButton_map = QtWidgets.QPushButton(DrawMapForm)
        self.pushButton_map.setObjectName("pushButton_map")
        self.gridLayout_2.addWidget(self.pushButton_map, 1, 0, 1, 1)

        self.retranslateUi(DrawMapForm)
        QtCore.QMetaObject.connectSlotsByName(DrawMapForm)

    def retranslateUi(self, DrawMapForm):
        _translate = QtCore.QCoreApplication.translate
        DrawMapForm.setWindowTitle(_translate("DrawMapForm", "Draw Map Form"))
        self.label_27.setText(_translate("DrawMapForm", "var_model:"))
        self.comboBox_var_model.setItemText(0, _translate("DrawMapForm", "linear"))
        self.comboBox_var_model.setItemText(1, _translate("DrawMapForm", "power"))
        self.comboBox_var_model.setItemText(2, _translate("DrawMapForm", "gaussian"))
        self.comboBox_var_model.setItemText(3, _translate("DrawMapForm", "spherical"))
        self.comboBox_var_model.setItemText(4, _translate("DrawMapForm", "exponential"))
        self.checkBox_weight.setToolTip(_translate("DrawMapForm", "Флаг, указывающий, следует ли при автоматическом расчете модели вариограммы более сильно взвешивать полувариантность с меньшими задержками. В настоящее время процедура жестко запрограммирована таким образом, что веса вычисляются на основе логистической функции, поэтому веса при малых задержках равны ~ 1, а веса при самых больших задержках равны ~ 0; центр логистического взвешивания жестко запрограммирован так, чтобы он находился на 70% расстояния от наименьшего запаздывания до самого большого. самое большое отставание. Установка этого параметра в значение True указывает на то, что будут применены веса. "))
        self.checkBox_weight.setText(_translate("DrawMapForm", "weight"))
        self.label_28.setToolTip(_translate("DrawMapForm", "Количество ячеек усреднения для вариограммы"))
        self.label_28.setText(_translate("DrawMapForm", "nlags:"))
        self.checkBox_vector.setToolTip(_translate("DrawMapForm", "подход при кригинге"))
        self.checkBox_vector.setText(_translate("DrawMapForm", "vectorized"))
        self.spinBox_grid.setToolTip(_translate("DrawMapForm", "Размер сетки"))
        self.label.setText(_translate("DrawMapForm", "cmap:"))
        self.label_2.setText(_translate("DrawMapForm", "grid:"))
        self.pushButton_map.setText(_translate("DrawMapForm", "DRAW"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    DrawMapForm = QtWidgets.QWidget()
    ui = Ui_DrawMapForm()
    ui.setupUi(DrawMapForm)
    DrawMapForm.show()
    sys.exit(app.exec_())
