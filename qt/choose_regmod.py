# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'qt/choose_regmod.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_FormRegMod(object):
    def setupUi(self, FormRegMod):
        FormRegMod.setObjectName("FormRegMod")
        FormRegMod.resize(258, 115)
        FormRegMod.setStyleSheet("")
        self.gridLayout_2 = QtWidgets.QGridLayout(FormRegMod)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.label = QtWidgets.QLabel(FormRegMod)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 0, 0, 1, 1)
        self.pushButton_calc_model = QtWidgets.QPushButton(FormRegMod)
        self.pushButton_calc_model.setStyleSheet("background-color: rgb(191, 255, 191);")
        self.pushButton_calc_model.setObjectName("pushButton_calc_model")
        self.gridLayout.addWidget(self.pushButton_calc_model, 2, 0, 1, 1)
        self.checkBox_color_marker = QtWidgets.QCheckBox(FormRegMod)
        self.checkBox_color_marker.setChecked(False)
        self.checkBox_color_marker.setObjectName("checkBox_color_marker")
        self.gridLayout.addWidget(self.checkBox_color_marker, 1, 0, 1, 1)
        self.gridLayout_2.addLayout(self.gridLayout, 0, 0, 1, 1)

        self.retranslateUi(FormRegMod)
        QtCore.QMetaObject.connectSlotsByName(FormRegMod)

    def retranslateUi(self, FormRegMod):
        _translate = QtCore.QCoreApplication.translate
        FormRegMod.setWindowTitle(_translate("FormRegMod", "RegMod"))
        self.label.setText(_translate("FormRegMod", "Выберите модель и нажмите ОК"))
        self.pushButton_calc_model.setText(_translate("FormRegMod", "OK"))
        self.checkBox_color_marker.setText(_translate("FormRegMod", "color_marker"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    FormRegMod = QtWidgets.QWidget()
    ui = Ui_FormRegMod()
    ui.setupUi(FormRegMod)
    FormRegMod.show()
    sys.exit(app.exec_())
