# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'qt/well_loader.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_WellLoader(object):
    def setupUi(self, WellLoader):
        WellLoader.setObjectName("WellLoader")
        WellLoader.resize(987, 265)
        self.gridLayout_14 = QtWidgets.QGridLayout(WellLoader)
        self.gridLayout_14.setObjectName("gridLayout_14")
        self.gridLayout_13 = QtWidgets.QGridLayout()
        self.gridLayout_13.setObjectName("gridLayout_13")
        self.gridLayout_12 = QtWidgets.QGridLayout()
        self.gridLayout_12.setObjectName("gridLayout_12")
        self.gridLayout_10 = QtWidgets.QGridLayout()
        self.gridLayout_10.setObjectName("gridLayout_10")
        self.gridLayout_5 = QtWidgets.QGridLayout()
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.label = QtWidgets.QLabel(WellLoader)
        self.label.setObjectName("label")
        self.gridLayout_5.addWidget(self.label, 0, 0, 1, 1)
        self.comboBox_name = QtWidgets.QComboBox(WellLoader)
        self.comboBox_name.setObjectName("comboBox_name")
        self.gridLayout_5.addWidget(self.comboBox_name, 0, 1, 1, 1)
        self.gridLayout_5.setColumnStretch(0, 1)
        self.gridLayout_5.setColumnStretch(1, 2)
        self.gridLayout_10.addLayout(self.gridLayout_5, 0, 0, 1, 1)
        self.gridLayout_6 = QtWidgets.QGridLayout()
        self.gridLayout_6.setObjectName("gridLayout_6")
        self.label_2 = QtWidgets.QLabel(WellLoader)
        self.label_2.setObjectName("label_2")
        self.gridLayout_6.addWidget(self.label_2, 0, 0, 1, 1)
        self.comboBox_x = QtWidgets.QComboBox(WellLoader)
        self.comboBox_x.setObjectName("comboBox_x")
        self.gridLayout_6.addWidget(self.comboBox_x, 0, 1, 1, 1)
        self.gridLayout_6.setColumnStretch(0, 1)
        self.gridLayout_6.setColumnStretch(1, 2)
        self.gridLayout_10.addLayout(self.gridLayout_6, 1, 0, 1, 1)
        self.gridLayout_7 = QtWidgets.QGridLayout()
        self.gridLayout_7.setObjectName("gridLayout_7")
        self.label_3 = QtWidgets.QLabel(WellLoader)
        self.label_3.setObjectName("label_3")
        self.gridLayout_7.addWidget(self.label_3, 0, 0, 1, 1)
        self.comboBox_y = QtWidgets.QComboBox(WellLoader)
        self.comboBox_y.setObjectName("comboBox_y")
        self.gridLayout_7.addWidget(self.comboBox_y, 0, 1, 1, 1)
        self.gridLayout_7.setColumnStretch(0, 1)
        self.gridLayout_7.setColumnStretch(1, 2)
        self.gridLayout_10.addLayout(self.gridLayout_7, 2, 0, 1, 1)
        self.gridLayout_8 = QtWidgets.QGridLayout()
        self.gridLayout_8.setObjectName("gridLayout_8")
        self.label_4 = QtWidgets.QLabel(WellLoader)
        self.label_4.setObjectName("label_4")
        self.gridLayout_8.addWidget(self.label_4, 0, 0, 1, 1)
        self.comboBox_alt = QtWidgets.QComboBox(WellLoader)
        self.comboBox_alt.setObjectName("comboBox_alt")
        self.gridLayout_8.addWidget(self.comboBox_alt, 0, 1, 1, 1)
        self.gridLayout_8.setColumnStretch(0, 1)
        self.gridLayout_8.setColumnStretch(1, 2)
        self.gridLayout_10.addLayout(self.gridLayout_8, 3, 0, 1, 1)
        self.gridLayout_9 = QtWidgets.QGridLayout()
        self.gridLayout_9.setObjectName("gridLayout_9")
        self.label_5 = QtWidgets.QLabel(WellLoader)
        self.label_5.setObjectName("label_5")
        self.gridLayout_9.addWidget(self.label_5, 0, 0, 1, 1)
        self.lineEdit_empty = QtWidgets.QLineEdit(WellLoader)
        self.lineEdit_empty.setObjectName("lineEdit_empty")
        self.gridLayout_9.addWidget(self.lineEdit_empty, 0, 1, 1, 1)
        self.gridLayout_9.setColumnStretch(0, 1)
        self.gridLayout_9.setColumnStretch(1, 2)
        self.gridLayout_10.addLayout(self.gridLayout_9, 4, 0, 1, 1)
        self.gridLayout_12.addLayout(self.gridLayout_10, 0, 0, 1, 1)
        self.gridLayout_11 = QtWidgets.QGridLayout()
        self.gridLayout_11.setObjectName("gridLayout_11")
        self.groupBox = QtWidgets.QGroupBox(WellLoader)
        self.groupBox.setObjectName("groupBox")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.groupBox)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.lineEdit_layers = QtWidgets.QLineEdit(self.groupBox)
        self.lineEdit_layers.setObjectName("lineEdit_layers")
        self.gridLayout.addWidget(self.lineEdit_layers, 1, 0, 1, 2)
        self.pushButton_add_layer = QtWidgets.QPushButton(self.groupBox)
        self.pushButton_add_layer.setStyleSheet("background-color: rgb(191, 255, 191);")
        self.pushButton_add_layer.setObjectName("pushButton_add_layer")
        self.gridLayout.addWidget(self.pushButton_add_layer, 0, 1, 1, 1)
        self.comboBox_layers = QtWidgets.QComboBox(self.groupBox)
        self.comboBox_layers.setObjectName("comboBox_layers")
        self.gridLayout.addWidget(self.comboBox_layers, 0, 0, 1, 1)
        self.gridLayout.setColumnStretch(0, 2)
        self.gridLayout.setColumnStretch(1, 1)
        self.gridLayout_2.addLayout(self.gridLayout, 0, 0, 1, 1)
        self.gridLayout_11.addWidget(self.groupBox, 0, 0, 1, 1)
        self.groupBox_2 = QtWidgets.QGroupBox(WellLoader)
        self.groupBox_2.setObjectName("groupBox_2")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.groupBox_2)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.gridLayout_3 = QtWidgets.QGridLayout()
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.comboBox_opt = QtWidgets.QComboBox(self.groupBox_2)
        self.comboBox_opt.setObjectName("comboBox_opt")
        self.gridLayout_3.addWidget(self.comboBox_opt, 0, 0, 1, 1)
        self.pushButton_add_opt = QtWidgets.QPushButton(self.groupBox_2)
        self.pushButton_add_opt.setStyleSheet("background-color: rgb(191, 255, 191);")
        self.pushButton_add_opt.setObjectName("pushButton_add_opt")
        self.gridLayout_3.addWidget(self.pushButton_add_opt, 0, 1, 1, 1)
        self.lineEdit_opt = QtWidgets.QLineEdit(self.groupBox_2)
        self.lineEdit_opt.setObjectName("lineEdit_opt")
        self.gridLayout_3.addWidget(self.lineEdit_opt, 1, 0, 1, 2)
        self.gridLayout_3.setColumnStretch(0, 2)
        self.gridLayout_3.setColumnStretch(1, 1)
        self.gridLayout_4.addLayout(self.gridLayout_3, 0, 0, 1, 1)
        self.gridLayout_11.addWidget(self.groupBox_2, 1, 0, 1, 1)
        self.gridLayout_12.addLayout(self.gridLayout_11, 0, 1, 1, 1)
        self.gridLayout_12.setColumnStretch(0, 1)
        self.gridLayout_12.setColumnStretch(1, 3)
        self.gridLayout_13.addLayout(self.gridLayout_12, 0, 0, 1, 1)
        self.buttonBox = QtWidgets.QDialogButtonBox(WellLoader)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.gridLayout_13.addWidget(self.buttonBox, 1, 0, 1, 1)
        self.gridLayout_14.addLayout(self.gridLayout_13, 0, 0, 1, 1)

        self.retranslateUi(WellLoader)
        QtCore.QMetaObject.connectSlotsByName(WellLoader)

    def retranslateUi(self, WellLoader):
        _translate = QtCore.QCoreApplication.translate
        WellLoader.setWindowTitle(_translate("WellLoader", "Well_loader"))
        self.label.setText(_translate("WellLoader", "Name:"))
        self.label_2.setText(_translate("WellLoader", "X:"))
        self.label_3.setText(_translate("WellLoader", "Y:"))
        self.label_4.setText(_translate("WellLoader", "Alt:"))
        self.label_5.setText(_translate("WellLoader", "Empty:"))
        self.groupBox.setTitle(_translate("WellLoader", "Layers"))
        self.pushButton_add_layer.setText(_translate("WellLoader", "add"))
        self.groupBox_2.setTitle(_translate("WellLoader", "Optionally"))
        self.pushButton_add_opt.setText(_translate("WellLoader", "add"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    WellLoader = QtWidgets.QWidget()
    ui = Ui_WellLoader()
    ui.setupUi(WellLoader)
    WellLoader.show()
    sys.exit(app.exec_())
