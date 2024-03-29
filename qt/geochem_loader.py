# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'qt/geochem_loader.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_GeochemLoader(object):
    def setupUi(self, GeochemLoader):
        GeochemLoader.setObjectName("GeochemLoader")
        GeochemLoader.resize(864, 212)
        self.gridLayout_12 = QtWidgets.QGridLayout(GeochemLoader)
        self.gridLayout_12.setObjectName("gridLayout_12")
        self.gridLayout_11 = QtWidgets.QGridLayout()
        self.gridLayout_11.setObjectName("gridLayout_11")
        self.gridLayout_8 = QtWidgets.QGridLayout()
        self.gridLayout_8.setObjectName("gridLayout_8")
        self.gridLayout_5 = QtWidgets.QGridLayout()
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.label = QtWidgets.QLabel(GeochemLoader)
        self.label.setObjectName("label")
        self.gridLayout_5.addWidget(self.label, 0, 0, 1, 1)
        self.comboBox_name = QtWidgets.QComboBox(GeochemLoader)
        self.comboBox_name.setObjectName("comboBox_name")
        self.gridLayout_5.addWidget(self.comboBox_name, 0, 1, 1, 1)
        self.gridLayout_5.setColumnStretch(0, 1)
        self.gridLayout_5.setColumnStretch(1, 2)
        self.gridLayout_8.addLayout(self.gridLayout_5, 0, 0, 1, 1)
        self.gridLayout_6 = QtWidgets.QGridLayout()
        self.gridLayout_6.setObjectName("gridLayout_6")
        self.label_2 = QtWidgets.QLabel(GeochemLoader)
        self.label_2.setObjectName("label_2")
        self.gridLayout_6.addWidget(self.label_2, 0, 0, 1, 1)
        self.comboBox_x = QtWidgets.QComboBox(GeochemLoader)
        self.comboBox_x.setObjectName("comboBox_x")
        self.gridLayout_6.addWidget(self.comboBox_x, 0, 1, 1, 1)
        self.gridLayout_6.setColumnStretch(0, 1)
        self.gridLayout_6.setColumnStretch(1, 2)
        self.gridLayout_8.addLayout(self.gridLayout_6, 1, 0, 1, 1)
        self.gridLayout_7 = QtWidgets.QGridLayout()
        self.gridLayout_7.setObjectName("gridLayout_7")
        self.label_3 = QtWidgets.QLabel(GeochemLoader)
        self.label_3.setObjectName("label_3")
        self.gridLayout_7.addWidget(self.label_3, 0, 0, 1, 1)
        self.comboBox_y = QtWidgets.QComboBox(GeochemLoader)
        self.comboBox_y.setObjectName("comboBox_y")
        self.gridLayout_7.addWidget(self.comboBox_y, 0, 1, 1, 1)
        self.gridLayout_7.setColumnStretch(0, 1)
        self.gridLayout_7.setColumnStretch(1, 2)
        self.gridLayout_8.addLayout(self.gridLayout_7, 2, 0, 1, 1)
        self.gridLayout_3 = QtWidgets.QGridLayout()
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.label_4 = QtWidgets.QLabel(GeochemLoader)
        self.label_4.setObjectName("label_4")
        self.gridLayout_3.addWidget(self.label_4, 0, 0, 1, 1)
        self.comboBox_class = QtWidgets.QComboBox(GeochemLoader)
        self.comboBox_class.setObjectName("comboBox_class")
        self.gridLayout_3.addWidget(self.comboBox_class, 0, 1, 1, 1)
        self.comboBox_classes = QtWidgets.QComboBox(GeochemLoader)
        self.comboBox_classes.setObjectName("comboBox_classes")
        self.gridLayout_3.addWidget(self.comboBox_classes, 0, 2, 1, 1)
        self.gridLayout_8.addLayout(self.gridLayout_3, 3, 0, 1, 1)
        self.gridLayout_4 = QtWidgets.QGridLayout()
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.pushButton_color = QtWidgets.QPushButton(GeochemLoader)
        self.pushButton_color.setObjectName("pushButton_color")
        self.gridLayout_4.addWidget(self.pushButton_color, 0, 0, 1, 1)
        self.pushButton_add_well = QtWidgets.QPushButton(GeochemLoader)
        self.pushButton_add_well.setStyleSheet("background-color: rgb(161, 245, 173);")
        self.pushButton_add_well.setObjectName("pushButton_add_well")
        self.gridLayout_4.addWidget(self.pushButton_add_well, 0, 1, 1, 1)
        self.gridLayout_8.addLayout(self.gridLayout_4, 4, 0, 1, 1)
        self.gridLayout_11.addLayout(self.gridLayout_8, 0, 0, 1, 1)
        self.gridLayout_9 = QtWidgets.QGridLayout()
        self.gridLayout_9.setObjectName("gridLayout_9")
        self.gridLayout_2 = QtWidgets.QGridLayout()
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.label_6 = QtWidgets.QLabel(GeochemLoader)
        self.label_6.setObjectName("label_6")
        self.gridLayout_2.addWidget(self.label_6, 0, 0, 1, 1)
        self.pushButton_del_well = QtWidgets.QPushButton(GeochemLoader)
        self.pushButton_del_well.setStyleSheet("background-color: rgb(253, 196, 196);")
        self.pushButton_del_well.setObjectName("pushButton_del_well")
        self.gridLayout_2.addWidget(self.pushButton_del_well, 0, 1, 1, 1)
        self.gridLayout_2.setColumnStretch(0, 3)
        self.gridLayout_9.addLayout(self.gridLayout_2, 0, 0, 1, 1)
        self.listWidget_well = QtWidgets.QListWidget(GeochemLoader)
        self.listWidget_well.setObjectName("listWidget_well")
        self.gridLayout_9.addWidget(self.listWidget_well, 1, 0, 1, 1)
        self.gridLayout_11.addLayout(self.gridLayout_9, 0, 1, 1, 1)
        self.gridLayout_10 = QtWidgets.QGridLayout()
        self.gridLayout_10.setObjectName("gridLayout_10")
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.label_5 = QtWidgets.QLabel(GeochemLoader)
        self.label_5.setObjectName("label_5")
        self.gridLayout.addWidget(self.label_5, 0, 0, 1, 1)
        self.pushButton_del_param = QtWidgets.QPushButton(GeochemLoader)
        self.pushButton_del_param.setStyleSheet("background-color: rgb(253, 196, 196);")
        self.pushButton_del_param.setObjectName("pushButton_del_param")
        self.gridLayout.addWidget(self.pushButton_del_param, 0, 1, 1, 1)
        self.gridLayout.setColumnStretch(0, 3)
        self.gridLayout_10.addLayout(self.gridLayout, 0, 0, 1, 1)
        self.listWidget_param = QtWidgets.QListWidget(GeochemLoader)
        self.listWidget_param.setObjectName("listWidget_param")
        self.gridLayout_10.addWidget(self.listWidget_param, 1, 0, 1, 1)
        self.gridLayout_11.addLayout(self.gridLayout_10, 0, 2, 1, 1)
        self.gridLayout_11.setColumnStretch(0, 1)
        self.gridLayout_11.setColumnStretch(1, 3)
        self.gridLayout_11.setColumnStretch(2, 3)
        self.gridLayout_12.addLayout(self.gridLayout_11, 0, 0, 1, 1)
        self.buttonBox = QtWidgets.QDialogButtonBox(GeochemLoader)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.gridLayout_12.addWidget(self.buttonBox, 1, 0, 1, 1)

        self.retranslateUi(GeochemLoader)
        QtCore.QMetaObject.connectSlotsByName(GeochemLoader)

    def retranslateUi(self, GeochemLoader):
        _translate = QtCore.QCoreApplication.translate
        GeochemLoader.setWindowTitle(_translate("GeochemLoader", "Geochem_loader"))
        self.label.setText(_translate("GeochemLoader", "Name:"))
        self.label_2.setText(_translate("GeochemLoader", "X:"))
        self.label_3.setText(_translate("GeochemLoader", "Y:"))
        self.label_4.setText(_translate("GeochemLoader", "class:"))
        self.pushButton_color.setText(_translate("GeochemLoader", "COLOR"))
        self.pushButton_add_well.setText(_translate("GeochemLoader", "ADD WELL"))
        self.label_6.setText(_translate("GeochemLoader", "well:"))
        self.pushButton_del_well.setText(_translate("GeochemLoader", "delete"))
        self.label_5.setText(_translate("GeochemLoader", "parameter:"))
        self.pushButton_del_param.setText(_translate("GeochemLoader", "delete"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    GeochemLoader = QtWidgets.QWidget()
    ui = Ui_GeochemLoader()
    ui.setupUi(GeochemLoader)
    GeochemLoader.show()
    sys.exit(app.exec_())
