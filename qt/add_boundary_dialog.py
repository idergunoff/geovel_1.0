# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'C:\Users\USER\PycharmProjects\geovel_1.0\qt\add_boundary_dialog.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_add_bondary(object):
    def setupUi(self, add_bondary):
        add_bondary.setObjectName("add_bondary")
        add_bondary.resize(400, 112)
        self.gridLayout_2 = QtWidgets.QGridLayout(add_bondary)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.label = QtWidgets.QLabel(add_bondary)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 0, 0, 1, 1)
        self.lineEdit_title = QtWidgets.QLineEdit(add_bondary)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.lineEdit_title.setFont(font)
        self.lineEdit_title.setInputMask("")
        self.lineEdit_title.setObjectName("lineEdit_title")
        self.gridLayout.addWidget(self.lineEdit_title, 1, 0, 1, 1)
        self.lineEdit_depth = QtWidgets.QLineEdit(add_bondary)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.lineEdit_depth.setFont(font)
        self.lineEdit_depth.setObjectName("lineEdit_depth")
        self.gridLayout.addWidget(self.lineEdit_depth, 1, 1, 1, 1)
        self.label_2 = QtWidgets.QLabel(add_bondary)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 0, 1, 1, 1)
        self.gridLayout_2.addLayout(self.gridLayout, 0, 0, 1, 1)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.buttonBox = QtWidgets.QDialogButtonBox(add_bondary)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.buttonBox.setFont(font)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.horizontalLayout_2.addWidget(self.buttonBox)
        self.horizontalLayout_2.setStretch(0, 2)
        self.gridLayout_2.addLayout(self.horizontalLayout_2, 1, 0, 1, 1)

        self.retranslateUi(add_bondary)
        QtCore.QMetaObject.connectSlotsByName(add_bondary)

    def retranslateUi(self, add_bondary):
        _translate = QtCore.QCoreApplication.translate
        add_bondary.setWindowTitle(_translate("add_bondary", "Добавить новую границу"))
        self.label.setText(_translate("add_bondary", "Введите название границы:"))
        self.lineEdit_title.setText(_translate("add_bondary", "layer"))
        self.lineEdit_depth.setText(_translate("add_bondary", "0"))
        self.label_2.setText(_translate("add_bondary", "и глубину в метрах:"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    add_bondary = QtWidgets.QWidget()
    ui = Ui_add_bondary()
    ui.setupUi(add_bondary)
    add_bondary.show()
    sys.exit(app.exec_())