# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'C:\Users\SayfutdinovaAMa\PycharmProjects\geovel_1.0\qt\model_lineup.ui'
#
# Created by: PyQt5 UI code generator 5.15.10
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Dialog_model_lineup(object):
    def setupUi(self, Dialog_model_lineup):
        Dialog_model_lineup.setObjectName("Dialog_model_lineup")
        Dialog_model_lineup.resize(506, 284)
        self.label = QtWidgets.QLabel(Dialog_model_lineup)
        self.label.setGeometry(QtCore.QRect(10, 10, 61, 16))
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(Dialog_model_lineup)
        self.label_2.setGeometry(QtCore.QRect(220, 10, 61, 16))
        self.label_2.setObjectName("label_2")
        self.listWidget_2 = QtWidgets.QListWidget(Dialog_model_lineup)
        self.listWidget_2.setGeometry(QtCore.QRect(10, 30, 201, 201))
        self.listWidget_2.setObjectName("listWidget_2")
        self.plainTextEdit_2 = QtWidgets.QPlainTextEdit(Dialog_model_lineup)
        self.plainTextEdit_2.setGeometry(QtCore.QRect(220, 30, 271, 201))
        self.plainTextEdit_2.setObjectName("plainTextEdit_2")
        self.widget = QtWidgets.QWidget(Dialog_model_lineup)
        self.widget.setGeometry(QtCore.QRect(10, 240, 201, 25))
        self.widget.setObjectName("widget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.widget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.pushButton = QtWidgets.QPushButton(self.widget)
        self.pushButton.setStyleSheet("background-color: rgb(255, 185, 185);")
        self.pushButton.setObjectName("pushButton")
        self.horizontalLayout.addWidget(self.pushButton)
        self.pushButton_2 = QtWidgets.QPushButton(self.widget)
        self.pushButton_2.setStyleSheet("background-color: rgb(191, 255, 191);")
        self.pushButton_2.setObjectName("pushButton_2")
        self.horizontalLayout.addWidget(self.pushButton_2)

        self.retranslateUi(Dialog_model_lineup)
        QtCore.QMetaObject.connectSlotsByName(Dialog_model_lineup)

    def retranslateUi(self, Dialog_model_lineup):
        _translate = QtCore.QCoreApplication.translate
        Dialog_model_lineup.setWindowTitle(_translate("Dialog_model_lineup", "Dialog"))
        self.label.setText(_translate("Dialog_model_lineup", "Lineup:"))
        self.label_2.setText(_translate("Dialog_model_lineup", "Model info:"))
        self.pushButton.setText(_translate("Dialog_model_lineup", "REMOVE"))
        self.pushButton_2.setText(_translate("Dialog_model_lineup", "START"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Dialog_model_lineup = QtWidgets.QDialog()
    ui = Ui_Dialog_model_lineup()
    ui.setupUi(Dialog_model_lineup)
    Dialog_model_lineup.show()
    sys.exit(app.exec_())
