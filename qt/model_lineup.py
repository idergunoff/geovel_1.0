# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'qt/model_lineup.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
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
        self.listWidget_lineup = QtWidgets.QListWidget(Dialog_model_lineup)
        self.listWidget_lineup.setGeometry(QtCore.QRect(10, 30, 201, 201))
        self.listWidget_lineup.setObjectName("listWidget_lineup")
        self.plainTextEdit_info = QtWidgets.QPlainTextEdit(Dialog_model_lineup)
        self.plainTextEdit_info.setGeometry(QtCore.QRect(220, 30, 271, 201))
        self.plainTextEdit_info.setObjectName("plainTextEdit_info")
        self.layoutWidget = QtWidgets.QWidget(Dialog_model_lineup)
        self.layoutWidget.setGeometry(QtCore.QRect(10, 240, 262, 27))
        self.layoutWidget.setObjectName("layoutWidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.layoutWidget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.pushButton_clear = QtWidgets.QPushButton(self.layoutWidget)
        self.pushButton_clear.setStyleSheet("background-color: rgb(255, 185, 185);")
        self.pushButton_clear.setObjectName("pushButton_clear")
        self.horizontalLayout.addWidget(self.pushButton_clear)
        self.pushButton_remove = QtWidgets.QPushButton(self.layoutWidget)
        self.pushButton_remove.setStyleSheet("background-color: rgb(255, 185, 185);")
        self.pushButton_remove.setObjectName("pushButton_remove")
        self.horizontalLayout.addWidget(self.pushButton_remove)
        self.pushButton_start = QtWidgets.QPushButton(self.layoutWidget)
        self.pushButton_start.setStyleSheet("background-color: rgb(191, 255, 191);")
        self.pushButton_start.setObjectName("pushButton_start")
        self.horizontalLayout.addWidget(self.pushButton_start)

        self.retranslateUi(Dialog_model_lineup)
        QtCore.QMetaObject.connectSlotsByName(Dialog_model_lineup)

    def retranslateUi(self, Dialog_model_lineup):
        _translate = QtCore.QCoreApplication.translate
        Dialog_model_lineup.setWindowTitle(_translate("Dialog_model_lineup", "Dialog"))
        self.label.setText(_translate("Dialog_model_lineup", "Lineup:"))
        self.label_2.setText(_translate("Dialog_model_lineup", "Model info:"))
        self.pushButton_clear.setText(_translate("Dialog_model_lineup", "CLEAR"))
        self.pushButton_remove.setText(_translate("Dialog_model_lineup", "REMOVE"))
        self.pushButton_start.setText(_translate("Dialog_model_lineup", "START"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Dialog_model_lineup = QtWidgets.QDialog()
    ui = Ui_Dialog_model_lineup()
    ui.setupUi(Dialog_model_lineup)
    Dialog_model_lineup.show()
    sys.exit(app.exec_())
