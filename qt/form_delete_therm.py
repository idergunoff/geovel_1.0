# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'qt/form_delete_therm.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Form_delete_therm_by_date(object):
    def setupUi(self, Form_delete_therm_by_date):
        Form_delete_therm_by_date.setObjectName("Form_delete_therm_by_date")
        Form_delete_therm_by_date.resize(322, 102)
        self.gridLayout_3 = QtWidgets.QGridLayout(Form_delete_therm_by_date)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.gridLayout_2 = QtWidgets.QGridLayout()
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.label_2 = QtWidgets.QLabel(Form_delete_therm_by_date)
        self.label_2.setObjectName("label_2")
        self.gridLayout_2.addWidget(self.label_2, 0, 0, 1, 1)
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.dateEdit_start = QtWidgets.QDateEdit(Form_delete_therm_by_date)
        self.dateEdit_start.setObjectName("dateEdit_start")
        self.gridLayout.addWidget(self.dateEdit_start, 0, 0, 1, 1)
        self.label = QtWidgets.QLabel(Form_delete_therm_by_date)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 0, 1, 1, 1)
        self.dateEdit_stop = QtWidgets.QDateEdit(Form_delete_therm_by_date)
        self.dateEdit_stop.setObjectName("dateEdit_stop")
        self.gridLayout.addWidget(self.dateEdit_stop, 0, 2, 1, 1)
        self.gridLayout_2.addLayout(self.gridLayout, 1, 0, 1, 1)
        self.buttonBox = QtWidgets.QDialogButtonBox(Form_delete_therm_by_date)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.gridLayout_2.addWidget(self.buttonBox, 2, 0, 1, 1)
        self.gridLayout_3.addLayout(self.gridLayout_2, 0, 0, 1, 1)

        self.retranslateUi(Form_delete_therm_by_date)
        QtCore.QMetaObject.connectSlotsByName(Form_delete_therm_by_date)

    def retranslateUi(self, Form_delete_therm_by_date):
        _translate = QtCore.QCoreApplication.translate
        Form_delete_therm_by_date.setWindowTitle(_translate("Form_delete_therm_by_date", "Form_delete_therm"))
        self.label_2.setText(_translate("Form_delete_therm_by_date", "Удалить термограммы за текущий период?"))
        self.label.setText(_translate("Form_delete_therm_by_date", "-"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Form_delete_therm_by_date = QtWidgets.QWidget()
    ui = Ui_Form_delete_therm_by_date()
    ui.setupUi(Form_delete_therm_by_date)
    Form_delete_therm_by_date.show()
    sys.exit(app.exec_())
