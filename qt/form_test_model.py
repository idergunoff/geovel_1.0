# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'C:\Users\SayfutdinovaAMa\PycharmProjects\geovel_1.0\qt\form_test_model.ui'
#
# Created by: PyQt5 UI code generator 5.15.10
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_FormTestModel(object):
    def setupUi(self, FormTestModel):
        FormTestModel.setObjectName("FormTestModel")
        FormTestModel.resize(494, 635)
        self.gridLayout_5 = QtWidgets.QGridLayout(FormTestModel)
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.gridLayout_4 = QtWidgets.QGridLayout()
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.gridLayout_2 = QtWidgets.QGridLayout()
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.pushButton_test = QtWidgets.QPushButton(FormTestModel)
        self.pushButton_test.setStyleSheet("background-color: rgb(200, 171, 252);")
        self.pushButton_test.setObjectName("pushButton_test")
        self.gridLayout.addWidget(self.pushButton_test, 0, 0, 1, 1)
        self.pushButton_test_all = QtWidgets.QPushButton(FormTestModel)
        self.pushButton_test_all.setStyleSheet("background-color: rgb(160, 125, 252);")
        self.pushButton_test_all.setObjectName("pushButton_test_all")
        self.gridLayout.addWidget(self.pushButton_test_all, 0, 1, 1, 1)
        self.checkBox_save_test = QtWidgets.QCheckBox(FormTestModel)
        self.checkBox_save_test.setObjectName("checkBox_save_test")
        self.gridLayout.addWidget(self.checkBox_save_test, 0, 2, 1, 1)
        self.gridLayout_2.addLayout(self.gridLayout, 0, 2, 1, 1)
        self.comboBox_test_analysis = QtWidgets.QComboBox(FormTestModel)
        self.comboBox_test_analysis.setObjectName("comboBox_test_analysis")
        self.gridLayout_2.addWidget(self.comboBox_test_analysis, 0, 0, 1, 1)
        self.comboBox_mark = QtWidgets.QComboBox(FormTestModel)
        self.comboBox_mark.setObjectName("comboBox_mark")
        self.gridLayout_2.addWidget(self.comboBox_mark, 0, 1, 1, 1)
        self.gridLayout_2.setColumnStretch(0, 1)
        self.gridLayout_4.addLayout(self.gridLayout_2, 0, 0, 1, 1)
        self.gridLayout_3 = QtWidgets.QGridLayout()
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.listWidget_test_model = QtWidgets.QListWidget(FormTestModel)
        self.listWidget_test_model.setObjectName("listWidget_test_model")
        self.gridLayout_3.addWidget(self.listWidget_test_model, 0, 0, 1, 1)
        self.listWidget_test_point = QtWidgets.QListWidget(FormTestModel)
        self.listWidget_test_point.setObjectName("listWidget_test_point")
        self.gridLayout_3.addWidget(self.listWidget_test_point, 0, 1, 1, 1)
        self.gridLayout_3.setColumnStretch(0, 1)
        self.gridLayout_3.setColumnStretch(1, 2)
        self.gridLayout_4.addLayout(self.gridLayout_3, 1, 0, 1, 1)
        self.textEdit_test_result = QtWidgets.QTextEdit(FormTestModel)
        self.textEdit_test_result.setObjectName("textEdit_test_result")
        self.gridLayout_4.addWidget(self.textEdit_test_result, 2, 0, 1, 1)
        self.gridLayout_4.setRowStretch(1, 1)
        self.gridLayout_4.setRowStretch(2, 3)
        self.gridLayout_5.addLayout(self.gridLayout_4, 0, 0, 1, 1)

        self.retranslateUi(FormTestModel)
        QtCore.QMetaObject.connectSlotsByName(FormTestModel)

    def retranslateUi(self, FormTestModel):
        _translate = QtCore.QCoreApplication.translate
        FormTestModel.setWindowTitle(_translate("FormTestModel", "Form"))
        self.pushButton_test.setText(_translate("FormTestModel", "TEST"))
        self.pushButton_test_all.setText(_translate("FormTestModel", "TEST ALL"))
        self.checkBox_save_test.setText(_translate("FormTestModel", "save"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    FormTestModel = QtWidgets.QWidget()
    ui = Ui_FormTestModel()
    ui.setupUi(FormTestModel)
    FormTestModel.show()
    sys.exit(app.exec_())
