# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'qt/form_well_log.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Form_well_log(object):
    def setupUi(self, Form_well_log):
        Form_well_log.setObjectName("Form_well_log")
        Form_well_log.resize(605, 684)
        self.gridLayout_3 = QtWidgets.QGridLayout(Form_well_log)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.label = QtWidgets.QLabel(Form_well_log)
        self.label.setObjectName("label")
        self.gridLayout_3.addWidget(self.label, 0, 0, 1, 2)
        self.widget_graph_well_log = PlotWidget(Form_well_log)
        self.widget_graph_well_log.setObjectName("widget_graph_well_log")
        self.gridLayout_3.addWidget(self.widget_graph_well_log, 1, 0, 1, 1)
        self.gridLayout_2 = QtWidgets.QGridLayout()
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.pushButton_add_dir_well_log = QtWidgets.QPushButton(Form_well_log)
        self.pushButton_add_dir_well_log.setStyleSheet("background-color: rgb(180, 255, 205);")
        self.pushButton_add_dir_well_log.setObjectName("pushButton_add_dir_well_log")
        self.gridLayout.addWidget(self.pushButton_add_dir_well_log, 2, 0, 1, 1)
        self.pushButton_rm_all_well_db = QtWidgets.QPushButton(Form_well_log)
        self.pushButton_rm_all_well_db.setStyleSheet("background-color: rgb(255, 162, 190);")
        self.pushButton_rm_all_well_db.setObjectName("pushButton_rm_all_well_db")
        self.gridLayout.addWidget(self.pushButton_rm_all_well_db, 2, 1, 1, 1)
        self.pushButton_rm_well_log = QtWidgets.QPushButton(Form_well_log)
        self.pushButton_rm_well_log.setStyleSheet("background-color: rgb(255, 185, 185);")
        self.pushButton_rm_well_log.setObjectName("pushButton_rm_well_log")
        self.gridLayout.addWidget(self.pushButton_rm_well_log, 1, 1, 1, 1)
        self.listWidget_well_log = QtWidgets.QListWidget(Form_well_log)
        self.listWidget_well_log.setObjectName("listWidget_well_log")
        self.gridLayout.addWidget(self.listWidget_well_log, 0, 0, 1, 2)
        self.pushButton_add_well_log = QtWidgets.QPushButton(Form_well_log)
        self.pushButton_add_well_log.setStyleSheet("background-color: rgb(191, 255, 191);")
        self.pushButton_add_well_log.setObjectName("pushButton_add_well_log")
        self.gridLayout.addWidget(self.pushButton_add_well_log, 1, 0, 1, 1)
        self.checkBox_raar_signal = QtWidgets.QCheckBox(Form_well_log)
        self.checkBox_raar_signal.setObjectName("checkBox_raar_signal")
        self.gridLayout.addWidget(self.checkBox_raar_signal, 3, 0, 1, 1)
        self.gridLayout_2.addLayout(self.gridLayout, 0, 0, 1, 1)
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout_2.addItem(spacerItem, 1, 0, 1, 1)
        self.gridLayout_3.addLayout(self.gridLayout_2, 1, 1, 1, 1)

        self.retranslateUi(Form_well_log)
        QtCore.QMetaObject.connectSlotsByName(Form_well_log)

    def retranslateUi(self, Form_well_log):
        _translate = QtCore.QCoreApplication.translate
        Form_well_log.setWindowTitle(_translate("Form_well_log", "Well Logging"))
        self.label.setText(_translate("Form_well_log", "TextLabel"))
        self.pushButton_add_dir_well_log.setText(_translate("Form_well_log", "ADD DIR"))
        self.pushButton_rm_all_well_db.setText(_translate("Form_well_log", "REMOVE ALL"))
        self.pushButton_rm_well_log.setText(_translate("Form_well_log", "REMOVE"))
        self.pushButton_add_well_log.setText(_translate("Form_well_log", "ADD"))
        self.checkBox_raar_signal.setText(_translate("Form_well_log", "radar"))
from pyqtgraph import PlotWidget


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Form_well_log = QtWidgets.QWidget()
    ui = Ui_Form_well_log()
    ui.setupUi(Form_well_log)
    Form_well_log.show()
    sys.exit(app.exec_())
