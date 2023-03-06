# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'C:\Users\USER\PycharmProjects\geovel_1.0\qt\new_window_dialog.ui'
#
# Created by: PyQt5 UI code generator 5.15.6
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_new_window(object):
    def setupUi(self, new_window):
        new_window.setObjectName("new_window")
        new_window.resize(1290, 286)
        self.gridLayout = QtWidgets.QGridLayout(new_window)
        self.gridLayout.setObjectName("gridLayout")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.radarogram = GraphicsLayoutWidget(new_window)
        self.radarogram.setObjectName("radarogram")
        self.horizontalLayout_2.addWidget(self.radarogram)
        self.signal = PlotWidget(new_window)
        self.signal.setObjectName("signal")
        self.horizontalLayout_2.addWidget(self.signal)
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.info = QtWidgets.QTextEdit(new_window)
        self.info.setObjectName("info")
        self.verticalLayout.addWidget(self.info)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label_id_nw_rad = QtWidgets.QLabel(new_window)
        self.label_id_nw_rad.setObjectName("label_id_nw_rad")
        self.horizontalLayout.addWidget(self.label_id_nw_rad)
        self.pushButton_rollback = QtWidgets.QPushButton(new_window)
        self.pushButton_rollback.setObjectName("pushButton_rollback")
        self.horizontalLayout.addWidget(self.pushButton_rollback)
        self.horizontalLayout.setStretch(0, 1)
        self.horizontalLayout.setStretch(1, 10)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.horizontalLayout_2.addLayout(self.verticalLayout)
        self.horizontalLayout_2.setStretch(0, 18)
        self.horizontalLayout_2.setStretch(1, 4)
        self.horizontalLayout_2.setStretch(2, 3)
        self.gridLayout.addLayout(self.horizontalLayout_2, 0, 0, 1, 1)

        self.retranslateUi(new_window)
        QtCore.QMetaObject.connectSlotsByName(new_window)

    def retranslateUi(self, new_window):
        _translate = QtCore.QCoreApplication.translate
        new_window.setWindowTitle(_translate("new_window", "Новое окно"))
        self.label_id_nw_rad.setText(_translate("new_window", "id"))
        self.pushButton_rollback.setText(_translate("new_window", "Rollback"))
from pyqtgraph import GraphicsLayoutWidget, PlotWidget


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    new_window = QtWidgets.QWidget()
    ui = Ui_new_window()
    ui.setupUi(new_window)
    new_window.show()
    sys.exit(app.exec_())
