# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'C:\Users\USER\PycharmProjects\geovel_1.0\qt\geovel_main_window.ui'
#
# Created by: PyQt5 UI code generator 5.15.6
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1550, 833)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.gridLayout_3 = QtWidgets.QGridLayout()
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.gridLayout_2 = QtWidgets.QGridLayout()
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setObjectName("label")
        self.gridLayout_2.addWidget(self.label, 0, 0, 1, 1)
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.radarogram = ImageView(self.centralwidget)
        self.radarogram.setObjectName("radarogram")
        self.gridLayout.addWidget(self.radarogram, 0, 0, 1, 1)
        self.signal = PlotWidget(self.centralwidget)
        self.signal.setObjectName("signal")
        self.gridLayout.addWidget(self.signal, 0, 1, 1, 1)
        self.graph = PlotWidget(self.centralwidget)
        self.graph.setObjectName("graph")
        self.gridLayout.addWidget(self.graph, 1, 0, 1, 1)
        self.info = QtWidgets.QTextEdit(self.centralwidget)
        self.info.setObjectName("info")
        self.gridLayout.addWidget(self.info, 1, 1, 1, 1)
        self.gridLayout.setColumnStretch(0, 9)
        self.gridLayout.setColumnStretch(1, 2)
        self.gridLayout.setRowStretch(0, 9)
        self.gridLayout.setRowStretch(1, 4)
        self.gridLayout_2.addLayout(self.gridLayout, 1, 0, 1, 1)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.tabWidget_2 = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget_2.setObjectName("tabWidget_2")
        self.tab_3 = QtWidgets.QWidget()
        self.tab_3.setObjectName("tab_3")
        self.gridLayout_5 = QtWidgets.QGridLayout(self.tab_3)
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.label_4 = QtWidgets.QLabel(self.tab_3)
        self.label_4.setObjectName("label_4")
        self.horizontalLayout_3.addWidget(self.label_4)
        self.comboBox_object = QtWidgets.QComboBox(self.tab_3)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.comboBox_object.sizePolicy().hasHeightForWidth())
        self.comboBox_object.setSizePolicy(sizePolicy)
        self.comboBox_object.setObjectName("comboBox_object")
        self.horizontalLayout_3.addWidget(self.comboBox_object)
        self.toolButton_add_obj = QtWidgets.QToolButton(self.tab_3)
        self.toolButton_add_obj.setStyleSheet("background-color: rgb(191, 255, 191);")
        self.toolButton_add_obj.setObjectName("toolButton_add_obj")
        self.horizontalLayout_3.addWidget(self.toolButton_add_obj)
        self.toolButton_del_obj = QtWidgets.QToolButton(self.tab_3)
        self.toolButton_del_obj.setStyleSheet("background-color: rgb(255, 185, 185);")
        self.toolButton_del_obj.setObjectName("toolButton_del_obj")
        self.horizontalLayout_3.addWidget(self.toolButton_del_obj)
        self.horizontalLayout_3.setStretch(0, 2)
        self.horizontalLayout_3.setStretch(1, 9)
        self.horizontalLayout_3.setStretch(2, 1)
        self.horizontalLayout_3.setStretch(3, 1)
        self.verticalLayout_3.addLayout(self.horizontalLayout_3)
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.label_5 = QtWidgets.QLabel(self.tab_3)
        self.label_5.setObjectName("label_5")
        self.horizontalLayout_4.addWidget(self.label_5)
        self.comboBox_profile = QtWidgets.QComboBox(self.tab_3)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.comboBox_profile.sizePolicy().hasHeightForWidth())
        self.comboBox_profile.setSizePolicy(sizePolicy)
        self.comboBox_profile.setObjectName("comboBox_profile")
        self.horizontalLayout_4.addWidget(self.comboBox_profile)
        self.toolButton_load_prof = QtWidgets.QToolButton(self.tab_3)
        self.toolButton_load_prof.setStyleSheet("background-color: rgb(191, 255, 191);")
        self.toolButton_load_prof.setObjectName("toolButton_load_prof")
        self.horizontalLayout_4.addWidget(self.toolButton_load_prof)
        self.toolButton_del_prof = QtWidgets.QToolButton(self.tab_3)
        self.toolButton_del_prof.setStyleSheet("background-color: rgb(255, 185, 185);")
        self.toolButton_del_prof.setObjectName("toolButton_del_prof")
        self.horizontalLayout_4.addWidget(self.toolButton_del_prof)
        self.horizontalLayout_4.setStretch(0, 2)
        self.horizontalLayout_4.setStretch(1, 9)
        self.horizontalLayout_4.setStretch(2, 1)
        self.horizontalLayout_4.setStretch(3, 1)
        self.verticalLayout_3.addLayout(self.horizontalLayout_4)
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.label_6 = QtWidgets.QLabel(self.tab_3)
        self.label_6.setObjectName("label_6")
        self.horizontalLayout_5.addWidget(self.label_6)
        self.comboBox_param_plast = QtWidgets.QComboBox(self.tab_3)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.comboBox_param_plast.sizePolicy().hasHeightForWidth())
        self.comboBox_param_plast.setSizePolicy(sizePolicy)
        self.comboBox_param_plast.setObjectName("comboBox_param_plast")
        self.horizontalLayout_5.addWidget(self.comboBox_param_plast)
        self.toolButton_load_plast = QtWidgets.QToolButton(self.tab_3)
        self.toolButton_load_plast.setStyleSheet("background-color: rgb(191, 255, 191);")
        self.toolButton_load_plast.setObjectName("toolButton_load_plast")
        self.horizontalLayout_5.addWidget(self.toolButton_load_plast)
        self.toolButton_del_plast = QtWidgets.QToolButton(self.tab_3)
        self.toolButton_del_plast.setStyleSheet("background-color: rgb(255, 185, 185);")
        self.toolButton_del_plast.setObjectName("toolButton_del_plast")
        self.horizontalLayout_5.addWidget(self.toolButton_del_plast)
        self.horizontalLayout_5.setStretch(0, 2)
        self.horizontalLayout_5.setStretch(1, 9)
        self.horizontalLayout_5.setStretch(2, 1)
        self.horizontalLayout_5.setStretch(3, 1)
        self.verticalLayout_3.addLayout(self.horizontalLayout_5)
        self.gridLayout_5.addLayout(self.verticalLayout_3, 1, 0, 1, 1)
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout_5.addItem(spacerItem, 3, 0, 1, 1)
        self.tabWidget_2.addTab(self.tab_3, "")
        self.tab_4 = QtWidgets.QWidget()
        self.tab_4.setObjectName("tab_4")
        self.tabWidget_2.addTab(self.tab_4, "")
        self.verticalLayout_2.addWidget(self.tabWidget_2)
        self.gridLayout_2.addLayout(self.verticalLayout_2, 1, 1, 1, 1)
        self.gridLayout_2.setColumnStretch(0, 7)
        self.gridLayout_2.setColumnStretch(1, 2)
        self.gridLayout_3.addLayout(self.gridLayout_2, 0, 0, 1, 1)
        self.gridLayout_3.setColumnStretch(0, 5)
        self.gridLayout_4.addLayout(self.gridLayout_3, 0, 0, 1, 1)
        self.progressBar = QtWidgets.QProgressBar(self.centralwidget)
        self.progressBar.setProperty("value", 24)
        self.progressBar.setAlignment(QtCore.Qt.AlignBottom|QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft)
        self.progressBar.setTextVisible(True)
        self.progressBar.setOrientation(QtCore.Qt.Vertical)
        self.progressBar.setInvertedAppearance(False)
        self.progressBar.setTextDirection(QtWidgets.QProgressBar.TopToBottom)
        self.progressBar.setObjectName("progressBar")
        self.gridLayout_4.addWidget(self.progressBar, 0, 1, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1550, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.tabWidget_2.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label.setText(_translate("MainWindow", "TextLabel"))
        self.label_4.setText(_translate("MainWindow", "????????????:"))
        self.toolButton_add_obj.setText(_translate("MainWindow", "add"))
        self.toolButton_del_obj.setText(_translate("MainWindow", "del"))
        self.label_5.setText(_translate("MainWindow", "??????????????:"))
        self.toolButton_load_prof.setText(_translate("MainWindow", "load"))
        self.toolButton_del_prof.setText(_translate("MainWindow", "del"))
        self.label_6.setText(_translate("MainWindow", "??????????. ????????????:"))
        self.toolButton_load_plast.setText(_translate("MainWindow", "load"))
        self.toolButton_del_plast.setText(_translate("MainWindow", "del"))
        self.tabWidget_2.setTabText(self.tabWidget_2.indexOf(self.tab_3), _translate("MainWindow", "Tab 1"))
        self.tabWidget_2.setTabText(self.tabWidget_2.indexOf(self.tab_4), _translate("MainWindow", "Tab 2"))
from pyqtgraph import ImageView, PlotWidget


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
