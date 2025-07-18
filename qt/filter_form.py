# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'qt/filter_form.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_FilterWellForm(object):
    def setupUi(self, FilterWellForm):
        FilterWellForm.setObjectName("FilterWellForm")
        FilterWellForm.resize(319, 561)
        self.gridLayout_6 = QtWidgets.QGridLayout(FilterWellForm)
        self.gridLayout_6.setObjectName("gridLayout_6")
        self.groupBox_4 = QtWidgets.QGroupBox(FilterWellForm)
        self.groupBox_4.setObjectName("groupBox_4")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.groupBox_4)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.checkBox_alt = QtWidgets.QCheckBox(self.groupBox_4)
        self.checkBox_alt.setText("")
        self.checkBox_alt.setObjectName("checkBox_alt")
        self.gridLayout_4.addWidget(self.checkBox_alt, 0, 0, 1, 1)
        self.gridLayout_2 = QtWidgets.QGridLayout()
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.spinBox_alt_from = QtWidgets.QSpinBox(self.groupBox_4)
        self.spinBox_alt_from.setMinimum(-1000)
        self.spinBox_alt_from.setMaximum(1000)
        self.spinBox_alt_from.setObjectName("spinBox_alt_from")
        self.gridLayout_2.addWidget(self.spinBox_alt_from, 0, 0, 1, 1)
        self.label_2 = QtWidgets.QLabel(self.groupBox_4)
        self.label_2.setObjectName("label_2")
        self.gridLayout_2.addWidget(self.label_2, 0, 1, 1, 1)
        self.spinBox_alt_to = QtWidgets.QSpinBox(self.groupBox_4)
        self.spinBox_alt_to.setMinimum(-1000)
        self.spinBox_alt_to.setMaximum(1000)
        self.spinBox_alt_to.setProperty("value", 250)
        self.spinBox_alt_to.setObjectName("spinBox_alt_to")
        self.gridLayout_2.addWidget(self.spinBox_alt_to, 0, 2, 1, 1)
        self.gridLayout_2.setColumnStretch(0, 1)
        self.gridLayout_2.setColumnStretch(2, 1)
        self.gridLayout_4.addLayout(self.gridLayout_2, 0, 1, 1, 1)
        self.label_4 = QtWidgets.QLabel(self.groupBox_4)
        self.label_4.setObjectName("label_4")
        self.gridLayout_4.addWidget(self.label_4, 0, 2, 1, 1)
        self.gridLayout_4.setColumnStretch(0, 1)
        self.gridLayout_4.setColumnStretch(1, 4)
        self.gridLayout_4.setColumnStretch(2, 1)
        self.gridLayout_6.addWidget(self.groupBox_4, 1, 0, 1, 1)
        self.pushButton_filter = QtWidgets.QPushButton(FilterWellForm)
        self.pushButton_filter.setStyleSheet("background-color: rgb(143, 240, 164);")
        self.pushButton_filter.setObjectName("pushButton_filter")
        self.gridLayout_6.addWidget(self.pushButton_filter, 3, 0, 1, 1)
        self.groupBox = QtWidgets.QGroupBox(FilterWellForm)
        self.groupBox.setObjectName("groupBox")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.groupBox)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.checkBox_distance = QtWidgets.QCheckBox(self.groupBox)
        self.checkBox_distance.setText("")
        self.checkBox_distance.setObjectName("checkBox_distance")
        self.gridLayout_3.addWidget(self.checkBox_distance, 0, 0, 1, 1)
        self.spinBox_distance = QtWidgets.QSpinBox(self.groupBox)
        self.spinBox_distance.setMaximum(10000000)
        self.spinBox_distance.setSingleStep(500)
        self.spinBox_distance.setProperty("value", 50000)
        self.spinBox_distance.setObjectName("spinBox_distance")
        self.gridLayout_3.addWidget(self.spinBox_distance, 0, 1, 1, 1)
        self.label_3 = QtWidgets.QLabel(self.groupBox)
        self.label_3.setObjectName("label_3")
        self.gridLayout_3.addWidget(self.label_3, 0, 2, 1, 1)
        self.gridLayout_3.setColumnStretch(0, 1)
        self.gridLayout_3.setColumnStretch(1, 4)
        self.gridLayout_3.setColumnStretch(2, 1)
        self.gridLayout_6.addWidget(self.groupBox, 0, 0, 1, 1)
        self.groupBox_2 = QtWidgets.QGroupBox(FilterWellForm)
        self.groupBox_2.setObjectName("groupBox_2")
        self.gridLayout_5 = QtWidgets.QGridLayout(self.groupBox_2)
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.checkBox_depth = QtWidgets.QCheckBox(self.groupBox_2)
        self.checkBox_depth.setText("")
        self.checkBox_depth.setObjectName("checkBox_depth")
        self.gridLayout_5.addWidget(self.checkBox_depth, 0, 0, 1, 1)
        self.label_5 = QtWidgets.QLabel(self.groupBox_2)
        self.label_5.setObjectName("label_5")
        self.gridLayout_5.addWidget(self.label_5, 0, 2, 1, 1)
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.label = QtWidgets.QLabel(self.groupBox_2)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 0, 1, 1, 1)
        self.spinBox_depth_from = QtWidgets.QSpinBox(self.groupBox_2)
        self.spinBox_depth_from.setMaximum(500)
        self.spinBox_depth_from.setProperty("value", 50)
        self.spinBox_depth_from.setObjectName("spinBox_depth_from")
        self.gridLayout.addWidget(self.spinBox_depth_from, 0, 0, 1, 1)
        self.spinBox_depth_to = QtWidgets.QSpinBox(self.groupBox_2)
        self.spinBox_depth_to.setMaximum(500)
        self.spinBox_depth_to.setProperty("value", 250)
        self.spinBox_depth_to.setObjectName("spinBox_depth_to")
        self.gridLayout.addWidget(self.spinBox_depth_to, 0, 2, 1, 1)
        self.gridLayout.setColumnStretch(0, 1)
        self.gridLayout.setColumnStretch(2, 1)
        self.gridLayout_5.addLayout(self.gridLayout, 0, 1, 1, 1)
        self.listWidget_title_layer = QtWidgets.QListWidget(self.groupBox_2)
        self.listWidget_title_layer.setObjectName("listWidget_title_layer")
        self.gridLayout_5.addWidget(self.listWidget_title_layer, 2, 0, 1, 3)
        self.label_6 = QtWidgets.QLabel(self.groupBox_2)
        self.label_6.setObjectName("label_6")
        self.gridLayout_5.addWidget(self.label_6, 1, 0, 1, 3)
        self.gridLayout_5.setColumnStretch(0, 1)
        self.gridLayout_5.setColumnStretch(1, 4)
        self.gridLayout_5.setColumnStretch(2, 1)
        self.gridLayout_6.addWidget(self.groupBox_2, 2, 0, 1, 1)

        self.retranslateUi(FilterWellForm)
        QtCore.QMetaObject.connectSlotsByName(FilterWellForm)

    def retranslateUi(self, FilterWellForm):
        _translate = QtCore.QCoreApplication.translate
        FilterWellForm.setWindowTitle(_translate("FilterWellForm", "Filter Well"))
        self.groupBox_4.setTitle(_translate("FilterWellForm", "Альтитуда"))
        self.label_2.setText(_translate("FilterWellForm", "  -  "))
        self.label_4.setText(_translate("FilterWellForm", "метров"))
        self.pushButton_filter.setText(_translate("FilterWellForm", "FILTER"))
        self.groupBox.setTitle(_translate("FilterWellForm", "Расстояние"))
        self.label_3.setText(_translate("FilterWellForm", "метров"))
        self.groupBox_2.setTitle(_translate("FilterWellForm", "Глубина"))
        self.label_5.setText(_translate("FilterWellForm", "метров"))
        self.label.setText(_translate("FilterWellForm", "  -  "))
        self.label_6.setText(_translate("FilterWellForm", "Варианты названия слоя:"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    FilterWellForm = QtWidgets.QWidget()
    ui = Ui_FilterWellForm()
    ui.setupUi(FilterWellForm)
    FilterWellForm.show()
    sys.exit(app.exec_())
