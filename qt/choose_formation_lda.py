# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'qt/choose_formation_lda.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_FormationLDA(object):
    def setupUi(self, FormationLDA):
        FormationLDA.setObjectName("FormationLDA")
        FormationLDA.resize(300, 300)
        self.gridLayout = QtWidgets.QGridLayout(FormationLDA)
        self.gridLayout.setObjectName("gridLayout")
        self.listWidget_form_lda = QtWidgets.QListWidget(FormationLDA)
        self.listWidget_form_lda.setObjectName("listWidget_form_lda")
        self.gridLayout.addWidget(self.listWidget_form_lda, 0, 0, 1, 2)
        self.checkBox_to_all = QtWidgets.QCheckBox(FormationLDA)
        self.checkBox_to_all.setObjectName("checkBox_to_all")
        self.gridLayout.addWidget(self.checkBox_to_all, 1, 0, 1, 1)
        self.pushButton_ok_form_lda = QtWidgets.QPushButton(FormationLDA)
        self.pushButton_ok_form_lda.setStyleSheet("background-color: rgb(191, 255, 191);")
        self.pushButton_ok_form_lda.setObjectName("pushButton_ok_form_lda")
        self.gridLayout.addWidget(self.pushButton_ok_form_lda, 1, 1, 1, 1)
        self.gridLayout.setColumnStretch(1, 1)

        self.retranslateUi(FormationLDA)
        QtCore.QMetaObject.connectSlotsByName(FormationLDA)

    def retranslateUi(self, FormationLDA):
        _translate = QtCore.QCoreApplication.translate
        FormationLDA.setWindowTitle(_translate("FormationLDA", "Formation LDA"))
        self.checkBox_to_all.setText(_translate("FormationLDA", "to all"))
        self.pushButton_ok_form_lda.setText(_translate("FormationLDA", "ОК"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    FormationLDA = QtWidgets.QWidget()
    ui = Ui_FormationLDA()
    ui.setupUi(FormationLDA)
    FormationLDA.show()
    sys.exit(app.exec_())
