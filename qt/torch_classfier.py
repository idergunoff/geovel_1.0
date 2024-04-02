# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'C:\Users\SayfutdinovaAMa\PycharmProjects\geovel_1.0\qt\torch_classfier.ui'
#
# Created by: PyQt5 UI code generator 5.15.10
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_TorchClassifierForm(object):
    def setupUi(self, TorchClassifierForm):
        TorchClassifierForm.setObjectName("TorchClassifierForm")
        TorchClassifierForm.resize(647, 390)
        TorchClassifierForm.setStyleSheet("")
        self.groupBox = QtWidgets.QGroupBox(TorchClassifierForm)
        self.groupBox.setGeometry(QtCore.QRect(20, 20, 281, 221))
        self.groupBox.setTitle("")
        self.groupBox.setObjectName("groupBox")
        self.label = QtWidgets.QLabel(self.groupBox)
        self.label.setGeometry(QtCore.QRect(50, 40, 71, 16))
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.groupBox)
        self.label_2.setGeometry(QtCore.QRect(70, 70, 47, 13))
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.groupBox)
        self.label_3.setGeometry(QtCore.QRect(50, 100, 71, 16))
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(self.groupBox)
        self.label_4.setGeometry(QtCore.QRect(50, 130, 71, 16))
        self.label_4.setObjectName("label_4")
        self.label_5 = QtWidgets.QLabel(self.groupBox)
        self.label_5.setGeometry(QtCore.QRect(50, 160, 71, 16))
        self.label_5.setObjectName("label_5")
        self.lineEdit_choose_layers = QtWidgets.QLineEdit(self.groupBox)
        self.lineEdit_choose_layers.setGeometry(QtCore.QRect(120, 160, 113, 20))
        self.lineEdit_choose_layers.setObjectName("lineEdit_choose_layers")
        self.checkBox_choose_param = QtWidgets.QCheckBox(self.groupBox)
        self.checkBox_choose_param.setGeometry(QtCore.QRect(10, 10, 121, 17))
        self.checkBox_choose_param.setStyleSheet("background-color: rgb(252, 204, 165);")
        self.checkBox_choose_param.setObjectName("checkBox_choose_param")
        self.doubleSpinBox_choose_lr = QtWidgets.QDoubleSpinBox(self.groupBox)
        self.doubleSpinBox_choose_lr.setGeometry(QtCore.QRect(120, 40, 111, 22))
        self.doubleSpinBox_choose_lr.setDecimals(6)
        self.doubleSpinBox_choose_lr.setSingleStep(0.01)
        self.doubleSpinBox_choose_lr.setProperty("value", 0.01)
        self.doubleSpinBox_choose_lr.setObjectName("doubleSpinBox_choose_lr")
        self.doubleSpinBox_choose_dropout = QtWidgets.QDoubleSpinBox(self.groupBox)
        self.doubleSpinBox_choose_dropout.setGeometry(QtCore.QRect(120, 70, 111, 22))
        self.doubleSpinBox_choose_dropout.setDecimals(6)
        self.doubleSpinBox_choose_dropout.setSingleStep(0.01)
        self.doubleSpinBox_choose_dropout.setProperty("value", 0.01)
        self.doubleSpinBox_choose_dropout.setObjectName("doubleSpinBox_choose_dropout")
        self.doubleSpinBox_choose_decay = QtWidgets.QDoubleSpinBox(self.groupBox)
        self.doubleSpinBox_choose_decay.setGeometry(QtCore.QRect(120, 100, 111, 22))
        self.doubleSpinBox_choose_decay.setDecimals(6)
        self.doubleSpinBox_choose_decay.setSingleStep(0.01)
        self.doubleSpinBox_choose_decay.setProperty("value", 0.01)
        self.doubleSpinBox_choose_decay.setObjectName("doubleSpinBox_choose_decay")
        self.doubleSpinBox_choose_reagular = QtWidgets.QDoubleSpinBox(self.groupBox)
        self.doubleSpinBox_choose_reagular.setGeometry(QtCore.QRect(120, 130, 111, 22))
        self.doubleSpinBox_choose_reagular.setDecimals(6)
        self.doubleSpinBox_choose_reagular.setSingleStep(0.01)
        self.doubleSpinBox_choose_reagular.setProperty("value", 0.01)
        self.doubleSpinBox_choose_reagular.setObjectName("doubleSpinBox_choose_reagular")
        self.groupBox_2 = QtWidgets.QGroupBox(TorchClassifierForm)
        self.groupBox_2.setGeometry(QtCore.QRect(340, 20, 281, 251))
        self.groupBox_2.setTitle("")
        self.groupBox_2.setObjectName("groupBox_2")
        self.label_6 = QtWidgets.QLabel(self.groupBox_2)
        self.label_6.setGeometry(QtCore.QRect(50, 100, 71, 16))
        self.label_6.setObjectName("label_6")
        self.lineEdit_tune_layers = QtWidgets.QLineEdit(self.groupBox_2)
        self.lineEdit_tune_layers.setGeometry(QtCore.QRect(120, 160, 113, 20))
        self.lineEdit_tune_layers.setObjectName("lineEdit_tune_layers")
        self.label_7 = QtWidgets.QLabel(self.groupBox_2)
        self.label_7.setGeometry(QtCore.QRect(50, 40, 71, 16))
        self.label_7.setObjectName("label_7")
        self.label_8 = QtWidgets.QLabel(self.groupBox_2)
        self.label_8.setGeometry(QtCore.QRect(70, 70, 47, 13))
        self.label_8.setObjectName("label_8")
        self.label_9 = QtWidgets.QLabel(self.groupBox_2)
        self.label_9.setGeometry(QtCore.QRect(50, 130, 71, 16))
        self.label_9.setObjectName("label_9")
        self.lineEdit_tune_dropout = QtWidgets.QLineEdit(self.groupBox_2)
        self.lineEdit_tune_dropout.setGeometry(QtCore.QRect(120, 70, 113, 20))
        self.lineEdit_tune_dropout.setObjectName("lineEdit_tune_dropout")
        self.lineEdit_tune_lr = QtWidgets.QLineEdit(self.groupBox_2)
        self.lineEdit_tune_lr.setGeometry(QtCore.QRect(120, 40, 113, 20))
        self.lineEdit_tune_lr.setObjectName("lineEdit_tune_lr")
        self.label_10 = QtWidgets.QLabel(self.groupBox_2)
        self.label_10.setGeometry(QtCore.QRect(50, 160, 71, 16))
        self.label_10.setObjectName("label_10")
        self.lineEdit_tune_decay = QtWidgets.QLineEdit(self.groupBox_2)
        self.lineEdit_tune_decay.setGeometry(QtCore.QRect(120, 100, 113, 20))
        self.lineEdit_tune_decay.setObjectName("lineEdit_tune_decay")
        self.checkBox_tune_param = QtWidgets.QCheckBox(self.groupBox_2)
        self.checkBox_tune_param.setGeometry(QtCore.QRect(20, 10, 111, 17))
        self.checkBox_tune_param.setStyleSheet("background-color: rgb(251, 228, 170);")
        self.checkBox_tune_param.setObjectName("checkBox_tune_param")
        self.doubleSpinBox_tune_regular = QtWidgets.QDoubleSpinBox(self.groupBox_2)
        self.doubleSpinBox_tune_regular.setGeometry(QtCore.QRect(120, 130, 111, 22))
        self.doubleSpinBox_tune_regular.setDecimals(6)
        self.doubleSpinBox_tune_regular.setSingleStep(0.01)
        self.doubleSpinBox_tune_regular.setProperty("value", 0.01)
        self.doubleSpinBox_tune_regular.setObjectName("doubleSpinBox_tune_regular")
        self.label_15 = QtWidgets.QLabel(self.groupBox_2)
        self.label_15.setGeometry(QtCore.QRect(30, 190, 91, 16))
        self.label_15.setObjectName("label_15")
        self.spinBox_layers_num = QtWidgets.QSpinBox(self.groupBox_2)
        self.spinBox_layers_num.setGeometry(QtCore.QRect(120, 190, 111, 22))
        self.spinBox_layers_num.setObjectName("spinBox_layers_num")
        self.label_16 = QtWidgets.QLabel(self.groupBox_2)
        self.label_16.setGeometry(QtCore.QRect(70, 220, 51, 16))
        self.label_16.setObjectName("label_16")
        self.spinBox_trials = QtWidgets.QSpinBox(self.groupBox_2)
        self.spinBox_trials.setGeometry(QtCore.QRect(120, 220, 111, 22))
        self.spinBox_trials.setObjectName("spinBox_trials")
        self.checkBox_threshold = QtWidgets.QCheckBox(TorchClassifierForm)
        self.checkBox_threshold.setGeometry(QtCore.QRect(190, 300, 70, 17))
        self.checkBox_threshold.setObjectName("checkBox_threshold")
        self.spinBox_epochs = QtWidgets.QSpinBox(TorchClassifierForm)
        self.spinBox_epochs.setGeometry(QtCore.QRect(90, 250, 81, 22))
        self.spinBox_epochs.setMaximum(10000)
        self.spinBox_epochs.setObjectName("spinBox_epochs")
        self.checkBox_early_stop = QtWidgets.QCheckBox(TorchClassifierForm)
        self.checkBox_early_stop.setGeometry(QtCore.QRect(190, 280, 91, 17))
        self.checkBox_early_stop.setObjectName("checkBox_early_stop")
        self.label_12 = QtWidgets.QLabel(TorchClassifierForm)
        self.label_12.setGeometry(QtCore.QRect(40, 280, 71, 16))
        self.label_12.setObjectName("label_12")
        self.label_13 = QtWidgets.QLabel(TorchClassifierForm)
        self.label_13.setGeometry(QtCore.QRect(20, 310, 71, 16))
        self.label_13.setObjectName("label_13")
        self.comboBox_loss = QtWidgets.QComboBox(TorchClassifierForm)
        self.comboBox_loss.setGeometry(QtCore.QRect(90, 310, 81, 22))
        self.comboBox_loss.setObjectName("comboBox_loss")
        self.comboBox_loss.addItem("")
        self.comboBox_loss.addItem("")
        self.comboBox_loss.addItem("")
        self.label_11 = QtWidgets.QLabel(TorchClassifierForm)
        self.label_11.setGeometry(QtCore.QRect(40, 250, 71, 16))
        self.label_11.setObjectName("label_11")
        self.comboBox_optimizer = QtWidgets.QComboBox(TorchClassifierForm)
        self.comboBox_optimizer.setGeometry(QtCore.QRect(90, 280, 81, 22))
        self.comboBox_optimizer.setObjectName("comboBox_optimizer")
        self.comboBox_optimizer.addItem("")
        self.comboBox_optimizer.addItem("")
        self.comboBox_optimizer.addItem("")
        self.label_14 = QtWidgets.QLabel(TorchClassifierForm)
        self.label_14.setGeometry(QtCore.QRect(190, 250, 71, 16))
        self.label_14.setObjectName("label_14")
        self.comboBox_dataset = QtWidgets.QComboBox(TorchClassifierForm)
        self.comboBox_dataset.setGeometry(QtCore.QRect(240, 250, 81, 22))
        self.comboBox_dataset.setObjectName("comboBox_dataset")
        self.comboBox_dataset.addItem("")
        self.comboBox_dataset.addItem("")
        self.pushButton_train = QtWidgets.QPushButton(TorchClassifierForm)
        self.pushButton_train.setGeometry(QtCore.QRect(460, 350, 111, 23))
        self.pushButton_train.setStyleSheet("background-color: rgb(252, 204, 165);")
        self.pushButton_train.setObjectName("pushButton_train")
        self.groupBox_9 = QtWidgets.QGroupBox(TorchClassifierForm)
        self.groupBox_9.setGeometry(QtCore.QRect(460, 270, 161, 61))
        self.groupBox_9.setObjectName("groupBox_9")
        self.gridLayout_9 = QtWidgets.QGridLayout(self.groupBox_9)
        self.gridLayout_9.setObjectName("gridLayout_9")
        self.spinBox_cross_val = QtWidgets.QSpinBox(self.groupBox_9)
        self.spinBox_cross_val.setMinimum(2)
        self.spinBox_cross_val.setProperty("value", 5)
        self.spinBox_cross_val.setObjectName("spinBox_cross_val")
        self.gridLayout_9.addWidget(self.spinBox_cross_val, 0, 1, 1, 1)
        self.pushButton_cross_val = QtWidgets.QPushButton(self.groupBox_9)
        self.pushButton_cross_val.setObjectName("pushButton_cross_val")
        self.gridLayout_9.addWidget(self.pushButton_cross_val, 0, 0, 1, 1)
        self.label_17 = QtWidgets.QLabel(TorchClassifierForm)
        self.label_17.setGeometry(QtCore.QRect(10, 340, 81, 16))
        self.label_17.setObjectName("label_17")
        self.comboBox_activation_func = QtWidgets.QComboBox(TorchClassifierForm)
        self.comboBox_activation_func.setGeometry(QtCore.QRect(90, 340, 81, 22))
        self.comboBox_activation_func.setObjectName("comboBox_activation_func")
        self.comboBox_activation_func.addItem("")
        self.comboBox_activation_func.addItem("")
        self.comboBox_activation_func.addItem("")
        self.checkBox_save_model = QtWidgets.QCheckBox(TorchClassifierForm)
        self.checkBox_save_model.setEnabled(True)
        self.checkBox_save_model.setGeometry(QtCore.QRect(580, 350, 70, 17))
        self.checkBox_save_model.setObjectName("checkBox_save_model")

        self.retranslateUi(TorchClassifierForm)
        QtCore.QMetaObject.connectSlotsByName(TorchClassifierForm)

    def retranslateUi(self, TorchClassifierForm):
        _translate = QtCore.QCoreApplication.translate
        TorchClassifierForm.setWindowTitle(_translate("TorchClassifierForm", "TorchNNClassifier"))
        self.label.setText(_translate("TorchClassifierForm", "learning rate"))
        self.label_2.setText(_translate("TorchClassifierForm", "dropout"))
        self.label_3.setText(_translate("TorchClassifierForm", "weight decay"))
        self.label_4.setText(_translate("TorchClassifierForm", "regularization"))
        self.label_5.setText(_translate("TorchClassifierForm", "hidden layers"))
        self.checkBox_choose_param.setText(_translate("TorchClassifierForm", "Choose parameters"))
        self.label_6.setText(_translate("TorchClassifierForm", "weight decay"))
        self.label_7.setText(_translate("TorchClassifierForm", "learning rate"))
        self.label_8.setText(_translate("TorchClassifierForm", "dropout"))
        self.label_9.setText(_translate("TorchClassifierForm", "regularization"))
        self.label_10.setText(_translate("TorchClassifierForm", "hidden layers"))
        self.checkBox_tune_param.setText(_translate("TorchClassifierForm", "Tune parameters"))
        self.label_15.setText(_translate("TorchClassifierForm", "hidden layers num"))
        self.label_16.setText(_translate("TorchClassifierForm", "num trials"))
        self.checkBox_threshold.setText(_translate("TorchClassifierForm", "Threshold"))
        self.checkBox_early_stop.setText(_translate("TorchClassifierForm", "Early stopping"))
        self.label_12.setText(_translate("TorchClassifierForm", "optimizer"))
        self.label_13.setText(_translate("TorchClassifierForm", "loss function"))
        self.comboBox_loss.setItemText(0, _translate("TorchClassifierForm", "BCELoss"))
        self.comboBox_loss.setItemText(1, _translate("TorchClassifierForm", "BCEWithLogitsLoss"))
        self.comboBox_loss.setItemText(2, _translate("TorchClassifierForm", "CrossEntropy"))
        self.label_11.setText(_translate("TorchClassifierForm", "epochs"))
        self.comboBox_optimizer.setItemText(0, _translate("TorchClassifierForm", "Adam"))
        self.comboBox_optimizer.setItemText(1, _translate("TorchClassifierForm", "SGD"))
        self.comboBox_optimizer.setItemText(2, _translate("TorchClassifierForm", "LBFGS"))
        self.label_14.setText(_translate("TorchClassifierForm", "dataset:"))
        self.comboBox_dataset.setItemText(0, _translate("TorchClassifierForm", "shuffle"))
        self.comboBox_dataset.setItemText(1, _translate("TorchClassifierForm", "well split"))
        self.pushButton_train.setText(_translate("TorchClassifierForm", "TRAIN"))
        self.groupBox_9.setToolTip(_translate("TorchClassifierForm", "Cross Validate by Well"))
        self.groupBox_9.setTitle(_translate("TorchClassifierForm", "Cross Validate"))
        self.pushButton_cross_val.setText(_translate("TorchClassifierForm", "CV"))
        self.label_17.setText(_translate("TorchClassifierForm", "activation func"))
        self.comboBox_activation_func.setItemText(0, _translate("TorchClassifierForm", "ReLU"))
        self.comboBox_activation_func.setItemText(1, _translate("TorchClassifierForm", "Sigmoid"))
        self.comboBox_activation_func.setItemText(2, _translate("TorchClassifierForm", "Tanh"))
        self.checkBox_save_model.setText(_translate("TorchClassifierForm", "save"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    TorchClassifierForm = QtWidgets.QWidget()
    ui = Ui_TorchClassifierForm()
    ui.setupUi(TorchClassifierForm)
    TorchClassifierForm.show()
    sys.exit(app.exec_())
