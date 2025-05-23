# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '.\qt\rem_db_window.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_rem_db(object):
    def setupUi(self, rem_db):
        rem_db.setObjectName("rem_db")
        rem_db.resize(537, 498)
        self.gridLayout_2 = QtWidgets.QGridLayout(rem_db)
        self.gridLayout_2.setObjectName("gridLayout_2")
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_2.addItem(spacerItem, 1, 0, 1, 1)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_2.addItem(spacerItem1, 1, 2, 1, 1)
        spacerItem2 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout_2.addItem(spacerItem2, 0, 1, 1, 1)
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.pushButton_load_formations = QtWidgets.QPushButton(rem_db)
        self.pushButton_load_formations.setStyleSheet("background-color: rgb(206, 222, 255);")
        self.pushButton_load_formations.setObjectName("pushButton_load_formations")
        self.gridLayout.addWidget(self.pushButton_load_formations, 5, 4, 1, 1)
        self.pushButton_sync_objects = QtWidgets.QPushButton(rem_db)
        self.pushButton_sync_objects.setStyleSheet("background-color: rgb(201, 196, 255);")
        self.pushButton_sync_objects.setObjectName("pushButton_sync_objects")
        self.gridLayout.addWidget(self.pushButton_sync_objects, 1, 5, 1, 1)
        self.comboBox_profile_rem = QtWidgets.QComboBox(rem_db)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.comboBox_profile_rem.sizePolicy().hasHeightForWidth())
        self.comboBox_profile_rem.setSizePolicy(sizePolicy)
        self.comboBox_profile_rem.setObjectName("comboBox_profile_rem")
        self.gridLayout.addWidget(self.comboBox_profile_rem, 1, 1, 1, 3)
        self.pushButton_unload_formations = QtWidgets.QPushButton(rem_db)
        self.pushButton_unload_formations.setStyleSheet("background-color: rgb(255, 233, 196);")
        self.pushButton_unload_formations.setObjectName("pushButton_unload_formations")
        self.gridLayout.addWidget(self.pushButton_unload_formations, 5, 5, 1, 1)
        self.pushButton_unload_obj_rem = QtWidgets.QPushButton(rem_db)
        self.pushButton_unload_obj_rem.setStyleSheet("background-color: rgb(255, 233, 196);")
        self.pushButton_unload_obj_rem.setObjectName("pushButton_unload_obj_rem")
        self.gridLayout.addWidget(self.pushButton_unload_obj_rem, 0, 5, 1, 1)
        self.pushButton_delete_obj_rem = QtWidgets.QPushButton(rem_db)
        self.pushButton_delete_obj_rem.setStyleSheet("background-color: rgb(255, 181, 182);")
        self.pushButton_delete_obj_rem.setObjectName("pushButton_delete_obj_rem")
        self.gridLayout.addWidget(self.pushButton_delete_obj_rem, 1, 4, 1, 1)
        self.label_wells_count = QtWidgets.QLabel(rem_db)
        self.label_wells_count.setObjectName("label_wells_count")
        self.gridLayout.addWidget(self.label_wells_count, 6, 0, 1, 3)
        self.pushButton_sync_wells = QtWidgets.QPushButton(rem_db)
        self.pushButton_sync_wells.setStyleSheet("background-color: rgb(255, 205, 169);")
        self.pushButton_sync_wells.setObjectName("pushButton_sync_wells")
        self.gridLayout.addWidget(self.pushButton_sync_wells, 6, 4, 1, 2)
        self.toolButton_cw = QtWidgets.QToolButton(rem_db)
        self.toolButton_cw.setObjectName("toolButton_cw")
        self.gridLayout.addWidget(self.toolButton_cw, 6, 3, 1, 1)
        self.pushButton_load_obj_rem = QtWidgets.QPushButton(rem_db)
        self.pushButton_load_obj_rem.setStyleSheet("background-color: rgb(206, 222, 255);")
        self.pushButton_load_obj_rem.setObjectName("pushButton_load_obj_rem")
        self.gridLayout.addWidget(self.pushButton_load_obj_rem, 0, 4, 1, 1)
        self.label_5 = QtWidgets.QLabel(rem_db)
        self.label_5.setStyleSheet("background-color: rgb(191, 255, 191);")
        self.label_5.setObjectName("label_5")
        self.gridLayout.addWidget(self.label_5, 1, 0, 1, 1)
        self.label_4 = QtWidgets.QLabel(rem_db)
        self.label_4.setStyleSheet("background-color: rgb(191, 255, 191);")
        self.label_4.setObjectName("label_4")
        self.gridLayout.addWidget(self.label_4, 0, 0, 1, 1)
        self.pushButton_load_well_rel = QtWidgets.QPushButton(rem_db)
        self.pushButton_load_well_rel.setStyleSheet("background-color: rgb(206, 222, 255);")
        self.pushButton_load_well_rel.setObjectName("pushButton_load_well_rel")
        self.gridLayout.addWidget(self.pushButton_load_well_rel, 7, 4, 1, 1)
        self.comboBox_object_rem = QtWidgets.QComboBox(rem_db)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.comboBox_object_rem.sizePolicy().hasHeightForWidth())
        self.comboBox_object_rem.setSizePolicy(sizePolicy)
        self.comboBox_object_rem.setToolTip("")
        self.comboBox_object_rem.setObjectName("comboBox_object_rem")
        self.gridLayout.addWidget(self.comboBox_object_rem, 0, 1, 1, 1)
        self.label_2 = QtWidgets.QLabel(rem_db)
        self.label_2.setStyleSheet("")
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 5, 0, 1, 4)
        self.label = QtWidgets.QLabel(rem_db)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 7, 0, 1, 4)
        self.comboBox_research_rem = QtWidgets.QComboBox(rem_db)
        self.comboBox_research_rem.setObjectName("comboBox_research_rem")
        self.gridLayout.addWidget(self.comboBox_research_rem, 0, 2, 1, 2)
        self.pushButton_unload_well_rel = QtWidgets.QPushButton(rem_db)
        self.pushButton_unload_well_rel.setStyleSheet("background-color: rgb(255, 233, 196);")
        self.pushButton_unload_well_rel.setObjectName("pushButton_unload_well_rel")
        self.gridLayout.addWidget(self.pushButton_unload_well_rel, 7, 5, 1, 1)
        self.groupBox = QtWidgets.QGroupBox(rem_db)
        self.groupBox.setObjectName("groupBox")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.groupBox)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.label_3 = QtWidgets.QLabel(self.groupBox)
        self.label_3.setStyleSheet("background-color: rgb(255, 206, 228);")
        self.label_3.setObjectName("label_3")
        self.gridLayout_3.addWidget(self.label_3, 0, 0, 1, 1)
        self.pushButton_load_model = QtWidgets.QPushButton(self.groupBox)
        self.pushButton_load_model.setStyleSheet("background-color: rgb(206, 222, 255);")
        self.pushButton_load_model.setObjectName("pushButton_load_model")
        self.gridLayout_3.addWidget(self.pushButton_load_model, 2, 3, 1, 1)
        self.comboBox_mlp_rdb = QtWidgets.QComboBox(self.groupBox)
        self.comboBox_mlp_rdb.setObjectName("comboBox_mlp_rdb")
        self.gridLayout_3.addWidget(self.comboBox_mlp_rdb, 0, 1, 1, 1)
        self.label_7 = QtWidgets.QLabel(self.groupBox)
        self.label_7.setStyleSheet("background-color: rgb(185, 228, 255);")
        self.label_7.setObjectName("label_7")
        self.gridLayout_3.addWidget(self.label_7, 2, 0, 1, 1)
        self.pushButton_load_mlp = QtWidgets.QPushButton(self.groupBox)
        self.pushButton_load_mlp.setStyleSheet("background-color: rgb(206, 222, 255);")
        self.pushButton_load_mlp.setObjectName("pushButton_load_mlp")
        self.gridLayout_3.addWidget(self.pushButton_load_mlp, 0, 3, 1, 1)
        self.checkBox_check_ga_params = QtWidgets.QCheckBox(self.groupBox)
        self.checkBox_check_ga_params.setObjectName("checkBox_check_ga_params")
        self.gridLayout_3.addWidget(self.checkBox_check_ga_params, 1, 1, 1, 2)
        self.pushButton_delete_model_rdb = QtWidgets.QPushButton(self.groupBox)
        self.pushButton_delete_model_rdb.setStyleSheet("background-color: rgb(255, 181, 182);")
        self.pushButton_delete_model_rdb.setObjectName("pushButton_delete_model_rdb")
        self.gridLayout_3.addWidget(self.pushButton_delete_model_rdb, 2, 2, 1, 1)
        self.pushButton_sync_ga_cls = QtWidgets.QPushButton(self.groupBox)
        self.pushButton_sync_ga_cls.setStyleSheet("background-color: rgb(255, 221, 215);")
        self.pushButton_sync_ga_cls.setObjectName("pushButton_sync_ga_cls")
        self.gridLayout_3.addWidget(self.pushButton_sync_ga_cls, 1, 3, 1, 2)
        self.pushButton_unload_model = QtWidgets.QPushButton(self.groupBox)
        self.pushButton_unload_model.setStyleSheet("background-color: rgb(255, 233, 196);")
        self.pushButton_unload_model.setObjectName("pushButton_unload_model")
        self.gridLayout_3.addWidget(self.pushButton_unload_model, 2, 4, 1, 1)
        self.pushButton_delete_mlp_rdb = QtWidgets.QPushButton(self.groupBox)
        self.pushButton_delete_mlp_rdb.setStyleSheet("background-color: rgb(255, 181, 182);")
        self.pushButton_delete_mlp_rdb.setObjectName("pushButton_delete_mlp_rdb")
        self.gridLayout_3.addWidget(self.pushButton_delete_mlp_rdb, 0, 2, 1, 1)
        self.pushButton_unload_mlp = QtWidgets.QPushButton(self.groupBox)
        self.pushButton_unload_mlp.setStyleSheet("background-color: rgb(255, 233, 196);")
        self.pushButton_unload_mlp.setObjectName("pushButton_unload_mlp")
        self.gridLayout_3.addWidget(self.pushButton_unload_mlp, 0, 4, 1, 1)
        self.label_6 = QtWidgets.QLabel(self.groupBox)
        self.label_6.setStyleSheet("background-color: rgb(124, 192, 255);")
        self.label_6.setObjectName("label_6")
        self.gridLayout_3.addWidget(self.label_6, 1, 0, 1, 1)
        self.comboBox_trained_model_rdb = QtWidgets.QComboBox(self.groupBox)
        self.comboBox_trained_model_rdb.setObjectName("comboBox_trained_model_rdb")
        self.gridLayout_3.addWidget(self.comboBox_trained_model_rdb, 2, 1, 1, 1)
        self.gridLayout.addWidget(self.groupBox, 8, 0, 1, 6)
        self.groupBox_2 = QtWidgets.QGroupBox(rem_db)
        self.groupBox_2.setObjectName("groupBox_2")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.groupBox_2)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.label_10 = QtWidgets.QLabel(self.groupBox_2)
        self.label_10.setStyleSheet("background-color: rgb(255, 206, 228);")
        self.label_10.setObjectName("label_10")
        self.gridLayout_4.addWidget(self.label_10, 0, 0, 1, 1)
        self.comboBox_regmod_rdb = QtWidgets.QComboBox(self.groupBox_2)
        self.comboBox_regmod_rdb.setObjectName("comboBox_regmod_rdb")
        self.gridLayout_4.addWidget(self.comboBox_regmod_rdb, 0, 1, 1, 1)
        self.pushButton_delete_regmod_rdb = QtWidgets.QPushButton(self.groupBox_2)
        self.pushButton_delete_regmod_rdb.setStyleSheet("background-color: rgb(255, 181, 182);")
        self.pushButton_delete_regmod_rdb.setObjectName("pushButton_delete_regmod_rdb")
        self.gridLayout_4.addWidget(self.pushButton_delete_regmod_rdb, 0, 2, 1, 1)
        self.pushButton_load_regmod = QtWidgets.QPushButton(self.groupBox_2)
        self.pushButton_load_regmod.setStyleSheet("background-color: rgb(206, 222, 255);")
        self.pushButton_load_regmod.setObjectName("pushButton_load_regmod")
        self.gridLayout_4.addWidget(self.pushButton_load_regmod, 0, 3, 1, 1)
        self.pushButton_unload_regmod = QtWidgets.QPushButton(self.groupBox_2)
        self.pushButton_unload_regmod.setStyleSheet("background-color: rgb(255, 233, 196);")
        self.pushButton_unload_regmod.setObjectName("pushButton_unload_regmod")
        self.gridLayout_4.addWidget(self.pushButton_unload_regmod, 0, 4, 1, 1)
        self.label_8 = QtWidgets.QLabel(self.groupBox_2)
        self.label_8.setStyleSheet("background-color: rgb(124, 192, 255);")
        self.label_8.setObjectName("label_8")
        self.gridLayout_4.addWidget(self.label_8, 1, 0, 1, 1)
        self.checkBox_check_ga_params_reg = QtWidgets.QCheckBox(self.groupBox_2)
        self.checkBox_check_ga_params_reg.setObjectName("checkBox_check_ga_params_reg")
        self.gridLayout_4.addWidget(self.checkBox_check_ga_params_reg, 1, 1, 1, 2)
        self.pushButton_sync_ga_reg = QtWidgets.QPushButton(self.groupBox_2)
        self.pushButton_sync_ga_reg.setStyleSheet("background-color: rgb(255, 221, 215);")
        self.pushButton_sync_ga_reg.setObjectName("pushButton_sync_ga_reg")
        self.gridLayout_4.addWidget(self.pushButton_sync_ga_reg, 1, 3, 1, 2)
        self.label_9 = QtWidgets.QLabel(self.groupBox_2)
        self.label_9.setStyleSheet("background-color: rgb(185, 228, 255);")
        self.label_9.setObjectName("label_9")
        self.gridLayout_4.addWidget(self.label_9, 2, 0, 1, 1)
        self.comboBox_trained_model_reg_rdb = QtWidgets.QComboBox(self.groupBox_2)
        self.comboBox_trained_model_reg_rdb.setObjectName("comboBox_trained_model_reg_rdb")
        self.gridLayout_4.addWidget(self.comboBox_trained_model_reg_rdb, 2, 1, 1, 1)
        self.pushButton_delete_model_reg_rdb = QtWidgets.QPushButton(self.groupBox_2)
        self.pushButton_delete_model_reg_rdb.setStyleSheet("background-color: rgb(255, 181, 182);")
        self.pushButton_delete_model_reg_rdb.setObjectName("pushButton_delete_model_reg_rdb")
        self.gridLayout_4.addWidget(self.pushButton_delete_model_reg_rdb, 2, 2, 1, 1)
        self.pushButton_load_model_reg = QtWidgets.QPushButton(self.groupBox_2)
        self.pushButton_load_model_reg.setStyleSheet("background-color: rgb(206, 222, 255);")
        self.pushButton_load_model_reg.setObjectName("pushButton_load_model_reg")
        self.gridLayout_4.addWidget(self.pushButton_load_model_reg, 2, 3, 1, 1)
        self.pushButton_unload_model_reg = QtWidgets.QPushButton(self.groupBox_2)
        self.pushButton_unload_model_reg.setStyleSheet("background-color: rgb(255, 233, 196);")
        self.pushButton_unload_model_reg.setObjectName("pushButton_unload_model_reg")
        self.gridLayout_4.addWidget(self.pushButton_unload_model_reg, 2, 4, 1, 1)
        self.gridLayout.addWidget(self.groupBox_2, 9, 0, 1, 6)
        self.gridLayout_2.addLayout(self.gridLayout, 1, 1, 1, 1)
        spacerItem3 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout_2.addItem(spacerItem3, 2, 1, 1, 1)

        self.retranslateUi(rem_db)
        QtCore.QMetaObject.connectSlotsByName(rem_db)

    def retranslateUi(self, rem_db):
        _translate = QtCore.QCoreApplication.translate
        rem_db.setWindowTitle(_translate("rem_db", "RemoteDB"))
        self.pushButton_load_formations.setToolTip(_translate("rem_db", "Загрузить пласты с удаленной БД на локальную"))
        self.pushButton_load_formations.setText(_translate("rem_db", "REM -> LOC"))
        self.pushButton_sync_objects.setToolTip(_translate("rem_db", "Синхронизировать все объекты вместе с исследованиями и профилями"))
        self.pushButton_sync_objects.setText(_translate("rem_db", "Sync all objects"))
        self.pushButton_unload_formations.setToolTip(_translate("rem_db", "Выгрузить пласты с локальной БД на удаленную"))
        self.pushButton_unload_formations.setText(_translate("rem_db", "LOC -> REM"))
        self.pushButton_unload_obj_rem.setToolTip(_translate("rem_db", "Выгрузить объект с локальной БД на удаленную"))
        self.pushButton_unload_obj_rem.setText(_translate("rem_db", "LOC -> REM"))
        self.pushButton_delete_obj_rem.setToolTip(_translate("rem_db", "Удалить объект в удаленной БД"))
        self.pushButton_delete_obj_rem.setText(_translate("rem_db", "Delete object"))
        self.label_wells_count.setText(_translate("rem_db", "Кол-во скважин: "))
        self.pushButton_sync_wells.setToolTip(_translate("rem_db", "Синхронизировать скважины в локальной и удаленной БД"))
        self.pushButton_sync_wells.setText(_translate("rem_db", "Synchronize wells"))
        self.toolButton_cw.setToolTip(_translate("rem_db", "Обновить кол-во скважин"))
        self.toolButton_cw.setText(_translate("rem_db", "?"))
        self.pushButton_load_obj_rem.setToolTip(_translate("rem_db", "Загрузить объект с удаленной БД на локальную"))
        self.pushButton_load_obj_rem.setText(_translate("rem_db", "REM -> LOC"))
        self.label_5.setText(_translate("rem_db", "Профиль:"))
        self.label_4.setText(_translate("rem_db", "Объект:"))
        self.pushButton_load_well_rel.setToolTip(_translate("rem_db", "Загрузить данные скважин с удаленной БД на локальную"))
        self.pushButton_load_well_rel.setText(_translate("rem_db", "REM -> LOC"))
        self.label_2.setText(_translate("rem_db", "Синхронизировать пласты:"))
        self.label.setText(_translate("rem_db", "Синхронизировать данные скважин:"))
        self.comboBox_research_rem.setToolTip(_translate("rem_db", "Дата исследования"))
        self.pushButton_unload_well_rel.setToolTip(_translate("rem_db", "Выгрузить данные скважин с локальной БД на удаленную"))
        self.pushButton_unload_well_rel.setText(_translate("rem_db", "LOC -> REM"))
        self.groupBox.setTitle(_translate("rem_db", "Classification"))
        self.label_3.setText(_translate("rem_db", "Анализ MLP:"))
        self.pushButton_load_model.setText(_translate("rem_db", "REM -> LOC"))
        self.label_7.setText(_translate("rem_db", "Trained model:"))
        self.pushButton_load_mlp.setToolTip(_translate("rem_db", "Загрузить анализ MLP с удаленной БД на локальную"))
        self.pushButton_load_mlp.setText(_translate("rem_db", "REM -> LOC"))
        self.checkBox_check_ga_params.setText(_translate("rem_db", "не учитывать параметры"))
        self.pushButton_delete_model_rdb.setToolTip(_translate("rem_db", "Удалить модель в удаленной БД"))
        self.pushButton_delete_model_rdb.setText(_translate("rem_db", "Delete"))
        self.pushButton_sync_ga_cls.setText(_translate("rem_db", "Synchronize genetic analysis"))
        self.pushButton_unload_model.setText(_translate("rem_db", "LOC -> REM"))
        self.pushButton_delete_mlp_rdb.setToolTip(_translate("rem_db", "Удалить анализ MLP в удаленной БД"))
        self.pushButton_delete_mlp_rdb.setText(_translate("rem_db", "Delete"))
        self.pushButton_unload_mlp.setToolTip(_translate("rem_db", "Выгрузить анализ с локальной БД на удаленную"))
        self.pushButton_unload_mlp.setText(_translate("rem_db", "LOC -> REM"))
        self.label_6.setText(_translate("rem_db", "Genetic:"))
        self.groupBox_2.setTitle(_translate("rem_db", "Regression"))
        self.label_10.setText(_translate("rem_db", "Анализ RegMod:"))
        self.pushButton_delete_regmod_rdb.setToolTip(_translate("rem_db", "Удалить анализ MLP в удаленной БД"))
        self.pushButton_delete_regmod_rdb.setText(_translate("rem_db", "Delete"))
        self.pushButton_load_regmod.setToolTip(_translate("rem_db", "Загрузить анализ MLP с удаленной БД на локальную"))
        self.pushButton_load_regmod.setText(_translate("rem_db", "REM -> LOC"))
        self.pushButton_unload_regmod.setToolTip(_translate("rem_db", "Выгрузить анализ с локальной БД на удаленную"))
        self.pushButton_unload_regmod.setText(_translate("rem_db", "LOC -> REM"))
        self.label_8.setText(_translate("rem_db", "Genetic:"))
        self.checkBox_check_ga_params_reg.setText(_translate("rem_db", "не учитывать параметры"))
        self.pushButton_sync_ga_reg.setText(_translate("rem_db", "Synchronize genetic analysis"))
        self.label_9.setText(_translate("rem_db", "Trained model:"))
        self.pushButton_delete_model_reg_rdb.setToolTip(_translate("rem_db", "Удалить модель в удаленной БД"))
        self.pushButton_delete_model_reg_rdb.setText(_translate("rem_db", "Delete"))
        self.pushButton_load_model_reg.setText(_translate("rem_db", "REM -> LOC"))
        self.pushButton_unload_model_reg.setText(_translate("rem_db", "LOC -> REM"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    rem_db = QtWidgets.QWidget()
    ui = Ui_rem_db()
    ui.setupUi(rem_db)
    rem_db.show()
    sys.exit(app.exec_())
