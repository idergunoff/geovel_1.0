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
        MainWindow.resize(1553, 833)
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
        self.info = QtWidgets.QTextEdit(self.centralwidget)
        self.info.setEnabled(True)
        self.info.setFocusPolicy(QtCore.Qt.NoFocus)
        self.info.setObjectName("info")
        self.gridLayout.addWidget(self.info, 1, 1, 1, 1)
        self.signal = PlotWidget(self.centralwidget)
        self.signal.setObjectName("signal")
        self.gridLayout.addWidget(self.signal, 0, 1, 1, 1)
        self.graph = PlotWidget(self.centralwidget)
        self.graph.setObjectName("graph")
        self.gridLayout.addWidget(self.graph, 1, 0, 1, 1)
        self.radarogram = GraphicsLayoutWidget(self.centralwidget)
        self.radarogram.setObjectName("radarogram")
        self.gridLayout.addWidget(self.radarogram, 0, 0, 1, 1)
        self.gridLayout.setColumnStretch(0, 9)
        self.gridLayout.setRowStretch(0, 9)
        self.gridLayout_2.addLayout(self.gridLayout, 1, 0, 1, 1)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.tabWidget_2 = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget_2.setObjectName("tabWidget_2")
        self.tab_3 = QtWidgets.QWidget()
        self.tab_3.setObjectName("tab_3")
        self.gridLayout_7 = QtWidgets.QGridLayout(self.tab_3)
        self.gridLayout_7.setObjectName("gridLayout_7")
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.checkBox_2degree = QtWidgets.QCheckBox(self.tab_3)
        self.checkBox_2degree.setObjectName("checkBox_2degree")
        self.horizontalLayout_6.addWidget(self.checkBox_2degree)
        self.spinBox_type_dct = QtWidgets.QSpinBox(self.tab_3)
        self.spinBox_type_dct.setMinimum(1)
        self.spinBox_type_dct.setMaximum(4)
        self.spinBox_type_dct.setProperty("value", 2)
        self.spinBox_type_dct.setObjectName("spinBox_type_dct")
        self.horizontalLayout_6.addWidget(self.spinBox_type_dct)
        self.pushButton_ceps_koef = QtWidgets.QPushButton(self.tab_3)
        self.pushButton_ceps_koef.setObjectName("pushButton_ceps_koef")
        self.horizontalLayout_6.addWidget(self.pushButton_ceps_koef)
        self.gridLayout_7.addLayout(self.horizontalLayout_6, 7, 0, 1, 1)
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout_7.addItem(spacerItem, 11, 0, 1, 1)
        self.gridLayout_10 = QtWidgets.QGridLayout()
        self.gridLayout_10.setObjectName("gridLayout_10")
        self.pushButton_find_oil = QtWidgets.QPushButton(self.tab_3)
        self.pushButton_find_oil.setObjectName("pushButton_find_oil")
        self.gridLayout_10.addWidget(self.pushButton_find_oil, 1, 1, 1, 1)
        self.pushButton_fft = QtWidgets.QPushButton(self.tab_3)
        self.pushButton_fft.setObjectName("pushButton_fft")
        self.gridLayout_10.addWidget(self.pushButton_fft, 0, 0, 1, 1)
        self.pushButton_reset = QtWidgets.QPushButton(self.tab_3)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_reset.sizePolicy().hasHeightForWidth())
        self.pushButton_reset.setSizePolicy(sizePolicy)
        self.pushButton_reset.setObjectName("pushButton_reset")
        self.gridLayout_10.addWidget(self.pushButton_reset, 0, 3, 1, 1)
        self.pushButton_add_fft = QtWidgets.QPushButton(self.tab_3)
        self.pushButton_add_fft.setObjectName("pushButton_add_fft")
        self.gridLayout_10.addWidget(self.pushButton_add_fft, 1, 0, 1, 1)
        self.pushButton_ifft = QtWidgets.QPushButton(self.tab_3)
        self.pushButton_ifft.setObjectName("pushButton_ifft")
        self.gridLayout_10.addWidget(self.pushButton_ifft, 0, 1, 1, 1)
        self.spinBox_fft_down = QtWidgets.QSpinBox(self.tab_3)
        self.spinBox_fft_down.setMaximum(267)
        self.spinBox_fft_down.setObjectName("spinBox_fft_down")
        self.gridLayout_10.addWidget(self.spinBox_fft_down, 1, 2, 1, 1)
        self.spinBox_ftt_up = QtWidgets.QSpinBox(self.tab_3)
        self.spinBox_ftt_up.setMaximum(267)
        self.spinBox_ftt_up.setObjectName("spinBox_ftt_up")
        self.gridLayout_10.addWidget(self.spinBox_ftt_up, 0, 2, 1, 1)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.checkBox_fft_2axes = QtWidgets.QCheckBox(self.tab_3)
        self.checkBox_fft_2axes.setObjectName("checkBox_fft_2axes")
        self.horizontalLayout_2.addWidget(self.checkBox_fft_2axes)
        self.checkBox_fft_int = QtWidgets.QCheckBox(self.tab_3)
        self.checkBox_fft_int.setObjectName("checkBox_fft_int")
        self.horizontalLayout_2.addWidget(self.checkBox_fft_int)
        self.gridLayout_10.addLayout(self.horizontalLayout_2, 1, 3, 1, 1)
        self.gridLayout_7.addLayout(self.gridLayout_10, 3, 0, 1, 1)
        self.gridLayout_9 = QtWidgets.QGridLayout()
        self.gridLayout_9.setObjectName("gridLayout_9")
        self.pushButton_rfft2 = QtWidgets.QPushButton(self.tab_3)
        self.pushButton_rfft2.setObjectName("pushButton_rfft2")
        self.gridLayout_9.addWidget(self.pushButton_rfft2, 0, 0, 1, 1)
        self.pushButton_rang = QtWidgets.QPushButton(self.tab_3)
        self.pushButton_rang.setObjectName("pushButton_rang")
        self.gridLayout_9.addWidget(self.pushButton_rang, 1, 2, 1, 1)
        self.pushButton_irfft2 = QtWidgets.QPushButton(self.tab_3)
        self.pushButton_irfft2.setObjectName("pushButton_irfft2")
        self.gridLayout_9.addWidget(self.pushButton_irfft2, 1, 0, 1, 1)
        self.pushButton_dct = QtWidgets.QPushButton(self.tab_3)
        self.pushButton_dct.setObjectName("pushButton_dct")
        self.gridLayout_9.addWidget(self.pushButton_dct, 0, 1, 1, 1)
        self.pushButton_log = QtWidgets.QPushButton(self.tab_3)
        self.pushButton_log.setObjectName("pushButton_log")
        self.gridLayout_9.addWidget(self.pushButton_log, 0, 2, 1, 1)
        self.pushButton_idct = QtWidgets.QPushButton(self.tab_3)
        self.pushButton_idct.setObjectName("pushButton_idct")
        self.gridLayout_9.addWidget(self.pushButton_idct, 1, 1, 1, 1)
        self.gridLayout_7.addLayout(self.gridLayout_9, 8, 0, 1, 1)
        self.label_9 = QtWidgets.QLabel(self.tab_3)
        font = QtGui.QFont()
        font.setPointSize(9)
        font.setItalic(True)
        self.label_9.setFont(font)
        self.label_9.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label_9.setAlignment(QtCore.Qt.AlignCenter)
        self.label_9.setObjectName("label_9")
        self.gridLayout_7.addWidget(self.label_9, 6, 0, 1, 1)
        self.label_7 = QtWidgets.QLabel(self.tab_3)
        font = QtGui.QFont()
        font.setPointSize(9)
        font.setItalic(True)
        font.setUnderline(False)
        font.setStrikeOut(False)
        font.setKerning(True)
        self.label_7.setFont(font)
        self.label_7.setAlignment(QtCore.Qt.AlignCenter)
        self.label_7.setObjectName("label_7")
        self.gridLayout_7.addWidget(self.label_7, 2, 0, 1, 1)
        self.gridLayout_6 = QtWidgets.QGridLayout()
        self.gridLayout_6.setObjectName("gridLayout_6")
        self.gridLayout_5 = QtWidgets.QGridLayout()
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.comboBox_atrib = QtWidgets.QComboBox(self.tab_3)
        self.comboBox_atrib.setObjectName("comboBox_atrib")
        self.comboBox_atrib.addItem("")
        self.comboBox_atrib.addItem("")
        self.comboBox_atrib.addItem("")
        self.comboBox_atrib.addItem("")
        self.comboBox_atrib.addItem("")
        self.comboBox_atrib.addItem("")
        self.gridLayout_5.addWidget(self.comboBox_atrib, 0, 0, 1, 1)
        self.pushButton_draw_rad = QtWidgets.QPushButton(self.tab_3)
        self.pushButton_draw_rad.setObjectName("pushButton_draw_rad")
        self.gridLayout_5.addWidget(self.pushButton_draw_rad, 0, 1, 1, 1)
        self.spinBox_rad_down = QtWidgets.QSpinBox(self.tab_3)
        self.spinBox_rad_down.setMaximum(99999999)
        self.spinBox_rad_down.setObjectName("spinBox_rad_down")
        self.gridLayout_5.addWidget(self.spinBox_rad_down, 1, 1, 1, 1)
        self.spinBox_rad_up = QtWidgets.QSpinBox(self.tab_3)
        self.spinBox_rad_up.setMaximum(99999999)
        self.spinBox_rad_up.setObjectName("spinBox_rad_up")
        self.gridLayout_5.addWidget(self.spinBox_rad_up, 1, 0, 1, 1)
        self.pushButton_draw_cur = QtWidgets.QPushButton(self.tab_3)
        self.pushButton_draw_cur.setObjectName("pushButton_draw_cur")
        self.gridLayout_5.addWidget(self.pushButton_draw_cur, 0, 2, 1, 1)
        self.checkBox_minmax = QtWidgets.QCheckBox(self.tab_3)
        self.checkBox_minmax.setObjectName("checkBox_minmax")
        self.gridLayout_5.addWidget(self.checkBox_minmax, 1, 2, 1, 1)
        self.gridLayout_6.addLayout(self.gridLayout_5, 5, 0, 1, 1)
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
        self.gridLayout_6.addLayout(self.horizontalLayout_3, 0, 0, 1, 1)
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
        self.gridLayout_6.addLayout(self.horizontalLayout_5, 4, 0, 1, 1)
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
        self.gridLayout_6.addLayout(self.horizontalLayout_4, 1, 0, 1, 1)
        self.label_2 = QtWidgets.QLabel(self.tab_3)
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setObjectName("label_2")
        self.gridLayout_6.addWidget(self.label_2, 2, 0, 1, 1)
        self.gridLayout_8 = QtWidgets.QGridLayout()
        self.gridLayout_8.setObjectName("gridLayout_8")
        self.pushButton_uf = QtWidgets.QPushButton(self.tab_3)
        self.pushButton_uf.setObjectName("pushButton_uf")
        self.gridLayout_8.addWidget(self.pushButton_uf, 0, 0, 1, 1)
        self.pushButton_m = QtWidgets.QPushButton(self.tab_3)
        self.pushButton_m.setObjectName("pushButton_m")
        self.gridLayout_8.addWidget(self.pushButton_m, 0, 1, 1, 1)
        self.pushButton_r = QtWidgets.QPushButton(self.tab_3)
        self.pushButton_r.setObjectName("pushButton_r")
        self.gridLayout_8.addWidget(self.pushButton_r, 0, 2, 1, 1)
        self.gridLayout_6.addLayout(self.gridLayout_8, 3, 0, 1, 1)
        self.gridLayout_7.addLayout(self.gridLayout_6, 0, 0, 1, 1)
        self.label_3 = QtWidgets.QLabel(self.tab_3)
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setUnderline(True)
        font.setWeight(75)
        self.label_3.setFont(font)
        self.label_3.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label_3.setAlignment(QtCore.Qt.AlignCenter)
        self.label_3.setObjectName("label_3")
        self.gridLayout_7.addWidget(self.label_3, 1, 0, 1, 1)
        self.label_8 = QtWidgets.QLabel(self.tab_3)
        font = QtGui.QFont()
        font.setPointSize(9)
        font.setItalic(True)
        self.label_8.setFont(font)
        self.label_8.setAlignment(QtCore.Qt.AlignCenter)
        self.label_8.setObjectName("label_8")
        self.gridLayout_7.addWidget(self.label_8, 4, 0, 1, 1)
        self.gridLayout_11 = QtWidgets.QGridLayout()
        self.gridLayout_11.setObjectName("gridLayout_11")
        self.pushButton_filtfilt = QtWidgets.QPushButton(self.tab_3)
        self.pushButton_filtfilt.setObjectName("pushButton_filtfilt")
        self.gridLayout_11.addWidget(self.pushButton_filtfilt, 0, 0, 1, 1)
        self.spinBox_a_filtfilt = QtWidgets.QSpinBox(self.tab_3)
        self.spinBox_a_filtfilt.setMinimum(1)
        self.spinBox_a_filtfilt.setProperty("value", 3)
        self.spinBox_a_filtfilt.setObjectName("spinBox_a_filtfilt")
        self.gridLayout_11.addWidget(self.spinBox_a_filtfilt, 0, 1, 1, 1)
        self.doubleSpinBox_b_filtfilt = QtWidgets.QDoubleSpinBox(self.tab_3)
        self.doubleSpinBox_b_filtfilt.setDecimals(3)
        self.doubleSpinBox_b_filtfilt.setMinimum(0.001)
        self.doubleSpinBox_b_filtfilt.setSingleStep(0.01)
        self.doubleSpinBox_b_filtfilt.setProperty("value", 0.05)
        self.doubleSpinBox_b_filtfilt.setObjectName("doubleSpinBox_b_filtfilt")
        self.gridLayout_11.addWidget(self.doubleSpinBox_b_filtfilt, 0, 2, 1, 1)
        self.spinBox_a_savgol = QtWidgets.QSpinBox(self.tab_3)
        self.spinBox_a_savgol.setMinimum(1)
        self.spinBox_a_savgol.setSingleStep(2)
        self.spinBox_a_savgol.setProperty("value", 5)
        self.spinBox_a_savgol.setObjectName("spinBox_a_savgol")
        self.gridLayout_11.addWidget(self.spinBox_a_savgol, 1, 1, 1, 1)
        self.spinBox_b_savgol = QtWidgets.QSpinBox(self.tab_3)
        self.spinBox_b_savgol.setMinimum(1)
        self.spinBox_b_savgol.setSingleStep(2)
        self.spinBox_b_savgol.setProperty("value", 3)
        self.spinBox_b_savgol.setObjectName("spinBox_b_savgol")
        self.gridLayout_11.addWidget(self.spinBox_b_savgol, 1, 2, 1, 1)
        self.spinBox_a_wiener = QtWidgets.QSpinBox(self.tab_3)
        self.spinBox_a_wiener.setMinimum(1)
        self.spinBox_a_wiener.setProperty("value", 3)
        self.spinBox_a_wiener.setObjectName("spinBox_a_wiener")
        self.gridLayout_11.addWidget(self.spinBox_a_wiener, 2, 1, 1, 1)
        self.pushButton_medfilt = QtWidgets.QPushButton(self.tab_3)
        self.pushButton_medfilt.setObjectName("pushButton_medfilt")
        self.gridLayout_11.addWidget(self.pushButton_medfilt, 3, 0, 1, 1)
        self.spinBox_a_medfilt = QtWidgets.QSpinBox(self.tab_3)
        self.spinBox_a_medfilt.setMinimum(1)
        self.spinBox_a_medfilt.setSingleStep(2)
        self.spinBox_a_medfilt.setProperty("value", 3)
        self.spinBox_a_medfilt.setObjectName("spinBox_a_medfilt")
        self.gridLayout_11.addWidget(self.spinBox_a_medfilt, 3, 1, 1, 1)
        self.pushButton_savgol = QtWidgets.QPushButton(self.tab_3)
        self.pushButton_savgol.setObjectName("pushButton_savgol")
        self.gridLayout_11.addWidget(self.pushButton_savgol, 1, 0, 1, 1)
        self.pushButton_wiener = QtWidgets.QPushButton(self.tab_3)
        self.pushButton_wiener.setObjectName("pushButton_wiener")
        self.gridLayout_11.addWidget(self.pushButton_wiener, 2, 0, 1, 1)
        self.spinBox_b_medfilt = QtWidgets.QSpinBox(self.tab_3)
        self.spinBox_b_medfilt.setMinimum(1)
        self.spinBox_b_medfilt.setSingleStep(2)
        self.spinBox_b_medfilt.setProperty("value", 3)
        self.spinBox_b_medfilt.setObjectName("spinBox_b_medfilt")
        self.gridLayout_11.addWidget(self.spinBox_b_medfilt, 3, 2, 1, 1)
        self.spinBox_b_wiener = QtWidgets.QSpinBox(self.tab_3)
        self.spinBox_b_wiener.setMinimum(1)
        self.spinBox_b_wiener.setProperty("value", 3)
        self.spinBox_b_wiener.setObjectName("spinBox_b_wiener")
        self.gridLayout_11.addWidget(self.spinBox_b_wiener, 2, 2, 1, 1)
        self.gridLayout_7.addLayout(self.gridLayout_11, 5, 0, 1, 1)
        self.pushButton_save_signal = QtWidgets.QPushButton(self.tab_3)
        self.pushButton_save_signal.setObjectName("pushButton_save_signal")
        self.gridLayout_7.addWidget(self.pushButton_save_signal, 9, 0, 1, 1)
        self.pushButton_maxmin = QtWidgets.QPushButton(self.tab_3)
        self.pushButton_maxmin.setObjectName("pushButton_maxmin")
        self.gridLayout_7.addWidget(self.pushButton_maxmin, 10, 0, 1, 1)
        self.tabWidget_2.addTab(self.tab_3, "")
        self.tab_4 = QtWidgets.QWidget()
        self.tab_4.setObjectName("tab_4")
        self.gridLayout_12 = QtWidgets.QGridLayout(self.tab_4)
        self.gridLayout_12.setObjectName("gridLayout_12")
        self.pushButton_vacuum = QtWidgets.QPushButton(self.tab_4)
        self.pushButton_vacuum.setObjectName("pushButton_vacuum")
        self.gridLayout_12.addWidget(self.pushButton_vacuum, 1, 1, 1, 1)
        self.gridLayout_13 = QtWidgets.QGridLayout()
        self.gridLayout_13.setObjectName("gridLayout_13")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.gridLayout_14 = QtWidgets.QGridLayout()
        self.gridLayout_14.setObjectName("gridLayout_14")
        self.label_10 = QtWidgets.QLabel(self.tab_4)
        font = QtGui.QFont()
        font.setPointSize(11)
        font.setItalic(True)
        font.setUnderline(True)
        self.label_10.setFont(font)
        self.label_10.setObjectName("label_10")
        self.gridLayout_14.addWidget(self.label_10, 0, 0, 1, 1)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_14.addItem(spacerItem1, 0, 1, 1, 1)
        self.checkBox_draw_layer = QtWidgets.QCheckBox(self.tab_4)
        self.checkBox_draw_layer.setStyleSheet("background-color: rgb(175, 255, 243);")
        self.checkBox_draw_layer.setObjectName("checkBox_draw_layer")
        self.gridLayout_14.addWidget(self.checkBox_draw_layer, 0, 2, 1, 1)
        self.verticalLayout_3.addLayout(self.gridLayout_14)
        self.gridLayout_17 = QtWidgets.QGridLayout()
        self.gridLayout_17.setObjectName("gridLayout_17")
        self.scrollArea_2 = QtWidgets.QScrollArea(self.tab_4)
        self.scrollArea_2.setWidgetResizable(True)
        self.scrollArea_2.setObjectName("scrollArea_2")
        self.scrollAreaWidgetContents_2 = QtWidgets.QWidget()
        self.scrollAreaWidgetContents_2.setGeometry(QtCore.QRect(0, 0, 214, 94))
        self.scrollAreaWidgetContents_2.setObjectName("scrollAreaWidgetContents_2")
        self.gridLayout_18 = QtWidgets.QGridLayout(self.scrollAreaWidgetContents_2)
        self.gridLayout_18.setObjectName("gridLayout_18")
        self.widget_layer_radio = QtWidgets.QWidget(self.scrollAreaWidgetContents_2)
        self.widget_layer_radio.setObjectName("widget_layer_radio")
        self.gridLayout_26 = QtWidgets.QGridLayout(self.widget_layer_radio)
        self.gridLayout_26.setObjectName("gridLayout_26")
        self.verticalLayout_layer_radio = QtWidgets.QVBoxLayout()
        self.verticalLayout_layer_radio.setObjectName("verticalLayout_layer_radio")
        spacerItem2 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_layer_radio.addItem(spacerItem2)
        self.gridLayout_26.addLayout(self.verticalLayout_layer_radio, 0, 0, 1, 1)
        self.gridLayout_18.addWidget(self.widget_layer_radio, 0, 1, 1, 1)
        self.widget_layer = QtWidgets.QWidget(self.scrollAreaWidgetContents_2)
        self.widget_layer.setObjectName("widget_layer")
        self.gridLayout_25 = QtWidgets.QGridLayout(self.widget_layer)
        self.gridLayout_25.setObjectName("gridLayout_25")
        self.verticalLayout_layer = QtWidgets.QVBoxLayout()
        self.verticalLayout_layer.setObjectName("verticalLayout_layer")
        spacerItem3 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_layer.addItem(spacerItem3)
        self.gridLayout_25.addLayout(self.verticalLayout_layer, 0, 0, 1, 1)
        self.gridLayout_18.addWidget(self.widget_layer, 0, 0, 1, 1)
        self.gridLayout_18.setColumnStretch(0, 4)
        self.gridLayout_18.setColumnStretch(1, 1)
        self.scrollArea_2.setWidget(self.scrollAreaWidgetContents_2)
        self.gridLayout_17.addWidget(self.scrollArea_2, 0, 0, 1, 1)
        self.gridLayout_layer_5 = QtWidgets.QGridLayout()
        self.gridLayout_layer_5.setObjectName("gridLayout_layer_5")
        self.pushButton_remove_layer = QtWidgets.QPushButton(self.tab_4)
        self.pushButton_remove_layer.setStyleSheet("background-color: rgb(255, 153, 153);")
        self.pushButton_remove_layer.setObjectName("pushButton_remove_layer")
        self.gridLayout_layer_5.addWidget(self.pushButton_remove_layer, 1, 0, 1, 1)
        self.pushButton_add_layer = QtWidgets.QPushButton(self.tab_4)
        self.pushButton_add_layer.setStyleSheet("background-color: rgb(170, 255, 127);")
        self.pushButton_add_layer.setObjectName("pushButton_add_layer")
        self.gridLayout_layer_5.addWidget(self.pushButton_add_layer, 0, 0, 1, 1)
        self.pushButton_edges_layer = QtWidgets.QPushButton(self.tab_4)
        self.pushButton_edges_layer.setStyleSheet("background-color: rgb(255, 204, 121);")
        self.pushButton_edges_layer.setObjectName("pushButton_edges_layer")
        self.gridLayout_layer_5.addWidget(self.pushButton_edges_layer, 2, 0, 1, 1)
        self.gridLayout_layer_5.setColumnStretch(0, 3)
        self.gridLayout_17.addLayout(self.gridLayout_layer_5, 0, 1, 1, 1)
        self.gridLayout_17.setColumnStretch(0, 4)
        self.gridLayout_17.setColumnStretch(1, 1)
        self.verticalLayout_3.addLayout(self.gridLayout_17)
        self.verticalLayout_3.setStretch(0, 1)
        self.verticalLayout_3.setStretch(1, 4)
        self.gridLayout_13.addLayout(self.verticalLayout_3, 0, 0, 1, 1)
        spacerItem4 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout_13.addItem(spacerItem4, 1, 0, 1, 1)
        self.gridLayout_13.setRowStretch(0, 1)
        self.gridLayout_13.setRowStretch(1, 4)
        self.gridLayout_12.addLayout(self.gridLayout_13, 0, 1, 1, 1)
        self.tabWidget_2.addTab(self.tab_4, "")
        self.verticalLayout_2.addWidget(self.tabWidget_2)
        self.gridLayout_2.addLayout(self.verticalLayout_2, 1, 1, 1, 1)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.pushButton_add_window = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_add_window.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_add_window.sizePolicy().hasHeightForWidth())
        self.pushButton_add_window.setSizePolicy(sizePolicy)
        self.pushButton_add_window.setIconSize(QtCore.QSize(16, 10))
        self.pushButton_add_window.setObjectName("pushButton_add_window")
        self.horizontalLayout.addWidget(self.pushButton_add_window)
        spacerItem5 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem5)
        self.spinBox_roi = QtWidgets.QSpinBox(self.centralwidget)
        self.spinBox_roi.setMinimum(1)
        self.spinBox_roi.setMaximum(500)
        self.spinBox_roi.setProperty("value", 20)
        self.spinBox_roi.setObjectName("spinBox_roi")
        self.horizontalLayout.addWidget(self.spinBox_roi)
        self.gridLayout_2.addLayout(self.horizontalLayout, 0, 1, 1, 1)
        self.gridLayout_2.setColumnStretch(0, 7)
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
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1553, 21))
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
        self.info.setWhatsThis(_translate("MainWindow", "Поле логгирования действий"))
        self.checkBox_2degree.setText(_translate("MainWindow", "^2"))
        self.pushButton_ceps_koef.setText(_translate("MainWindow", "koef"))
        self.pushButton_find_oil.setText(_translate("MainWindow", "Найти нефть"))
        self.pushButton_fft.setText(_translate("MainWindow", "FTT"))
        self.pushButton_reset.setText(_translate("MainWindow", "Сборс"))
        self.pushButton_add_fft.setText(_translate("MainWindow", "+ FFT"))
        self.pushButton_ifft.setText(_translate("MainWindow", "IFFT"))
        self.checkBox_fft_2axes.setText(_translate("MainWindow", "2 оси"))
        self.checkBox_fft_int.setText(_translate("MainWindow", "int"))
        self.pushButton_rfft2.setText(_translate("MainWindow", "RFFT2"))
        self.pushButton_rang.setText(_translate("MainWindow", "RANG"))
        self.pushButton_irfft2.setText(_translate("MainWindow", "IRFFT2"))
        self.pushButton_dct.setText(_translate("MainWindow", "DCT"))
        self.pushButton_log.setText(_translate("MainWindow", "LOG"))
        self.pushButton_idct.setText(_translate("MainWindow", "IDCT"))
        self.label_9.setText(_translate("MainWindow", "Кепструм"))
        self.label_7.setText(_translate("MainWindow", "Фурье"))
        self.comboBox_atrib.setItemText(0, _translate("MainWindow", "A"))
        self.comboBox_atrib.setItemText(1, _translate("MainWindow", "diff"))
        self.comboBox_atrib.setItemText(2, _translate("MainWindow", "At"))
        self.comboBox_atrib.setItemText(3, _translate("MainWindow", "Vt"))
        self.comboBox_atrib.setItemText(4, _translate("MainWindow", "Pht"))
        self.comboBox_atrib.setItemText(5, _translate("MainWindow", "Wt"))
        self.pushButton_draw_rad.setText(_translate("MainWindow", "draw"))
        self.pushButton_draw_cur.setText(_translate("MainWindow", "draw current"))
        self.checkBox_minmax.setText(_translate("MainWindow", "min/max"))
        self.label_4.setText(_translate("MainWindow", "Объект:"))
        self.toolButton_add_obj.setText(_translate("MainWindow", "add"))
        self.toolButton_del_obj.setText(_translate("MainWindow", "del"))
        self.label_6.setText(_translate("MainWindow", "Парам. пласта:"))
        self.toolButton_load_plast.setText(_translate("MainWindow", "load"))
        self.toolButton_del_plast.setText(_translate("MainWindow", "del"))
        self.label_5.setText(_translate("MainWindow", "Профиль:"))
        self.toolButton_load_prof.setText(_translate("MainWindow", "load"))
        self.toolButton_del_prof.setText(_translate("MainWindow", "del"))
        self.label_2.setText(_translate("MainWindow", "Загрузка GRID"))
        self.pushButton_uf.setText(_translate("MainWindow", "P2uf"))
        self.pushButton_m.setText(_translate("MainWindow", "мощность"))
        self.pushButton_r.setText(_translate("MainWindow", "рельеф"))
        self.label_3.setText(_translate("MainWindow", "Фильтрация"))
        self.label_8.setText(_translate("MainWindow", "Другие"))
        self.pushButton_filtfilt.setText(_translate("MainWindow", "filtfilt"))
        self.pushButton_medfilt.setText(_translate("MainWindow", "medfilt"))
        self.pushButton_savgol.setText(_translate("MainWindow", "savgol"))
        self.pushButton_wiener.setText(_translate("MainWindow", "wiener"))
        self.pushButton_save_signal.setText(_translate("MainWindow", "Выгрузить"))
        self.pushButton_maxmin.setText(_translate("MainWindow", "max/min"))
        self.tabWidget_2.setTabText(self.tabWidget_2.indexOf(self.tab_3), _translate("MainWindow", "Tab 1"))
        self.pushButton_vacuum.setText(_translate("MainWindow", "VACUUM"))
        self.label_10.setText(_translate("MainWindow", "Слои:"))
        self.checkBox_draw_layer.setText(_translate("MainWindow", "draw"))
        self.pushButton_remove_layer.setText(_translate("MainWindow", "remove"))
        self.pushButton_add_layer.setText(_translate("MainWindow", "add"))
        self.pushButton_edges_layer.setText(_translate("MainWindow", "edges"))
        self.tabWidget_2.setTabText(self.tabWidget_2.indexOf(self.tab_4), _translate("MainWindow", "Tab 2"))
        self.pushButton_add_window.setText(_translate("MainWindow", "to window"))
from pyqtgraph import GraphicsLayoutWidget, PlotWidget


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
