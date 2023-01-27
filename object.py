import sys
from sqlite3 import connect

from PyQt5.QtCore import QRect

from qt.geovel_main_window import *
from model import *
import numpy as np
import pyqtgraph as pg
import pandas as pd
import json
from scipy.stats import skew, kurtosis
from PyQt5.QtWidgets import QFileDialog, QCheckBox
from scipy.signal import savgol_filter, hilbert, wiener, medfilt, medfilt2d, filtfilt, butter
from scipy.fft import rfft2, irfft2



app = QtWidgets.QApplication(sys.argv)
MainWindow = QtWidgets.QMainWindow()
ui = Ui_MainWindow()
ui.setupUi(MainWindow)

radarogramma = ui.radarogram.addPlot()
img = pg.ImageItem()
radarogramma.invertY(True)
radarogramma.addItem(img)

hist = pg.HistogramLUTItem(gradientPosition='left')

ui.radarogram.addItem(hist)


roi = pg.ROI(pos=[0, 0], size=[20, 512], maxBounds=QRect(0, 0, 100000000, 512))
radarogramma.addItem(roi)



session = Session()