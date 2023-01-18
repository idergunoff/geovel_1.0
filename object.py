import sys
from qt.geovel_main_window import *
from model import *
import numpy as np
import pyqtgraph as pg
from scipy.stats import skew, kurtosis

app = QtWidgets.QApplication(sys.argv)
MainWindow = QtWidgets.QMainWindow()
ui = Ui_MainWindow()
ui.setupUi(MainWindow)

img = pg.ImageItem()
ui.radarogram.addItem(img)

# roi = pg.ROI(pos=[0, 0], size=[ui.spinBox_roi.value(), 512], maxBounds=QRect(0, 0, 100000000, 512))
# ui.graphicsView.addItem(roi)

session = Session()