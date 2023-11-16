import sys, os
import random, math
from sqlite3 import connect

from PyQt5.QtCore import QRect, Qt
from PyQt5.QtWidgets import QFileDialog, QCheckBox, QListWidgetItem, QApplication, QMessageBox
from PyQt5.QtGui import QBrush, QColor

from qt.geovel_main_window import *
from qt.classifier_form import *
from qt.regressor_form import *
from qt.test import *
from qt.lof_form import *
from qt.formation_ai_form import *
from qt.choose_formation_lda import *
from qt.choose_regmod import *
from qt.random_search_form import *
from qt.comment_form import *
from model import *
import numpy as np
import pyqtgraph as pg
import pandas as pd
import json

from scipy.stats import skew, kurtosis, rankdata, f_oneway, spearmanr

from scipy.signal import savgol_filter, hilbert, wiener, medfilt, medfilt2d, filtfilt, butter, argrelmin, argrelmax, find_peaks
from scipy.fft import rfft2, irfft2, rfft, irfft, rfftfreq, fftfreq, fft, dct
from scipy.interpolate import splrep, splev
from scipy.signal.windows import cosine
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import normalize
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier,
                              ExtraTreesClassifier, StackingClassifier, VotingClassifier, GradientBoostingRegressor,
                              AdaBoostRegressor, RandomForestRegressor, ExtraTreesRegressor, StackingRegressor,
                              VotingRegressor)
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import train_test_split, KFold, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.manifold import TSNE
from sklearn.linear_model import LinearRegression, ElasticNet, Lasso
from sklearn.neighbors import KNeighborsRegressor, LocalOutlierFactor
from sklearn.svm import SVR, SVC
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE, RFECV
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB, CategoricalNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import mean_squared_error, roc_curve, auc

from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from python_speech_features import mfcc
import lasio as ls
from skgstat import Variogram, OrdinaryKriging


import pickle

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


roi = pg.ROI(pos=[0, 0], size=[ui.spinBox_roi.value(), 512], maxBounds=QRect(0, 0, 100000000, 512))
radarogramma.addItem(roi)

imputer = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0) # заполнение пропусков для LDA

