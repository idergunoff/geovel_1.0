import sys, os
import random, math
from sqlite3 import connect
from itertools import combinations
from collections import Counter
import re
import inspect
import chardet


import tqdm as tqdm
from PIL import Image, ImageDraw, ImageFont
from PyQt5.QtCore import QRect, Qt
from PyQt5.QtWidgets import QFileDialog, QCheckBox, QListWidgetItem, QApplication, QMessageBox, QColorDialog
from PyQt5.QtGui import QBrush, QColor, QCursor
from pyqtgraph.Qt import QtWidgets


from qt.geovel_main_window import *
from qt.classifier_form import *
from qt.regressor_form import *
from qt.anova_form import *
from qt.lof_form import *
from qt.formation_ai_form import *
from qt.choose_formation_lda import *
from qt.choose_regmod import *
from qt.random_search_form import *
from qt.comment_form import *
from qt.choose_formation_map import *
from qt.load_points import *
from qt.form_delete_therm import *
from qt.add_profile_class import *
from qt.geochem_loader import *
from qt.tsne_form import *
from qt.point_param_graph_form import *
from qt.form_test_model import *
from qt.model_lineup import *
from qt.torch_classfier import *
from qt.torch_regressor import *
from qt.rename_model import *
from qt.random_param import *

from model import *
import numpy as np
import pyqtgraph as pg
from pyqtgraph.exporters import ImageExporter
import pandas as pd
import json
import zipfile
import shutil
from tqdm import tqdm

from scipy.stats import skew, kurtosis, rankdata, f_oneway, spearmanr

from scipy.fftpack import fft2, ifft2, dctn, idctn
from scipy.signal import savgol_filter, hilbert, wiener, medfilt, medfilt2d, filtfilt, butter, argrelmin, argrelmax, find_peaks
from scipy.fft import rfft2, irfft2, rfft, irfft, rfftfreq, fftfreq, fft, dct
from scipy.interpolate import splrep, splev
from scipy.signal.windows import cosine
from scipy.special import softmax
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
                              VotingRegressor, BaggingClassifier, BaggingRegressor, IsolationForest)
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, cross_val_score, RandomizedSearchCV, cross_validate
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.manifold import TSNE
from sklearn.linear_model import LinearRegression, ElasticNet, Lasso
from sklearn.neighbors import KNeighborsRegressor, LocalOutlierFactor
from sklearn.svm import SVR, SVC
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE, RFECV
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB, CategoricalNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import mean_squared_error, roc_curve, auc
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import r2_score, precision_score, recall_score, f1_score, accuracy_score
from imblearn.over_sampling import SMOTE, ADASYN

from yellowbrick.classifier import DiscriminationThreshold

from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from python_speech_features import mfcc
import lasio as ls
from skgstat import Variogram, OrdinaryKriging

from screeninfo import get_monitors
from numpy.linalg import LinAlgError
import pickle
from scipy.signal import butter, filtfilt

from functools import lru_cache


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

