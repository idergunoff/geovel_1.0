import sys, os
import random, math
from pathlib import Path
from sqlite3 import connect
from itertools import combinations, permutations
from collections import Counter
import re
import inspect
import chardet

from models_db.model import *
from models_db.model_profile_features import *

import tqdm as tqdm
from PIL import Image, ImageDraw, ImageFont
from PyQt5.QtCore import QRect, Qt
from PyQt5.QtWidgets import QFileDialog, QCheckBox, QListWidgetItem, QApplication, QMessageBox, QColorDialog, QTableWidget, QTableWidgetItem
from PyQt5.QtGui import QBrush, QColor, QCursor
from pyqtgraph.Qt import QtWidgets

from platypus import (Problem, Real, Binary, NSGAII, GDE3, RandomGenerator, SBX, PM, GAOperator, ProcessPoolEvaluator,
                      GeneticAlgorithm, InjectedPopulation, Solution, HUX, BitFlip, CompoundOperator, nondominated,
                      crowding_distance, nondominated_sort)


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
from qt.show_model_params import *
from qt.choose_formation_map import *
from qt.load_points import *
from qt.form_delete_therm import *
from qt.add_profile_class import *
from qt.geochem_loader import *
from qt.tsne_form import *
from qt.point_param_graph_form import *
from qt.form_test_model import *
from qt.torch_classfier import *
from qt.torch_regressor import *
from qt.rename_model import *
from qt.random_param import *
from qt.random_search_reg import *
from qt.feature_selection import *
from qt.parameter_dependence import *
from qt.pareto_form import *
from qt.corrected_model_pred import *
from qt.rem_db_window import *
from qt.form_well_log import *
from qt.gen_alg_form import *
from qt.filter_form import *
from qt.form_map_well_log import *
from qt.form_upgrade_predict_model import *

import numpy as np
import pyqtgraph as pg
from pyqtgraph.exporters import ImageExporter
import pandas as pd
import json
from json import JSONDecodeError
import pywt
from nolds import hurst_rs, dfa, corr_dim, sampen, lyap_r
from pyentrp import entropy as entr
from numpy.lib.stride_tricks import as_strided
import zipfile
import shutil
from tqdm import tqdm

from scipy.stats import skew, kurtosis, rankdata, f_oneway, spearmanr, norm, entropy, linregress
from scipy.fftpack import fft2, ifft2, dctn, idctn
from scipy.signal import (savgol_filter, hilbert, wiener, medfilt, medfilt2d, filtfilt, butter, argrelmin, argrelmax,
                          find_peaks, peak_widths, correlate)
from scipy.fft import rfft2, irfft2, rfft, irfft, rfftfreq, fftfreq, fft, dct
from scipy.interpolate import splrep, splev, splprep
from scipy.signal.windows import cosine
from scipy.special import softmax
from scipy.spatial import cKDTree, ConvexHull
from scipy.spatial.distance import pdist, squareform
from scipy.ndimage import binary_erosion, binary_dilation

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
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, cross_val_score, RandomizedSearchCV, cross_validate, LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler, Normalizer, PowerTransformer
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor
import lightgbm as lgb
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
from sklearn.pipeline import FeatureUnion
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB, CategoricalNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import mean_squared_error, roc_curve, auc
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import r2_score, precision_score, recall_score, f1_score, accuracy_score
from imblearn.over_sampling import SMOTE, ADASYN

from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest, f_regression, chi2, mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import PartialDependenceDisplay
from sklearn.cluster import KMeans

from boruta import BorutaPy
from collections import defaultdict


from difflib import SequenceMatcher

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split, RandomSampler, SubsetRandomSampler
from skorch import NeuralNetClassifier, NeuralNetRegressor

from skorch.dataset import ValidSplit
from skorch.callbacks import EarlyStopping

from yellowbrick.classifier import DiscriminationThreshold

from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.patches import Rectangle

import matplotlib.cm as cm
import matplotlib.colors as mcolors

from mpl_toolkits.mplot3d import Axes3D

import seaborn as sns
from python_speech_features import mfcc
import lasio as ls
from PyEMD import EMD
from skgstat import Variogram, OrdinaryKriging

from screeninfo import get_monitors
from numpy.linalg import LinAlgError
import pickle

from pyts.image import RecurrencePlot
import optuna

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


Base.metadata.create_all(engine)
