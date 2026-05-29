from __future__ import annotations

import pandas as pd
import base64
import gzip
import hashlib
import json
import random
import gc
import multiprocessing as mp
from collections import Counter, OrderedDict
from itertools import product
from time import monotonic
from typing import Any, Dict, Literal, Optional, TypedDict

import build_table
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from PyQt5.QtCore import Qt
from scipy.interpolate import griddata
from sklearn.random_projection import SparseRandomProjection
from draw import draw_radarogram
from func import *
import datetime as dt
from krige import draw_map
