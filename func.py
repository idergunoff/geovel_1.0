from PyQt5.QtWidgets import QFileDialog, QCheckBox
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter, hilbert, wiener
from object import *


def set_info(text, color):
    ui.info.append(f'<span style =\"color:{color};\" >{text}</span>')


def get_object_id():
    return int(ui.comboBox_object.currentText().split('id')[-1])


def query_to_list(query):
    """ результаты запроса в список """
    return sum(list(map(list, query)), [])

