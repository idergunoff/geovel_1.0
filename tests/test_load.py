import json
import pandas as pd
import numpy as np
import tempfile
from load import read_interval_file, validate_interval_data, compute_parameters

def test_read_interval_file(tmp_path):
    content = 'T01;D02;Latd;Long\n1,0;2,0;10;20\n'
    p = tmp_path / 'interval.txt'
    p.write_text(content, encoding='utf-8')
    df = read_interval_file(str(p))
    assert df.loc[0, 'T01'] == 1.0
    assert df.loc[0, 'D02'] == 2.0

def test_validate_interval_data_success():
    signals = [[0]*10, [1]*10]
    df = pd.DataFrame({'T01':[40,40], 'D02':[80,80]})
    layer_top, layer_bottom = validate_interval_data(signals, df)
    assert layer_top == [1,1]
    assert layer_bottom == [2,2]

def test_validate_interval_data_mismatch():
    signals = [[0]*10]
    df = pd.DataFrame({'T01':[40,40], 'D02':[80,80]})
    try:
        validate_interval_data(signals, df)
    except ValueError:
        assert True
    else:
        assert False

def test_compute_parameters_basic():
    signals = [
        [0,1,2,3,4],
        [4,3,2,1,0]
    ]
    layer_top = [1,1]
    layer_bottom = [3,4]
    result = compute_parameters(signals, layer_top, layer_bottom, [], [])
    assert result['T_top'] == [8,8]
    assert result['A_top'][0] == 1
    assert result['A_bottom'][1] == 0
