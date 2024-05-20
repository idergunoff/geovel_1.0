import json
from math import sqrt

import matplotlib.pyplot as plt
import pandas as pd

from func import *



# def interpolate(grid, x, y):
#     # Find the four nearest points in the grid
#     nearest_points = sorted(grid, key=lambda point: (point[0] - x) ** 2 + (point[1] - y) ** 2)[:4]
#
#     # If the point is exactly on one of the grid points, return its value
#     for point in nearest_points:
#         if point[0] == x and point[1] == y:
#             return point[2]
#
#     # Otherwise, perform linear interpolation
#     total_value = 0
#     total_weight = 0
#
#     for point in nearest_points:
#         distance = ((point[0] - x) ** 2 + (point[1] - y) ** 2) ** 0.5
#         weight = 1 / distance
#         total_value += point[2] * weight
#         total_weight += weight
#
#     value = total_value / total_weight
#
#     return value
#
#
#
# def interpolate_pandas(pd_grid, x, y):
#     # Find the four nearest points in the grid
#     pd_grid['dist'] = ((pd_grid['x'] - x) ** 2 + (pd_grid['y'] - y) ** 2) ** 0.5
#     near_grid = pd_grid.sort_values('dist', ascending=True)[:20]
#     near_grid['weight'] = 1 / near_grid['dist']
#     value = (near_grid['value'] * near_grid['weight']).sum() / near_grid['weight'].sum()
#
#     yield value
#
#
#
# time = datetime.datetime.now()
# grid = session.query(Grid).filter_by(id=1).first()
# grid_relief = json.loads(grid.grid_table_r)
# pd_grid = pd.DataFrame(grid_relief, columns=['x', 'y', 'value'])
#
# pr = session.query(Profile).filter_by(id=2).first()
# list_x = json.loads(pr.x_pulc)
# list_y = json.loads(pr.y_pulc)
#
# list_relief = []
#
# for i in range(len(list_x)):
#     list_relief.append(next(interpolate_pandas(pd_grid, list_x[i], list_y[i])))
#
# print(datetime.datetime.now() - time)
# plt.plot(list_relief)
# plt.show()

from collections import Counter
import re

def result_analysis(results):
    sorted_result = sorted(results, key=lambda x: x[0], reverse=True)
    for item in sorted_result:
        print(item)

    twenty_percent = int(len(sorted_result) * 1)
    sorted_result = sorted_result[:twenty_percent]
    result_param = [item[4][0] for item in sorted_result]
    flattened_list = [item for sublist in result_param for item in sublist]
    print('flat length: ', len(flattened_list))
    processed_list = []
    for s in flattened_list:
        parts = s.split('_')
        processed_parts = [re.sub(r'\d+', '', part) for part in parts]
        new_string = '_'.join(processed_parts)
        if new_string.endswith('_') or new_string.endswith('__'):
            new_string = new_string[:-1]
        processed_list.append(new_string)

    # processed_list = list(set)
    print('length: ', len(processed_list))
    param_count = Counter(processed_list)
    print(param_count)
    part_list = int(len(param_count) * 0.2)
    common_param = param_count.most_common(part_list)
    top_params = [item[0] for item in common_param]
    for param in common_param:
        print(f'{param[0]}: {param[1]}')
    # print("top parameters: ", top_params)
    # print("len top parameters: ", len(top_params))
    print("common top parameters: ", common_param)


results = [[0.4562786522233527, [0.47552217367885563, 0.44604704051708655, 0.49205218744389245, 0.4774552636303788, 0.39031659584654976], 0.4697954271961492, [0.4981949458483754, 0.457280385078219, 0.49217809867629364, 0.49217809867629364, 0.4091456077015644], [['sig_Abase_50_20', 'distr_At_10', 'sep_At_10', 'distr_Vt_10', 'mfcc_Vt_13', 'sig_Vt_50_20', 'sig_Pht_50_20', 'distr_Wt_10', 'sig_Wt_50_20', 'sig_diff_50_20', 'sig_CRLNF_50_20', 'distr_CRL_10', 'mfcc_CRL_13', 'sig_CRL_50_20', 'A_top', 'A_bottom', 'dA', 'A_sum', 'A_mean', 'A_max', 'A_T_max', 'A_Sn', 'A_wmf', 'A_Qf', 'A_Sn_wmf', 'T_top', 'T_bottom', 'dT', 'CRL_skew', 'CRL_kurt', 'CRL_std', 'CRL_k_var']]], [0.4891339996409122, [0.47480399784547256, 0.513633371237058, 0.5428272188640854, 0.457561793045664, 0.45684361721228084], 0.4974729241877256, [0.5138387484957883, 0.5042117930204573, 0.5318892900120337, 0.470517448856799, 0.46690734055354993], [['distr_Abase_10', 'sig_Abase_50_20', 'distr_At_10', 'sep_At_10', 'mfcc_At_13', 'sig_At_50_20', 'distr_Vt_10', 'mfcc_Vt_13', 'sig_Vt_50_20', 'sep_Pht_10', 'distr_Wt_10', 'sep_Wt_10', 'mfcc_Wt_13', 'distr_diff_10', 'sep_diff_10', 'mfcc_diff_13', 'sig_diff_50_20', 'distr_CRL_10', 'sep_CRL_10', 'mfcc_CRL_13', 'A_top', 'A_bottom', 'dA', 'A_sum', 'A_mean', 'A_max', 'A_T_max', 'A_Sn', 'A_wmf', 'A_Qf', 'A_Sn_wmf', 'Pht_top', 'dPht', 'Pht_sum', 'Pht_mean', 'Pht_max', 'Pht_T_max', 'Pht_Sn', 'Pht_wmf', 'Pht_Qf', 'Pht_Sn_wmf', 'Wt_top', 'Wt_sum', 'Wt_mean', 'Wt_max', 'Wt_T_max', 'Wt_Sn', 'Wt_wmf', 'Wt_Qf', 'Wt_Sn_wmf', 'skew', 'kurt', 'std', 'k_var', 'CRL_skew', 'CRL_kurt', 'CRL_std', 'CRL_k_var']]], [0.4709749236938177, [0.47302651265784906, 0.4957627625830391, 0.46937578550481784, 0.5312107247591118, 0.3854988329642708], 0.4779783393501805, [0.4825511432009627, 0.48856799037304455, 0.5018050541516246, 0.5126353790613718, 0.4043321299638989], [['sep_Abase_10', 'mfcc_Abase_13', 'sig_Abase_50_20', 'mfcc_At_13', 'sig_At_50_20', 'sep_Vt_10', 'sig_Vt_50_20', 'distr_Pht_10', 'sig_Pht_50_20', 'distr_Wt_10', 'mfcc_Wt_13', 'sep_diff_10', 'mfcc_CRL_13', 'sig_CRL_50_20', 'At_top', 'dAt', 'At_sum', 'At_mean', 'At_max', 'At_T_max', 'At_Sn', 'At_wmf', 'At_Qf', 'At_Sn_wmf', 'Vt_top', 'dVt', 'Vt_sum', 'Vt_mean', 'Vt_max', 'Vt_T_max', 'Vt_Sn', 'Vt_wmf', 'Vt_Qf', 'Vt_Sn_wmf', 'T_top', 'T_bottom', 'dT', 'skew', 'kurt', 'std', 'k_var', 'CRL_skew', 'CRL_kurt', 'CRL_std', 'CRL_k_var']]], [0.5004811778083667, [0.44897360703812317, 0.45580226225387516, 0.5714225866299599, 0.5194326410916272, 0.5067747920282483], 0.5010830324909747, [0.4825511432009627, 0.4416365824308063, 0.542719614921781, 0.5342960288808665, 0.5042117930204573], [['distr_Abase_10', 'sig_Abase_50_20', 'sep_At_10', 'distr_Vt_10', 'distr_Wt_10', 'Vt_top', 'dVt', 'Vt_sum', 'Vt_mean', 'Vt_max', 'Vt_T_max', 'Vt_Sn', 'Vt_wmf', 'Vt_Qf', 'Vt_Sn_wmf', 'T_top', 'T_bottom', 'dT']]], [0.47565982404692086, [0.4290322580645161, 0.49774971572206594, 0.5189837811957627, 0.5032916392363397, 0.4292417260159196], 0.49410348977135987, [0.5006016847172082, 0.5042117930204573, 0.5006016847172082, 0.5054151624548736, 0.45968712394705175], [['distr_Abase_10', 'sig_Abase_50_20', 'sep_At_10', 'mfcc_At_13', 'distr_Vt_10', 'sep_Vt_10', 'mfcc_Vt_13', 'distr_Pht_10', 'sig_Pht_50_20', 'At_top', 'dAt', 'At_sum', 'At_mean', 'At_max', 'At_T_max', 'At_Sn', 'At_wmf', 'At_Qf', 'At_Sn_wmf', 'Pht_top', 'dPht', 'Pht_sum', 'Pht_mean', 'Pht_max', 'Pht_T_max', 'Pht_Sn', 'Pht_wmf', 'Pht_Qf', 'Pht_Sn_wmf', 'CRL_skew', 'CRL_kurt', 'CRL_std', 'CRL_k_var']]]]
[0.5004811778083667, [0.44897360703812317, 0.45580226225387516, 0.5714225866299599, 0.5194326410916272, 0.5067747920282483], 0.5010830324909747, [0.4825511432009627, 0.4416365824308063, 0.542719614921781, 0.5342960288808665, 0.5042117930204573], [['distr_Abase_10', 'sig_Abase_50_20', 'sep_At_10', 'distr_Vt_10', 'distr_Wt_10', 'Vt_top', 'dVt', 'Vt_sum', 'Vt_mean', 'Vt_max', 'Vt_T_max', 'Vt_Sn', 'Vt_wmf', 'Vt_Qf', 'Vt_Sn_wmf', 'T_top', 'T_bottom', 'dT']]]
[0.4891339996409122, [0.47480399784547256, 0.513633371237058, 0.5428272188640854, 0.457561793045664, 0.45684361721228084], 0.4974729241877256, [0.5138387484957883, 0.5042117930204573, 0.5318892900120337, 0.470517448856799, 0.46690734055354993], [['distr_Abase_10', 'sig_Abase_50_20', 'distr_At_10', 'sep_At_10', 'mfcc_At_13', 'sig_At_50_20', 'distr_Vt_10', 'mfcc_Vt_13', 'sig_Vt_50_20', 'sep_Pht_10', 'distr_Wt_10', 'sep_Wt_10', 'mfcc_Wt_13', 'distr_diff_10', 'sep_diff_10', 'mfcc_diff_13', 'sig_diff_50_20', 'distr_CRL_10', 'sep_CRL_10', 'mfcc_CRL_13', 'A_top', 'A_bottom', 'dA', 'A_sum', 'A_mean', 'A_max', 'A_T_max', 'A_Sn', 'A_wmf', 'A_Qf', 'A_Sn_wmf', 'Pht_top', 'dPht', 'Pht_sum', 'Pht_mean', 'Pht_max', 'Pht_T_max', 'Pht_Sn', 'Pht_wmf', 'Pht_Qf', 'Pht_Sn_wmf', 'Wt_top', 'Wt_sum', 'Wt_mean', 'Wt_max', 'Wt_T_max', 'Wt_Sn', 'Wt_wmf', 'Wt_Qf', 'Wt_Sn_wmf', 'skew', 'kurt', 'std', 'k_var', 'CRL_skew', 'CRL_kurt', 'CRL_std', 'CRL_k_var']]]
[0.47565982404692086, [0.4290322580645161, 0.49774971572206594, 0.5189837811957627, 0.5032916392363397, 0.4292417260159196], 0.49410348977135987, [0.5006016847172082, 0.5042117930204573, 0.5006016847172082, 0.5054151624548736, 0.45968712394705175], [['distr_Abase_10', 'sig_Abase_50_20', 'sep_At_10', 'mfcc_At_13', 'distr_Vt_10', 'sep_Vt_10', 'mfcc_Vt_13', 'distr_Pht_10', 'sig_Pht_50_20', 'At_top', 'dAt', 'At_sum', 'At_mean', 'At_max', 'At_T_max', 'At_Sn', 'At_wmf', 'At_Qf', 'At_Sn_wmf', 'Pht_top', 'dPht', 'Pht_sum', 'Pht_mean', 'Pht_max', 'Pht_T_max', 'Pht_Sn', 'Pht_wmf', 'Pht_Qf', 'Pht_Sn_wmf', 'CRL_skew', 'CRL_kurt', 'CRL_std', 'CRL_k_var']]]
[0.4709749236938177, [0.47302651265784906, 0.4957627625830391, 0.46937578550481784, 0.5312107247591118, 0.3854988329642708], 0.4779783393501805, [0.4825511432009627, 0.48856799037304455, 0.5018050541516246, 0.5126353790613718, 0.4043321299638989], [['sep_Abase_10', 'mfcc_Abase_13', 'sig_Abase_50_20', 'mfcc_At_13', 'sig_At_50_20', 'sep_Vt_10', 'sig_Vt_50_20', 'distr_Pht_10', 'sig_Pht_50_20', 'distr_Wt_10', 'mfcc_Wt_13', 'sep_diff_10', 'mfcc_CRL_13', 'sig_CRL_50_20', 'At_top', 'dAt', 'At_sum', 'At_mean', 'At_max', 'At_T_max', 'At_Sn', 'At_wmf', 'At_Qf', 'At_Sn_wmf', 'Vt_top', 'dVt', 'Vt_sum', 'Vt_mean', 'Vt_max', 'Vt_T_max', 'Vt_Sn', 'Vt_wmf', 'Vt_Qf', 'Vt_Sn_wmf', 'T_top', 'T_bottom', 'dT', 'skew', 'kurt', 'std', 'k_var', 'CRL_skew', 'CRL_kurt', 'CRL_std', 'CRL_k_var']]]
[0.4562786522233527, [0.47552217367885563, 0.44604704051708655, 0.49205218744389245, 0.4774552636303788, 0.39031659584654976], 0.4697954271961492, [0.4981949458483754, 0.457280385078219, 0.49217809867629364, 0.49217809867629364, 0.4091456077015644], [['sig_Abase_50_20', 'distr_At_10', 'sep_At_10', 'distr_Vt_10', 'mfcc_Vt_13', 'sig_Vt_50_20', 'sig_Pht_50_20', 'distr_Wt_10', 'sig_Wt_50_20', 'sig_diff_50_20', 'sig_CRLNF_50_20', 'distr_CRL_10', 'mfcc_CRL_13', 'sig_CRL_50_20', 'A_top', 'A_bottom', 'dA', 'A_sum', 'A_mean', 'A_max', 'A_T_max', 'A_Sn', 'A_wmf', 'A_Qf', 'A_Sn_wmf', 'T_top', 'T_bottom', 'dT', 'CRL_skew', 'CRL_kurt', 'CRL_std', 'CRL_k_var']]]

result = result_analysis(results)






