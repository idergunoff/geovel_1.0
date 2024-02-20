import json
from math import sqrt

import matplotlib.pyplot as plt
import pandas as pd

from func import *


def interpolate(grid, x, y):
    # Find the four nearest points in the grid
    nearest_points = sorted(grid, key=lambda point: (point[0] - x) ** 2 + (point[1] - y) ** 2)[:4]

    # If the point is exactly on one of the grid points, return its value
    for point in nearest_points:
        if point[0] == x and point[1] == y:
            return point[2]

    # Otherwise, perform linear interpolation
    total_value = 0
    total_weight = 0

    for point in nearest_points:
        distance = ((point[0] - x) ** 2 + (point[1] - y) ** 2) ** 0.5
        weight = 1 / distance
        total_value += point[2] * weight
        total_weight += weight

    value = total_value / total_weight

    return value



def interpolate_pandas(pd_grid, x, y):
    # Find the four nearest points in the grid
    pd_grid['dist'] = ((pd_grid['x'] - x) ** 2 + (pd_grid['y'] - y) ** 2) ** 0.5
    near_grid = pd_grid.sort_values('dist', ascending=True)[:20]
    near_grid['weight'] = 1 / near_grid['dist']
    value = (near_grid['value'] * near_grid['weight']).sum() / near_grid['weight'].sum()

    return value



grid = session.query(Grid).filter_by(id=1).first()
grid_relief = json.loads(grid.grid_table_r)
pd_grid = pd.DataFrame(grid_relief, columns=['x', 'y', 'value'])

pr = session.query(Profile).filter_by(id=2).first()
list_x = json.loads(pr.x_pulc)
list_y = json.loads(pr.y_pulc)

list_relief = []

for i in range(len(list_x)):
    list_relief.append(interpolate_pandas(pd_grid, list_x[i], list_y[i]))


plt.plot(list_relief)
plt.show()







