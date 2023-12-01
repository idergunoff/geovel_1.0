def create_grid_points(x, y):
    x_new = []
    y_new = []

    for i in range(len(x)):
        x_point = x[i]
        y_point = y[i]

        # Создаем новые точки сетки вокруг заданных координат
        for x_grid in range(x_point - 25, x_point + 26, 5):
            for y_grid in range(y_point - 25, y_point + 26, 5):
                x_new.append(x_grid)
                y_new.append(y_grid)

    return x_new, y_new

list_x, list_y = [25], [125]

x_new, y_new = create_grid_points(list_x, list_y)
print(len(x_new))
print(len(y_new))