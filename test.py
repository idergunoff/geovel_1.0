# def create_grid_points(x, y):
#     x_new = []
#     y_new = []
#
#     for i in range(len(x)):
#         x_point = x[i]
#         y_point = y[i]
#
#         # Создаем новые точки сетки вокруг заданных координат
#         for x_grid in range(x_point - 25, x_point + 26, 5):
#             for y_grid in range(y_point - 25, y_point + 26, 5):
#                 x_new.append(x_grid)
#                 y_new.append(y_grid)
#
#     return x_new, y_new
#
# list_x, list_y = [25], [125]
#
# x_new, y_new = create_grid_points(list_x, list_y)
# print(len(x_new))
# print(len(y_new))
#
#
# target_train = [1, 2, 3]
#
# target_train_new = [[i] * 5 for i in target_train]
# target_train = []
# for i in target_train_new:
#     target_train.extend(i)
#
# print(target_train)

def parse_range(input_str):
    result = set()

    ranges = input_str.split(',')
    for i in ranges:
        try:
            if '-' in i:
                start, end = map(int, i.split('-'))
                if 1 <= start <= 512 and 1 <= end <= 512:
                    result.update(range(start, end + 1))
            else:
                num = int(i)
                if 1 <= num <= 512:
                    result.add(num)
        except:
            print("Incorrect input")
            return
    res_list = list(result)
    return res_list


print(parse_range("512,1,3,3-8,10,22"))
