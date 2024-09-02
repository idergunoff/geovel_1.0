from func import *

from regression import *
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest, SelectPercentile, f_regression, chi2


def feature_selection(data, target):
    FeatSelect = QtWidgets.QDialog()
    ui_fs = Ui_FeatureSelection()
    ui_fs.setupUi(FeatSelect)
    FeatSelect.show()
    FeatSelect.setAttribute(QtCore.Qt.WA_DeleteOnClose)

    print('training_sample\n', data)
    print(type(data))
    print('target\n', target)
    print(type(target))

    print('shape:', data.shape)

    def choose_method():
        if ui_fs.comboBox_method.currentText() == 'Quasi-constant':
            print('Quasi-constant: ')
            sel = VarianceThreshold(threshold=0.01)
            sel.fit(data)

            print('get support')
            print(len(data.columns[sel.get_support()]))

            print('print the constant features')
            print(
                len([
                    x for x in data.columns
                    if x not in data.columns[sel.get_support()]
                ]))

            print([x for x in data.columns if x not in data.columns[sel.get_support()]])
            print('\n\n')

        elif ui_fs.comboBox_method.currentText() == 'SelectKBest':
            print('SelectKBest: ')
            sel = SelectKBest(f_regression, k=20)
            sel.fit(data, target)

            selected_indices = sel.get_support(indices=True)
            selected_feature_names = data.columns[selected_indices]
            selected_scores = sel.scores_[selected_indices]

            sorted_indices = np.argsort(selected_scores)[::-1]
            sorted_feature_names = selected_feature_names[sorted_indices]

            print("Индексы отобранных признаков:", selected_indices)
            print("Имена отобранных признаков:", sorted_feature_names.to_list())
            print('\n\n')

        elif ui_fs.comboBox_method.currentText() == 'SelectPercentile':
            print('SelectPercentile: ')
            sel = SelectPercentile(f_regression, percentile=20)
            sel.fit(data, target)

            selected_indices = sel.get_support(indices=True)
            selected_feature_names = data.columns[selected_indices]
            selected_scores = sel.scores_[selected_indices]

            sorted_indices = np.argsort(selected_scores)[::-1]
            sorted_feature_names = selected_feature_names[sorted_indices]

            print("Индексы отобранных признаков:", selected_indices)
            print("Имена отобранных признаков:", sorted_feature_names.to_list())
            print('\n\n')

        elif ui_fs.comboBox_method.currentText() == 'Correlation':
            print('Correlation: ')
            corr = data.corr()
            print(corr)

            plt.figure(figsize=(20, 20))
            plt.title('Correlation Heatmap of Iris Dataset')
            a = sns.heatmap(corr, square=True, annot=True, fmt='.2f', linecolor='black')
            a.set_xticklabels(a.get_xticklabels(), rotation=30)
            a.set_yticklabels(a.get_yticklabels(), rotation=30)
            plt.show()

            threshold = 0.8

            # Фильтрация пар с корреляцией выше порога, исключая диагональные элементы (корреляция признака с самим собой)
            strong_corr = np.where((corr > threshold) | (corr < -threshold))
            strong_pairs = [(data.columns[i], data.columns[j], corr.iloc[i, j])
                            for i, j in zip(*strong_corr) if i != j and i < j]

            # Вывод пар признаков с сильной корреляцией
            print("Признаки с сильной корреляцией (выше порога):")
            for pair in strong_pairs:
                print(f"{pair[0]} и {pair[1]}: корреляция = {pair[2]:.2f}")

            print('\n\n')

    ui_fs.pushButton_select_features.clicked.connect(choose_method)
    FeatSelect.exec_()

