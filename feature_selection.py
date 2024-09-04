from func import *

from regression import *
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest, SelectPercentile, f_regression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler
from boruta import BorutaPy

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)



# def plot_correlation_subplots(corr_matrix, step=50):
#     n_features = corr_matrix.shape[0]
#     n_plots = (n_features // step) + (1 if n_features % step != 0 else 0)  # Количество графиков
#
#     fig, axes = plt.subplots(n_plots, 1, figsize=(20, 20 * n_plots))
#     fig.suptitle('Correlation Heatmap Subplots', fontsize=20)
#
#     for i in range(n_plots):
#         start_idx = i * step
#         end_idx = min((i + 1) * step, n_features)
#         sub_corr = corr_matrix.iloc[start_idx:end_idx, start_idx:end_idx]
#
#         ax = axes[i] if n_plots > 1 else axes  # если график один, `axes` не является списком
#         sns.heatmap(sub_corr, square=True, annot=True, fmt='.2f', linecolor='black', ax=ax)
#         ax.set_title(f'Features {start_idx} to {end_idx - 1}')
#         ax.set_xticklabels(sub_corr.columns, rotation=45)
#         ax.set_yticklabels(sub_corr.index, rotation=45)
#
#     plt.tight_layout(rect=(0, 0.03, 1, 0.95)) # чтобы разместить заголовок сверху
#     plt.show()

def add_param_to_list(list_param, ui_fs):
    ui_fs.listWidget_features.clear()
    for i in list_param:
        ui_fs.listWidget_features.addItem(i)


def feature_selection(data, target):
    FeatSelect = QtWidgets.QDialog()
    ui_fs = Ui_FeatureSelection()
    ui_fs.setupUi(FeatSelect)
    FeatSelect.show()
    FeatSelect.setAttribute(QtCore.Qt.WA_DeleteOnClose)

    data = data.fillna(data.median())
    print('missing values', data.isna().sum())
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

            data['target'] = target
            corr = data.corr()
            corr_with_target = corr['target']

            k = 15
            top_k = corr_with_target.abs().sort_values(ascending=True)[:k].index
            selected_features = data[top_k]
            print("Отобранные признаки:", top_k.to_list())
            print(len(top_k.to_list()))

            selected_corr_matrix = selected_features.corr()
            plt.figure(figsize=(10, 8))
            sns.heatmap(selected_corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
            plt.title('Heatmap for Top Selected Features')
            plt.xticks(rotation=45)
            plt.yticks(rotation=45)
            plt.show()



            # print(corr)
            #
            # threshold = 0.95
            # strong_corr = np.where((corr > threshold) | (corr < -threshold))
            # strong_pairs = set([(data.columns[i], data.columns[j], corr.iloc[i, j])
            #                     for i, j in zip(*strong_corr) if i != j and i < j])
            #
            # print("Признаки с сильной корреляцией (выше порога):")
            # for pair in strong_pairs:
            #     print(f"{pair[0]} и {pair[1]}: корреляция = {pair[2]:.2f}")
            #
            # print('Количество уникальных пар с сильной корреляцией:', len(strong_pairs))
            # print('\n\n')
            #
            # columns_to_drop = set()
            # for col1, col2, _ in strong_pairs:
            #     if col1 not in columns_to_drop and col2 not in columns_to_drop:
            #         columns_to_drop.add(col2)
            # data_reduced = data.drop(columns=columns_to_drop)
            #
            # print('Оставшиеся признаки после удаления коррелированных:', data_reduced.columns.to_list())
            # print('Количество оставшихся признаков:', len(data_reduced.columns))

        elif ui_fs.comboBox_method.currentText() == 'Forward Selection':
            print('Forward Selection: ')

            X_train, X_test, y_train, y_test = train_test_split(
                data,
                target,
                test_size=0.3,
                random_state=0)

            print('X_train.shape', X_train.shape)
            X_train.fillna(X_train.median(), inplace=True)
            X_test.fillna(X_train.median(), inplace=True)

            rf = RandomForestRegressor(n_estimators=100, random_state=0)
            rf.fit(X_train, y_train)

            y_pred = rf.predict(X_test)
            print('R^2:', r2_score(y_test, y_pred))

            sfs1 = SFS(RandomForestRegressor(),
                       k_features=50,
                       forward=True,
                       floating=False,
                       verbose=2,
                       scoring='r2',
                       cv=3)

            X_new = sfs1.fit_transform(X_train, y_train)
            X_test_new = sfs1.transform(X_test)

            print('sfs1.k_feature_names_:', sfs1.k_feature_names_)
            print('sfs1.k_feature_idx_:', sfs1.k_feature_idx_)

            rf1 = RandomForestRegressor(n_estimators=100, random_state=0)
            rf1.fit(X_new, y_train)

            y_pred = rf1.predict(X_test_new)
            print('R^2 after feature selection:', r2_score(y_test, y_pred))

        elif ui_fs.comboBox_method.currentText() == 'Backward Selection':
            print('Backward Selection: ')

            X_train, X_test, y_train, y_test = train_test_split(
                data.iloc[:, :100],
                target,
                test_size=0.3,
                random_state=0)

            print('X_train.shape', X_train.shape)
            X_train.fillna(X_train.median(), inplace=True)
            X_test.fillna(X_train.median(), inplace=True)

            rf = RandomForestRegressor(n_estimators=100, random_state=0)
            rf.fit(X_train, y_train)

            y_pred = rf.predict(X_test)
            print('R^2:', r2_score(y_test, y_pred))

            sfs1 = SFS(RandomForestRegressor(),
                       k_features=20,
                       forward=False,
                       floating=False,
                       verbose=2,
                       scoring='r2',
                       cv=3)

            X_new = sfs1.fit_transform(X_train, y_train)
            X_test_new = sfs1.transform(X_test)

            print('sfs1.k_feature_names_:', sfs1.k_feature_names_)
            print('sfs1.k_feature_idx_:', sfs1.k_feature_idx_)

            rf1 = RandomForestRegressor(n_estimators=100, random_state=0)
            rf1.fit(X_new, y_train)

            y_pred = rf1.predict(X_test_new)
            print('R^2 after feature selection:', r2_score(y_test, y_pred))


        elif ui_fs.comboBox_method.currentText() == 'LASSO':
            print('LASSO Regression: ')

            X_train, X_test, y_train, y_test = train_test_split(
                data,
                target,
                test_size=0.3,
                random_state=0)

            print('X_train.shape', X_train.shape)

            scaler = StandardScaler()
            scaler.fit(X_train.fillna(0))

            sel_ = SelectFromModel(Lasso(alpha=10))
            sel_.fit(scaler.transform(X_train.fillna(0)), y_train)

            selected_feat = X_train.columns[(sel_.get_support())]

            print('total features: {}'.format((X_train.shape[1])))
            print('selected features: {}'.format(len(selected_feat)))
            print('features with coefficients shrank to zero: {}'.format(
                np.sum(sel_.estimator_.coef_ == 0)))

            add_param_to_list(selected_feat, ui_fs)
            ui_fs.label_params.setText(f'Отобранные параметры ({len(selected_feat)})')
            ui_fs.plainTextEdit_results.setPlainText(f'total features: {X_train.shape[1]}\n'
                                                     f'selected features: {len(selected_feat)}\n'
                                                     f'features with coefficients shrank to zero: '
                                                     f'{np.sum(sel_.estimator_.coef_ == 0)}')

        elif ui_fs.comboBox_method.currentText() == 'Random Forest':
            print('Random Forest: ')
            X_train, X_test, y_train, y_test = train_test_split(
                data,
                target,
                test_size=0.3,
                random_state=0)

            rf = RandomForestRegressor(n_estimators=100, random_state=0)
            rf.fit(X_train, y_train)

            y_pred = rf.predict(X_test)
            r2_all = r2_score(y_test, y_pred)
            print('R^2:', r2_all)

            plt.figure(num=None, figsize=(15, 10), dpi=80, facecolor='w', edgecolor='k')
            feat_importances = pd.Series(rf.feature_importances_, index=X_train.columns)
            feat_importances.nlargest(20).plot(kind='barh')
            plt.show()

            num_params = ui_fs.spinBox_num_param.value()

            top_features = feat_importances.nlargest(num_params).index
            list_param = feat_importances.nlargest(num_params).index.tolist()

            X_train_top = X_train[top_features]
            X_test_top = X_test[top_features]

            rf1 = RandomForestRegressor(n_estimators=100, random_state=0)
            rf1.fit(X_train_top, y_train)
            y_pred = rf1.predict(X_test_top)

            print('R^2 after feature selection:', r2_score(y_test, y_pred))
            add_param_to_list(list_param, ui_fs)
            ui_fs.label_params.setText(f'Отобранные параметры, ранжированные ({len(list_param)})')
            ui_fs.plainTextEdit_results.setPlainText('Все параметры R^2: ' + str(round(r2_all, 4)) + \
                                        '\n' + 'Отобранные параметры R^2: ' + str(round(r2_score(y_test, y_pred), 4)))

        elif ui_fs.comboBox_method.currentText() == 'Boruta':
            print('Boruta: ')

            X_train, X_test, y_train, y_test = train_test_split(
                data,
                target,
                test_size=0.3,
                random_state=0)

            rf_all_features = RandomForestRegressor(random_state=123, n_estimators=1000, max_depth=5)
            rf_all_features.fit(X_train, y_train)
            r2_all = r2_score(y_test, rf_all_features.predict(X_test))
            print('All features R^2: ', r2_all)

            rfc = RandomForestRegressor(random_state=1, n_estimators=1000, max_depth=5)
            boruta_selector = BorutaPy(rfc, n_estimators='auto', verbose=1, random_state=1, max_iter=10)
            boruta_selector.fit(np.array(data), np.array(target))

            print("No. of significant features: ", boruta_selector.n_features_)
            selected_rf_features = pd.DataFrame({'Feature': list(data.columns),
                                                 'Ranking': boruta_selector.ranking_}).sort_values(by='Ranking')
            ranking_counts = selected_rf_features['Ranking'].value_counts().sort_index()
            print("Количество признаков в категории 1:")
            features_with_ranking_1 = selected_rf_features[selected_rf_features["Ranking"] == 1]['Feature']
            list_param = features_with_ranking_1.tolist()
            print(list_param)

            new_columns = selected_rf_features[selected_rf_features["Ranking"] == 1].index
            X_new = data.iloc[:, new_columns]

            X_train, X_test, y_train, y_test = train_test_split(
                X_new,
                target,
                test_size=0.3,
                random_state=0)

            rf = RandomForestRegressor(n_estimators=100, random_state=0)
            rf.fit(X_train, y_train)

            y_pred = rf.predict(X_test)
            print('Final R^2:', r2_score(y_test, y_pred))
            print('columns: ', new_columns)

            ui_fs.label_params.setText(f'Отобранные параметры, ранжированные ({len(list_param)})')
            add_param_to_list(list_param, ui_fs)
            confirmed = np.sum(boruta_selector.support_)
            tentative = np.sum(boruta_selector.support_weak_)
            rejected = len(boruta_selector.support_) - confirmed - tentative
            ui_fs.plainTextEdit_results.setPlainText(f'confirmed: {confirmed}\ntentative: {tentative}\nrejected: {rejected}\n')
            ui_fs.plainTextEdit_results.appendPlainText('Все параметры R^2: ' + str(round(r2_all, 4)) + \
                                                    '\n' + 'Отобранные параметры R^2: ' + str(round(r2_score(y_test, y_pred), 4)))




    ui_fs.pushButton_select_features.clicked.connect(choose_method)
    FeatSelect.exec_()

