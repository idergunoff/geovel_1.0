from func import *
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest, f_regression, chi2, mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler
from boruta import BorutaPy
from collections import defaultdict

from build_table import *

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)


def update_list_param_mlp(db=False):
    start_time = datetime.datetime.now()
    data_train, list_param = build_table_train(db, 'mlp')
    list_marker = get_list_marker_mlp('georadar')
    ui.listWidget_param_mlp.clear()
    list_param_mlp = data_train.columns.tolist()[2:]
    print('list_param_mlp', list_param_mlp)
    for param in list_param_mlp:
        if ui.checkBox_kf.isChecked():
            groups = []
            for mark in list_marker:
                groups.append(data_train[data_train['mark'] == mark][param].values.tolist())
            F, p = f_oneway(*groups)
            if np.isnan(F) or np.isnan(p):
                ui.listWidget_param_mlp.addItem(param)
                continue
            ui.listWidget_param_mlp.addItem(f'{param} \t\tF={round(F, 2)} p={round(p, 3)}')
            if F < 1 or p > 0.05:
                i_item = ui.listWidget_param_mlp.findItems(f'{param} \t\tF={round(F, 2)} p={round(p, 3)}', Qt.MatchContains)[0]
                i_item.setBackground(QBrush(QColor('red')))
        else:
            ui.listWidget_param_mlp.addItem(param)
    ui.label_count_param_mlp.setText(f'<i><u>{ui.listWidget_param_mlp.count()}</u></i> параметров')
    set_color_button_updata()



def set_color_button_updata():
    mlp = session.query(AnalysisMLP).filter(AnalysisMLP.id == get_MLP_id()).first()
    btn_color = 'background-color: rgb(191, 255, 191);' if mlp.up_data else 'background-color: rgb(255, 185, 185);'
    ui.pushButton_updata_mlp.setStyleSheet(btn_color)

def update_all_params(list_param):
    session.query(ParameterMLP).filter_by(analysis_id=get_MLP_id()).delete()
    session.query(AnalysisMLP).filter_by(id=get_MLP_id()).update({'up_data': False}, synchronize_session='fetch')
    session.commit()

    count = session.query(ParameterMLP).filter_by(analysis_id=get_MLP_id()).count()
    print('clean count', count)

    for param in list_param:
        new_param = f'{param}'
        print(new_param)
        new_param_mlp = ParameterMLP(analysis_id=get_MLP_id(), parameter=new_param)
        session.add(new_param_mlp)
    session.query(AnalysisMLP).filter_by(id=get_MLP_id()).update({'up_data': False}, synchronize_session='fetch')
    session.commit()

    count = session.query(ParameterMLP).filter_by(analysis_id=get_MLP_id()).count()
    print('added count', count)

    set_color_button_updata()
    update_list_param_mlp()

def add_param_to_list(list_param, ui_fs):
    ui_fs.listWidget_features.clear()
    for i in list_param:
        ui_fs.listWidget_features.addItem(i)

def train_model(X_train, X_test, y_train, y_test, mode):
    if mode == 'reg':
        rf = RandomForestRegressor(n_estimators=1000, random_state=0)
        rf.fit(X_train, y_train)
        res = r2_score(y_test, rf.predict(X_test))
    elif mode == 'classif':
        rf = RandomForestClassifier(n_estimators=1000, random_state=0)
        rf.fit(X_train, y_train)
        res = accuracy_score(y_test, rf.predict(X_test))

    return(res, rf)

list_param = []

def feature_selection_calc(data, target, mode):
    FeatSelect = QtWidgets.QDialog()
    ui_fs = Ui_FeatureSelection()
    ui_fs.setupUi(FeatSelect)
    FeatSelect.show()
    FeatSelect.setAttribute(QtCore.Qt.WA_DeleteOnClose)

    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.fillna(data.median())
    target = pd.to_numeric(target, errors='coerce')
    target = target.replace([np.inf, -np.inf], np.nan)
    list_param = []

    if mode == 'reg':
        metric = f_regression
    else:
        metric = mutual_info_classif

    def choose_method():
        global list_param
        if ui_fs.comboBox_method.currentText() == 'Quasi-constant':
            threshold = ui_fs.doubleSpinBox_threshold.value()
            print('Quasi-constant: ')
            sel = VarianceThreshold(threshold=threshold)
            sel.fit(data)

            print('get support')
            print(len(data.columns[sel.get_support()]))

            print('print the constant features')
            num_null = len([
                    x for x in data.columns
                    if x not in data.columns[sel.get_support()]
                ])

            null_param = [x for x in data.columns if x not in data.columns[sel.get_support()]]
            list_param = [x for x in data.columns if x in data.columns[sel.get_support()]]
            ui_fs.label_params.setText(f'Отобранные параметры с дисперсией больше погора ({len(list_param)})')
            add_param_to_list(list_param, ui_fs)
            ui_fs.plainTextEdit_results.setPlainText(f'Параметры с низкой дисперсией ({num_null}):\n{null_param}')
            print('\n\n')

        elif ui_fs.comboBox_method.currentText() == 'SelectKBest':
            ui_fs.plainTextEdit_results.clear()
            print('SelectKBest: ')
            param = ui_fs.spinBox_num_param.value()
            sel = SelectKBest(metric, k=param)
            sel.fit(data, target)

            selected_indices = sel.get_support(indices=True)
            selected_feature_names = data.columns[selected_indices]
            selected_scores = sel.scores_[selected_indices]

            sorted_indices = np.argsort(selected_scores)[::-1]
            sorted_feature_names = selected_feature_names[sorted_indices]

            print("Индексы отобранных признаков:", selected_indices)
            print("Имена отобранных признаков:", sorted_feature_names.to_list())
            list_param = sorted_feature_names.to_list()

            ui_fs.label_params.setText(f'Отобранные параметры, ранжированные ({len(list_param)})')
            add_param_to_list(list_param, ui_fs)
            print('\n\n')

        elif ui_fs.comboBox_method.currentText() == 'Correlation':
            ui_fs.plainTextEdit_results.clear()
            print('Correlation: ')
            data_c = data.copy()
            data_no_target = data.copy()

            data_c['target'] = target
            corr = data_c.corr()
            corr_with_target = corr['target']

            top_k = corr_with_target.abs().sort_values(ascending=False)[:ui_fs.spinBox_num_param.value()].index
            selected_features = data_c[top_k]
            print("Отобранные признаки с высокой корреляцией с target:", top_k.to_list())
            print(len(top_k.to_list()))

            selected_corr_matrix = selected_features.corr()
            plt.figure(figsize=(10, 8))
            sns.heatmap(selected_corr_matrix, annot=False, cmap='coolwarm', fmt='.1f', linewidths=0.1)
            plt.title('Тепловая карта корреляции k признаков с target с наиболее высокой корреляцией')
            plt.xticks(rotation=45)
            plt.yticks(rotation=45)
            plt.show()

            corr_no_target = data_no_target.corr()

            threshold = 0.95
            strong_corr = np.where((corr_no_target > threshold) | (corr_no_target < -threshold))
            strong_pairs = set([(data.columns[i], data.columns[j], corr_no_target.iloc[i, j])
                                for i, j in zip(*strong_corr) if i != j and i < j])

            print("Признаки с сильной корреляцией (выше порога):")
            for pair in strong_pairs:
                print(f"{pair[0]} и {pair[1]}: корреляция = {pair[2]:.2f}")

            print('Количество уникальных пар с сильной корреляцией:', len(strong_pairs))
            print('\n\n')

            correlation_counts = defaultdict(int)
            for col1, col2, _ in strong_pairs:
                correlation_counts[col1] += 1
                correlation_counts[col2] += 1

            columns_to_drop = {col for col, count in correlation_counts.items() if count >= 2}

            data_reduced = data.drop(columns=columns_to_drop)
            list_param = data_reduced.columns.to_list()
            corr_reduced = data_reduced.iloc[:, :ui_fs.spinBox_num_param.value()].corr()

            print('Оставшиеся признаки после удаления коррелированных больше чем с n признаками:', data_reduced.columns.to_list())
            print('Количество оставшихся признаков:', len(data_reduced.columns))
            print(f'Количество элементов в columns_to_drop: {len(columns_to_drop)}')

            corr_sum = corr_reduced.abs().sum().sort_values(ascending=False)
            top_features = corr_sum.index[:ui_fs.spinBox_num_param.value()]
            corr_top_reduced = data_reduced[top_features].corr()

            plt.figure(figsize=(10, 8))
            sns.heatmap(corr_top_reduced, annot=False, cmap='coolwarm', linewidths=0.1)
            plt.title('Тепловая карта сильной корреляции для признаков после удаления коррелированных больше чем с n признаками:')
            plt.xticks(rotation=45)
            plt.yticks(rotation=45)
            plt.show()

            ui_fs.label_params.setText(f'Отобранные параметры, ранжированные ({len(list_param)})')
            add_param_to_list(list_param, ui_fs)


            # plt.figure(figsize=(10, 8))
            # sns.heatmap(corr_reduced, annot=False, cmap='coolwarm', linewidths=0.1)
            # plt.title('Тепловая карта корреляции после удаления признаков')
            # plt.xticks(rotation=45)
            # plt.yticks(rotation=45)
            # plt.show()

        elif ui_fs.comboBox_method.currentText() == 'Forward Selection':
            print('Forward Selection: ')
            param = ui_fs.spinBox_num_param.value()
            X_train, X_test, y_train, y_test = train_test_split(
                data,
                target,
                test_size=0.3,
                random_state=0)

            print('X_train.shape', X_train.shape)
            X_train.fillna(X_train.median(), inplace=True)
            X_test.fillna(X_train.median(), inplace=True)

            r2_all, _ = train_model(X_train, X_test, y_train, y_test, mode)
            print('R^2:', r2_all)

            if mode == 'reg':
                model = RandomForestRegressor()
            elif mode == 'classif':
                model = RandomForestClassifier()
            sfs1 = SFS(model,
                       k_features=param,
                       forward=True,
                       floating=False,
                       verbose=2,
                       scoring='r2',
                       cv=3)

            X_new = sfs1.fit_transform(X_train, y_train)
            X_test_new = sfs1.transform(X_test)

            print('sfs1.k_feature_names_:', sfs1.k_feature_names_)
            print('sfs1.k_feature_idx_:', sfs1.k_feature_idx_)

            r2_param, rf1 = train_model(X_new, X_test_new, y_train, y_test, mode)

            feature_importances = pd.Series(rf1.feature_importances_, index=list(sfs1.k_feature_names_))
            list_param = feature_importances.sort_values(ascending=False)
            list_param = list(list_param.index)

            print('R^2 after feature selection:', r2_param)
            add_param_to_list(list_param, ui_fs)
            ui_fs.label_params.setText(f'Отобранные параметры, ранжированные ({len(list_param)})')
            ui_fs.plainTextEdit_results.setPlainText('Все параметры R^2: ' + str(round(r2_all, 4)) + \
                                                     '\n' + 'Отобранные параметры R^2: ' + str(
                round(r2_param, 4)))

        elif ui_fs.comboBox_method.currentText() == 'Backward Selection':
            print('Backward Selection: ')
            param = ui_fs.spinBox_num_param.value()
            X_train, X_test, y_train, y_test = train_test_split(
                data,
                target,
                test_size=0.3,
                random_state=0)

            print('X_train.shape', X_train.shape)
            X_train.fillna(X_train.median(), inplace=True)
            X_test.fillna(X_train.median(), inplace=True)

            r2_all, _ = train_model(X_train, X_test, y_train, y_test, mode)
            print('R^2:', r2_all)

            if mode == 'reg':
                model = RandomForestRegressor()
            elif mode == 'classif':
                model = RandomForestClassifier()
            sfs1 = SFS(model,
                       k_features=param,
                       forward=False,
                       floating=False,
                       verbose=2,
                       scoring='r2',
                       cv=3)

            X_new = sfs1.fit_transform(X_train, y_train)
            X_test_new = sfs1.transform(X_test)

            print('sfs1.k_feature_names_:', sfs1.k_feature_names_)
            print('sfs1.k_feature_idx_:', sfs1.k_feature_idx_)

            r2_param, rf1 = train_model(X_new, X_test_new, y_train, y_test, mode)
            feature_importances = pd.Series(rf1.feature_importances_, index=list(sfs1.k_feature_names_))
            list_param = feature_importances.sort_values(ascending=False)
            list_param = list(list_param.index)

            print('R^2 after feature selection:', r2_param)

            add_param_to_list(list_param, ui_fs)
            ui_fs.label_params.setText(f'Отобранные параметры, ранжированные ({len(list_param)})')
            ui_fs.plainTextEdit_results.setPlainText('Все параметры R^2: ' + str(round(r2_all, 4)) + \
                                                     '\n' + 'Отобранные параметры R^2: ' + str(
                round(r2_param, 4)))

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

            list_param = X_train.columns[(sel_.get_support())]

            print('total features: {}'.format((X_train.shape[1])))
            print('selected features: {}'.format(len(list_param)))
            print('features with coefficients shrank to zero: {}'.format(
                np.sum(sel_.estimator_.coef_ == 0)))

            add_param_to_list(list_param, ui_fs)
            ui_fs.label_params.setText(f'Отобранные параметры ({len(list_param)})')
            ui_fs.plainTextEdit_results.setPlainText(f'total features: {X_train.shape[1]}\n'
                                                     f'selected features: {len(list_param)}\n'
                                                     f'features with coefficients shrank to zero: '
                                                     f'{np.sum(sel_.estimator_.coef_ == 0)}')

        elif ui_fs.comboBox_method.currentText() == 'Random Forest':
            print('Random Forest: ')
            X_train, X_test, y_train, y_test = train_test_split(
                data,
                target,
                test_size=0.3,
                random_state=0)

            r2_all, rf = train_model(X_train, X_test, y_train, y_test, mode)
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

            r2_param, _ = train_model(X_train_top, X_test_top, y_train, y_test, mode)
            print('R^2 after feature selection:', r2_param)
            add_param_to_list(list_param, ui_fs)
            ui_fs.label_params.setText(f'Отобранные параметры, ранжированные ({len(list_param)})')
            ui_fs.plainTextEdit_results.setPlainText('Все параметры R^2: ' + str(round(r2_all, 4)) + \
                                        '\n' + 'Отобранные параметры R^2: ' + str(round(r2_param, 4)))

        elif ui_fs.comboBox_method.currentText() == 'Boruta':
            print('Boruta: ')
            iter = ui_fs.spinBox_num_param.value()

            X_train, X_test, y_train, y_test = train_test_split(
                data,
                target,
                test_size=0.3,
                random_state=0)

            r2_all, _ = train_model(X_train, X_test, y_train, y_test, mode)
            print('All features R^2: ', r2_all)

            if mode == 'reg':
                rfc = RandomForestRegressor(random_state=1, n_estimators=1000, max_depth=5)
            elif mode =='classif':
                rfc = RandomForestClassifier(random_state=1, n_estimators=1000, max_depth=5)
            boruta_selector = BorutaPy(rfc, n_estimators='auto', verbose=1, random_state=1, max_iter=iter)
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
            if len(selected_rf_features[selected_rf_features["Ranking"] == 1]) == 0:
                ui_fs.plainTextEdit_results.appendPlainText('Количество отобранных параметров равно 0')
                return

            X_new = data.iloc[:, new_columns]

            X_train, X_test, y_train, y_test = train_test_split(
                X_new,
                target,
                test_size=0.3,
                random_state=0)


            r2_param, _ = train_model(X_train, X_test, y_train, y_test, mode)

            print('Final R^2:', r2_param)
            print('columns: ', new_columns)

            ui_fs.label_params.setText(f'Отобранные параметры, ранжированные ({len(list_param)})')
            add_param_to_list(list_param, ui_fs)
            confirmed = np.sum(boruta_selector.support_)
            tentative = np.sum(boruta_selector.support_weak_)
            rejected = len(boruta_selector.support_) - confirmed - tentative
            ui_fs.plainTextEdit_results.setPlainText(f'confirmed: {confirmed}\ntentative: {tentative}\nrejected: {rejected}\n')
            ui_fs.plainTextEdit_results.appendPlainText('Все параметры R^2: ' + str(round(r2_all, 4)) + \
                                                    '\n' + 'Отобранные параметры R^2: ' + str(round(r2_param, 4)))

    def import_params():
        global list_param
        print('list_param: ', list_param)
        update_all_params(list_param)

    ui_fs.pushButton_select_features.clicked.connect(choose_method)
    ui_fs.pushButton_import_param.clicked.connect(import_params)
    FeatSelect.exec_()

