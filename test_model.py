import math

import numpy as np
import pandas as pd
from func import *
from decimal import *


def test_start():
    test_classifModel = QtWidgets.QDialog()
    ui_tm = Ui_FormTestModel()
    ui_tm.setupUi(test_classifModel)
    test_classifModel.show()
    test_classifModel.setAttribute(QtCore.Qt.WA_DeleteOnClose)

    m_width, m_height = get_width_height_monitor()
    test_classifModel.resize(int(m_width/3), m_height - 200)

    def get_test_MLP_id():
        return ui_tm.comboBox_test_analysis.currentText().split(' id')[-1]

    def update_list_test_well():
        ui_tm.listWidget_test_point.clear()
        count_markup, count_measure, count_fake = 0, 0, 0
        for i in session.query(MarkupMLP).filter(MarkupMLP.analysis_id == get_test_MLP_id()).all():
            try:
                fake = len(json.loads(i.list_fake)) if i.list_fake else 0
                measure = len(json.loads(i.list_measure))
                if i.type_markup == 'intersection':
                    try:
                        inter_name = session.query(Intersection.name).filter(Intersection.id == i.well_id).first()[0]
                    except TypeError:
                        session.query(MarkupMLP).filter(MarkupMLP.id == i.id).delete()
                        session.commit()
                        continue
                    item = f'{i.profile.research.object.title} - {i.profile.title} | {i.formation.title} | {inter_name} | {measure - fake} из {measure} | id{i.id}'
                elif i.type_markup == 'profile':
                    item = f'{i.profile.research.object.title} - {i.profile.title} | {i.formation.title} | | {measure - fake} из {measure} | id{i.id}'
                else:
                    item = f'{i.profile.research.object.title} - {i.profile.title} | {i.formation.title} | {i.well.name} | {measure - fake} из {measure} | id{i.id}'
                ui_tm.listWidget_test_point.addItem(item)
                i_item = ui_tm.listWidget_test_point.findItems(item, Qt.MatchContains)[0]
                i_item.setBackground(QBrush(QColor(i.marker.color)))
                count_markup += 1
                count_measure += measure - fake
                count_fake += fake
            except AttributeError:
                session.delete(i)
                session.commit()
        update_analysis_marker()

    def update_test_analysis_combobox():
        ui_tm.comboBox_test_analysis.clear()
        for i in session.query(AnalysisMLP).order_by(AnalysisMLP.title).all():
            ui_tm.comboBox_test_analysis.addItem(f'{i.title} id{i.id}')
            update_list_test_well()
            update_analysis_marker()


    def update_analysis_marker():
        ui_tm.comboBox_mark.clear()
        for i in get_test_list_marker_mlp():
            ui_tm.comboBox_mark.addItem(i)

    def update_test_model_list():
        ui_tm.listWidget_test_model.clear()
        models = session.query(TrainedModelClass).filter_by(analysis_id=get_MLP_id()).all()
        for model in models:
            item_text = model.title + ' id' + str(model.id)
            item = QListWidgetItem(item_text)
            item.setData(Qt.UserRole, model.id)
            item.setToolTip(model.comment)
            ui_tm.listWidget_test_model.addItem(item)
        ui_tm.listWidget_test_model.setCurrentRow(0)

    def get_test_list_marker_mlp():
        markers = session.query(MarkerMLP).filter_by(analysis_id=get_test_MLP_id()).order_by(MarkerMLP.title).all()
        return [m.title for m in markers]

    update_test_analysis_combobox()
    update_test_model_list()
    update_list_test_well()

    def test_classif_model():
        global list_cat
        ui_tm.textEdit_test_result.clear()
        model = session.query(TrainedModelClass).filter_by(
            id=ui_tm.listWidget_test_model.currentItem().text().split(' id')[-1]).first()
        list_param = json.loads(model.list_params)

        try:
            with open(model.path_model, 'rb') as f:
                class_model = pickle.load(f)
            list_param_num = get_list_param_numerical(json.loads(model.list_params), model)
        except:
            set_info('Не удалось загрузить модель. Выберите нужную модель и рассчитайте заново', 'red')
            QMessageBox.critical(MainWindow, 'Ошибка', 'Не удалось загрузить модель.')
            return

        list_cat = list(class_model.classes_)
        if set(get_test_list_marker_mlp()) != set(list_cat):
            set_info('Не совпадают названия меток для данной модели.', 'red')

            ChooseMark = QtWidgets.QDialog()
            ui_cm = Ui_FormRegMod()
            ui_cm.setupUi(ChooseMark)
            ChooseMark.show()
            ChooseMark.setAttribute(Qt.WA_DeleteOnClose)

            ui_cm.label.setText('Не совпадают названия меток для данной модели.\n'
                                f'Выберите маркер соответствующий метке: {list_cat[0]}')
            ui_cm.checkBox_color_marker.hide()

            def update_markers():
                global list_cat
                mark1 = ui_tm.comboBox_mark.currentText()
                print(mark1)
                list_cat = get_test_list_marker_mlp()
                list_cat.remove(mark1)
                list_cat.insert(0, mark1)
                print(list_cat)

                ChooseMark.close()

            ui_cm.pushButton_calc_model.clicked.connect(update_markers)
            ChooseMark.exec()

        # print('list marks: ', list_cat, get_test_list_marker_mlp())

        data_table, params = build_table_test_no_db("mlp", get_test_MLP_id(), list_param)
        data_test = data_table.copy()

        try:
            working_sample = data_test[list_param_num].values.tolist()
        except KeyError:
            set_info('Не совпадает количество признаков для данной модели. Выберите нужную модель и '
                     'рассчитайте заново', 'red')
            QMessageBox.critical(MainWindow, 'Ошибка', 'Не совпадает количество признаков для данной модели.')
            return

        try:
            mark = class_model.predict(working_sample)
            probability = class_model.predict_proba(working_sample)
        except ValueError:
            working_sample = [[np.nan if np.isinf(x) else x for x in y] for y in working_sample]

            data = imputer.fit_transform(working_sample)
            try:
                mark = class_model.predict(data)
            except ValueError:
                set_info('Не совпадает количество признаков для данной модели. Выберите нужную модель и '
                         'рассчитайте заново', 'red')
                QMessageBox.critical(MainWindow, 'Ошибка', 'Не совпадает количество признаков для данной модели.')
                return
            probability = class_model.predict_proba(data)

            for i in data_test.index:
                p_nan = [data_test.columns[ic + 3] for ic, v in enumerate(data_test.iloc[i, 3:].tolist()) if
                         np.isnan(v)]
                if len(p_nan) > 0:
                    set_info(f'Внимание для измерения "{i}" отсутствуют параметры "{", ".join(p_nan)}", поэтому расчёт для'
                             f' этого измерения может быть не корректен', 'red')


        result_df = pd.concat([data_test, pd.DataFrame(probability, columns=list_cat)], axis=1)


        new_list_cat = list(class_model.classes_)
        result_df['mark_probability'] = mark
        result_df['mark_probability'] = result_df['mark_probability'].replace(
            {new_list_cat[0]: list_cat[0], new_list_cat[1]: list_cat[1]})
        # result_df['mark'] = result_df['mark'].replace({'bitum': 'нефть', 'empty': 'пусто'})

        result_df['совпадение'] = result_df['mark'].eq(result_df['mark_probability']).astype(int)
        correct_matches = result_df['совпадение'].sum()
        # print('\nMarked')
        # pd.set_option('display.max_columns', None)
        print(f'\n Cовпало: {correct_matches}/{len(result_df)}')
        ui_tm.textEdit_test_result.setTextColor(Qt.darkGreen)
        ui_tm.textEdit_test_result.append(f"Тестирование модели {model.title}:\n")
        index = 0
        while index + 1 < len(result_df):
            comp, total = 0, 0
            nulls, ones = 0, 0
            while index + 1 < len(result_df) and \
                    result_df.loc[index, 'prof_well_index'].split('_')[0] == result_df.loc[index + 1, 'prof_well_index'].split('_')[0] and\
                    result_df.loc[index, 'prof_well_index'].split('_')[1] == result_df.loc[index + 1, 'prof_well_index'].split('_')[1]:
                if result_df.loc[index, 'совпадение'] == 1:
                    comp += 1
                nulls = nulls + result_df.loc[index, list_cat[1]]
                ones = ones + result_df.loc[index, list_cat[0]]
                total += 1
                index += 1
            if result_df.loc[index, 'prof_well_index'].split('_')[1] == result_df.loc[index - 1, 'prof_well_index'].split('_')[1]:
                if result_df.loc[index, 'совпадение'] == 1:
                    comp += 1
                total += 1
            print('total comp:', total, comp)
            profile = session.query(Profile).filter(Profile.id == result_df.loc[index, 'prof_well_index'].split('_')[0]).first()
            well = session.query(Well).filter(Well.id == result_df.loc[index, 'prof_well_index'].split('_')[1]).first()

            color_text = Qt.black
            if comp / total < 0.5:
                color_text = Qt.red
            if 0.9 > comp / total >= 0.5:
                color_text = Qt.darkYellow

            ui_tm.textEdit_test_result.setTextColor(color_text)
            ui_tm.textEdit_test_result.insertPlainText(f'{profile.research.object.title} - {profile.title} | {well.name} |'
                                                       f'  {list_cat[0]} {ones/total:.3f} | {list_cat[1]} {nulls/total:.3f} | {comp}/{total} \n')

            index += 1

        percent = correct_matches / len(result_df) * 100
        color_text = Qt.green
        if percent < 80:
            color_text = Qt.darkYellow
        if percent < 50:
            color_text = Qt.red
        ui_tm.textEdit_test_result.setTextColor(color_text)
        ui_tm.textEdit_test_result.insertPlainText(f'\nВсего совпало: {correct_matches}/{len(result_df)} - {percent:.1f}%\n\n')

    def test_all_classif_models():
        global list_cat
        ui_tm.textEdit_test_result.clear()
        curr_list_cat, curr_list_param, curr_data_table = [], [], pd.DataFrame()
        for model in session.query(TrainedModelClass).filter_by(analysis_id=get_MLP_id()).all():
            list_param = json.loads(model.list_params)
            try:
                with open(model.path_model, 'rb') as f:
                    class_model = pickle.load(f)
                list_param_num = get_list_param_numerical(json.loads(model.list_params), model)
            except:
                set_info('Не удалось загрузить модель. Выберите нужную модель и рассчитайте заново', 'red')
                QMessageBox.critical(MainWindow, 'Ошибка', 'Не удалось загрузить модель.')
                return
            if set(curr_list_cat) != set(get_test_list_marker_mlp()):
                list_cat = list(class_model.classes_)
                if set(get_test_list_marker_mlp()) != set(list_cat):
                    set_info('Не совпадают названия меток для данной модели.', 'red')

                    ChooseMark = QtWidgets.QDialog()
                    ui_cm = Ui_FormRegMod()
                    ui_cm.setupUi(ChooseMark)
                    ChooseMark.show()
                    ChooseMark.setAttribute(Qt.WA_DeleteOnClose)

                    ui_cm.label.setText('Не совпадают названия меток для данной модели.\n'
                                        f'Выберите маркер соответствующий метке: {list_cat[0]}')
                    ui_cm.checkBox_color_marker.hide()

                    def update_markers():
                        global list_cat
                        mark1 = ui_tm.comboBox_mark.currentText()
                        ui_cm.label.setText(f'Выберите маркер соответствующий метке: {mark1}')
                        list_cat = get_test_list_marker_mlp()
                        list_cat.remove(mark1)
                        list_cat.insert(0, mark1)


                        ChooseMark.close()

                    ui_cm.pushButton_calc_model.clicked.connect(update_markers)
                    ChooseMark.exec()
                curr_list_cat = list_cat.copy()
            else:
                list_cat = curr_list_cat.copy()

            if curr_list_param == list_param_num:
                data_table = curr_data_table.copy()
            else:
                data_table, params = build_table_test_no_db("mlp", get_test_MLP_id(), list_param)
            curr_list_param = list_param_num.copy()
            curr_data_table = data_table.copy()
            data_test = data_table.copy()

            try:
                working_sample = data_test[list_param_num].values.tolist()
            except KeyError:
                set_info('Не совпадает количество признаков для данной модели. Выберите нужную модель и '
                         'рассчитайте заново', 'red')
                QMessageBox.critical(MainWindow, 'Ошибка', 'Не совпадает количество признаков для данной модели.')
                return

            try:
                mark = class_model.predict(working_sample)
                probability = class_model.predict_proba(working_sample)
            except ValueError:
                working_sample = [[np.nan if np.isinf(x) else x for x in y] for y in working_sample]

                data = imputer.fit_transform(working_sample)
                try:
                    mark = class_model.predict(data)
                except ValueError:
                    set_info('Не совпадает количество признаков для данной модели. Выберите нужную модель и '
                             'рассчитайте заново', 'red')
                    QMessageBox.critical(MainWindow, 'Ошибка', 'Не совпадает количество признаков для данной модели.')
                    return
                probability = class_model.predict_proba(data)

                for i in data_test.index:
                    p_nan = [data_test.columns[ic + 3] for ic, v in enumerate(data_test.iloc[i, 3:].tolist()) if
                             np.isnan(v)]
                    if len(p_nan) > 0:
                        set_info(f'Внимание для измерения "{i}" отсутствуют параметры "{", ".join(p_nan)}", поэтому расчёт для'
                                 f' этого измерения может быть не корректен', 'red')

            result_df = pd.concat([data_test, pd.DataFrame(probability, columns=list_cat)], axis=1)

            new_list_cat = list(class_model.classes_)
            result_df['mark_probability'] = mark
            result_df['mark_probability'] = result_df['mark_probability'].replace(
                {new_list_cat[0]: list_cat[0], new_list_cat[1]: list_cat[1]})

            result_df['совпадение'] = result_df['mark'].eq(result_df['mark_probability']).astype(int)
            correct_matches = result_df['совпадение'].sum()

            ui_tm.textEdit_test_result.setTextColor(QColor("darkgreen"))
            ui_tm.textEdit_test_result.append(f"Тестирование модели {model.title}:\n")
            index = 0
            while index + 1 < len(result_df):
                comp, total = 0, 0
                nulls, ones = 0, 0
                while index + 1 < len(result_df) and \
                        result_df.loc[index, 'prof_well_index'].split('_')[0] == \
                        result_df.loc[index + 1, 'prof_well_index'].split('_')[0] and \
                        result_df.loc[index, 'prof_well_index'].split('_')[1] == \
                        result_df.loc[index + 1, 'prof_well_index'].split('_')[1]:
                    if result_df.loc[index, 'совпадение'] == 1:
                        comp += 1
                    nulls = nulls + result_df.loc[index, list_cat[1]]
                    ones = ones + result_df.loc[index, list_cat[0]]
                    total += 1
                    index += 1
                if result_df.loc[index, 'prof_well_index'].split('_')[1] == \
                        result_df.loc[index - 1, 'prof_well_index'].split('_')[1]:
                    if result_df.loc[index, 'совпадение'] == 1:
                        comp += 1
                    total += 1

                profile = session.query(Profile).filter(
                    Profile.id == result_df.loc[index, 'prof_well_index'].split('_')[0]).first()
                well = session.query(Well).filter(Well.id == result_df.loc[index, 'prof_well_index'].split('_')[1]).first()

                color_text = Qt.black
                if comp/total < 0.5:
                    color_text = Qt.red
                if 0.9 > comp/total >= 0.5:
                    color_text = Qt.darkYellow

                ui_tm.textEdit_test_result.setTextColor(color_text)
                ui_tm.textEdit_test_result.append(
                    f'{profile.research.object.title} - {profile.title} | {well.name} |'
                    f'  {list_cat[0]} {ones / total:.3f} | {list_cat[1]} {nulls / total:.3f} | {comp}/{total}')

                index += 1
            percent = correct_matches / len(result_df) * 100
            color_text = Qt.green
            if percent < 80:
                color_text = Qt.darkYellow
            if percent < 50:
                color_text = Qt.red
            ui_tm.textEdit_test_result.setTextColor(color_text)
            ui_tm.textEdit_test_result.append(
                f'\nВсего совпало: {correct_matches}/{len(result_df)} - {percent:.1f}%\n\n')


    ui_tm.pushButton_test.clicked.connect(test_classif_model)
    ui_tm.pushButton_test_all.clicked.connect(test_all_classif_models)
    ui_tm.comboBox_test_analysis.activated.connect(update_list_test_well)
    test_classifModel.exec_()



def regression_test():
    test_regressModel = QtWidgets.QDialog()
    ui_tr = Ui_FormTestModel()
    ui_tr.setupUi(test_regressModel)
    test_regressModel.show()
    test_regressModel.setAttribute(QtCore.Qt.WA_DeleteOnClose)

    m_width, m_height = get_width_height_monitor()
    test_regressModel.resize(int(m_width/2.2), m_height - 200)
    ui_tr.comboBox_mark.hide()

    def update_test_reg_model_list():
        ui_tr.listWidget_test_model.clear()
        models = session.query(TrainedModelReg).filter(TrainedModelReg.analysis_id == get_regmod_id()).all()
        for model in models:
            item_text = model.title
            item = QListWidgetItem(item_text)
            item.setData(Qt.UserRole, model.id)
            item.setToolTip(model.comment)
            ui_tr.listWidget_test_model.addItem(item)
        ui_tr.listWidget_test_model.setCurrentRow(0)

    def get_test_regmod_id():
        return ui_tr.comboBox_test_analysis.currentText().split(' id')[-1]

    def update_test_reg_list_well():
        ui_tr.listWidget_test_point.clear()
        count_markup, count_measure, count_fake = 0, 0, 0

        for i in session.query(MarkupReg).filter(MarkupReg.analysis_id == get_test_regmod_id()).all():
            try:
                fake = len(json.loads(i.list_fake)) if i.list_fake else 0
                measure = len(json.loads(i.list_measure))
                if i.type_markup == 'intersection':
                    try:
                        inter_name = session.query(Intersection.name).filter(Intersection.id == i.well_id).first()[0]
                    except TypeError:
                        # session.query(MarkupReg).filter(MarkupReg.id == i.id).delete()
                        session.commit()
                        set_info(f'Обучающая скважина удалена из-за отсутствия пересечения', 'red')
                        continue
                    item = (
                        f'{i.profile.research.object.title} - {i.profile.title} | {i.formation.title} | {inter_name.split("_")[0]} | '
                        f'{measure - fake} | {i.target_value} | id{i.id}')
                else:
                    item = (
                        f'{i.profile.research.object.title} - {i.profile.title} | {i.formation.title} | {i.well.name} | '
                        f'{measure - fake} | {i.target_value} | id{i.id}')
                ui_tr.listWidget_test_point.addItem(item)
                count_markup += 1
                count_measure += measure - fake
                count_fake += fake
            except AttributeError:
                set_info(f'Параметр для профиля {i.profile.title} удален из-за отсутствия одного из параметров', 'red')
                # session.delete(i)
                session.commit()

    def update_test_reg_analysis_combobox():
        ui_tr.comboBox_test_analysis.clear()
        for i in session.query(AnalysisReg).order_by(AnalysisReg.title).all():
            ui_tr.comboBox_test_analysis.addItem(f'{i.title} id{i.id}')
        update_test_reg_list_well()

    update_test_reg_analysis_combobox()
    update_test_reg_list_well()
    update_test_reg_model_list()

    def test_regress_model():
        ui_tr.textEdit_test_result.clear()
        model = session.query(TrainedModelReg).filter_by(
            id=ui_tr.listWidget_test_model.currentItem().data(Qt.UserRole)).first()
        list_param = json.loads(model.list_params)
        list_param_num = get_list_param_numerical(json.loads(model.list_params), model)

        working_data, curr_form = build_table_test_no_db('regmod', get_test_regmod_id(), list_param)
        working_sample = working_data[list_param_num].values.tolist()

        with open(model.path_model, 'rb') as f:
            reg_model = pickle.load(f)

        try:
            y_pred = reg_model.predict(working_sample)
        except ValueError:
            data = imputer.fit_transform(working_sample)
            y_pred = reg_model.predict(data)

            for i in working_data.index:
                p_nan = [working_data.columns[ic + 3] for ic, v in enumerate(working_data.iloc[i, 3:].tolist()) if
                         np.isnan(v)]
                if len(p_nan) > 0:
                    set_info(f'Внимание для измерения "{i}" отсутствуют параметры "{", ".join(p_nan)}", поэтому расчёт для'
                             f' этого измерения может быть не корректен', 'red')

        working_data['y_pred'] = y_pred
        working_data['diff'] = working_data['target_value'] - working_data['y_pred']

        # print(working_data[['target_value', 'y_pred', 'diff']].head(20))

        accuracy = reg_model.score(working_sample, working_data['target_value'].values.tolist())
        mse = round(mean_squared_error(working_data['target_value'].values.tolist(), working_data['y_pred'].values.tolist()), 5)
        # print(accuracy)
        #
        # print(working_data['prof_well_index'])

        ui_tr.textEdit_test_result.setTextColor(Qt.black)
        ui_tr.textEdit_test_result.append(f"Тестирование модели {model.title}:")
        ui_tr.textEdit_test_result.append(f'Точность: {round(accuracy, 2)} Mean Squared Error: {round(mse, 2)}\n\n')

        index = 0
        while index + 1 < len(working_data):
            comp, total = 0, 0
            diff_list, list_y = [], []
            while index + 1 < len(working_data) and \
                    working_data.loc[index, 'prof_well_index'].split('_')[0] == \
                    working_data.loc[index + 1, 'prof_well_index'].split('_')[0] and \
                    working_data.loc[index, 'prof_well_index'].split('_')[1] == \
                    working_data.loc[index + 1, 'prof_well_index'].split('_')[1]:
                diff_list.append(working_data.loc[index, 'diff'])
                list_y.append(working_data.loc[index, 'y_pred'])
                total += 1
                index += 1
            if working_data.loc[index, 'prof_well_index'].split('_')[1] == \
                    working_data.loc[index - 1, 'prof_well_index'].split('_')[1]:
                total += 1
            if total == 0: total = 1
            profile = session.query(Profile).filter(
                Profile.id == working_data.loc[index, 'prof_well_index'].split('_')[0]).first()
            well = session.query(Well).filter(Well.id == working_data.loc[index, 'prof_well_index'].split('_')[1]).first()


            min_target, max_target = working_data['target_value'].min(), working_data['target_value'].max()
            lin_target = np.linspace(0, max_target - min_target, working_data['target_value'].size)
            print('lin_target: ', lin_target)
            percentile_20 = np.percentile(lin_target, 20)
            percentile_50 = np.percentile(lin_target, 50)
            print('20%: ', percentile_20)
            print('50%: ', percentile_50)
            mistake = math.fabs(round(working_data.loc[index, "target_value"] - sum(list_y)/total, 2))
            mean_pred = round(sum(list_y)/total, 2)

            color_text = Qt.black
            if percentile_20 <= mistake < percentile_50:
                color_text = Qt.darkYellow
            if mistake >= percentile_50:
                color_text = Qt.red
            ui_tr.textEdit_test_result.setTextColor(color_text)
            ui_tr.textEdit_test_result.insertPlainText(
                f'{profile.research.object.title} - {profile.title} | Скв. {well.name} |'
                f' predict {mean_pred} | target {round(working_data.loc[index, "target_value"], 2)} '
                f'| погрешность: {mistake}\n')
            index += 1

        def regress_test_graphs():
            data_graph = pd.DataFrame({
                'y_test': working_data['target_value'].values.tolist(),
                'y_pred': working_data['y_pred'].values.tolist(),
                'y_remain': working_data['diff'].values.tolist()
            })

            fig, axes = plt.subplots(nrows=2, ncols=2)
            fig.set_size_inches(15, 10)
            fig.suptitle(f'Модель {model.title}:\n точность: {accuracy} '
                         f' Mean Squared Error:\n {mse},')
            sns.scatterplot(data=data_graph, x='y_test', y='y_pred', ax=axes[0, 0])
            sns.regplot(data=data_graph, x='y_test', y='y_pred', ax=axes[0, 0])
            sns.scatterplot(data=data_graph, x='y_test', y='y_remain', ax=axes[1, 0])
            sns.regplot(data=data_graph, x='y_test', y='y_remain', ax=axes[1, 0])
            try:
                sns.histplot(data=data_graph, x='y_remain', kde=True, ax=axes[1, 1])
            except MemoryError:
                pass
            fig.tight_layout()
            fig.show()

        regress_test_graphs()

    def test_all_regress_models():
        ui_tr.textEdit_test_result.clear()
        curr_list_param, current_data_table = [], pd.DataFrame()
        all_models = session.query(TrainedModelReg).filter(TrainedModelReg.analysis_id == get_regmod_id()).all()
        cell_grid = math.sqrt(len(all_models))
        cell_grid = int(cell_grid) if cell_grid.is_integer() else int(cell_grid) + 1
        fig, axes = plt.subplots(nrows=cell_grid, ncols=cell_grid)
        fig.set_size_inches(cell_grid * 3.5, cell_grid * 3.5)
        fig.suptitle(f'Тестирование модели регрессии {ui_tr.comboBox_test_analysis.currentText().split(" id")[0]}')
        nr, nc = 0, 0
        for model in all_models:
            list_param = json.loads(model.list_params)
            list_param_num = get_list_param_numerical(json.loads(model.list_params), model)

            if curr_list_param == list_param_num:
                working_data = current_data_table.copy()
            else:
                working_data, curr_form = build_table_test_no_db('regmod', get_test_regmod_id(), list_param)
            curr_list_param = list_param_num.copy()
            current_data_table = working_data.copy()

            working_sample = working_data[list_param_num].values.tolist()
            with open(model.path_model, 'rb') as f:
                reg_model = pickle.load(f)

            try:
                y_pred = reg_model.predict(working_sample)
            except ValueError:
                data = imputer.fit_transform(working_sample)
                y_pred = reg_model.predict(data)

                for i in working_data.index:
                    p_nan = [working_data.columns[ic + 3] for ic, v in enumerate(working_data.iloc[i, 3:].tolist()) if
                             np.isnan(v)]
                    if len(p_nan) > 0:
                        set_info(
                            f'Внимание для измерения "{i}" отсутствуют параметры "{", ".join(p_nan)}", поэтому расчёт для'
                            f' этого измерения может быть не корректен', 'red')

            working_data['y_pred'] = y_pred
            working_data['diff'] = working_data['target_value'] - working_data['y_pred']
            accuracy = reg_model.score(working_sample, working_data['target_value'].values.tolist())
            mse = round(mean_squared_error(working_data['target_value'].values.tolist(),
                                           working_data['y_pred'].values.tolist()), 5)

            ui_tr.textEdit_test_result.setTextColor(Qt.black)
            ui_tr.textEdit_test_result.append(f"Тестирование модели {model.title}:")
            ui_tr.textEdit_test_result.append(f'Точность: {round(accuracy, 2)} Mean Squared Error: {round(mse, 2)}\n\n')
            index = 0
            while index + 1 < len(working_data):
                comp, total = 0, 0
                diff_list, list_y = [], []
                while index + 1 < len(working_data) and \
                        working_data.loc[index, 'prof_well_index'].split('_')[0] == \
                        working_data.loc[index + 1, 'prof_well_index'].split('_')[0] and \
                        working_data.loc[index, 'prof_well_index'].split('_')[1] == \
                        working_data.loc[index + 1, 'prof_well_index'].split('_')[1]:
                    list_y.append(working_data.loc[index, 'y_pred'])
                    diff_list.append(working_data.loc[index, 'diff'])
                    total += 1
                    index += 1
                if working_data.loc[index, 'prof_well_index'].split('_')[1] == \
                        working_data.loc[index - 1, 'prof_well_index'].split('_')[1]:
                    total += 1
                if total == 0: total = 1
                profile = session.query(Profile).filter(
                    Profile.id == working_data.loc[index, 'prof_well_index'].split('_')[0]).first()
                well = session.query(Well).filter(
                    Well.id == working_data.loc[index, 'prof_well_index'].split('_')[1]).first()

                min_target, max_target = working_data['target_value'].min(), working_data['target_value'].max()
                lin_target = np.linspace(0, max_target - min_target, working_data['target_value'].size)
                percentile_20 = np.percentile(lin_target, 20)
                percentile_50 = np.percentile(lin_target, 50)
                mistake = math.fabs(round(working_data.loc[index, "target_value"] - sum(list_y) / total, 2))
                mean_pred = round(sum(list_y) / total, 2)

                color_text = Qt.black
                if percentile_20 <= mistake < percentile_50:
                    color_text = Qt.darkYellow
                if mistake >= percentile_50:
                    color_text = Qt.red
                ui_tr.textEdit_test_result.setTextColor(color_text)
                ui_tr.textEdit_test_result.insertPlainText(
                    f'{profile.research.object.title} - {profile.title} | Скв. {well.name} |'
                    f' predict {mean_pred} | target {round(working_data.loc[index, "target_value"], 2)} '
                    f'| погрешность: {mistake}\n')
                index += 1

            data_graph = pd.DataFrame({
                'y_test': working_data['target_value'].values.tolist(),
                'y_pred': working_data['y_pred'].values.tolist(),
            })

            axes[nc, nr].set_title(f'Модель {model.title}:\n точность: {round(accuracy, 2)}')
            sns.scatterplot(data=data_graph, x='y_test', y='y_pred', ax=axes[nc, nr])
            sns.regplot(data=data_graph, x='y_test', y='y_pred', ax=axes[nc, nr])
            nr += 1
            if nr == cell_grid:
                nc += 1
                nr = 0
        fig.tight_layout()
        fig.show()

    ui_tr.comboBox_test_analysis.activated.connect(update_test_reg_list_well)
    ui_tr.pushButton_test.clicked.connect(test_regress_model)
    ui_tr.pushButton_test_all.clicked.connect(test_all_regress_models)

    test_regressModel.exec_()

