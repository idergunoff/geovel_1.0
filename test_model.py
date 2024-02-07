from decimal import Decimal

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
    # update_list_test_well()

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
                    f'  {list_cat[1]} {ones / total:.3f} | {list_cat[0]} {nulls / total:.3f} | {comp}/{total}')

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

