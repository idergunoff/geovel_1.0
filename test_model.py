from decimal import Decimal

from func import *
from decimal import *


def get_test_MLP_id(ui_tm):
    return ui_tm.comboBox_test_analysis.currentText().split(' id')[-1]

def test_start():
    test_classifModel = QtWidgets.QDialog()
    ui_tm = Ui_FormTestModel()
    ui_tm.setupUi(test_classifModel)
    test_classifModel.show()
    test_classifModel.setAttribute(QtCore.Qt.WA_DeleteOnClose)
    def update_list_test_well():
        ui_tm.listWidget_test_point.clear()
        count_markup, count_measure, count_fake = 0, 0, 0
        for i in session.query(MarkupMLP).filter(MarkupMLP.analysis_id == get_test_MLP_id(ui_tm)).all():
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

    def update_test_analysis_combobox(ui_tm):
        ui_tm.comboBox_test_analysis.clear()
        for i in session.query(AnalysisMLP).order_by(AnalysisMLP.title).all():
            ui_tm.comboBox_test_analysis.addItem(f'{i.title} id{i.id}')
            update_list_test_well()

    def update_test_model_list(ui_tm):
        ui_tm.listWidget_test_model.clear()
        models = session.query(TrainedModelClass).filter_by(analysis_id=get_MLP_id()).all()
        for model in models:
            item_text = model.title + ' id' + str(model.id)
            item = QListWidgetItem(item_text)
            item.setData(Qt.UserRole, model.id)
            item.setToolTip(model.comment)
            ui_tm.listWidget_test_model.addItem(item)
        ui_tm.listWidget_test_model.setCurrentRow(0)


    update_test_analysis_combobox(ui_tm)
    update_test_model_list(ui_tm)
    update_list_test_well()


    def test_classif_model():
        ui_tm.textEdit_test_result.clear()
        model = session.query(TrainedModelClass).filter_by(
            id=ui_tm.listWidget_test_model.currentItem().text().split(' id')[-1]).first()
        list_param = json.loads(model.list_params)

        data_table, params = build_table_test_no_db("mlp", get_test_MLP_id(ui_tm), list_param)
        data_test = data_table.copy()

        with open(model.path_model, 'rb') as f:
            class_model = pickle.load(f)
        list_param_num = get_list_param_numerical(json.loads(model.list_params))

        try:
            working_sample = data_test[list_param_num].values.tolist()
        except KeyError:
            set_info('Не совпадает количество признаков для данной модели. Выберите нужную модель и '
                     'рассчитайте заново', 'red')
            QMessageBox.critical(MainWindow, 'Ошибка', 'Не совпадает количество признаков для данной модели.')
            return

        list_cat = list(class_model.classes_)
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
        print('Probability')
        # print(pd.DataFrame(probability, columns=list_cat))

        result_df = pd.concat([data_test, pd.DataFrame(probability, columns=list_cat)], axis=1)

        result_df['mark_probability'] = mark
        result_df['mark_probability'] = result_df['mark_probability'].replace({'bitum': 'нефть', 'empty': 'пусто'})
        result_df['mark'] = result_df['mark'].replace({'bitum': 'нефть', 'empty': 'пусто'})

        result_df['совпадение'] = result_df['mark'].eq(result_df['mark_probability']).astype(int)
        correct_matches = result_df['совпадение'].sum()
        print('\nMarked')
        pd.set_option('display.max_columns', None)
        print(result_df)
        print(f'\n Cовпало: {correct_matches}/{len(result_df)}')

        index = 0
        while index + 1 < len(result_df):
            comp, total = 0, 0
            nulls, ones = 0, 0
            while index + 1 < len(result_df) and \
                    result_df.loc[index, 'prof_well_index'].split('_')[0] == result_df.loc[index + 1, 'prof_well_index'].split('_')[0] and\
                    result_df.loc[index, 'prof_well_index'].split('_')[1] == result_df.loc[index + 1, 'prof_well_index'].split('_')[1]:
                if result_df.loc[index, 'совпадение'] == 1:
                    comp += 1
                nulls = nulls + result_df.loc[index, 'empty']
                ones = ones + result_df.loc[index, 'bitum']
                total += 1
                index += 1
            if result_df.loc[index, 'prof_well_index'].split('_')[1] == result_df.loc[index - 1, 'prof_well_index'].split('_')[1]:
                if result_df.loc[index, 'совпадение'] == 1:
                    comp += 1
                total += 1

            profile = session.query(Profile).filter(Profile.id == result_df.loc[index, 'prof_well_index'].split('_')[0]).first()
            well = session.query(Well).filter(Well.id == result_df.loc[index, 'prof_well_index'].split('_')[1]).first()

            if (comp != total):
                ui_tm.textEdit_test_result.setTextColor(Qt.red)
                ui_tm.textEdit_test_result.insertPlainText(f'{profile.research.object.title} - {profile.title} | {well.name} |'
                                                           f'  bitum {ones/total:.3f} | empty {nulls/total:.3f} | {comp}/{total} \n')
            else:
                ui_tm.textEdit_test_result.setTextColor(Qt.black)
                ui_tm.textEdit_test_result.insertPlainText(f'{profile.research.object.title} - {profile.title} | {well.name} |'
                                                           f'  bitum {Decimal(ones)/Decimal(total):.3f} | empty {nulls/total:.3f} | {comp}/{total} \n')

            index += 1
        percent = correct_matches / len(result_df) * 100
        ui_tm.textEdit_test_result.insertPlainText(f'\nВсего совпало: {correct_matches}/{len(result_df)} - {percent:.1f}%\n\n')




    ui_tm.pushButton_test.clicked.connect(test_classif_model)
    ui_tm.comboBox_test_analysis.activated.connect(update_list_test_well)
    test_classifModel.exec_()

