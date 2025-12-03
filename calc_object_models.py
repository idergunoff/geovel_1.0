from func import *
from build_table import *
from qt.calc_object import Ui_Dialog_calc_object
from regression import update_list_trained_models_regmod
from PyQt5.QtWidgets import QDialog

def calc_object():
    CalcObjectWindow = QtWidgets.QDialog()
    ui_co = Ui_Dialog_calc_object()
    ui_co.setupUi(CalcObjectWindow)
    CalcObjectWindow.show()
    CalcObjectWindow.setAttribute(QtCore.Qt.WA_DeleteOnClose)

    def update_co_models_list():
        ui_co.listWidget_objects.clear()
        for i in session.query(CalcObject).all():
            try:
                formations = list(json.loads(i.list_formation))
                item_text = f'research: id{i.research_id}, model: {i.type_ml} id{i.model_id}, formations: {len(formations)}'
                item = QListWidgetItem(item_text)
                item.setData(Qt.UserRole, i.id)
                ui_co.listWidget_objects.addItem(item)
                ui_co.listWidget_objects.setCurrentRow(0)
            except AttributeError:
                pass

    def remove_co_model():
        co = session.query(CalcObject).filter_by(id=ui_co.listWidget_objects.currentItem().data(Qt.UserRole)).first()
        session.delete(co)
        session.commit()
        update_co_models_list()
        set_info(f'Запись id{co.id} удалена из списка', 'green')

    def clear_co_list():
        for co in session.query(CalcObject).all():
            session.delete(co)
            session.commit()
        update_co_models_list()
        set_info('Список очищен', 'green')

    def add_cls_model():
        global flag_break
        try:
            model = session.query(TrainedModelClass).filter_by(
                id=ui.listWidget_trained_model_class.currentItem().data(Qt.UserRole)).first()
        except AttributeError:
            set_info('Не выбрана модель', 'red')
        list_formation = []
        profiles = session.query(Profile).filter(Profile.research_id == get_research_id()).all()
        flag_break = []
        for n, prof in enumerate(profiles):
            if flag_break:
                if flag_break[0] == 'stop':
                    break
                else:
                    set_info(f'Нет пласта с названием {flag_break[1]} для профиля {flag_break[0]}', 'red')
                    QMessageBox.critical(MainWindow, 'Ошибка', f'Нет пласта с названием {flag_break[1]} для профиля '
                                                               f'{flag_break[0]}, выберите пласты для каждого профиля.')
                    return
            if len(prof.formations) == 1:
                list_formation.append(prof.formations[0].id)
            elif len(prof.formations) > 1:
                Choose_Formation = QtWidgets.QDialog()
                ui_cf = Ui_FormationLDA()
                ui_cf.setupUi(Choose_Formation)
                Choose_Formation.show()
                Choose_Formation.setAttribute(QtCore.Qt.WA_DeleteOnClose)  # атрибут удаления виджета после закрытия
                for f in prof.formations:
                    ui_cf.listWidget_form_lda.addItem(f'{f.title} id{f.id}')
                ui_cf.listWidget_form_lda.setCurrentRow(0)

                def form_mlp_ok():
                    global flag_break
                    if ui_cf.checkBox_to_all.isChecked():
                        title_form = ui_cf.listWidget_form_lda.currentItem().text().split(' id')[0]
                        for prof in profiles:
                            prof_form = session.query(Formation).filter_by(
                                profile_id=prof.id,
                                title=title_form
                            ).first()
                            if prof_form:
                                list_formation.append(prof_form.id)
                            else:
                                flag_break = [prof.title, title_form]
                                Choose_Formation.close()
                                return
                        flag_break = ['stop', 'stop']
                        Choose_Formation.close()
                    else:
                        list_formation.append(int(ui_cf.listWidget_form_lda.currentItem().text().split('id')[-1]))
                        Choose_Formation.close()

                ui_cf.pushButton_ok_form_lda.clicked.connect(form_mlp_ok)
                Choose_Formation.exec_()
        new_co_cls = CalcObject(
            research_id=get_research_id(),
            type_ml = 'cls',
            model_id = model.id,
            list_formation = json.dumps(list_formation)
        )
        session.add(new_co_cls)
        session.commit()
        update_co_models_list()

    def add_reg_model():
        global flag_break
        list_formation = []
        profiles = session.query(Profile).filter(Profile.research_id == get_research_id()).all()
        flag_break = []
        for n, prof in enumerate(profiles):
            if flag_break:
                if flag_break[0] == 'stop':
                    break
                else:
                    set_info(f'Нет пласта с названием {flag_break[1]} для профиля {flag_break[0]}', 'red')
                    QMessageBox.critical(MainWindow, 'Ошибка', f'Нет пласта с названием {flag_break[1]} для профиля '
                                                               f'{flag_break[0]}, выберите пласты для каждого профиля.')
                    return
            if len(prof.formations) == 1:
                list_formation.append(prof.formations[0].id)
            elif len(prof.formations) > 1:
                Choose_Formation = QtWidgets.QDialog()
                ui_cf = Ui_FormationLDA()
                ui_cf.setupUi(Choose_Formation)
                Choose_Formation.show()
                Choose_Formation.setAttribute(QtCore.Qt.WA_DeleteOnClose)  # атрибут удаления виджета после закрытия
                for f in prof.formations:
                    ui_cf.listWidget_form_lda.addItem(f'{f.title} id{f.id}')
                ui_cf.listWidget_form_lda.setCurrentRow(0)

                def form_mlp_ok():
                    global flag_break
                    if ui_cf.checkBox_to_all.isChecked():
                        title_form = ui_cf.listWidget_form_lda.currentItem().text().split(' id')[0]
                        for prof in profiles:
                            prof_form = session.query(Formation).filter_by(
                                profile_id=prof.id,
                                title=title_form
                            ).first()
                            if prof_form:
                                list_formation.append(prof_form.id)
                            else:
                                flag_break = [prof.title, title_form]
                                Choose_Formation.close()
                                return
                        flag_break = ['stop', 'stop']
                        Choose_Formation.close()
                    else:
                        list_formation.append(int(ui_cf.listWidget_form_lda.currentItem().text().split('id')[-1]))
                        Choose_Formation.close()

                ui_cf.pushButton_ok_form_lda.clicked.connect(form_mlp_ok)
                Choose_Formation.exec_()
        model = session.query(TrainedModelReg).filter_by(
            id=ui.listWidget_trained_model_reg.currentItem().data(Qt.UserRole)).first()
        new_co_reg = CalcObject(
            research_id=get_research_id(),
            type_ml = 'reg',
            model_id = model.id,
            list_formation = json.dumps(list_formation)
        )
        session.add(new_co_reg)
        session.commit()
        update_co_models_list()



    def start_co_class():
        """ Расчет объекта по модели """

        working_data_result = pd.DataFrame()

        co_cls = session.query(CalcObject).filter_by(
            id=ui_co.listWidget_objects.currentItem().data(Qt.UserRole)).first()

        model = session.query(TrainedModelClass).filter_by(
            id=co_cls.model_id).first()

        with open(model.path_model, 'rb') as f:
            class_model = pickle.load(f)

        list_param_num = get_list_param_numerical(json.loads(model.list_params), model)

        model_mask = session.query(TrainedModelClassMask).filter_by(model_id=model.id).first()
        if model_mask:
            list_param_num = sorted(
                json.loads(session.query(ParameterMask).filter_by(id=model_mask.mask_id).first().mask))

        for n, prof in enumerate(session.query(Profile).filter(Profile.research_id == co_cls.research_id).all()):

            if session.query(ProfileModelPrediction).filter_by(
                    profile_id=prof.id, type_model='cls', model_id=model.id).count() == 0:

                list_formation_loaded = json.loads(co_cls.list_formation)
                curr_form = session.query(Formation).filter(Formation.id == list_formation_loaded[n]).first()
                working_data, curr_form = build_table_test('mlp', model=model, curr_form=curr_form)

                working_data_profile = working_data.copy()
                working_sample_profile = working_data_profile[list_param_num].values.tolist()

                try:
                    probability = class_model.predict_proba(working_sample_profile)
                except ValueError:
                    working_sample_profile = [[np.nan if np.isinf(x) else x for x in y] for y in
                                              working_sample_profile]
                    data = imputer.fit_transform(working_sample_profile)
                    probability = class_model.predict_proba(data)

                list_result = [round(p[0], 6) for p in probability]
                new_prof_model_pred = ProfileModelPrediction(
                    profile_id=prof.id,
                    type_model='cls',
                    model_id=model.id,
                    prediction=json.dumps(list_result)
                )

                session.add(new_prof_model_pred)
                session.commit()
                set_info(f'Результат расчета модели "{model.title}" для профиля {prof.title} сохранен', 'green')

        update_list_model_prediction()

    def start_co_reg():
        """ Расчет объекта обученной моделью """

        working_data_result = pd.DataFrame()

        co_reg = session.query(CalcObject).filter_by(
            id=ui_co.listWidget_objects.currentItem().data(Qt.UserRole)).first()

        model = session.query(TrainedModelReg).filter_by(
            id=co_reg.model_id).first()

        with open(model.path_model, 'rb') as f:
            reg_model = pickle.load(f)

        list_param_num = get_list_param_numerical(json.loads(model.list_params), model)

        model_mask = session.query(TrainedModelRegMask).filter_by(model_id=model.id).first()
        if model_mask:
            list_param_num = sorted(
                json.loads(session.query(ParameterMask).filter_by(id=model_mask.mask_id).first().mask))

        for n, prof in enumerate(session.query(Profile).filter(Profile.research_id == co_reg.research_id).all()):

            if session.query(ProfileModelPrediction).filter_by(
                    profile_id=prof.id, model_id=model.id, type_model='reg').count() == 0:

                list_formation_loaded = json.loads(co_reg.list_formation)
                curr_form = session.query(Formation).filter(Formation.id == list_formation_loaded[n]).first()
                working_data, curr_form = build_table_test('regmod', model=model, curr_form=curr_form)

                working_data_profile = working_data.copy()
                working_sample_profile = working_data_profile[list_param_num].values.tolist()

                try:
                    y_pred = reg_model.predict(working_sample_profile)
                except ValueError:
                    working_sample = [[np.nan if np.isinf(x) else x for x in y] for y in working_sample_profile]
                    data = imputer.fit_transform(working_sample)
                    y_pred = reg_model.predict(data)

                new_prof_model_pred = ProfileModelPrediction(
                    profile_id=prof.id,
                    type_model='reg',
                    model_id=model.id,
                    prediction=json.dumps(y_pred.tolist())
                )
                session.add(new_prof_model_pred)
                session.commit()
                set_info(f'Результат расчета модели "{model.title}" для профиля {prof.title} сохранен', 'green')

        update_list_model_prediction()

    def start_co():
        set_info('Начало расчета объектов по моделям', 'blue')
        # co = session.query(CalcObject).filter_by(id=ui_co.listWidget_objects.currentItem().data(Qt.UserRole)).first()
        while ui_co.listWidget_objects.count() > 0:
            item = ui_co.listWidget_objects.item(0)
            if item is None:
                break

            co = session.query(CalcObject).filter_by(id=item.data(Qt.UserRole)).first()
            if co is None:
                ui_co.listWidget_objects.takeItem(0)
                continue

            if co.type_ml == 'cls':
                start_co_class()
            if co.type_ml == 'reg':
                start_co_reg()

            session.delete(co)
            session.commit()

            ui_co.listWidget_objects.takeItem(0)

        set_info('Расчет объектов по моделям завершен', 'blue')

    update_co_models_list()
    ui_co.pushButton_start_co.clicked.connect(start_co)
    ui_co.pushButton_delete_co.clicked.connect(remove_co_model)
    ui_co.pushButton_clear_co.clicked.connect(clear_co_list)
    ui_co.pushButton_calc_cls.clicked.connect(add_cls_model)
    ui_co.pushButton_calc_reg.clicked.connect(add_reg_model)

    CalcObjectWindow.exec_()
