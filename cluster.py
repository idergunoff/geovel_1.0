from build_table import build_table_test
from func import *


def calc_object_cluster():
    """ Расчет объекта по модели """
    global flag_break
    working_data_result = pd.DataFrame()
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
        count_measure = len(json.loads(session.query(Profile.signal).filter(Profile.id == prof.id).first()[0]))
        ui.comboBox_profile.setCurrentText(f'{prof.title} ({count_measure} измерений) id{prof.id}')
        set_info(f'Профиль {prof.title} ({count_measure} измерений)', 'blue')
        update_formation_combobox()
        if len(prof.formations) == 1:
            # ui.comboBox_plast.setCurrentText(f'{prof.formations[0].title} id{prof.formations[0].id}')
            list_formation.append(f'{prof.formations[0].title} id{prof.formations[0].id}')
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
                # ui.comboBox_plast.setCurrentText(ui_cf.listWidget_form_lda.currentItem().text())
                if ui_cf.checkBox_to_all.isChecked():
                    title_form = ui_cf.listWidget_form_lda.currentItem().text().split(' id')[0]
                    for prof in profiles:
                        prof_form = session.query(Formation).filter_by(
                            profile_id=prof.id,
                            title=title_form
                        ).first()
                        if prof_form:
                            list_formation.append(f'{prof_form.title} id{prof_form.id}')
                        else:
                            flag_break = [prof.title, title_form]
                            Choose_Formation.close()
                            return
                    flag_break = ['stop', 'stop']
                    Choose_Formation.close()
                else:
                    list_formation.append(ui_cf.listWidget_form_lda.currentItem().text())
                    Choose_Formation.close()

            ui_cf.pushButton_ok_form_lda.clicked.connect(form_mlp_ok)
            Choose_Formation.exec_()

    if ui.checkBox_save_prof_mlp.isChecked():
        model = session.query(TrainedModelClass).filter_by(
            id=ui.listWidget_trained_model_class.currentItem().data(Qt.UserRole)).first()

        with open(model.path_model, 'rb') as f:
            class_model = pickle.load(f)

        list_param_num = get_list_param_numerical(json.loads(model.list_params), model)

        model_mask = session.query(TrainedModelClassMask).filter_by(model_id=model.id).first()
        if model_mask:
            list_param_num = sorted(json.loads(session.query(ParameterMask).filter_by(id=model_mask.mask_id).first().mask))

    for n, prof in enumerate(session.query(Profile).filter(Profile.research_id == get_research_id()).all()):
        count_measure = len(json.loads(session.query(Profile.signal).filter(Profile.id == prof.id).first()[0]))
        ui.comboBox_profile.setCurrentText(f'{prof.title} ({count_measure} измерений) id{prof.id}')
        update_formation_combobox()
        ui.comboBox_plast.setCurrentText(list_formation[n])
        working_data, curr_form = build_table_test('mlp')

        if ui.checkBox_save_prof_mlp.isChecked():
            if session.query(ProfileModelPrediction).filter_by(
                    profile_id=prof.id, type_model='cls', model_id=model.id).count() == 0:

                working_data_profile = working_data.copy()
                working_sample_profile = working_data_profile[list_param_num].values.tolist()

                try:
                    probability = class_model.predict_proba(working_sample_profile)
                except ValueError:
                    working_sample_profile = [[np.nan if np.isinf(x) else x for x in y] for y in working_sample_profile]
                    data = imputer.fit_transform(working_sample_profile)
                    probability = class_model.predict_proba(data)

                list_result = [round(p[0], 6) for p in probability]
                new_prof_model_pred = ProfileModelPrediction(
                    profile_id=get_profile_id(),
                    type_model='cls',
                    model_id=model.id,
                    prediction=json.dumps(list_result)
                )

                session.add(new_prof_model_pred)
                session.commit()
                set_info(f'Результат расчета модели "{model.title}" для профиля {prof.title} сохранен', 'green')

        working_data_result = pd.concat([working_data_result, working_data], axis=0, ignore_index=True)

    update_list_model_prediction()
    working_data_result_copy = working_data_result.copy()

    model = session.query(TrainedModelClass).filter_by(
        id=ui.listWidget_trained_model_class.currentItem().data(Qt.UserRole)).first()

    list_param_num = get_list_param_numerical(json.loads(model.list_params), model)

    model_mask = session.query(TrainedModelClassMask).filter_by(model_id=model.id).first()
    if model_mask:
        list_param_num = sorted(json.loads(session.query(ParameterMask).filter_by(id=model_mask.mask_id).first().mask))

    working_sample = working_data_result_copy[list_param_num].values.tolist()

    from sklearn.preprocessing import RobustScaler
    from sklearn.decomposition import PCA

    # Робастная нормализация (устойчива к выбросам)
    scaler = RobustScaler()
    try:
        X_scaled = scaler.fit_transform(working_sample)

    except ValueError:
        working_sample = [[np.nan if np.isinf(x) else x for x in y] for y in working_sample]

        data = imputer.fit_transform(working_sample)


        try:
            X_scaled = scaler.fit_transform(data)
        except ValueError:
            set_info('Не совпадает количество признаков для данной модели. Выберите нужную модель и '
                     'рассчитайте заново', 'red')
            QMessageBox.critical(MainWindow, 'Ошибка', 'Не совпадает количество признаков для данной модели.')
            return


    # Уменьшаем размерность (шум часто сидит в мелких компонентах)
    pca = PCA(n_components=50, random_state=42)  # сохраняем 90% дисперсии
    X_pca = pca.fit_transform(X_scaled)

    print("Компонент до PCA:", X_scaled.shape[1])
    print("Компонент после PCA:", X_pca.shape[1])

    import hdbscan

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=30,  # попробуй 15, 30, 50
        min_samples=10,  # попробуй 5, 10, 20
        metric="correlation",
    )

    labels = clusterer.fit_predict(X_pca)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print("Найдено кластеров:", n_clusters)
    print("Шумовых точек:", (labels == -1).sum())

    for mcs in [10, 20, 30, 50, 80]:
        for ms in [1, 5, 10, 20]:
            cl = hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=ms, metric = "correlation")
            lab = cl.fit_predict(X_pca)
            ncl = len(set(lab)) - (1 if -1 in lab else 0)
            noise = (lab == -1).mean()
            print(f"mcs={mcs:3d} ms={ms:2d} clusters={ncl:2d} noise={noise:.2%}")

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=30,  # попробуй 15, 30, 50
        min_samples=10,  # попробуй 5, 10, 20
        metric="correlation",
    )

    labels = clusterer.fit_predict(X_scaled)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print("Найдено кластеров:", n_clusters)
    print("Шумовых точек:", (labels == -1).sum())

    for mcs in [10, 20, 30, 50, 80]:
        for ms in [1, 5, 10, 20]:
            cl = hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=ms, metric = "correlation")
            lab = cl.fit_predict(X_scaled)
            ncl = len(set(lab)) - (1 if -1 in lab else 0)
            noise = (lab == -1).mean()
            print(f"mcs={mcs:3d} ms={ms:2d} clusters={ncl:2d} noise={noise:.2%}")
