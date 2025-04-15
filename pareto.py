import json

import numpy as np

from build_table import build_table_train
from func import *


def pareto_start():

    Pareto = QtWidgets.QDialog()
    ui_prt = Ui_PARETO()
    ui_prt.setupUi(Pareto)
    Pareto.show()
    Pareto.setAttribute(QtCore.Qt.WA_DeleteOnClose)  # атрибут удаления виджета после закрытия

    # ui_anova.graphicsView.setBackground('w')
    m_width, m_height = get_width_height_monitor()
    Pareto.resize(m_width - 200, m_height - 200)

    data_plot, list_param = build_table_train(True, 'mlp')
    markers = list(set(data_plot['mark']))
    list_markers = list(set(data_plot['mark']))
    pallet = {i: session.query(MarkerMLP).filter(MarkerMLP.title == i, MarkerMLP.analysis_id == get_MLP_id()).first().color
              for i in markers}

    list_param = data_plot.columns.tolist()
    list_param.remove('mark')
    list_param.remove('prof_well_index')

    # Убираем NaN и inf значения
    nan_mask = data_plot[list_param].isna()
    inf_mask = np.isinf(data_plot[list_param])
    # Подсчет NaN и inf значений
    count_nan = nan_mask.sum().sum() + inf_mask.sum().sum()
    # Замена inf на 0
    data_plot[list_param] = data_plot[list_param].replace([np.inf, -np.inf], 0)
    if count_nan > 0:
        list_col = data_plot.columns.tolist()
        data_plot = pd.DataFrame(imputer.fit_transform(data_plot), columns=list_col)


    def update_list_pareto():
        ui_prt.listWidget_pareto_analysis.clear()
        pareto_analysis = session.query(ParetoAnalysis).filter(ParetoAnalysis.analysis_mlp_id == get_MLP_id()).all()

        for i in pareto_analysis:
            ui_prt.listWidget_pareto_analysis.addItem(f'{len(json.loads(i.start_params))}_niter{i.n_iter}_{i.problem_type}_id{i.id}')
    # ui_prt.listWidget_pareto_analysis.setCurrentRow(0)

    def update_list_saved_mask():
        ui_prt.listWidget_saved_mask.clear()
        for i in session.query(ParameterMask).all():
            item = QListWidgetItem(f'{i.count_param} id{i.id}')
            item.setToolTip(i.mask_info)
            ui_prt.listWidget_saved_mask.addItem(item)


    for i_param in list_param:
        check_box_widget = QCheckBox(i_param)
        check_box_widget.setChecked(True)
        list_item = QListWidgetItem()
        ui_prt.listWidget_check_param.addItem(list_item)
        ui_prt.listWidget_check_param.setItemWidget(list_item, check_box_widget)

    def set_list_point():
        ui_prt.listWidget_point.clear()
        # list_well = get_list_check_checkbox(ui_prt.listWidget_checkbox_well)
        # data_plot_new = data_plot.loc[data_plot['well'].isin(list_well)]

        for i in data_plot.index:
            prof = get_profile_by_id(data_plot['prof_well_index'][i].split('_')[0])
            well = get_well_by_id(data_plot['prof_well_index'][i].split('_')[1])
            item = QListWidgetItem(str(data_plot['prof_well_index'][i]))
            try:
                tooltip_text = f'{prof.research.object.title} {prof.title} {well.name}'
                item.setToolTip(tooltip_text)
            except AttributeError:
                item.setToolTip(f'{prof.research.object.title} {prof.title}')
            ui_prt.listWidget_point.addItem(item)
            ui_prt.listWidget_point.findItems(str(data_plot['prof_well_index'][i]), Qt.MatchExactly)[0].setBackground(
                QColor(pallet[data_plot['mark'][i]]))
        ui_prt.listWidget_point.setCurrentRow(0)


    def draw_graph_tsne():
        clear_layout(ui_prt.verticalLayout_graph)
        figure_tsne = plt.figure()
        canvas_tsne = FigureCanvas(figure_tsne)
        mpl_toolbar = NavigationToolbar(canvas_tsne)
        ui_prt.verticalLayout_graph.addWidget(mpl_toolbar)
        ui_prt.verticalLayout_graph.addWidget(canvas_tsne)

        # list_well = get_list_check_checkbox(ui_tsne.listWidget_checkbox_well)
        # data_plot_new = data_plot.loc[data_plot['well'].isin(list_well)]
        #
        # data_plot_new.reset_index(inplace=True, drop=True)

        # list_drop_point = get_list_check_checkbox(ui_tsne.listWidget_check_point)
        # data_plot_new = data_plot_new.loc[data_plot_new['point'].isin(list_drop_point)]
        #
        # data_plot_new.reset_index(inplace=True, drop=True)

        # list_drop_param = get_list_check_checkbox(ui_tsne.listWidget_check_param, is_checked=False)
        # data_plot_new = data_plot_new.drop(list_drop_param, axis=1)
        #
        # Сохраняем индекс исходных данных
        original_index = data_plot.index

        data_point = data_plot[['prof_well_index', 'mark']]
        data_tsne = data_plot.drop(['prof_well_index', 'mark'], axis=1)

        list_checked_param = get_list_check_checkbox(ui_prt.listWidget_check_param)
        data_tsne = data_tsne[list_checked_param]

        if ui_prt.checkBox_standart.isChecked():
            scaler = StandardScaler()
            data_tsne = scaler.fit_transform(data_tsne)
        if ui_prt.radioButton_tsne.isChecked():
            name_graph = 't-SNE'
            tsne = TSNE(
                n_components=2,
                perplexity=ui_prt.spinBox_perplexity.value(),
                learning_rate=200,
                random_state=42
            )
            data_tsne_result = tsne.fit_transform(data_tsne)
        if ui_prt.radioButton_pca.isChecked():
            name_graph = 'PCA'
            pca = PCA(
                n_components=2,
                random_state=42
            )
            data_tsne_result = pca.fit_transform(data_tsne)
        # Создаем DataFrame с правильным индексом
        data_tsne_result = pd.DataFrame(
            data_tsne_result,
            columns=['0', '1'],
            index=original_index
        )
        data_plot_new = pd.concat([data_point, data_tsne_result], axis=1)

        sns.scatterplot(data=data_plot_new, x='0', y='1', hue='mark', s=100, palette=pallet)

        # if ui_tsne.checkBox_name_point.isChecked():
        #     # Добавление подписей к точкам
        #     for i_data in data_plot_new.index:
        #         plt.text(data_plot_new['0'][i_data], data_plot_new['1'][i_data],
        #                  data_plot_new['point'][i_data], horizontalalignment='left',
        #                  size='medium', color='black', weight='semibold')
        # try:
        #     plt.vlines(
        #         x=data_plot_new['0'].loc[data_plot_new['point'] == ui_tsne.listWidget_point.currentItem().text()],
        #         ymin=data_plot_new['1'].min(),
        #         ymax=data_plot_new['1'].max(),
        #         color='black',
        #         linestyle='--'
        #     )
        #     plt.hlines(
        #         y=data_plot_new['1'].loc[data_plot_new['point'] == ui_tsne.listWidget_point.currentItem().text()],
        #         xmin=data_plot_new['0'].min(),
        #         xmax=data_plot_new['0'].max(),
        #         color='black',
        #         linestyle='--'
        #     )
        # except AttributeError:
        #     print('AttributeError')
        #     pass

        figure_tsne.suptitle(f'{name_graph}\n{len(data_plot_new.index)} точек')
        figure_tsne.tight_layout()
        canvas_tsne.draw()

    def calc_pareto():

        clear_layout(ui_prt.verticalLayout_anova)
        figure_pareto = plt.figure()
        canvas_pareto = FigureCanvas(figure_pareto)
        mpl_toolbar = NavigationToolbar(canvas_pareto)
        ui_prt.verticalLayout_anova.addWidget(mpl_toolbar)
        ui_prt.verticalLayout_anova.addWidget(canvas_pareto)

        time_start = datetime.datetime.now()
        ui_prt.progressBar_1.setMaximum(ui_prt.spinBox_poreto.value())
        ui_prt.progressBar_1.setValue(0)
        n = 0
        # Загрузка данных
        X = data_plot.iloc[:, 2:]  # Все параметры
        y = data_plot.iloc[:, 1]  # Метки классов

        # Определение задачи
        n_features = X.shape[1]

        problem = Problem(n_features, 1 if ui_prt.radioButton_pareto_no.isChecked() else 2)  # n_features переменных, 2 цели

        # Важно: создаем отдельный объект Binary для каждой переменной
        for i in range(n_features):
            problem.types[i] = Binary(1)  # Указываем размерность 1 для каждой переменной

        problem.directions[0] = Problem.MAXIMIZE  # Максимизация расстояния
        if ui_prt.radioButton_pareto_min.isChecked():
            problem.directions[1] = Problem.MINIMIZE  # Минимизация числа признаков
        elif ui_prt.radioButton_pareto_max.isChecked():
            problem.directions[1] = Problem.MAXIMIZE
        else:
            pass

        # Целевая функция остается без изменений
        def objectives(features):
            ui_prt.progressBar_1.setValue(ui_prt.progressBar_1.value() + 1)
            ui_prt.lcdNumber.display(ui_prt.progressBar_1.value())
            selected_features = np.array(features, dtype=int)
            if np.sum(selected_features) == 0:
                return [0, n_features]

            # Выбор активных признаков
            X_subset = X.loc[:, selected_features == 1]

            # Стандартизация данных
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_subset)

            # Применяем PCA
            pca = PCA(n_components=min(2, X_subset.shape[1]))  # Используем 2 компоненты или меньше
            X_pca = pca.fit_transform(X_scaled)

            # Разделяем данные по классам после PCA
            X_pca_0 = X_pca[y == list_markers[0]]
            X_pca_1 = X_pca[y == list_markers[1]]

            # Расчет центроидов в пространстве PCA
            centroid_0 = X_pca_0.mean(axis=0)
            centroid_1 = X_pca_1.mean(axis=0)

            # Евклидово расстояние между центроидами
            distance = np.linalg.norm(centroid_0 - centroid_1)
            count = np.sum(selected_features)

            print(distance, count, ui_prt.progressBar_1.value())

            if ui_prt.radioButton_pareto_no.isChecked():
                return [distance]
            else:
                return [distance, count]

        problem.function = objectives



        # # Создаем алгоритм с параметрами
        # algorithm = NSGAII(
        #     problem,
        #     population_size=200,
        #     generator=RandomGenerator(),
        #     variator=GAOperator(
        #         SBX(probability=1.0, distribution_index=15.0),
        #         PM(probability=1.0 / n_features)
        #     )
        # )
        if ui_prt.radioButton_pareto_no.isChecked():
            algorithm = GeneticAlgorithm(problem)
        else:
            algorithm = NSGAII(problem)


        # Запускаем оптимизацию
        algorithm.run(ui_prt.spinBox_poreto.value())

        # Остальной код без изменений
        solutions = []
        for solution in algorithm.result:
            solutions.append({
                "features": solution.variables,
                "distance": solution.objectives[0],
                "count": np.sum(solution.variables)
            })

        # Создание осей внутри фигуры
        ax = figure_pareto.add_subplot(111)

        # Построение точек на графике
        counts = [s["count"] for s in solutions]
        distances = [s["distance"] for s in solutions]

        ax.scatter(counts, distances, alpha=0.5)
        ax.set_xlabel("Количество признаков")
        ax.set_ylabel("Евклидово расстояние")
        ax.set_title("Парето-фронт")
        ax.grid(True)

        # Обновление канвы
        canvas_pareto.draw()


        print(datetime.datetime.now() - time_start)

        result = QtWidgets.QMessageBox.question(
            MainWindow,
            'Парето-фронт',
            'Сохранить результат?',
            QtWidgets.QMessageBox.Yes,
            QtWidgets.QMessageBox.No)


        if result == QtWidgets.QMessageBox.Yes:

            if ui_prt.radioButton_pareto_min.isChecked():
                p_type = 'MIN'
            elif ui_prt.radioButton_pareto_max.isChecked():
                p_type = 'MAX'
            else:
                p_type = 'NO'

            new_pareto = ParetoAnalysis(
                analysis_mlp_id = get_MLP_id(),
                n_iter = ui_prt.spinBox_poreto.value(),
                problem_type = p_type,
                start_params = json.dumps(list_param)
            )
            session.add(new_pareto)
            session.commit()


            for solution in solutions:
                new_pareto_result = ParetoResult(
                    pareto_analysis_id = new_pareto.id,
                    pareto_data = json.dumps(list(solution["features"])),
                    distance = solution["distance"]
                )
                session.add(new_pareto_result)
            session.commit()

            update_list_pareto()


    def set_list_pareto_analysis():
        id_pareto = ui_prt.listWidget_pareto_analysis.currentItem().text().split('_id')[-1]
        if not id_pareto:
            QMessageBox.critical(MainWindow, 'Парето-фронт', 'Не выбран анализ')
            return
        pareto_analysis = session.query(ParetoAnalysis).filter_by(
            id=id_pareto
        ).first()
        solutions = session.query(ParetoResult).filter_by(
            pareto_analysis_id=pareto_analysis.id
        ).order_by(desc(ParetoResult.distance)).all()

        ui_prt.listWidget_population.clear()
        list_distance, list_count = [], []
        for solution in solutions:
            dist = solution.distance
            n_count = np.sum(json.loads(solution.pareto_data))
            list_distance.append(dist)
            list_count.append(n_count)

            ui_prt.listWidget_population.addItem(f'{n_count} - {dist}_id{solution.id}')

        clear_layout(ui_prt.verticalLayout_anova)
        figure_pareto = plt.figure()
        canvas_pareto = FigureCanvas(figure_pareto)
        mpl_toolbar = NavigationToolbar(canvas_pareto)
        ui_prt.verticalLayout_anova.addWidget(mpl_toolbar)
        ui_prt.verticalLayout_anova.addWidget(canvas_pareto)

        ax = figure_pareto.add_subplot(111)

        ax.scatter(list_count, list_distance, alpha=0.5)
        ax.set_xlabel("Количество признаков")
        ax.set_ylabel("Евклидово расстояние")
        ax.set_title("Парето-фронт")
        ax.grid(True)

        # Обновление канвы
        canvas_pareto.draw()

    def set_list_pareto_result():
        try:
            pareto_result = session.query(ParetoResult).filter_by(
                id=ui_prt.listWidget_population.currentItem().text().split('_id')[-1]
            ).first()
        except AttributeError:
            return
        list_check = json.loads(pareto_result.pareto_data)
        for i in range(ui_prt.listWidget_check_param.count()):
            checkbox = ui_prt.listWidget_check_param.itemWidget(ui_prt.listWidget_check_param.item(i))
            checkbox.setChecked(list_check[i][0])


    def best_params():
        pareto_analysis = session.query(ParetoAnalysis).filter_by(
            id=ui_prt.listWidget_pareto_analysis.currentItem().text().split('_id')[-1]
        ).first()
        solutions = session.query(ParetoResult).filter_by(
            pareto_analysis_id=pareto_analysis.id
        ).all()

        dict_params = {}
        for i in list_param:
            dict_params[i] = 0

        for solution in solutions:
            dict_p = json.loads(solution.pareto_data)
            for i in range(len(list_param)):
                if dict_p[i][0]:
                    dict_params[list_param[i]] += 1

        dict_params = {k: v for k, v in sorted(dict_params.items(), key=lambda item: item[1], reverse=True) if v > 80}

        clear_layout(ui_prt.verticalLayout_anova)
        figure_best_param = plt.figure()
        canvas_best_param = FigureCanvas(figure_best_param)
        mpl_toolbar = NavigationToolbar(canvas_best_param)
        ui_prt.verticalLayout_anova.addWidget(mpl_toolbar)
        ui_prt.verticalLayout_anova.addWidget(canvas_best_param)

        ax = figure_best_param.add_subplot(111)

        ax.bar(list(dict_params.keys()), list(dict_params.values()))
        ax.set_xlabel("Количество признаков")
        ax.set_ylabel("Параметр")
        ax.set_title("Лучшие параметры")
        ax.grid(True)

        # Обновление канвы
        canvas_best_param.draw()


    def calc_correlation():
        pareto_analysis = session.query(ParetoAnalysis).filter_by(
            id=ui_prt.listWidget_pareto_analysis.currentItem().text().split('_id')[-1]
        ).first()
        solutions = session.query(ParetoResult).filter_by(
            pareto_analysis_id=pareto_analysis.id
        ).all()

        dict_params = {}
        for i in solutions:
            data_param = json.loads(i.pareto_data)
            dict_params[i.id] = [int(i[0]) for i in data_param]

        corr_data = pd.DataFrame.from_dict(dict_params, orient='index')
        corr_data = corr_data.transpose()
        corr_result = corr_data.corr()

        dict_corr = {}
        for i in corr_result.columns:
            dict_corr[i] = np.sum(corr_result[i])

        clear_layout(ui_prt.verticalLayout_anova)
        figure_correlation = plt.figure()
        canvas_correlation = FigureCanvas(figure_correlation)
        mpl_toolbar = NavigationToolbar(canvas_correlation)
        ui_prt.verticalLayout_anova.addWidget(mpl_toolbar)
        ui_prt.verticalLayout_anova.addWidget(canvas_correlation)

        ax = figure_correlation.add_subplot(111)

        ax.bar(list(dict_corr.keys()), list(dict_corr.values()))

        # ax.set_xlabel("Количество признаков")
        # ax.set_ylabel("Евклидово расстояние")
        # ax.set_title("Парето-фронт")
        ax.grid(True)

        # Обновление канвы
        canvas_correlation.draw()


    def set_list_check_param():
        all_check(ui_prt.listWidget_check_param, ui_prt.checkBox_param_all)

    def remove_pareto_analysis():
        id_pareto = ui_prt.listWidget_pareto_analysis.currentItem().text().split('_id')[-1]
        if not id_pareto:
            QMessageBox.critical(MainWindow, 'Ошибка', 'Выберите анализ')
            return
        # if session.query(ParameterMask).filter_by(pareto_analysis_id=id_pareto).count() > 0:
        #     QMessageBox.critical(MainWindow, 'Ошибка', 'Невозможно удалить этот анализ так как по нему есть сохраненные маски параметров')
        #     return
        result = QMessageBox.question(
            MainWindow, "Удаление анализа",
            "Вы действительно хотите удалить анализ?",
            QMessageBox.Yes | QMessageBox.No
        )

        if result == QMessageBox.No:
            return
        else:
            pareto_analysis = session.query(ParetoAnalysis).filter_by(
                id=id_pareto
            ).first()
            for i in session.query(ParetoResult).filter_by(pareto_analysis_id=pareto_analysis.id).all():
                session.delete(i)
            session.delete(pareto_analysis)
            session.commit()
            update_list_pareto()


    def save_mask():
        info = (f'cls analysis: {ui.comboBox_mlp_analysis.currentText().split(" id")[0]}\n'
                f'pareto analysis: {ui_prt.listWidget_pareto_analysis.currentItem().text().split("_id")[0]}')
        list_checked_param = get_list_check_checkbox(ui_prt.listWidget_check_param)
        print(list_checked_param)
        new_mask = ParameterMask(
            count_param = len(list_checked_param),
            mask = json.dumps(list_checked_param),
            mask_info = info
        )
        session.add(new_mask)
        session.commit()
        QMessageBox.information(MainWindow, 'Info', 'Маска сохранена')
        update_list_saved_mask()


    def remove_mask():
        id_mask = ui_prt.listWidget_saved_mask.currentItem().text().split(' id')[-1]
        if not id_mask:
            QMessageBox.critical(MainWindow, 'Ошибка', 'Выберите маску')
            return
        result = QMessageBox.question(
            MainWindow, "Удаление маски",
            "Вы действительно хотите удалить маску?",
            QMessageBox.Yes | QMessageBox.No
        )

        if result == QMessageBox.No:
            return
        else:
            session.query(ParameterMask).filter_by(id=id_mask).delete()
            session.commit()
            update_list_saved_mask()


    def export_mask():
        mask_to_export = session.query(ParameterMask).filter_by(id=ui_prt.listWidget_saved_mask.currentItem().text().split(' id')[-1]).first()

        file_name = QFileDialog.getSaveFileName(caption='Export mask', filter="*.pkl")[0]
        if not file_name:
            return
        with open(file_name, 'wb') as f:
            pickle.dump(mask_to_export, f)
        QMessageBox.information(MainWindow, 'Info', 'Маска экспортирована')


    def import_mask():
        file_name = QFileDialog.getOpenFileName(caption='Import mask', filter="*.pkl")[0]
        if not file_name:
            return
        with open(file_name, 'rb') as f:
            mask = pickle.load(f)

        import_mask = ParameterMask(
            count_param = mask.count_param,
            mask = mask.mask,
            mask_info = mask.mask_info
        )
        session.add(import_mask)
        session.commit()
        QMessageBox.information(MainWindow, 'Info', 'Маска импортирована')
        update_list_saved_mask()

    ui_prt.pushButton_apply.clicked.connect(draw_graph_tsne)
    ui_prt.pushButton_pareto.clicked.connect(calc_pareto)
    ui_prt.listWidget_pareto_analysis.clicked.connect(set_list_pareto_analysis)
    ui_prt.listWidget_population.clicked.connect(set_list_pareto_result)
    ui_prt.checkBox_param_all.clicked.connect(set_list_check_param)
    ui_prt.pushButton_best_params.clicked.connect(best_params)
    ui_prt.pushButton_correlation.clicked.connect(calc_correlation)
    ui_prt.pushButton_rm_pareto.clicked.connect(remove_pareto_analysis)
    ui_prt.pushButton_save_mask.clicked.connect(save_mask)
    ui_prt.pushButton_rm_mask.clicked.connect(remove_mask)
    ui_prt.pushButton_export_mask.clicked.connect(export_mask)
    ui_prt.pushButton_import_mask.clicked.connect(import_mask)



    update_list_pareto()
    set_list_point()
    update_list_saved_mask()

    Pareto.exec_()

