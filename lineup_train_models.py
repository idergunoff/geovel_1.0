from func import *

def model_lineup():
    lineupModel = QtWidgets.QDialog()
    ui_l = Ui_Dialog_model_lineup()
    ui_l.setupUi(lineupModel)
    lineupModel.show()
    lineupModel.setAttribute(QtCore.Qt.WA_DeleteOnClose)

    m_width, m_height = get_width_height_monitor()
    lineupModel.resize(int(m_width/3), int(m_height/3))

    def update_lineup_list():
        ui_l.listWidget_lineup.clear()
        for i in session.query(LineupTrain).all():
            try:
                params = list(json.loads(i.list_param))
                item_text = f'{i.type_ml} {i.text_model.split(":")[0]} params: {len(params)}'
                item = QListWidgetItem(item_text)
                item.setData(Qt.UserRole, i.id)
                ui_l.listWidget_lineup.addItem(item)
            except AttributeError:
                pass

    def update_lineup_info():
        ui_l.plainTextEdit_info.clear()
        model = session.query(LineupTrain).filter_by(id=ui_l.listWidget_lineup.currentItem().data(Qt.UserRole)).first()
        ui_l.plainTextEdit_info.insertPlainText(f'{model.text_model} \n\n {model.list_param}')
        # ui_l.plainTextEdit_info.insertPlainText(model.list_param)

    def remove_lineup_model():
        model = session.query(LineupTrain).filter_by(id=ui_l.listWidget_lineup.currentItem().data(Qt.UserRole)).first()
        session.delete(model)
        session.commit()
        update_lineup_list()
        set_info(f'Модель {model.text_model.split(":")[0]} удалена из списка', 'green')


    def clear_lineup_model():
        for model in session.query(LineupTrain).all():
            session.delete(model)
            session.commit()
        update_lineup_list()
        ui_l.plainTextEdit_info.clear()
        set_info('Список моделей очищен', 'green')


    def train_lineup_models():
        models = session.query(LineupTrain).all()
        ui.progressBar.setMaximum(len(models))
        for n, i in enumerate(models):
            train_lineup_model(i)
            session.query(LineupTrain).filter_by(id=i.id).delete()
            session.commit()
            update_lineup_list()
            ui.progressBar.setValue(n + 1)


    update_lineup_list()
    ui_l.listWidget_lineup.clicked.connect(update_lineup_info)
    ui_l.pushButton_remove.clicked.connect(remove_lineup_model)
    ui_l.pushButton_clear.clicked.connect(clear_lineup_model)
    ui_l.pushButton_start.clicked.connect(train_lineup_models)
    lineupModel.exec_()


def train_lineup_model(model):
    if model.type_ml == 'cls':
        train_cls_model(model)
    if model.type_ml == 'reg':
        train_reg_model(model)


def train_cls_model(model):

    start_time = datetime.datetime.now()
    analysis_cls = session.query(AnalysisMLP).filter_by(id=model.analysis_id).first()

    ui.comboBox_mlp_analysis.setCurrentText(f'{analysis_cls.title} id{analysis_cls.id}')

    data_train, _ = build_table_train(True, 'mlp')
    list_param = json.loads(model.list_param)

    list_nan_param, count_nan = [], 0
    for i in data_train.index:
        for param in list_param:
            if pd.isna(data_train[param][i]):
                count_nan += 1
                list_nan_param.append(param)
    if count_nan > 0:
        list_col = data_train.columns.tolist()
        data_train = pd.DataFrame(imputer.fit_transform(data_train), columns=list_col)
        set_info(f'Заполнены пропуски в {count_nan} образцах в параметрах {", ".join(list_nan_param)}', 'red')

    training_sample = np.array(data_train[list_param].values.tolist())
    markup = np.array(sum(data_train[['mark']].values.tolist(), []))

    # Разделение данных на обучающую и тестовую выборки
    training_sample_train, training_sample_test, markup_train, markup_test = train_test_split(
        training_sample, markup, test_size=0.20, random_state=1, stratify=markup)

    if model.over_sampling == 'smote':
        smote = SMOTE(random_state=0)
        training_sample_train, markup_train = smote.fit_resample(training_sample_train, markup_train)

    if model.over_sampling == 'adasyn':
        try:
            adasyn = ADASYN(random_state=0)
            training_sample_train, markup_train = adasyn.fit_resample(training_sample_train, markup_train)
        except ValueError:
            set_info('Невозможно применить ADASYN c n_neighbors=5, значение уменьшено до n_neighbors=3', 'red')
            adasyn = ADASYN(random_state=0, n_neighbors=3)
            training_sample_train, markup_train = adasyn.fit_resample(training_sample_train, markup_train)

    pipe = pickle.loads(model.pipe)
    pipe.fit(training_sample_train, markup_train)

    # Оценка точности на всей обучающей выборке
    train_accuracy = pipe.score(training_sample, markup)
    test_accuracy = pipe.score(training_sample_test, markup_test)

    train_time = datetime.datetime.now() - start_time
    text_model = model.text_model
    text_model += f'\ntrain_accuracy: {round(train_accuracy, 4)}, test_accuracy: {round(test_accuracy, 4)}, \nвремя обучения: {train_time}'
    set_info(text_model, 'blue')

    preds_train = pipe.predict(training_sample)

    preds_proba_train = pipe.predict_proba(training_sample)
    try:
        tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=42)
        train_tsne = tsne.fit_transform(preds_proba_train)
        data_tsne = pd.DataFrame(train_tsne)
        data_tsne['mark'] = preds_train
    except ValueError:
        tsne = TSNE(n_components=2, perplexity=len(data_train.index) - 1, learning_rate=200, random_state=42)
        train_tsne = tsne.fit_transform(preds_proba_train)
        data_tsne = pd.DataFrame(train_tsne)
        data_tsne['mark'] = preds_train

    (fig_train, axes) = plt.subplots(nrows=1, ncols=2)
    fig_train.set_size_inches(20, 10)

    sns.scatterplot(data=data_tsne, x=0, y=1, hue='mark', s=200, ax=axes[0])
    axes[0].grid()
    axes[0].xaxis.grid(True, "minor", linewidth=.25)
    axes[0].yaxis.grid(True, "minor", linewidth=.25)
    axes[0].set_title(f'Диаграмма рассеяния для канонических значений {model.model_name}\nдля обучающей выборки и тестовой выборки')
    list_marker = list(pipe.classes_)
    if len(list_marker) == 2:
        # Вычисляем ROC-кривую и AUC
        preds_test = pipe.predict_proba(training_sample_test)[:, 0]
        fpr, tpr, thresholds = roc_curve(markup_test, preds_test, pos_label=list_marker[0])
        roc_auc = auc(fpr, tpr)

        # Строим ROC-кривую
        axes[1].plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        axes[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        axes[1].set_xlim([0.0, 1.0])
        axes[1].set_ylim([0.0, 1.05])
        axes[1].set_xlabel('False Positive Rate')
        axes[1].set_ylabel('True Positive Rate')
        axes[1].set_title('ROC-кривая')
        axes[1].legend(loc="lower right")

    title_graph = text_model

    fig_train.suptitle(title_graph)
    fig_train.tight_layout()
    fig_train.savefig(f'lineup_pictures/{model.model_name}_{train_accuracy}.png', dpi=300)

    # Сохранение модели в файл с помощью pickle
    path_model = f'models/classifier/{model.model_name}_{round(test_accuracy, 3)}_{datetime.datetime.now().strftime("%d%m%y")}.pkl'
    if os.path.exists(path_model):
        path_model = f'models/classifier/{model.model_name}_{round(test_accuracy, 3)}_{datetime.datetime.now().strftime("%d%m%y_%H%M%S")}.pkl'
    with open(path_model, 'wb') as f:
        pickle.dump(pipe, f)

    new_trained_model = TrainedModelClass(
        analysis_id=get_MLP_id(),
        title=f'{model.model_name}_{round(test_accuracy, 3)}_{datetime.datetime.now().strftime("%d%m%y")}',
        path_model=path_model,
        list_params=json.dumps(list_param),
        except_signal=ui.lineEdit_signal_except.text(),
        except_crl=ui.lineEdit_crl_except.text(),
        comment=text_model
    )
    session.add(new_trained_model)
    session.commit()

def train_reg_model(model):
    pass

















