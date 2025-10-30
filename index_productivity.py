from func import *
from krige import draw_map


def index_prod_add():
    new_ix_prod = IndexProductivity(
        research_id=get_research_id(),
        prediction_id=ui.listWidget_model_pred.currentItem().text().split(' id')[-1]
    )
    session.add(new_ix_prod)
    session.commit()

    index_prod_list_update()


def index_prod_remove():
    session.query(IndexProductivity).filter_by(id=ui.listWidget_ix_prod.currentItem().text().split('_id')[-1]).delete()
    session.commit()

    index_prod_list_update()


def index_prod_list_update():
    ui.listWidget_ix_prod.clear()
    try:
        for i in session.query(IndexProductivity).filter_by(research_id=get_research_id()).all():
            try:
                if i.prediction.type_model == 'cls':
                    model = session.query(TrainedModelClass).filter_by(id=i.prediction.model_id).first()
                else:
                    model = session.query(TrainedModelReg).filter_by(id=i.prediction.model_id).first()
                item = f'{i.prediction.type_model}-{model.title}_id{i.id}'
                ui.listWidget_ix_prod.addItem(item)
            except AttributeError:
                session.delete(i)
                session.commit()
                continue
    except ValueError:
        return

def index_prod_clear():
    for i in session.query(IndexProductivity).filter_by(research_id=get_research_id()).all():
        session.delete(i)
    session.commit()
    set_info('Список индекса продуктивности очищен', 'blue')
    index_prod_list_update()


def index_prod_draw():
    pd_ix_prod = build_table_index_productivity()

    pd_ix_prod = calc_index_productivity(pd_ix_prod)

    draw_map(pd_ix_prod['x_pulc'], pd_ix_prod['y_pulc'], pd_ix_prod['norm'], 'Индекс производительности')


def index_prod_save():
    pass


def build_table_index_productivity():
    list_prediction = []
    for i in session.query(IndexProductivity).filter_by(research_id=get_research_id()).all():
        if i.prediction:
            list_prediction.append(i.prediction)
        else:
            session.delete(i)
            session.commit()
    print(list_prediction)

    pd_ix_prod = pd.DataFrame(columns=['x_pulc', 'y_pulc'])
    for pr in session.query(Profile).filter_by(research_id=get_research_id()).all():
        ix_prod_dict = {}
        ix_prod_dict['x_pulc'] = json.loads(pr.x_pulc)
        ix_prod_dict['y_pulc'] = json.loads(pr.y_pulc)

        for ni, i in enumerate(list_prediction):
            pred = session.query(ProfileModelPrediction).filter_by(profile_id=pr.id, model_id=i.model_id).first()
            if not pred:
                set_info(f'Модель не расчитана для профиля {pr.title}', 'red')
                QtWidgets.QMessageBox.critical(MainWindow, 'Error', f'Модель не расчитана для профиля {pr.title}')
                return

            if ui.checkBox_corr_pred.isChecked() and pred.corrected:
                value = json.loads(pred.corrected[0].correct)
            else:
                value = json.loads(pred.prediction)
            ix_prod_dict[f'value_{ni}'] = value

        pd_ix_prod = pd.concat([pd_ix_prod, pd.DataFrame(ix_prod_dict)], ignore_index=True)

    return pd_ix_prod


def calc_index_productivity(df):
    # Копируем DataFrame, чтобы не менять оригинал
    result_df = df.copy()

    # Пропускаем первые два столбца (координаты)
    columns_to_process = df.columns[2:]

    # Создаем копию для нормализации
    normalized_columns = []

    # Нормализуем каждый столбец по методу min-max
    for col in columns_to_process:
        if ui.checkBox_ix_prod_norm.isChecked():
            col_min = df[col].min()
            col_max = df[col].max()

            # Избегаем деления на ноль
            if col_min == col_max:
                normalized_col = np.zeros_like(df[col], dtype=float)
            else:
                normalized_col = (df[col] - col_min) / (col_max - col_min)
        else:
            normalized_col = df[col]

        normalized_columns.append(normalized_col)

    # Перемножаем нормализованные столбцы
    multiplication_result = np.prod(normalized_columns, axis=0)

    # Добавляем результат в DataFrame
    result_df['norm'] = multiplication_result

    return result_df