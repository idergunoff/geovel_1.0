from remote_db.model_remote_db import *
from models_db.model import *
from qt.rem_db_window import *
from func import *

def unload_formation_feature():
    """Выгрузка параметро таблицы Formation с локальной БД на удаленную"""

    set_info('Начало выгрузки парметров пластов с локальной БД на удаленную', 'blue')

    set_info('Проверка наличия связанных пластов в удаленной БД', 'blue')

    # Предзагрузка данных из удаленной БД для проверки
    with get_session() as remote_session:
        remote_formations = {}
        for f in remote_session.query(FormationRDB.up_hash, FormationRDB.down_hash, FormationRDB.id).all():
            remote_formations[f.up_hash] = f.id
            remote_formations[f.down_hash] = f.id

        local_features = session.query(
            Formation
        ).all()

        problems = False

        for local_feature in tqdm(local_features, desc='Проверка наличия связанных пластов в удаленной БД'):

            if (local_feature.up_hash not in remote_formations and
                    local_feature.down_hash not in remote_formations):
                problems = True

        if problems:
            error_info = (f'Отсутствуют данные в таблице FormationRDB.')
            error_info += "\n\nНеобходимо сначала выгрузить пласты с локальной БД."
            set_info('Обнаружены проблемы с зависимостями', 'red')
            QMessageBox.critical(MainWindow, 'Ошибка зависимостей. Выгрузка данных прекращена.', error_info)
            return
        else:
            set_info('Проблем с зависимостями нет', 'green')

        try:
            session.commit()
        except Exception as e:
            session.rollback()
            error_info = 'Ошибка при удалении записей без связанных пластов в локальной БД'
            set_info('Обнаружены проблемы с зависимостями в локальной БД', 'red')
            QMessageBox.critical(MainWindow, 'Ошибка зависимостей', error_info)
            return


    # Если проверка пройдена, выполняем выгрузку
    with get_session() as remote_session:
        local_features = session.query(
            Formation
        ).all()
        remote_formations = {}
        for f in remote_session.query(FormationRDB.up_hash, FormationRDB.down_hash, FormationRDB.id).all():
            remote_formations[f.up_hash] = [f.id]
            remote_formations[f.down_hash] = [f.id]

        ui.progressBar.setMaximum(len(local_features))
        new_feature_count = 0

        for n, local_feature in tqdm(enumerate(local_features), desc='Выгрузка параметров пластов'):
            ui.progressBar.setValue(n + 1)

            # Получаем ID пласта
            remote_formation_list = remote_formations.get(local_feature.up_hash)
            if not remote_formation_list:
                remote_formation_list = remote_formations.get(local_feature.down_hash)

            remote_feature_count = remote_session.query(FormationFeatureRDB).filter_by(
                formation_id=remote_formation_list[0]
            ).count()

            if remote_feature_count == 0:
                new_feature = FormationFeatureRDB(
                    formation_id=remote_formation_list[0],
                    T_top=local_feature.T_top,
                    T_bottom=local_feature.T_bottom,
                    dT=local_feature.dT,

                    A_top=local_feature.A_top,
                    A_bottom=local_feature.A_bottom,
                    dA=local_feature.dA,
                    A_sum=local_feature.A_sum,
                    A_mean=local_feature.A_mean,

                    dVt=local_feature.dVt,
                    Vt_top=local_feature.Vt_top,
                    Vt_sum=local_feature.Vt_sum,
                    Vt_mean=local_feature.Vt_mean,

                    dAt=local_feature.dAt,
                    At_top=local_feature.At_top,
                    At_sum=local_feature.At_sum,
                    At_mean=local_feature.At_mean,

                    dPht=local_feature.dPht,
                    Pht_top=local_feature.Pht_top,
                    Pht_sum=local_feature.Pht_sum,
                    Pht_mean=local_feature.Pht_mean,

                    Wt_top=local_feature.Wt_top,
                    Wt_mean=local_feature.Wt_mean,
                    Wt_sum=local_feature.Wt_sum,

                    width=local_feature.width,
                    top=local_feature.top,
                    land=local_feature.land,
                    speed=local_feature.speed,
                    speed_cover=local_feature.speed_cover,

                    skew=local_feature.skew,
                    kurt=local_feature.kurt,
                    std=local_feature.std,
                    k_var=local_feature.k_var,

                    A_max=local_feature.A_max,
                    Vt_max=local_feature.Vt_max,
                    At_max=local_feature.At_max,
                    Pht_max=local_feature.Pht_max,
                    Wt_max=local_feature.Wt_max,

                    A_T_max=local_feature.A_T_max,
                    Vt_T_max=local_feature.Vt_T_max,
                    At_T_max=local_feature.At_T_max,
                    Pht_T_max=local_feature.Pht_T_max,
                    Wt_T_max=local_feature.Wt_T_max,

                    A_Sn=local_feature.A_Sn,
                    Vt_Sn=local_feature.Vt_Sn,
                    At_Sn=local_feature.At_Sn,
                    Pht_Sn=local_feature.Pht_Sn,
                    Wt_Sn=local_feature.Wt_Sn,

                    A_wmf=local_feature.A_wmf,
                    Vt_wmf=local_feature.Vt_wmf,
                    At_wmf=local_feature.At_wmf,
                    Pht_wmf=local_feature.Pht_wmf,
                    Wt_wmf=local_feature.Wt_wmf,

                    A_Qf=local_feature.A_Qf,
                    Vt_Qf=local_feature.Vt_Qf,
                    At_Qf=local_feature.At_Qf,
                    Pht_Qf=local_feature.Pht_Qf,
                    Wt_Qf=local_feature.Wt_Qf,

                    A_Sn_wmf=local_feature.A_Sn_wmf,
                    Vt_Sn_wmf=local_feature.Vt_Sn_wmf,
                    At_Sn_wmf=local_feature.At_Sn_wmf,
                    Pht_Sn_wmf=local_feature.Pht_Sn_wmf,
                    Wt_Sn_wmf=local_feature.Wt_Sn_wmf,

                    CRL_top=local_feature.CRL_top,
                    CRL_bottom=local_feature.CRL_bottom,
                    dCRL=local_feature.dCRL,
                    CRL_sum=local_feature.CRL_sum,
                    CRL_mean=local_feature.CRL_mean,
                    CRL_max=local_feature.CRL_max,
                    CRL_T_max=local_feature.CRL_T_max,
                    CRL_Sn=local_feature.CRL_Sn,
                    CRL_wmf=local_feature.CRL_wmf,
                    CRL_Qf=local_feature.CRL_Qf,
                    CRL_Sn_wmf=local_feature.CRL_Sn_wmf,

                    CRL_skew=local_feature.CRL_skew,
                    CRL_kurt=local_feature.CRL_kurt,
                    CRL_std=local_feature.CRL_std,
                    CRL_k_var=local_feature.CRL_k_var,

                    k_r=local_feature.k_r
                )
                remote_session.add(new_feature)
                new_feature_count += 1

        remote_session.commit()
        added_word = "Добавлена" if new_feature_count == 1 else "Добавлено"
        set_info(
            f'{added_word} {pluralize(new_feature_count, ["новая запись", "новых записи", "новых записей"])} в таблицу FormationFeatureRDB',
            'green')

    set_info('Выгрузка данных с локальной БД на удаленную завершена', 'blue')


# def load_formation_feature():
#     """Загрузка таблицы FormationFeature с удаленной БД на локальную (в таблицу Formation)"""
#
#     set_info('Начало загрузки данных с удаленной БД на локальную', 'blue')
#
#     set_info('Проверка наличия связанных пластов в локальной БД', 'blue')
#     with get_session() as remote_session:
#
#         local_formations = {}
#         for f in session.query(Formation.up_hash, Formation.down_hash, Formation.id).all():
#             local_formations[f.up_hash] = f.id
#             local_formations[f.down_hash] = f.id
#
#         remote_features = remote_session.query(
#             FormationFeatureRDB.id,
#             FormationRDB.up_hash,
#             FormationRDB.down_hash
#         ).join(
#             FormationRDB, FormationFeatureRDB.formation_id == FormationRDB.id
#         ).all()
#
#         problems = False
#
#         for remote_feature, formation_up_hash, formation_down_hash in tqdm(remote_features, desc='Проверка наличия связанных пластов в локальной БД'):
#
#             # Проверяем пласт
#             if (formation_up_hash not in local_formations and
#                     formation_down_hash not in local_formations):
#                 problems = True
#
#         if problems:
#             error_info = (f'Отсутствуют данные в таблице FormationRDB.')
#             error_info += "\n\nНеобходимо сначала загрузить пласты с удаленной БД."
#             set_info('Обнаружены проблемы с зависимостями', 'red')
#             QMessageBox.critical(MainWindow, 'Ошибка зависимостей. Загрузка данных прекращена.', error_info)
#             return
#         else:
#             set_info('Проблем с зависимостями нет', 'green')
#
#     # Если проверка пройдена, выполняем выгрузку
#     with get_session() as remote_session:
#         remote_features = remote_session.query(
#             FormationFeatureRDB.id,
#             FormationRDB.up_hash,
#             FormationRDB.down_hash
#         ).join(
#             FormationRDB, FormationFeatureRDB.formation_id == FormationRDB.id
#         ).all()
#         local_formations = {}
#         for f in session.query(Formation.up_hash, Formation.down_hash, Formation.id).all():
#             local_formations[(f.up_hash, f.down_hash)] = f.id
#             # local_formations[f.up_hash] = f.id
#             # local_formations[f.down_hash] = f.id
#
#         ui.progressBar.setMaximum(len(remote_features))
#         new_feature_count = 0
#
#         for n, (remote_feature_id, formation_up_hash, formation_down_hash) in tqdm(enumerate(remote_features), desc='Загрузка параметров пластов'):
#             ui.progressBar.setValue(n+1)
#
#             # Получаем ID пласта
#             # local_formation_id = local_formations.get((formation_up_hash, formation_down_hash))
#
#             # local_formation_list = local_formations.get(formation_up_hash)
#             # if not local_formation_list:
#             #     local_formation_list = local_formations.get(formation_down_hash)
#             #
#             # local_feature_count = session.query(Formation).filter_by(
#             #     id=local_formation_list[0]
#             # ).count()
#             #
#             # # if local_formation_id:
#             #
#             # if local_feature_count == 0:
#             #     # local_formation = session.query(Formation).get(local_formation_list[0])
#             #     remote_feature = remote_session.get(FormationFeatureRDB, remote_feature_id)
#
#             # Получаем ID пласта по up_hash и down_hash
#             local_formation_id = local_formations.get((formation_up_hash, formation_down_hash))
#
#             if local_formation_id:
#                 remote_feature = remote_session.get(FormationFeatureRDB, remote_feature_id)
#
#                 # Обновляем существующую запись в Formation
#                 session.query(Formation).filter_by(id=local_formation_id).update({
#                     'T_top': remote_feature.T_top,
#                     'T_bottom': remote_feature.T_bottom,
#                     'dT': remote_feature.dT,
#
#                     'A_top': remote_feature.A_top,
#                     'A_bottom': remote_feature.A_bottom,
#                     'dA': remote_feature.dA,
#                     'A_sum': remote_feature.A_sum,
#                     'A_mean': remote_feature.A_mean,
#
#                     'dVt': remote_feature.dVt,
#                     'Vt_top': remote_feature.Vt_top,
#                     'Vt_sum': remote_feature.Vt_sum,
#                     'Vt_mean': remote_feature.Vt_mean,
#
#                     'dAt': remote_feature.dAt,
#                     'At_top': remote_feature.At_top,
#                     'At_sum': remote_feature.At_sum,
#                     'At_mean': remote_feature.At_mean,
#
#                     'dPht': remote_feature.dPht,
#                     'Pht_top': remote_feature.Pht_top,
#                     'Pht_sum': remote_feature.Pht_sum,
#                     'Pht_mean': remote_feature.Pht_mean,
#
#                     'Wt_top': remote_feature.Wt_top,
#                     'Wt_mean': remote_feature.Wt_mean,
#                     'Wt_sum': remote_feature.Wt_sum,
#
#                     'width': remote_feature.width,
#                     'top': remote_feature.top,
#                     'land': remote_feature.land,
#                     'speed': remote_feature.speed,
#                     'speed_cover': remote_feature.speed_cover,
#
#                     'skew': remote_feature.skew,
#                     'kurt': remote_feature.kurt,
#                     'std': remote_feature.std,
#                     'k_var': remote_feature.k_var,
#
#                     'A_max': remote_feature.A_max,
#                     'Vt_max': remote_feature.Vt_max,
#                     'At_max': remote_feature.At_max,
#                     'Pht_max': remote_feature.Pht_max,
#                     'Wt_max': remote_feature.Wt_max,
#
#                     'A_T_max': remote_feature.A_T_max,
#                     'Vt_T_max': remote_feature.Vt_T_max,
#                     'At_T_max': remote_feature.At_T_max,
#                     'Pht_T_max': remote_feature.Pht_T_max,
#                     'Wt_T_max': remote_feature.Wt_T_max,
#
#                     'A_Sn': remote_feature.A_Sn,
#                     'Vt_Sn': remote_feature.Vt_Sn,
#                     'At_Sn': remote_feature.At_Sn,
#                     'Pht_Sn': remote_feature.Pht_Sn,
#                     'Wt_Sn': remote_feature.Wt_Sn,
#
#                     'A_wmf': remote_feature.A_wmf,
#                     'Vt_wmf': remote_feature.Vt_wmf,
#                     'At_wmf': remote_feature.At_wmf,
#                     'Pht_wmf': remote_feature.Pht_wmf,
#                     'Wt_wmf': remote_feature.Wt_wmf,
#
#                     'A_Qf': remote_feature.A_Qf,
#                     'Vt_Qf': remote_feature.Vt_Qf,
#                     'At_Qf': remote_feature.At_Qf,
#                     'Pht_Qf': remote_feature.Pht_Qf,
#                     'Wt_Qf': remote_feature.Wt_Qf,
#
#                     'A_Sn_wmf': remote_feature.A_Sn_wmf,
#                     'Vt_Sn_wmf': remote_feature.Vt_Sn_wmf,
#                     'At_Sn_wmf': remote_feature.At_Sn_wmf,
#                     'Pht_Sn_wmf': remote_feature.Pht_Sn_wmf,
#                     'Wt_Sn_wmf': remote_feature.Wt_Sn_wmf,
#
#                     'CRL_top': remote_feature.CRL_top,
#                     'CRL_bottom': remote_feature.CRL_bottom,
#                     'dCRL': remote_feature.dCRL,
#                     'CRL_sum': remote_feature.CRL_sum,
#                     'CRL_mean': remote_feature.CRL_mean,
#                     'CRL_max': remote_feature.CRL_max,
#                     'CRL_T_max': remote_feature.CRL_T_max,
#                     'CRL_Sn': remote_feature.CRL_Sn,
#                     'CRL_wmf': remote_feature.CRL_wmf,
#                     'CRL_Qf': remote_feature.CRL_Qf,
#                     'CRL_Sn_wmf': remote_feature.CRL_Sn_wmf,
#
#                     'CRL_skew': remote_feature.CRL_skew,
#                     'CRL_kurt': remote_feature.CRL_kurt,
#                     'CRL_std': remote_feature.CRL_std,
#                     'CRL_k_var': remote_feature.CRL_k_var,
#
#                     'k_r': remote_feature.k_r
#                 })
#                 new_feature_count += 1
#
#         session.commit()
#         added_word = "Добавлена" if new_feature_count == 1 else "Добавлено"
#         set_info(
#             f'{added_word} {pluralize(new_feature_count, ["новая запись", "новых записи", "новых записей"])} в таблицу Formation',
#             'green')
#
#     set_info('Загрузка параметров пластов с удаленной БД на локальную завершена', 'blue')