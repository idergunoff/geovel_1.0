from remote_db.model_remote_db import *
from models_db.model import *
from qt.rem_db_window import *
from func import *

def unload_emd_feature():
    """Выгрузка таблицы EMDFeature с локальной БД на удаленную"""

    # set_info('Начало выгрузки данных с локальной БД на удаленную', 'blue')

    set_info('Проверка наличия связанных пластов для характеристик EMD в удаленной БД', 'blue')

    # Предзагрузка данных из удаленной БД для проверки
    with get_session() as remote_session:
        remote_formations = {}
        for f in remote_session.query(FormationRDB.up_hash, FormationRDB.down_hash, FormationRDB.id).all():
            remote_formations[f.up_hash] = f.id
            remote_formations[f.down_hash] = f.id

        local_features = session.query(
            EMDFeature,
            Formation.up_hash,
            Formation.down_hash
        ).join(
            Formation, EMDFeature.formation_id == Formation.id
        ).all()

        problems = False

        for local_feature, formation_up_hash, formation_down_hash in tqdm(local_features, desc='Проверка наличия связанных пластов в удаленной БД'):

            if local_feature.formation is None:
                session.delete(local_feature)
                set_info(f'В таблице EMDFeature удалена запись id{local_feature.id}, для которой отстутствует связанный пласт', 'black')
                continue

            if (formation_up_hash not in remote_formations and
                    formation_down_hash not in remote_formations):
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
            EMDFeature,
            Formation.up_hash,
            Formation.down_hash
        ).join(
            Formation, EMDFeature.formation_id == Formation.id
        ).all()
        remote_formations = {}
        for f in remote_session.query(FormationRDB.up_hash, FormationRDB.down_hash, FormationRDB.id).all():
            remote_formations[f.up_hash] = [f.id]
            remote_formations[f.down_hash] = [f.id]

        ui.progressBar.setMaximum(len(local_features))
        new_feature_count = 0

        for n, (local_feature, formation_up_hash, formation_down_hash) in tqdm(enumerate(local_features), desc='Выгрузка характеристик EMD'):
            ui.progressBar.setValue(n + 1)

            # Получаем ID пласта
            remote_formation_list = remote_formations.get(local_feature.formation.up_hash)
            if not remote_formation_list:
                remote_formation_list = remote_formations.get(local_feature.formation.down_hash)

            remote_feature_count = remote_session.query(EMDFeatureRDB).filter_by(
                formation_id=remote_formation_list[0]
            ).count()

            if remote_feature_count == 0:
                new_feature = EMDFeatureRDB(
                    formation_id=remote_formation_list[0],
                    emd_num_imfs=local_feature.emd_num_imfs,
                    emd_energ_mean=local_feature.emd_energ_mean,
                    emd_energ_med=local_feature.emd_energ_med,
                    emd_energ_max=local_feature.emd_energ_max,
                    emd_energ_min=local_feature.emd_energ_min,
                    emd_energ_std=local_feature.emd_energ_std,
                    emd_rel_energ_mean=local_feature.emd_rel_energ_mean,
                    emd_rel_energ_med=local_feature.emd_rel_energ_med,
                    emd_rel_energ_max=local_feature.emd_rel_energ_max,
                    emd_rel_energ_min=local_feature.emd_rel_energ_min,
                    emd_rel_energ_std=local_feature.emd_rel_energ_std,
                    emd_dom_freqs_mean=local_feature.emd_dom_freqs_mean,
                    emd_dom_freqs_med=local_feature.emd_dom_freqs_med,
                    emd_dom_freqs_max=local_feature.emd_dom_freqs_max,
                    emd_dom_freqs_min=local_feature.emd_dom_freqs_min,
                    emd_dom_freqs_std=local_feature.emd_dom_freqs_std,
                    emd_mean_corr=local_feature.emd_mean_corr,
                    emd_median_corr=local_feature.emd_median_corr,
                    emd_max_corr=local_feature.emd_max_corr,
                    emd_min_corr=local_feature.emd_min_corr,
                    emd_std_corr=local_feature.emd_std_corr,
                    emd_corr_25=local_feature.emd_corr_25,
                    emd_corr_50=local_feature.emd_corr_50,
                    emd_corr_75=local_feature.emd_corr_75,
                    emd_energ_entropy=local_feature.emd_energ_entropy,
                    emd_oi=local_feature.emd_oi,
                    emd_hi=local_feature.emd_hi
                )
                remote_session.add(new_feature)
                new_feature_count += 1

        remote_session.commit()
        added_word = "Добавлена" if new_feature_count == 1 else "Добавлено"
        set_info(
            f'{added_word} {pluralize(new_feature_count, ["новая запись", "новых записи", "новых записей"])} в таблицу EMDFeatureRDB',
            'green')

    # set_info('Выгрузка данных с локальной БД на удаленную завершена', 'blue')


def load_emd_feature():
    """Загрузка таблицы EMDFeature с удаленной БД на локальную"""

    # set_info('Начало загрузки данных с удаленной БД на локальную', 'blue')

    set_info('Проверка наличия связанных пластов для характеристик EMD в локальной БД', 'blue')
    with get_session() as remote_session:

        local_formations = {}
        for f in session.query(Formation.up_hash, Formation.down_hash, Formation.id).all():
            local_formations[f.up_hash] = f.id
            local_formations[f.down_hash] = f.id

        remote_features = remote_session.query(
            EMDFeatureRDB.id,
            FormationRDB.up_hash,
            FormationRDB.down_hash
        ).join(
            FormationRDB, EMDFeatureRDB.formation_id == FormationRDB.id
        ).all()

        problems = False

        for remote_feature, formation_up_hash, formation_down_hash in tqdm(remote_features, desc='Проверка наличия связанных пластов в локальной БД'):

            # Проверяем пласт
            if (formation_up_hash not in local_formations and
                    formation_down_hash not in local_formations):
                problems = True

        if problems:
            error_info = (f'Отсутствуют данные в таблице FormationRDB.')
            error_info += "\n\nНеобходимо сначала загрузить пласты с удаленной БД."
            set_info('Обнаружены проблемы с зависимостями', 'red')
            QMessageBox.critical(MainWindow, 'Ошибка зависимостей. Загрузка данных прекращена.', error_info)
            return
        else:
            set_info('Проблем с зависимостями нет', 'green')

    # Если проверка пройдена, выполняем выгрузку
    with get_session() as remote_session:
        remote_features = remote_session.query(
            EMDFeatureRDB.id,
            FormationRDB.up_hash,
            FormationRDB.down_hash
        ).join(
            FormationRDB, EMDFeatureRDB.formation_id == FormationRDB.id
        ).all()
        local_formations = {}
        for f in session.query(Formation.up_hash, Formation.down_hash, Formation.id).all():
            local_formations[f.up_hash] = [f.id]
            local_formations[f.down_hash] = [f.id]

        ui.progressBar.setMaximum(len(remote_features))
        new_feature_count = 0

        for n, (remote_feature_id, formation_up_hash, formation_down_hash) in tqdm(enumerate(remote_features), desc='Загрузка характеристик EMD'):
            ui.progressBar.setValue(n+1)

            # Получаем ID пласта
            local_formation_list = local_formations.get(formation_up_hash)
            if not local_formation_list:
                local_formation_list = local_formations.get(formation_down_hash)

            local_feature_count = session.query(EMDFeature).filter_by(
                formation_id=local_formation_list[0]
            ).count()

            if local_feature_count == 0:
                remote_feature = remote_session.get(EMDFeatureRDB, remote_feature_id)
                new_feature = EMDFeature(
                    formation_id=local_formation_list[0],
                    emd_num_imfs=remote_feature.emd_num_imfs,
                    emd_energ_mean=remote_feature.emd_energ_mean,
                    emd_energ_med=remote_feature.emd_energ_med,
                    emd_energ_max=remote_feature.emd_energ_max,
                    emd_energ_min=remote_feature.emd_energ_min,
                    emd_energ_std=remote_feature.emd_energ_std,
                    emd_rel_energ_mean=remote_feature.emd_rel_energ_mean,
                    emd_rel_energ_med=remote_feature.emd_rel_energ_med,
                    emd_rel_energ_max=remote_feature.emd_rel_energ_max,
                    emd_rel_energ_min=remote_feature.emd_rel_energ_min,
                    emd_rel_energ_std=remote_feature.emd_rel_energ_std,
                    emd_dom_freqs_mean=remote_feature.emd_dom_freqs_mean,
                    emd_dom_freqs_med=remote_feature.emd_dom_freqs_med,
                    emd_dom_freqs_max=remote_feature.emd_dom_freqs_max,
                    emd_dom_freqs_min=remote_feature.emd_dom_freqs_min,
                    emd_dom_freqs_std=remote_feature.emd_dom_freqs_std,
                    emd_mean_corr=remote_feature.emd_mean_corr,
                    emd_median_corr=remote_feature.emd_median_corr,
                    emd_max_corr=remote_feature.emd_max_corr,
                    emd_min_corr=remote_feature.emd_min_corr,
                    emd_std_corr=remote_feature.emd_std_corr,
                    emd_corr_25=remote_feature.emd_corr_25,
                    emd_corr_50=remote_feature.emd_corr_50,
                    emd_corr_75=remote_feature.emd_corr_75,
                    emd_energ_entropy=remote_feature.emd_energ_entropy,
                    emd_oi=remote_feature.emd_oi,
                    emd_hi=remote_feature.emd_hi
                )
                session.add(new_feature)
                new_feature_count += 1

        session.commit()
        added_word = "Добавлена" if new_feature_count == 1 else "Добавлено"
        set_info(
            f'{added_word} {pluralize(new_feature_count, ["новая запись", "новых записи", "новых записей"])} в таблицу EMDFeature',
            'green')

    # set_info('Загрузка параметров с удаленной БД на локальную завершена', 'blue')