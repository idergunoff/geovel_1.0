from remote_db.model_remote_db import *
from models_db.model import *
from qt.rem_db_window import *
from func import *

def unload_hht_feature():
    """Выгрузка таблицы HHTFeature с локальной БД на удаленную"""

    # set_info('Начало выгрузки данных с локальной БД на удаленную', 'blue')

    set_info('Проверка наличия связанных пластов для характеристик HHT в удаленной БД', 'blue')

    # Предзагрузка данных из удаленной БД для проверки
    with get_session() as remote_session:
        remote_formations = {}
        for f in remote_session.query(FormationRDB.up_hash, FormationRDB.down_hash, FormationRDB.id).all():
            remote_formations[f.up_hash] = f.id
            remote_formations[f.down_hash] = f.id

        local_features = session.query(
            HHTFeature,
            Formation.up_hash,
            Formation.down_hash
        ).join(
            Formation, HHTFeature.formation_id == Formation.id
        ).all()

        problems = False

        for local_feature, formation_up_hash, formation_down_hash in tqdm(local_features, desc='Проверка наличия связанных пластов в удаленной БД'):

            if local_feature.formation is None:
                session.delete(local_feature)
                set_info(f'В таблице HHTFeature удалена запись id{local_feature.id}, для которой отстутствует связанный пласт', 'black')
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
            HHTFeature,
            Formation.up_hash,
            Formation.down_hash
        ).join(
            Formation, HHTFeature.formation_id == Formation.id
        ).all()
        remote_formations = {}
        for f in remote_session.query(FormationRDB.up_hash, FormationRDB.down_hash, FormationRDB.id).all():
            remote_formations[f.up_hash] = [f.id]
            remote_formations[f.down_hash] = [f.id]

        ui.progressBar.setMaximum(len(local_features))
        new_feature_count = 0

        for n, (local_feature, formation_up_hash, formation_down_hash) in tqdm(enumerate(local_features), desc='Выгрузка характеристик HHT'):
            ui.progressBar.setValue(n + 1)

            # Получаем ID пласта
            remote_formation_list = remote_formations.get(local_feature.formation.up_hash)
            if not remote_formation_list:
                remote_formation_list = remote_formations.get(local_feature.formation.down_hash)

            remote_feature_count = remote_session.query(HHTFeatureRDB).filter_by(
                formation_id=remote_formation_list[0]
            ).count()

            if remote_feature_count == 0:
                new_feature = HHTFeatureRDB(
                    formation_id=remote_formation_list[0],
                    hht_inst_freq_mean=local_feature.hht_inst_freq_mean,
                    hht_inst_freq_med=local_feature.hht_inst_freq_med,
                    hht_inst_freq_max=local_feature.hht_inst_freq_max,
                    hht_inst_freq_min=local_feature.hht_inst_freq_min,
                    hht_inst_freq_std=local_feature.hht_inst_freq_std,

                    hht_inst_amp_mean=local_feature.hht_inst_amp_mean,
                    hht_inst_amp_med=local_feature.hht_inst_amp_med,
                    hht_inst_amp_max=local_feature.hht_inst_amp_max,
                    hht_inst_amp_min=local_feature.hht_inst_amp_min,
                    hht_inst_amp_std=local_feature.hht_inst_amp_std,

                    hht_mean_freq_mean=local_feature.hht_mean_freq_mean,
                    hht_mean_freq_med=local_feature.hht_mean_freq_med,
                    hht_mean_freq_max=local_feature.hht_mean_freq_max,
                    hht_mean_freq_min=local_feature.hht_mean_freq_min,
                    hht_mean_freq_std=local_feature.hht_mean_freq_std,

                    hht_mean_amp_mean=local_feature.hht_mean_amp_mean,
                    hht_mean_amp_med=local_feature.hht_mean_amp_med,
                    hht_mean_amp_max=local_feature.hht_mean_amp_max,
                    hht_mean_amp_min=local_feature.hht_mean_amp_min,
                    hht_mean_amp_std=local_feature.hht_mean_amp_std,

                    hht_marg_spec_mean=local_feature.hht_marg_spec_mean,
                    hht_marg_spec_med=local_feature.hht_marg_spec_med,
                    hht_marg_spec_max=local_feature.hht_marg_spec_max,
                    hht_marg_spec_min=local_feature.hht_marg_spec_min,
                    hht_marg_spec_std=local_feature.hht_marg_spec_std,

                    hht_teager_energ_mean=local_feature.hht_teager_energ_mean,
                    hht_teager_energ_med=local_feature.hht_teager_energ_med,
                    hht_teager_energ_max=local_feature.hht_teager_energ_max,
                    hht_teager_energ_min=local_feature.hht_teager_energ_min,
                    hht_teager_energ_std=local_feature.hht_teager_energ_std,

                    hht_hi=local_feature.hht_hi,

                    hht_dos_mean=local_feature.hht_dos_mean,
                    hht_dos_med=local_feature.hht_dos_med,
                    hht_dos_max=local_feature.hht_dos_max,
                    hht_dos_min=local_feature.hht_dos_min,
                    hht_dos_std=local_feature.hht_dos_std,

                    hht_oi=local_feature.hht_oi,

                    hht_hsd_mean=local_feature.hht_hsd_mean,
                    hht_hsd_med=local_feature.hht_hsd_med,
                    hht_hsd_max=local_feature.hht_hsd_max,
                    hht_hsd_min=local_feature.hht_hsd_min,
                    hht_hsd_std=local_feature.hht_hsd_std,

                    hht_ci=local_feature.hht_ci
                )
                remote_session.add(new_feature)
                new_feature_count += 1

        remote_session.commit()
        added_word = "Добавлена" if new_feature_count == 1 else "Добавлено"
        set_info(
            f'{added_word} {pluralize(new_feature_count, ["новая запись", "новых записи", "новых записей"])} в таблицу HHTFeatureRDB',
            'green')

    # set_info('Выгрузка данных с локальной БД на удаленную завершена', 'blue')


def load_hht_feature():
    """Загрузка таблицы HHTFeature с удаленной БД на локальную"""

    # set_info('Начало загрузки данных с удаленной БД на локальную', 'blue')

    set_info('Проверка наличия связанных пластов для характеристик HHT в локальной БД', 'blue')
    with get_session() as remote_session:

        local_formations = {}
        for f in session.query(Formation.up_hash, Formation.down_hash, Formation.id).all():
            local_formations[f.up_hash] = f.id
            local_formations[f.down_hash] = f.id

        remote_features = remote_session.query(
            HHTFeatureRDB.id,
            FormationRDB.up_hash,
            FormationRDB.down_hash
        ).join(
            FormationRDB, HHTFeatureRDB.formation_id == FormationRDB.id
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
            HHTFeatureRDB.id,
            FormationRDB.up_hash,
            FormationRDB.down_hash
        ).join(
            FormationRDB, HHTFeatureRDB.formation_id == FormationRDB.id
        ).all()
        local_formations = {}
        for f in session.query(Formation.up_hash, Formation.down_hash, Formation.id).all():
            local_formations[f.up_hash] = [f.id]
            local_formations[f.down_hash] = [f.id]

        ui.progressBar.setMaximum(len(remote_features))
        new_feature_count = 0

        for n, (remote_feature_id, formation_up_hash, formation_down_hash) in tqdm(enumerate(remote_features), desc='Загрузка характеристик HHT'):
            ui.progressBar.setValue(n+1)

            # Получаем ID пласта
            local_formation_list = local_formations.get(formation_up_hash)
            if not local_formation_list:
                local_formation_list = local_formations.get(formation_down_hash)

            local_feature_count = session.query(HHTFeature).filter_by(
                formation_id=local_formation_list[0]
            ).count()

            if local_feature_count == 0:
                remote_feature = remote_session.get(HHTFeatureRDB, remote_feature_id)
                new_feature = HHTFeature(
                    formation_id=local_formation_list[0],
                    hht_inst_freq_mean=remote_feature.hht_inst_freq_mean,
                    hht_inst_freq_med=remote_feature.hht_inst_freq_med,
                    hht_inst_freq_max=remote_feature.hht_inst_freq_max,
                    hht_inst_freq_min=remote_feature.hht_inst_freq_min,
                    hht_inst_freq_std=remote_feature.hht_inst_freq_std,

                    hht_inst_amp_mean=remote_feature.hht_inst_amp_mean,
                    hht_inst_amp_med=remote_feature.hht_inst_amp_med,
                    hht_inst_amp_max=remote_feature.hht_inst_amp_max,
                    hht_inst_amp_min=remote_feature.hht_inst_amp_min,
                    hht_inst_amp_std=remote_feature.hht_inst_amp_std,

                    hht_mean_freq_mean=remote_feature.hht_mean_freq_mean,
                    hht_mean_freq_med=remote_feature.hht_mean_freq_med,
                    hht_mean_freq_max=remote_feature.hht_mean_freq_max,
                    hht_mean_freq_min=remote_feature.hht_mean_freq_min,
                    hht_mean_freq_std=remote_feature.hht_mean_freq_std,

                    hht_mean_amp_mean=remote_feature.hht_mean_amp_mean,
                    hht_mean_amp_med=remote_feature.hht_mean_amp_med,
                    hht_mean_amp_max=remote_feature.hht_mean_amp_max,
                    hht_mean_amp_min=remote_feature.hht_mean_amp_min,
                    hht_mean_amp_std=remote_feature.hht_mean_amp_std,

                    hht_marg_spec_mean=remote_feature.hht_marg_spec_mean,
                    hht_marg_spec_med=remote_feature.hht_marg_spec_med,
                    hht_marg_spec_max=remote_feature.hht_marg_spec_max,
                    hht_marg_spec_min=remote_feature.hht_marg_spec_min,
                    hht_marg_spec_std=remote_feature.hht_marg_spec_std,

                    hht_teager_energ_mean=remote_feature.hht_teager_energ_mean,
                    hht_teager_energ_med=remote_feature.hht_teager_energ_med,
                    hht_teager_energ_max=remote_feature.hht_teager_energ_max,
                    hht_teager_energ_min=remote_feature.hht_teager_energ_min,
                    hht_teager_energ_std=remote_feature.hht_teager_energ_std,

                    hht_hi=remote_feature.hht_hi,

                    hht_dos_mean=remote_feature.hht_dos_mean,
                    hht_dos_med=remote_feature.hht_dos_med,
                    hht_dos_max=remote_feature.hht_dos_max,
                    hht_dos_min=remote_feature.hht_dos_min,
                    hht_dos_std=remote_feature.hht_dos_std,

                    hht_oi=remote_feature.hht_oi,

                    hht_hsd_mean=remote_feature.hht_hsd_mean,
                    hht_hsd_med=remote_feature.hht_hsd_med,
                    hht_hsd_max=remote_feature.hht_hsd_max,
                    hht_hsd_min=remote_feature.hht_hsd_min,
                    hht_hsd_std=remote_feature.hht_hsd_std,

                    hht_ci=remote_feature.hht_ci
                )
                session.add(new_feature)
                new_feature_count += 1

        session.commit()
        added_word = "Добавлена" if new_feature_count == 1 else "Добавлено"
        set_info(
            f'{added_word} {pluralize(new_feature_count, ["новая запись", "новых записи", "новых записей"])} в таблицу HHTFeature',
            'green')

    # set_info('Загрузка параметров с удаленной БД на локальную завершена', 'blue')