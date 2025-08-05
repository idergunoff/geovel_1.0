from remote_db.model_remote_db import *
from models_db.model import *
from qt.rem_db_window import *
from func import *

def unload_entropy_feature():
    """Выгрузка таблицы EntropyFeature с локальной БД на удаленную"""

    # set_info('Начало выгрузки данных с локальной БД на удаленную', 'blue')

    set_info('Проверка наличия связанных пластов для параметров энтропии в удаленной БД', 'blue')

    # Предзагрузка данных из удаленной БД для проверки
    with get_session() as remote_session:
        remote_formations = {}
        for f in remote_session.query(FormationRDB.up_hash, FormationRDB.down_hash, FormationRDB.id).all():
            remote_formations[f.up_hash] = f.id
            remote_formations[f.down_hash] = f.id

        local_ent_features = session.query(
            EntropyFeature,
            Formation.up_hash,
            Formation.down_hash
        ).join(
            Formation, EntropyFeature.formation_id == Formation.id
        ).all()

        problems = False

        for local_ent_feature, formation_up_hash, formation_down_hash in tqdm(local_ent_features, desc='Проверка наличия связанных пластов в удаленной БД'):

            if local_ent_feature.formation is None:
                session.delete(local_ent_feature)
                set_info(f'В таблице EntopyFeature удалена запись id{local_ent_feature.id}, для которой отстутствует связанный пласт', 'black')
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
        local_ent_features = session.query(
            EntropyFeature,
            Formation.up_hash,
            Formation.down_hash
        ).join(
            Formation, EntropyFeature.formation_id == Formation.id
        ).all()
        remote_formations = {}
        for f in remote_session.query(FormationRDB.up_hash, FormationRDB.down_hash, FormationRDB.id).all():
            remote_formations[f.up_hash] = [f.id]
            remote_formations[f.down_hash] = [f.id]

        ui.progressBar.setMaximum(len(local_ent_features))
        new_ent_feature_count = 0

        for n, (local_ent_feature, formation_up_hash, formation_down_hash) in tqdm(enumerate(local_ent_features), desc='Выгрузка параметров энтропии'):
            ui.progressBar.setValue(n + 1)

            # Получаем ID пласта
            remote_formation_list = remote_formations.get(local_ent_feature.formation.up_hash)
            if not remote_formation_list:
                remote_formation_list = remote_formations.get(local_ent_feature.formation.down_hash)

            remote_ent_feature_count = remote_session.query(EntropyFeatureRDB).filter_by(
                formation_id=remote_formation_list[0]
            ).count()

            if remote_ent_feature_count == 0:
                new_ent_feature = EntropyFeatureRDB(
                    formation_id=remote_formation_list[0],
                    ent_sh=local_ent_feature.ent_sh,
                    ent_perm=local_ent_feature.ent_perm,
                    ent_appr=local_ent_feature.ent_appr,
                    ent_sample1=local_ent_feature.ent_sample1,
                    ent_sample2=local_ent_feature.ent_sample2,
                    ent_ms1=local_ent_feature.ent_ms1,
                    ent_ms2=local_ent_feature.ent_ms2,
                    ent_ms3=local_ent_feature.ent_ms3,
                    ent_ms4=local_ent_feature.ent_ms4,
                    ent_ms5=local_ent_feature.ent_ms5,
                    ent_ms6=local_ent_feature.ent_ms6,
                    ent_ms7=local_ent_feature.ent_ms7,
                    ent_ms8=local_ent_feature.ent_ms8,
                    ent_ms9=local_ent_feature.ent_ms9,
                    ent_ms10=local_ent_feature.ent_ms10,
                    ent_fft=local_ent_feature.ent_fft
                )
                remote_session.add(new_ent_feature)
                new_ent_feature_count += 1

        remote_session.commit()
        added_word = "Добавлена" if new_ent_feature_count == 1 else "Добавлено"
        set_info(
            f'{added_word} {pluralize(new_ent_feature_count, ["новая запись", "новых записи", "новых записей"])} в таблицу EntropyFeatureRDB',
            'green')

    # set_info('Выгрузка данных с локальной БД на удаленную завершена', 'blue')


def load_entropy_feature():
    """Загрузка таблицs EntropyFeature с удаленной БД на локальную"""

    # set_info('Начало загрузки данных с удаленной БД на локальную', 'blue')

    set_info('Проверка наличия связанных пластов для параметров энтропии в локальной БД', 'blue')
    with get_session() as remote_session:

        local_formations = {}
        for f in session.query(Formation.up_hash, Formation.down_hash, Formation.id).all():
            local_formations[f.up_hash] = f.id
            local_formations[f.down_hash] = f.id

        remote_ent_features = remote_session.query(
            EntropyFeatureRDB.id,
            FormationRDB.up_hash,
            FormationRDB.down_hash
        ).join(
            FormationRDB, EntropyFeatureRDB.formation_id == FormationRDB.id
        ).all()

        problems = False

        for remote_ent_feature, formation_up_hash, formation_down_hash in tqdm(remote_ent_features, desc='Проверка наличия связанных пластов в локальной БД'):

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
        remote_ent_features = remote_session.query(
            EntropyFeatureRDB.id,
            FormationRDB.up_hash,
            FormationRDB.down_hash
        ).join(
            FormationRDB, EntropyFeatureRDB.formation_id == FormationRDB.id
        ).all()
        local_formations = {}
        for f in session.query(Formation.up_hash, Formation.down_hash, Formation.id).all():
            local_formations[f.up_hash] = [f.id]
            local_formations[f.down_hash] = [f.id]

        ui.progressBar.setMaximum(len(remote_ent_features))
        new_ent_feature_count = 0

        for n, (remote_ent_feature_id, formation_up_hash, formation_down_hash) in tqdm(enumerate(remote_ent_features), desc='Загрузка параметров энтропии'):
            ui.progressBar.setValue(n+1)

            # Получаем ID пласта
            local_formation_list = local_formations.get(formation_up_hash)
            if not local_formation_list:
                local_formation_list = local_formations.get(formation_down_hash)

            local_ent_feature_count = session.query(EntropyFeature).filter_by(
                formation_id=local_formation_list[0]
            ).count()

            if local_ent_feature_count == 0:
                remote_ent_feature = remote_session.get(EntropyFeatureRDB, remote_ent_feature_id)
                new_ent_feature = EntropyFeature(
                    formation_id=local_formation_list[0],
                    ent_sh=remote_ent_feature.ent_sh,
                    ent_perm=remote_ent_feature.ent_perm,
                    ent_appr=remote_ent_feature.ent_appr,
                    ent_sample1=remote_ent_feature.ent_sample1,
                    ent_sample2=remote_ent_feature.ent_sample2,
                    ent_ms1=remote_ent_feature.ent_ms1,
                    ent_ms2=remote_ent_feature.ent_ms2,
                    ent_ms3=remote_ent_feature.ent_ms3,
                    ent_ms4=remote_ent_feature.ent_ms4,
                    ent_ms5=remote_ent_feature.ent_ms5,
                    ent_ms6=remote_ent_feature.ent_ms6,
                    ent_ms7=remote_ent_feature.ent_ms7,
                    ent_ms8=remote_ent_feature.ent_ms8,
                    ent_ms9=remote_ent_feature.ent_ms9,
                    ent_ms10=remote_ent_feature.ent_ms10,
                    ent_fft=remote_ent_feature.ent_fft
                )
                session.add(new_ent_feature)
                new_ent_feature_count += 1

        session.commit()
        added_word = "Добавлена" if new_ent_feature_count == 1 else "Добавлено"
        set_info(
            f'{added_word} {pluralize(new_ent_feature_count, ["новая запись", "новых записи", "новых записей"])} в таблицу EntropyFeature',
            'green')

    # set_info('Загрузка данных с удаленной БД на локальную завершена', 'blue')


