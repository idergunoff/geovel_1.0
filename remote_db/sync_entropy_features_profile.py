from remote_db.model_remote_db import *
from models_db.model import *
from qt.rem_db_window import *
from func import *

def unload_entropy_feature_profile(Window):
    """Выгрузка таблицы EntropyFeatureProfile с локальной БД на удаленную"""

    set_info('Начало выгрузки данных с локальной БД на удаленную', 'blue')

    set_info('Проверка наличия связанных профилей в удаленной БД', 'blue')

    # Предзагрузка данных из удаленной БД для проверки
    with get_session() as remote_session:
        remote_profiles = {
            p.signal_hash_md5: p.id
            for p in remote_session.query(ProfileRDB.signal_hash_md5, ProfileRDB.id).all()
        }

        local_ent_features = session.query(
            EntropyFeatureProfile,
            Profile.signal_hash_md5
        ).join(
            Profile, EntropyFeatureProfile.profile_id == Profile.id
        ).all()

        problems = False

        for local_ent_feature, profile_hash in tqdm(local_ent_features, desc='Проверка наличия связанных профилей в удаленной БД'):

            if local_ent_feature.profile is None:
                session.delete(local_ent_feature)
                set_info(f'В таблице EntopyFeature удалена запись id{local_ent_feature.id}, для которой отстутствует связанный профиль', 'black')
                continue

            if profile_hash not in remote_profiles:
                problems = True

        if problems:
            error_info = (f'Отсутствуют данные в таблице ProfileRDB.')
            error_info += "\n\nНеобходимо сначала выгрузить профили с локальной БД."
            set_info('Обнаружены проблемы с зависимостями', 'red')
            QMessageBox.critical(Window, 'Ошибка зависимостей', error_info)
        else:
            set_info('Проблем с зависимостями нет', 'green')

        try:
            session.commit()
        except Exception as e:
            session.rollback()
            error_info = 'Ошибка при удалении записей без связанных профилей в локальной БД'
            set_info('Обнаружены проблемы с зависимостями в локальной БД', 'red')
            QMessageBox.critical(Window, 'Ошибка зависимостей', error_info)
            return


    # Если проверка пройдена, выполняем выгрузку
    with get_session() as remote_session:

        local_ent_features = session.query(
            EntropyFeatureProfile,
            Profile.signal_hash_md5
        ).join(
            Profile, EntropyFeatureProfile.profile_id == Profile.id
        ).all()

        remote_profiles = {
            p.signal_hash_md5: p.id
            for p in remote_session.query(ProfileRDB.signal_hash_md5, ProfileRDB.id).all()
        }

        ui.progressBar.setMaximum(len(local_ent_features))
        new_ent_feature_count = 0

        for n, (local_ent_feature, profile_hash) in tqdm(enumerate(local_ent_features), desc='Выгрузка энтропии'):
            ui.progressBar.setValue(n + 1)

            # Получаем ID профиля
            remote_profile_id = remote_profiles[profile_hash] if local_ent_feature.profile_id != 0 else None

            remote_ent_feature_count = remote_session.query(EntropyFeatureProfileRDB).filter_by(
                profile_id=remote_profile_id
            ).count()

            if remote_ent_feature_count == 0:
                new_ent_feature = EntropyFeatureProfileRDB(
                    profile_id=remote_profile_id,
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
            f'{added_word} {pluralize(new_ent_feature_count, ["новая запись", "новых записи", "новых записей"])} в таблицу EntropyFeatureProfileRDB',
            'green')

    set_info('Выгрузка данных с локальной БД на удаленную завершена', 'blue')


def load_entropy_feature_profile(Window):
    """Загрузка таблиц EntropyFeatureProfile с удаленной БД на локальную"""

    set_info('Начало загрузки данных с удаленной БД на локальную', 'blue')

    set_info('Проверка наличия связанных профилей в локальной БД', 'blue')
    with get_session() as remote_session:

        local_profiles = {
            p.signal_hash_md5: p.id
            for p in session.query(Profile.signal_hash_md5, Profile.id).all()
        }

        remote_ent_features = remote_session.query(
            EntropyFeatureProfileRDB.profile_id,
            ProfileRDB.signal_hash_md5
        ).join(
            ProfileRDB, EntropyFeatureProfileRDB.profile_id == ProfileRDB.id
        ).all()

        problems = False

        for remote_ent_feature, profile_hash in tqdm(remote_ent_features, desc='Проверка наличия связанных профилей в локальной БД'):

            # Проверяем профиль
            if profile_hash not in local_profiles:
                problems = True

        if problems:
            error_info = (f'Отсутствуют данные в таблице ProfileRDB.')
            error_info += "\n\nНеобходимо сначала загрузить профили с удаленной БД."
            set_info('Обнаружены проблемы с зависимостями', 'red')
            QMessageBox.critical(Window, 'Ошибка зависимостей', error_info)
        else:
            set_info('Проблем с зависимостями нет', 'green')

    # Если проверка пройдена, выполняем загрузку
    with get_session() as remote_session:

        remote_ent_features = remote_session.query(
            EntropyFeatureProfileRDB.id,
            EntropyFeatureProfileRDB.profile_id,
            ProfileRDB.signal_hash_md5
        ).join(
            ProfileRDB, EntropyFeatureProfileRDB.profile_id == ProfileRDB.id
        ).all()

        local_profiles = {
            p.signal_hash_md5: p.id
            for p in session.query(Profile.signal_hash_md5, Profile.id).all()
        }

        ui.progressBar.setMaximum(len(remote_ent_features))
        new_ent_feature_count = 0

        for n, (remote_ent_feature_id, profile_id, profile_hash) in tqdm(enumerate(remote_ent_features), desc='Загрузка энтропии'):
            ui.progressBar.setValue(n+1)

            # Получаем ID профиля
            local_profile_id = local_profiles[profile_hash] if profile_id != 0 else None

            if profile_hash is None:
                continue

            local_ent_feature_count = session.query(EntropyFeatureProfile).filter_by(
                profile_id=local_profile_id
            ).count()

            if local_ent_feature_count == 0:
                remote_ent_feature = remote_session.get(EntropyFeatureProfileRDB, remote_ent_feature_id)
                new_ent_feature = EntropyFeatureProfile(
                    profile_id=local_profile_id,
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
            f'{added_word} {pluralize(new_ent_feature_count, ["новая запись", "новых записи", "новых записей"])} в таблицу EntropyFeatureProfile',
            'green')

    update_list_trained_models_class()

    set_info('Загрузка данных с удаленной БД на локальную завершена', 'blue')