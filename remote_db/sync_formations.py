from remote_db.model_remote_db import *
from models_db.model import *
from qt.rem_db_window import *
from func import *
from layer import add_param_in_new_formation
import hashlib
from sqlalchemy.orm import selectinload

def load_formations():
    """Загрузка пластов с удаленной БД на локальную"""
    set_info('Начало загрузки пластов с удаленной БД на локальную', 'blue')

    set_info(f'Обновление хэшей...', 'blue')
    with get_session() as remote_session:
        try:
            update_formation_hashes(session)
            update_formation_hashes_rdb(remote_session)

            # Загружаем все удаленные пласты
            remote_formations = remote_session.query(
                FormationRDB.id,
                FormationRDB.title,
                FormationRDB.profile_id,
                FormationRDB.up_hash,
                FormationRDB.down_hash
            ).all()

            # Предзагрузка профилей из локальной базы данных
            local_profiles = {
                p.signal_hash_md5: p.id
                for p in session.query(Profile.signal_hash_md5, Profile.id).all()
            }

            # Предзагрузка существующих пластов из локальной базы данных
            existing_formations = {
                (up_hash, down_hash) for up_hash, down_hash in
                session.query(Formation.up_hash, Formation.down_hash).all()
            }

            added_formations_count = 0
            added_layers_count = 0
            skipped_profiles_count = 0

            ui.progressBar.setMaximum(len(remote_formations))

            for n, remote_formation in tqdm(enumerate(remote_formations), desc="Загрузка пластов"):

                ui.progressBar.setValue(n + 1)
                QApplication.processEvents()  # Принудительное обновление

                profile = remote_session.query(ProfileRDB.signal_hash_md5).filter_by(
                    id=remote_formation.profile_id).first()

                # Ищем соответствующий профиль в локальной БД по хэшу
                remote_profile_hash = profile[0]
                local_profile_id = local_profiles.get(remote_profile_hash)

                if not local_profile_id:
                    set_info(f'Пропуск пласта {remote_formation.title}: профиль не найден в локальной БД', 'yellow')
                    skipped_profiles_count += 1
                    continue

                # Проверяем, существует ли пласт в локальной БД
                key = (remote_formation.up_hash, remote_formation.down_hash)
                if key in existing_formations:
                    continue

                remote_formation_all = remote_session.query(FormationRDB).filter_by(id=remote_formation.id).first()

                # Создаем новые слои
                new_layer_1 = Layers(
                    profile_id=local_profile_id,
                    layer_title=remote_formation_all.title + '_top',
                    layer_line=remote_formation_all.up
                )

                new_layer_2 = Layers(
                    profile_id=local_profile_id,
                    layer_title=remote_formation_all.title + '_bottom',
                    layer_line=remote_formation_all.down
                )

                # Добавляем слои в базу данных по одному
                session.add(new_layer_1)
                session.add(new_layer_2)
                session.flush()  # Фиксируем изменения, чтобы получить ID
                added_layers_count += 2

                # Создаем новый пласт
                new_formation = Formation(
                    profile_id=local_profile_id,
                    title=remote_formation_all.title,
                    up=new_layer_1.id,
                    down=new_layer_2.id,
                    up_hash=remote_formation_all.up_hash,
                    down_hash=remote_formation_all.down_hash
                )

                session.add(new_formation)
                session.flush()
                add_param_in_new_formation(new_formation.id, new_formation.profile_id)

                ui.progressBar.setValue(len(remote_formations))  # Гарантированное завершение
                added_formations_count += 1

            session.commit()
            set_info(f'Загружено: {pluralize(added_formations_count, ["пласт", "пласта", "пластов"])}',
                     'green')
            set_info(f'В локальную БД добавлено:'
                     f'{pluralize(added_layers_count, ["новый слой", "новых слоя", "новых слоев"])}',
                     'green')
            if skipped_profiles_count:
                set_info(f'{pluralize(skipped_profiles_count, ["пласт пропущен", "пласта пропущено", "пластов пропущено"])} '
                         f'из-за несоответствия профилей', 'yellow')

        except Exception as e:
            remote_session.rollback()
            set_info(f'Ошибка при выгрузке пластов: {str(e)}', 'red')
            raise

    set_info(f'Загрузка пластов с удаленной БД на локальную завершена', 'blue')


def unload_formations():
    """Выгрузка слоев с локальной БД на удаленную"""
    set_info(f'Начало выгрузки пластов с локальной БД на удаленную', 'blue')

    set_info(f'Обновление хэшей...', 'blue')
    with get_session() as remote_session:
        try:
            update_formation_hashes(session)
            update_formation_hashes_rdb(remote_session)

            local_formations = session.query(Formation)\
                .options(
                    selectinload(Formation.profile),
                    selectinload(Formation.layer_up),
                    selectinload(Formation.layer_down)
                ).all()

            # Предзагрузка профилей из удаленной базы данных
            remote_profiles = {
                p.signal_hash_md5: p.id
                for p in remote_session.query(ProfileRDB.signal_hash_md5, ProfileRDB.id).all()
            }

            # Предзагрузка существующих пластов из удаленной базы данных
            existing_remote_formations = {
                (up_hash, down_hash) for up_hash, down_hash in
                remote_session.query(FormationRDB.up_hash, FormationRDB.down_hash).all()
            }

            added_formations_count = 0
            skipped_profiles_count = 0

            ui.progressBar.setMaximum(len(local_formations))

            for n, local_formation in tqdm(enumerate(local_formations), desc="Выгрузка пластов"):

                ui.progressBar.setValue(n + 1)

                if not local_formation.layer_up or not local_formation.layer_down:
                    set_info(f'Пропуск пласта {local_formation.title}: нет слоёв up/down', 'yellow')
                    skipped_profiles_count += 1
                    continue

                # Ищем соответствующий профиль в удаленной БД по хэшу
                local_profile_hash = local_formation.profile.signal_hash_md5
                remote_profile_id = remote_profiles.get(local_profile_hash)

                if not remote_profile_id:
                    set_info(f'Пропуск пласта {local_formation.title}: профиль не найден в удаленной БД', 'yellow')
                    skipped_profiles_count += 1
                    continue

                # Проверяем, существует ли пласт в удаленной БД
                key = (local_formation.up_hash, local_formation.down_hash)
                if key in existing_remote_formations:
                    continue

                # Создаем новый пласт
                new_formation = FormationRDB(
                    profile_id=remote_profile_id,
                    title=local_formation.title,
                    up=local_formation.layer_up.layer_line,
                    down=local_formation.layer_down.layer_line,
                    up_hash=local_formation.up_hash,
                    down_hash=local_formation.down_hash
                )
                remote_session.add(new_formation)
                added_formations_count += 1

            remote_session.commit()
            set_info(f'Выгружено: {pluralize(added_formations_count, ["пласт", "пласта", "пластов"])}', 'green')
            if skipped_profiles_count:
                set_info(
                    f'{pluralize(skipped_profiles_count, ["пласт пропущен", "пласта пропущено", "пластов пропущено"])} '
                    f'из-за несоответствия профилей', 'yellow')

        except Exception as e:
            remote_session.rollback()
            set_info(f'Ошибка при выгрузке пластов: {str(e)}', 'red')
            raise

    set_info(f'Выгрузка пластов с локальной БД на удаленную завершена', 'blue')


# Функция вычисления хэш-суммы
def calculate_hash(value):
    return hashlib.md5(str(value).encode()).hexdigest()

def update_formation_hashes(session):
    """Обновляет только пустые хэши в таблице Formation"""
    # Выбираем только записи с пустыми хэшами
    formations = session.query(Formation)\
        .options(
            selectinload(Formation.layer_up),
            selectinload(Formation.layer_down)
        )\
        .filter(
            or_(
                Formation.up_hash.is_(None),
                Formation.down_hash.is_(None)
            )
        ).all()

    updated_count = 0

    for formation in tqdm(formations, desc="Обновление хэшей Formation"):

        if formation.up_hash is None and formation.layer_up and formation.layer_up.layer_line:
            formation.up_hash = calculate_hash(formation.layer_up.layer_line)
            updated_count += 1

        if formation.down_hash is None and formation.layer_down and formation.layer_down.layer_line:
            formation.down_hash = calculate_hash(formation.layer_down.layer_line)
            updated_count += 1

    session.commit()
    set_info(f'{pluralize(updated_count, ["хэш обновлен", "хэша обновлено", "хэшей обновлено"])} в таблице Formation',
             'blue')


def update_formation_hashes_rdb(session):
    """Обновляет хэши up_hash и down_hash в таблице FormationRDB,"""
    # Загружаем только те записи, где нужно обновить хэши
    formations = session.query(FormationRDB) \
        .filter(
        or_(
            FormationRDB.up_hash.is_(None),
            FormationRDB.down_hash.is_(None),
        )
    ).all()

    updated_count = 0

    for formation in tqdm(formations, desc="Обновление хэшей FormationRDB"):

        # Проверяем и обновляем up_hash
        if formation.up_hash is None:
            formation.up_hash = calculate_hash(formation.up)
            updated_count += 1

        # Проверяем и обновляем down_hash
        if formation.down_hash is None:
            formation.down_hash = calculate_hash(formation.down)
            updated_count += 1

    session.commit()
    set_info(f'{pluralize(updated_count, ["хэш обновлен", "хэша обновлено", "хэшей обновлено"])} в таблице FormationRDB',
             'blue')
