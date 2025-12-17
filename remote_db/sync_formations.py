import hashlib
from remote_db.sync_features.sync_formation_features import *
from remote_db.sync_features.sync_wavelet_features import *
from remote_db.sync_features.sync_fractal_features import *
from remote_db.sync_features.sync_entropy_features import *
from remote_db.sync_features.sync_nonlinear_features import *
from remote_db.sync_features.sync_morphology_features import *
from remote_db.sync_features.sync_frequency_feature import *
from remote_db.sync_features.sync_envelope_features import *
from remote_db.sync_features.sync_autocorr_features import *
from remote_db.sync_features.sync_emd_features import *
from remote_db.sync_features.sync_hht_features import *
from func import update_formation_combobox

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
                    set_info(f'Пропуск пласта {remote_formation.title}: профиль не найден в локальной БД', 'black')
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

                remote_feature = remote_session.query(FormationFeatureRDB).filter_by(formation_id=remote_formation.id).first()

                # Создаем новый пласт
                new_formation = Formation(
                    profile_id=local_profile_id,
                    title=remote_formation_all.title,
                    up=new_layer_1.id,
                    down=new_layer_2.id,
                    up_hash=remote_formation_all.up_hash,
                    down_hash=remote_formation_all.down_hash,

                    T_top = remote_feature.T_top,
                    T_bottom = remote_feature.T_bottom,
                    dT = remote_feature.dT,

                    A_top = remote_feature.A_top,
                    A_bottom = remote_feature.A_bottom,
                    dA = remote_feature.dA,
                    A_sum = remote_feature.A_sum,
                    A_mean = remote_feature.A_mean,

                    dVt = remote_feature.dVt,
                    Vt_top = remote_feature.Vt_top,
                    Vt_sum = remote_feature.Vt_sum,
                    Vt_mean = remote_feature.Vt_mean,

                    dAt = remote_feature.dAt,
                    At_top = remote_feature.At_top,
                    At_sum = remote_feature.At_sum,
                    At_mean = remote_feature.At_mean,

                    dPht = remote_feature.dPht,
                    Pht_top = remote_feature.Pht_top,
                    Pht_sum = remote_feature.Pht_sum,
                    Pht_mean = remote_feature.Pht_mean,

                    Wt_top = remote_feature.Wt_top,
                    Wt_mean = remote_feature.Wt_mean,
                    Wt_sum = remote_feature.Wt_sum,

                    width = remote_feature.width,
                    top = remote_feature.top,
                    land = remote_feature.land,
                    speed = remote_feature.speed,
                    speed_cover = remote_feature.speed_cover,

                    skew = remote_feature.skew,
                    kurt = remote_feature.kurt,
                    std = remote_feature.std,
                    k_var = remote_feature.k_var,

                    A_max = remote_feature.A_max,
                    Vt_max = remote_feature.Vt_max,
                    At_max = remote_feature.At_max,
                    Pht_max = remote_feature.Pht_max,
                    Wt_max = remote_feature.Wt_max,

                    A_T_max = remote_feature.A_T_max,
                    Vt_T_max = remote_feature.Vt_T_max,
                    At_T_max = remote_feature.At_T_max,
                    Pht_T_max = remote_feature.Pht_T_max,
                    Wt_T_max = remote_feature.Wt_T_max,

                    A_Sn = remote_feature.A_Sn,
                    Vt_Sn = remote_feature.Vt_Sn,
                    At_Sn = remote_feature.At_Sn,
                    Pht_Sn = remote_feature.Pht_Sn,
                    Wt_Sn = remote_feature.Wt_Sn,

                    A_wmf = remote_feature.A_wmf,
                    Vt_wmf = remote_feature.Vt_wmf,
                    At_wmf = remote_feature.At_wmf,
                    Pht_wmf = remote_feature.Pht_wmf,
                    Wt_wmf = remote_feature.Wt_wmf,

                    A_Qf = remote_feature.A_Qf,
                    Vt_Qf = remote_feature.Vt_Qf,
                    At_Qf = remote_feature.At_Qf,
                    Pht_Qf = remote_feature.Pht_Qf,
                    Wt_Qf = remote_feature.Wt_Qf,

                    A_Sn_wmf = remote_feature.A_Sn_wmf,
                    Vt_Sn_wmf = remote_feature.Vt_Sn_wmf,
                    At_Sn_wmf = remote_feature.At_Sn_wmf,
                    Pht_Sn_wmf = remote_feature.Pht_Sn_wmf,
                    Wt_Sn_wmf = remote_feature.Wt_Sn_wmf,

                    CRL_top = remote_feature.CRL_top,
                    CRL_bottom = remote_feature.CRL_bottom,
                    dCRL = remote_feature.dCRL,
                    CRL_sum = remote_feature.CRL_sum,
                    CRL_mean = remote_feature.CRL_mean,
                    CRL_max = remote_feature.CRL_max,
                    CRL_T_max = remote_feature.CRL_T_max,
                    CRL_Sn = remote_feature.CRL_Sn,
                    CRL_wmf = remote_feature.CRL_wmf,
                    CRL_Qf = remote_feature.CRL_Qf,
                    CRL_Sn_wmf = remote_feature.CRL_Sn_wmf,

                    CRL_skew = remote_feature.CRL_skew,
                    CRL_kurt = remote_feature.CRL_kurt,
                    CRL_std = remote_feature.CRL_std,
                    CRL_k_var = remote_feature.CRL_k_var,

                    k_r = remote_feature.k_r
                )

                session.add(new_formation)
                session.flush()
                # add_param_in_new_formation(new_formation.id, new_formation.profile_id)

                ui.progressBar.setValue(len(remote_formations))  # Гарантированное завершение
                added_formations_count += 1

            session.commit()
            set_info('Начало загрузки параметров с удаленной БД на локальную', 'blue')
            if added_formations_count > 0:
                load_wavelet_feature()
                load_fractal_feature()
                load_entropy_feature()
                load_nonlinear_feature()
                load_morphology_feature()
                load_frequency_feature()
                load_envelope_feature()
                load_autocorr_feature()
                load_emd_feature()
                load_hht_feature()

            set_info('Загрузка параметров с удаленной БД на локальную завершена', 'blue')
            set_info(f'Загружено: {pluralize(added_formations_count, ["пласт", "пласта", "пластов"])}',
                     'green')
            set_info(f'В локальную БД добавлено: '
                     f'{pluralize(added_layers_count, ["новый слой", "новых слоя", "новых слоев"])}',
                     'green')
            if skipped_profiles_count:
                set_info(f'{pluralize(skipped_profiles_count, ["пласт пропущен", "пласта пропущено", "пластов пропущено"])} '
                         f'из-за несоответствия профилей', 'black')

        except Exception as e:
            remote_session.rollback()
            set_info(f'Ошибка при выгрузке пластов: {str(e)}', 'red')
            raise

    update_formation_combobox()
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
                    set_info(f'Пропуск пласта {local_formation.title}: нет слоёв up/down', 'black')
                    skipped_profiles_count += 1
                    continue

                # Ищем соответствующий профиль в удаленной БД по хэшу
                local_profile_hash = local_formation.profile.signal_hash_md5
                remote_profile_id = remote_profiles.get(local_profile_hash)

                if not remote_profile_id:
                    set_info(f'Пропуск пласта {local_formation.title}: профиль не найден в удаленной БД', 'black')
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
            set_info('Начало выгрузки параметров с локальной БД на удаленную', 'blue')

            if added_formations_count > 0:
                unload_formation_feature()
                unload_wavelet_feature()
                unload_fractal_feature()
                unload_entropy_feature()
                unload_nonlinear_feature()
                unload_morphology_feature()
                unload_frequency_feature()
                unload_envelope_feature()
                unload_autocorr_feature()
                unload_emd_feature()
                unload_hht_feature()


            set_info('Выгрузка параметров с локальной БД на удаленную завершена', 'blue')
            set_info(f'Выгружено: {pluralize(added_formations_count, ["пласт", "пласта", "пластов"])}', 'green')
            if skipped_profiles_count:
                set_info(
                    f'{pluralize(skipped_profiles_count, ["пласт пропущен", "пласта пропущено", "пластов пропущено"])} '
                    f'из-за несоответствия профилей', 'black')

        except Exception as e:
            remote_session.rollback()
            set_info(f'Ошибка при выгрузке пластов: {str(e)}', 'red')
            raise

    set_info(f'Выгрузка пластов с локальной БД на удаленную завершена', 'blue')


# Функция вычисления хэш-суммы
def calculate_hash(value):
    return hashlib.md5(str(value).encode()).hexdigest()

def update_formation_hashes(session):
    """Обновление пустых хэшей в таблице Formation"""
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
    """Обновление хэшей up_hash и down_hash в таблице FormationRDB,"""
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
