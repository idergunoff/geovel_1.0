from remote_db.model_remote_db import *
from models_db.model import *
from qt.rem_db_window import *
from func import *

def unload_wavelet_feature():
    """Выгрузка таблицы WaveletFeature с локальной БД на удаленную"""

    # set_info('Начало выгрузки данных с локальной БД на удаленную', 'blue')

    set_info('Проверка наличия связанных пластов для вейвлет параметров в удаленной БД', 'blue')

    # Предзагрузка данных из удаленной БД для проверки
    with get_session() as remote_session:
        remote_formations = {}
        for f in remote_session.query(FormationRDB.up_hash, FormationRDB.down_hash, FormationRDB.id).all():
            remote_formations[f.up_hash] = f.id
            remote_formations[f.down_hash] = f.id

        local_features = session.query(
            WaveletFeature,
            Formation.up_hash,
            Formation.down_hash
        ).join(
            Formation, WaveletFeature.formation_id == Formation.id
        ).all()

        problems = False

        for local_feature, formation_up_hash, formation_down_hash in tqdm(local_features, desc='Проверка наличия связанных пластов в удаленной БД'):

            if local_feature.formation is None:
                session.delete(local_feature)
                set_info(f'В таблице WaveletFeature удалена запись id{local_feature.id}, для которой отстутствует связанный пласт', 'black')
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
            WaveletFeature,
            Formation.up_hash,
            Formation.down_hash
        ).join(
            Formation, WaveletFeature.formation_id == Formation.id
        ).all()
        remote_formations = {}
        for f in remote_session.query(FormationRDB.up_hash, FormationRDB.down_hash, FormationRDB.id).all():
            remote_formations[f.up_hash] = [f.id]
            remote_formations[f.down_hash] = [f.id]

        ui.progressBar.setMaximum(len(local_features))
        new_feature_count = 0

        for n, (local_feature, formation_up_hash, formation_down_hash) in tqdm(enumerate(local_features), desc='Выгрузка вейвлет параметров'):
            ui.progressBar.setValue(n + 1)

            # Получаем ID пласта
            remote_formation_list = remote_formations.get(local_feature.formation.up_hash)
            if not remote_formation_list:
                remote_formation_list = remote_formations.get(local_feature.formation.down_hash)

            remote_feature_count = remote_session.query(WaveletFeatureRDB).filter_by(
                formation_id=remote_formation_list[0]
            ).count()

            if remote_feature_count == 0:
                new_feature = WaveletFeatureRDB(
                    formation_id=remote_formation_list[0],

                    # Энергия вейвлета для каждого уровня декомпозиции
                    wvt_energ_D1=local_feature.wvt_energ_D1,
                    wvt_energ_D2=local_feature.wvt_energ_D2,
                    wvt_energ_D3=local_feature.wvt_energ_D3,
                    wvt_energ_D4=local_feature.wvt_energ_D4,
                    wvt_energ_D5=local_feature.wvt_energ_D5,
                    wvt_energ_A5=local_feature.wvt_energ_A5,

                    # Среднее вейвлета для каждого уровня декомпозиции
                    wvt_mean_D1=local_feature.wvt_mean_D1,
                    wvt_mean_D2=local_feature.wvt_mean_D2,
                    wvt_mean_D3=local_feature.wvt_mean_D3,
                    wvt_mean_D4=local_feature.wvt_mean_D4,
                    wvt_mean_D5=local_feature.wvt_mean_D5,
                    wvt_mean_A5=local_feature.wvt_mean_A5,

                    # Максимум вейвлета для каждого уровня декомпозиции
                    wvt_max_D1=local_feature.wvt_max_D1,
                    wvt_max_D2=local_feature.wvt_max_D2,
                    wvt_max_D3=local_feature.wvt_max_D3,
                    wvt_max_D4=local_feature.wvt_max_D4,
                    wvt_max_D5=local_feature.wvt_max_D5,
                    wvt_max_A5=local_feature.wvt_max_A5,

                    # Минимум вейвлета для каждого уровня декомпозиции
                    wvt_min_D1=local_feature.wvt_min_D1,
                    wvt_min_D2=local_feature.wvt_min_D2,
                    wvt_min_D3=local_feature.wvt_min_D3,
                    wvt_min_D4=local_feature.wvt_min_D4,
                    wvt_min_D5=local_feature.wvt_min_D5,
                    wvt_min_A5=local_feature.wvt_min_A5,

                    # Стандартное отклонение вейвлета для каждого уровня декомпозиции
                    wvt_std_D1=local_feature.wvt_std_D1,
                    wvt_std_D2=local_feature.wvt_std_D2,
                    wvt_std_D3=local_feature.wvt_std_D3,
                    wvt_std_D4=local_feature.wvt_std_D4,
                    wvt_std_D5=local_feature.wvt_std_D5,
                    wvt_std_A5=local_feature.wvt_std_A5,

                    # Коэффициент асимметрии вейвлета для каждого уровня декомпозиции
                    wvt_skew_D1=local_feature.wvt_skew_D1,
                    wvt_skew_D2=local_feature.wvt_skew_D2,
                    wvt_skew_D3=local_feature.wvt_skew_D3,
                    wvt_skew_D4=local_feature.wvt_skew_D4,
                    wvt_skew_D5=local_feature.wvt_skew_D5,
                    wvt_skew_A5=local_feature.wvt_skew_A5,

                    # Коэффициент эксцесса вейвлета для каждого уровня декомпозиции
                    wvt_kurt_D1=local_feature.wvt_kurt_D1,
                    wvt_kurt_D2=local_feature.wvt_kurt_D2,
                    wvt_kurt_D3=local_feature.wvt_kurt_D3,
                    wvt_kurt_D4=local_feature.wvt_kurt_D4,
                    wvt_kurt_D5=local_feature.wvt_kurt_D5,
                    wvt_kurt_A5=local_feature.wvt_kurt_A5,

                    # Энтропия вейвлета для каждого уровня декомпозиции
                    wvt_entr_D1=local_feature.wvt_entr_D1,
                    wvt_entr_D2=local_feature.wvt_entr_D2,
                    wvt_entr_D3=local_feature.wvt_entr_D3,
                    wvt_entr_D4=local_feature.wvt_kurt_D4,
                    wvt_entr_D5=local_feature.wvt_kurt_D5,
                    wvt_entr_A5=local_feature.wvt_kurt_A5,

                    # Отношение энергий между различными уровнями декомпозиции
                    wvt_energ_D1D2=local_feature.wvt_energ_D1D2,
                    wvt_energ_D2D3=local_feature.wvt_energ_D2D3,
                    wvt_energ_D3D4=local_feature.wvt_energ_D3D4,
                    wvt_energ_D4D5=local_feature.wvt_energ_D4D5,
                    wvt_energ_D5A5=local_feature.wvt_energ_D5A5,

                    # Отношение энергии высокочастотных компонент к низкочастотным
                    wvt_HfLf_Ratio=local_feature.wvt_HfLf_Ratio,

                    # Соотношение высоких и низких частот на разных масштабах
                    wvt_HfLf_D1=local_feature.wvt_HfLf_D1,
                    wvt_HfLf_D2=local_feature.wvt_HfLf_D2,
                    wvt_HfLf_D3=local_feature.wvt_HfLf_D3,
                    wvt_HfLf_D4=local_feature.wvt_HfLf_D4,
                    wvt_HfLf_D5=local_feature.wvt_HfLf_D5,
                )
                remote_session.add(new_feature)
                new_feature_count += 1

        remote_session.commit()
        added_word = "Добавлена" if new_feature_count == 1 else "Добавлено"
        set_info(
            f'{added_word} {pluralize(new_feature_count, ["новая запись", "новых записи", "новых записей"])} в таблицу WaveletFeatureRDB',
            'green')

    # set_info('Выгрузка данных с локальной БД на удаленную завершена', 'blue')


def load_wavelet_feature():
    """Загрузка таблицs WaveletFeature с удаленной БД на локальную"""

    # set_info('Начало загрузки данных с удаленной БД на локальную', 'blue')

    set_info('Проверка наличия связанных пластов для вейвлет параметров в локальной БД', 'blue')
    with get_session() as remote_session:

        local_formations = {}
        for f in session.query(Formation.up_hash, Formation.down_hash, Formation.id).all():
            local_formations[f.up_hash] = f.id
            local_formations[f.down_hash] = f.id

        remote_features = remote_session.query(
            WaveletFeatureRDB.id,
            FormationRDB.up_hash,
            FormationRDB.down_hash
        ).join(
            FormationRDB, WaveletFeatureRDB.formation_id == FormationRDB.id
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
            WaveletFeatureRDB.id,
            FormationRDB.up_hash,
            FormationRDB.down_hash
        ).join(
            FormationRDB, WaveletFeatureRDB.formation_id == FormationRDB.id
        ).all()
        local_formations = {}
        for f in session.query(Formation.up_hash, Formation.down_hash, Formation.id).all():
            local_formations[f.up_hash] = [f.id]
            local_formations[f.down_hash] = [f.id]

        ui.progressBar.setMaximum(len(remote_features))
        new_feature_count = 0

        for n, (remote_feature_id, formation_up_hash, formation_down_hash) in tqdm(enumerate(remote_features), desc='Загрузка вейвлет параметров'):
            ui.progressBar.setValue(n+1)

            # Получаем ID пласта
            local_formation_list = local_formations.get(formation_up_hash)
            if not local_formation_list:
                local_formation_list = local_formations.get(formation_down_hash)

            local_feature_count = session.query(WaveletFeature).filter_by(
                formation_id=local_formation_list[0]
            ).count()

            if local_feature_count == 0:
                remote_feature = remote_session.get(WaveletFeatureRDB, remote_feature_id)
                new_feature = WaveletFeature(
                    formation_id=local_formation_list[0],

                    # Энергия вейвлета для каждого уровня декомпозиции
                    wvt_energ_D1=remote_feature.wvt_energ_D1,
                    wvt_energ_D2=remote_feature.wvt_energ_D2,
                    wvt_energ_D3=remote_feature.wvt_energ_D3,
                    wvt_energ_D4=remote_feature.wvt_energ_D4,
                    wvt_energ_D5=remote_feature.wvt_energ_D5,
                    wvt_energ_A5=remote_feature.wvt_energ_A5,

                    # Среднее вейвлета для каждого уровня декомпозиции
                    wvt_mean_D1=remote_feature.wvt_mean_D1,
                    wvt_mean_D2=remote_feature.wvt_mean_D2,
                    wvt_mean_D3=remote_feature.wvt_mean_D3,
                    wvt_mean_D4=remote_feature.wvt_mean_D4,
                    wvt_mean_D5=remote_feature.wvt_mean_D5,
                    wvt_mean_A5=remote_feature.wvt_mean_A5,

                    # Максимум вейвлета для каждого уровня декомпозиции
                    wvt_max_D1=remote_feature.wvt_max_D1,
                    wvt_max_D2=remote_feature.wvt_max_D2,
                    wvt_max_D3=remote_feature.wvt_max_D3,
                    wvt_max_D4=remote_feature.wvt_max_D4,
                    wvt_max_D5=remote_feature.wvt_max_D5,
                    wvt_max_A5=remote_feature.wvt_max_A5,

                    # Минимум вейвлета для каждого уровня декомпозиции
                    wvt_min_D1=remote_feature.wvt_min_D1,
                    wvt_min_D2=remote_feature.wvt_min_D2,
                    wvt_min_D3=remote_feature.wvt_min_D3,
                    wvt_min_D4=remote_feature.wvt_min_D4,
                    wvt_min_D5=remote_feature.wvt_min_D5,
                    wvt_min_A5=remote_feature.wvt_min_A5,

                    # Стандартное отклонение вейвлета для каждого уровня декомпозиции
                    wvt_std_D1=remote_feature.wvt_std_D1,
                    wvt_std_D2=remote_feature.wvt_std_D2,
                    wvt_std_D3=remote_feature.wvt_std_D3,
                    wvt_std_D4=remote_feature.wvt_std_D4,
                    wvt_std_D5=remote_feature.wvt_std_D5,
                    wvt_std_A5=remote_feature.wvt_std_A5,

                    # Коэффициент асимметрии вейвлета для каждого уровня декомпозиции
                    wvt_skew_D1=remote_feature.wvt_skew_D1,
                    wvt_skew_D2=remote_feature.wvt_skew_D2,
                    wvt_skew_D3=remote_feature.wvt_skew_D3,
                    wvt_skew_D4=remote_feature.wvt_skew_D4,
                    wvt_skew_D5=remote_feature.wvt_skew_D5,
                    wvt_skew_A5=remote_feature.wvt_skew_A5,

                    # Коэффициент эксцесса вейвлета для каждого уровня декомпозиции
                    wvt_kurt_D1=remote_feature.wvt_kurt_D1,
                    wvt_kurt_D2=remote_feature.wvt_kurt_D2,
                    wvt_kurt_D3=remote_feature.wvt_kurt_D3,
                    wvt_kurt_D4=remote_feature.wvt_kurt_D4,
                    wvt_kurt_D5=remote_feature.wvt_kurt_D5,
                    wvt_kurt_A5=remote_feature.wvt_kurt_A5,

                    # Энтропия вейвлета для каждого уровня декомпозиции
                    wvt_entr_D1=remote_feature.wvt_entr_D1,
                    wvt_entr_D2=remote_feature.wvt_entr_D2,
                    wvt_entr_D3=remote_feature.wvt_entr_D3,
                    wvt_entr_D4=remote_feature.wvt_kurt_D4,
                    wvt_entr_D5=remote_feature.wvt_kurt_D5,
                    wvt_entr_A5=remote_feature.wvt_kurt_A5,

                    # Отношение энергий между различными уровнями декомпозиции
                    wvt_energ_D1D2=remote_feature.wvt_energ_D1D2,
                    wvt_energ_D2D3=remote_feature.wvt_energ_D2D3,
                    wvt_energ_D3D4=remote_feature.wvt_energ_D3D4,
                    wvt_energ_D4D5=remote_feature.wvt_energ_D4D5,
                    wvt_energ_D5A5=remote_feature.wvt_energ_D5A5,

                    # Отношение энергии высокочастотных компонент к низкочастотным
                    wvt_HfLf_Ratio=remote_feature.wvt_HfLf_Ratio,

                    # Соотношение высоких и низких частот на разных масштабах
                    wvt_HfLf_D1=remote_feature.wvt_HfLf_D1,
                    wvt_HfLf_D2=remote_feature.wvt_HfLf_D2,
                    wvt_HfLf_D3=remote_feature.wvt_HfLf_D3,
                    wvt_HfLf_D4=remote_feature.wvt_HfLf_D4,
                    wvt_HfLf_D5=remote_feature.wvt_HfLf_D5,
                )
                session.add(new_feature)
                new_feature_count += 1

        session.commit()
        added_word = "Добавлена" if new_feature_count == 1 else "Добавлено"
        set_info(
            f'{added_word} {pluralize(new_feature_count, ["новая запись", "новых записи", "новых записей"])} в таблицу WaveletFeature',
            'green')

    set_info('Загрузка данных с удаленной БД на локальную завершена', 'blue')