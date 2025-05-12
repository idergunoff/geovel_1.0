from remote_db.model_remote_db import *
from models_db.model import *
from func import *
from sqlalchemy.orm import selectinload

# Функция для проверки зависимостей
def check_rdb_reg_dependencies():
    set_info('Проверка наличия всех связанных данных в удаленной БД', 'blue')
    errors = []
    local_analyzes = session.query(AnalysisReg).filter(AnalysisReg.id == get_regmod_id())

    # Предзагрузка данных из удаленной БД для проверки
    with get_session() as remote_session:
        # Проверяем наличие всех связанных данных в удаленной БД
        remote_wells = {
            w.well_hash: w.id
            for w in remote_session.query(WellRDB.well_hash, WellRDB.id).all()
        }

        remote_profiles = {
            p.signal_hash_md5: p.id
            for p in remote_session.query(ProfileRDB.signal_hash_md5, ProfileRDB.id).all()
        }

        remote_formations = {}
        for f in remote_session.query(FormationRDB.up_hash, FormationRDB.down_hash, FormationRDB.id).all():
            remote_formations[f.up_hash] = f.id
            remote_formations[f.down_hash] = f.id

        for local_analysis in tqdm(local_analyzes, desc='Проверка зависимостей RegMod'):

            local_markups = session.query(MarkupReg).filter_by(analysis_id=local_analysis.id).all()

            for local_markup in local_markups:
                related_tables = []

                # Проверяем скважину
                try:
                    local_well_hash = local_markup.well.well_hash
                    if local_well_hash not in remote_wells:
                        related_tables.append('WellRDB')
                except AttributeError:
                    pass

                # Проверяем профиль
                try:
                    local_profile_hash = local_markup.profile.signal_hash_md5
                    if local_profile_hash not in remote_profiles:
                        related_tables.append('ProfileRDB')
                except AttributeError:
                    pass

                # Проверяем пласт
                local_formation_up_hash = local_markup.formation.up_hash
                local_formation_down_hash = local_markup.formation.down_hash
                if (local_formation_up_hash not in remote_formations and
                        local_formation_down_hash not in remote_formations):
                    related_tables.append('FormationRDB')

            if related_tables:
                try:
                    error_msg = (
                        f'Для анализа "{local_analysis.title}" '
                        f'отсутствуют данные в таблицах: {", ".join(related_tables)}. '
                    )
                    errors.append(error_msg)
                except AttributeError:
                    pass
    return errors

def unload_regmod_func(Window):
    """Выгрузка таблиц AnalysisReg, MarkupReg с локальной БД на удаленную"""

    set_info('Начало выгрузки данных с локальной БД на удаленную', 'blue')

    # Сначала выполняем проверку
    dependency_errors = check_rdb_reg_dependencies()

    if dependency_errors:
        error_info = "Обнаружены следующие проблемы:\n\n" + "\n\n".join(dependency_errors)
        error_info += "\n\nНеобходимо сначала синхронизировать эти данные с локальной БД."
        set_info('Обнаружены проблемы с зависимостями', 'red')
        QMessageBox.critical(Window, 'Ошибка зависимостей', error_info)
        return
    else:
        set_info('Проблем с зависимостями нет', 'green')

    # Если проверка пройдена, выполняем выгрузку
    with get_session() as remote_session:
        local_analyzes = session.query(AnalysisReg).filter(AnalysisReg.id == get_regmod_id())

        for local_analysis in tqdm(local_analyzes, desc='Выгрузка анализа'):
            remote_analysis = remote_session.query(AnalysisRegRDB).filter_by(title=local_analysis.title).first()

            if not remote_analysis:
                new_analysis = AnalysisRegRDB(title=local_analysis.title)
                remote_session.add(new_analysis)
                remote_session.commit()
                remote_analysis = new_analysis
                set_info(f'Анализ "{local_analysis.title}" выгружен на удаленную БД', 'green')
            else:
                set_info(f'Анализ "{local_analysis.title}" есть в удаленной БД', 'blue')

            local_markups = session.query(MarkupReg) \
               .filter_by(
                analysis_id=local_analysis.id
            ).all()

            # Повторно загружаем данные
            remote_wells = {
                w.well_hash: w.id
                for w in remote_session.query(WellRDB.well_hash, WellRDB.id).all()
            }

            remote_formations = {}
            for f in remote_session.query(FormationRDB.up_hash, FormationRDB.down_hash, FormationRDB.id,
                                          FormationRDB.profile_id).all():
                remote_formations[f.up_hash] = [f.id, f.profile_id]
                remote_formations[f.down_hash] = [f.id, f.profile_id]

            added_markups_count = 0
            ui.progressBar.setMaximum(len(local_markups))

            for n, local_markup in tqdm(enumerate(local_markups), desc='Выгрузка обучающих скважин'):
                ui.progressBar.setValue(n + 1)

                remote_well_id = remote_wells[local_markup.well.well_hash] if local_markup.well_id != 0 else None

                # Получаем ID пласта
                remote_formation_list = remote_formations.get(local_markup.formation.up_hash)
                if not remote_formation_list:
                    remote_formation_list = remote_formations.get(local_markup.formation.down_hash)

                remote_markup = remote_session.query(MarkupRegRDB).filter_by(
                    analysis_id=remote_analysis.id,
                    well_id=remote_well_id,
                    profile_id=remote_formation_list[1],
                    formation_id=remote_formation_list[0]
                ).first()

                if not remote_markup:
                    new_markup = MarkupRegRDB(
                        analysis_id=remote_analysis.id,
                        well_id=remote_well_id,
                        profile_id=remote_formation_list[1],
                        formation_id=remote_formation_list[0],
                        target_value=local_markup.target_value,
                        list_measure=local_markup.list_measure,
                        type_markup=local_markup.type_markup
                    )
                    remote_session.add(new_markup)
                    added_markups_count += 1

            remote_session.commit()
            set_info(
                f'Выгружено: '
                f'{pluralize(added_markups_count, ["обучающая скважина", "обучающие скважины", "обучающих скважин"])}',
                'green')

    set_info('Выгрузка данных с локальной БД на удаленную завершена', 'blue')
