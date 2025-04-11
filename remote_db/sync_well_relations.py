from remote_db.model_remote_db import *
from models_db.model import *
from qt.rem_db_window import *
from func import *
from sqlalchemy.orm import selectinload
from remote_db.sync_wells import create_sync_func


def sync_well_relations(source_session, target_session, source_well_model, target_well_model, target_bound_model, 
                        target_opt_model, target_log_model, batch_size):
    """
    Синхронизация связанных данных
    :param source_session: Исходная сессия (откуда берём данные).
    :param target_session: Целевая сессия (куда добавляем или обновляем данные).
    :param source_well_model: Модель таблицы well источника данных.
    :param target_well_model: Модель таблицы well целевой базы данных.
    :param target_bound_model: Модель таблицы boundary целевой базы данных.
    :param target_opt_model: Модель таблицы well_optionally целевой базы данных.
    :param target_log_model: Модель таблицы well_log целевой базы данных.
    :param batch_size: Размер пакета для обработки данных.
    """

    total_wells = source_session.query(source_well_model).count()
    ui.progressBar.setMaximum((total_wells + batch_size - 1) // batch_size)
    n = 0
    offset = 0

    while True:
        ui.progressBar.setValue(n)
        source_wells = source_session.query(source_well_model) \
            .order_by(source_well_model.id) \
            .offset(offset).limit(batch_size) \
            .options(
            selectinload(source_well_model.boundaries),
            selectinload(source_well_model.well_optionally),
            selectinload(source_well_model.well_logs)
        ).all()

        if not source_wells:
            break

        try:
            # Получаем хеши всех скважин в текущем batch
            well_hashes = [well.well_hash for well in source_wells]

            # Предзагружаем целевые скважины со связанными данными
            target_wells_map = {
                well.well_hash: well for well in
                target_session.query(target_well_model)
                .filter(target_well_model.well_hash.in_(well_hashes))
                .options(
                    selectinload(target_well_model.boundaries),
                    selectinload(target_well_model.well_optionally),
                    selectinload(target_well_model.well_logs)
                )
            }

            for source_well in tqdm(source_wells):
                target_well = target_wells_map.get(source_well.well_hash)

                if not target_well:
                    create_sync_func()
                    set_info(f'Произведена синхронизация скважин, надо заново выполнить синхронизацию данных '
                             f'скважин!!!', 'red')
                    return

                # Синхронизация границ
                if hasattr(source_well, 'boundaries') and source_well.boundaries:
                    existing_boundaries = {b.depth: b for b in target_well.boundaries}
                    new_boundaries = [
                        {'well_id': target_well.id, 'depth': b.depth, 'title': b.title}
                        for b in source_well.boundaries
                        if b.depth not in existing_boundaries
                    ]
                    if new_boundaries:
                        new_b_count = len(new_boundaries)
                        target_session.bulk_insert_mappings(target_bound_model, new_boundaries)
                        set_info(
                            f'Добавлено {new_b_count} границ для скважины {target_well.name}',
                            'blue'
                        )

                # Синхронизация опций
                if hasattr(source_well, 'well_optionally') and source_well.well_optionally:
                    existing_options = {o.option: o for o in target_well.well_optionally}
                    new_options = [
                        {'well_id': target_well.id, 'option': o.option, 'value': o.value}
                        for o in source_well.well_optionally
                        if o.option not in existing_options
                    ]
                    if new_options:
                        new_o_count = len(new_options)
                        target_session.bulk_insert_mappings(target_opt_model, new_options)
                        set_info(f'Добавлено {new_o_count} опций для скважины {target_well.name}', 'blue')

                # Синхронизация каротажа
                if hasattr(source_well, 'well_logs') and source_well.well_logs:
                    existing_logs = {l.curve_name for l in target_well.well_logs}
                    new_logs = [
                        {
                                    'well_id': target_well.id,
                                    'curve_name': l.curve_name,
                                    'begin': l.begin,
                                    'end': l.end,
                                    'step': l.step,
                                    'curve_data': l.curve_data
                                }
                        for l in source_well.well_logs
                        if l.curve_name not in existing_logs
                    ]
                    if new_logs:
                        new_l_count = len(new_logs)
                        target_session.bulk_insert_mappings(target_log_model, new_logs)
                        set_info(f'Добавлено {new_l_count} каротажей для скважины {target_well.name}', 'blue')

            target_session.commit()

            offset += batch_size
            set_info(f'Обновлены данные {min(offset, total_wells)} из {total_wells} скважин', 'green')
            n += 1

        except Exception as e:
            target_session.rollback()
            set_info(f'Ошибка при обработке batch {offset}-{offset + batch_size}: {str(e)}', 'red')
            raise


def load_well_relations():
    """ Загрузка данных скважин с удаленной БД на локальную"""
    batch_size=5000

    try:
        with get_session() as remote_session:

            # Синхронизация данных скважин (удаленная -> локальная)
            set_info(f'Обновление данных скважин в локальной БД...', 'blue')
            sync_well_relations(remote_session, session, WellRDB, Well, Boundary, WellOptionally, WellLog, batch_size)
            set_info(f'Обновление данных скважин в локальной БД завершено', 'blue')

    except Exception as e:
        # Откат изменений в случае ошибки
        session.rollback()
        set_info(f'Синхронизация прервалась: {str(e)}', 'red')
        raise # Проброс исключения для дальнейшей обработки
    finally:
        session.close()

def unload_well_relations():
    """ Выгрузка данных скважин с локальной БД на удаленную"""
    batch_size=5000

    try:
        with get_session() as remote_session:

            # Синхронизация данных скважин (локальная -> удаленная)
            set_info(f'Обновление данных скважин в удаленной БД...', 'blue')
            sync_well_relations(session, remote_session, Well, WellRDB, BoundaryRDB, WellOptionallyRDB, WellLogRDB,
                                     batch_size)
            set_info(f'Обновление данных скважин в удаленной БД завершено', 'blue')


    except Exception as e:
        # Откат изменений в случае ошибки
        session.rollback()
        set_info(f'Синхронизация прервалась: {str(e)}', 'red')
        raise # Проброс исключения для дальнейшей обработки
    finally:
        session.close()