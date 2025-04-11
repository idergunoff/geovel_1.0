from remote_db.model_remote_db import *
from models_db.model import *
from qt.rem_db_window import *
from func import *
import hashlib

def sync_direction(source_session, target_session, source_model, target_model, batch_size):
    """
    Универсальная функция для синхронизации данных между двумя базами данных.
    :param source_session: Исходная сессия (откуда берём данные).
    :param target_session: Целевая сессия (куда добавляем или обновляем данные).
    :param source_model: Модель таблицы источника данных.
    :param target_model: Модель таблицы целевой базы данных.
    :param batch_size: Размер пакета для обработки данных.
    """
    total_wells = source_session.query(source_model).count()
    ui.progressBar.setMaximum((total_wells + batch_size - 1) // batch_size)
    n = 0
    offset = 0

    while True:
        ui.progressBar.setValue(n)
        source_well = source_session.query(source_model).order_by(source_model.id).offset(offset).limit(
            batch_size).all()

        if not source_well:
            break

        source_hashes = {w.well_hash: w for w in source_well}
        hash_list = list(source_hashes.keys())

        existing = target_session.query(target_model).filter(target_model.well_hash.in_(hash_list)).all()

        existing_hashes = {e.well_hash for e in existing}

        new_wells = [w for h, w in source_hashes.items() if h not in existing_hashes]

        # Пакетное добавление
        if new_wells:
            target_session.bulk_save_objects([
                target_model(
                    name=w.name,
                    x_coord=w.x_coord,
                    y_coord=w.y_coord,
                    alt=w.alt,
                    well_hash=w.well_hash
                ) for w in new_wells
            ])
            target_session.commit()

        offset += batch_size
        set_info(f'Обновлено {min(offset, total_wells)} скважин', 'green')

        n += 1

def create_sync_func():
    batch_size=5000

    try:
        with get_session() as remote_session:

            update_well_hashes(session, Well)  # Обновляем хэши для локальной таблицы
            update_well_hashes(remote_session, WellRDB)  # Обновляем хэши для удалённой таблицы

            set_info('Начало синхронизации...', 'blue')

            # Синхронизация скважин (удаленная -> локальная)
            set_info(f'Обновление скважин в локальной БД...', 'blue')
            sync_direction(remote_session, session, WellRDB, Well, batch_size)
            set_info(f'Обновление скважин в локальной БД завершено', 'blue')

            # Синхронизация скважин (локальная -> удаленная)
            set_info(f'Обновление скважин в удаленной БД...', 'blue')
            sync_direction(session, remote_session, Well, WellRDB, batch_size)
            set_info(f'Обновление скважин в удаленной БД завершено', 'blue')

            set_info('Синхронизация завершена', 'blue')


    except Exception as e:
        # Откат изменений в случае ошибки
        session.rollback()
        set_info(f'Синхронизация прервалась: {str(e)}', 'red')
        raise # Проброс исключения для дальнейшей обработки
    finally:
        session.close()


def calculate_well_hash(x_coord, y_coord):
    """
    Вычисляет хэш-сумму на основе координат x и y.
    """
    hash_input = f"{x_coord}:{y_coord}".encode('utf-8')
    return hashlib.md5(hash_input).hexdigest()


def update_well_hashes(session, model):
    """
    Обновляет поле well_hash для всех записей в таблице.
    """
    records = session.query(model).all()
    for record in records:
        record.well_hash = calculate_well_hash(record.x_coord, record.y_coord)
    session.commit()
