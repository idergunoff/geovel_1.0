from remote_db.model_remote_db import *
from models_db.model import *
from qt.rem_db_window import *
from func import *
from sqlalchemy.orm import selectinload
from remote_db.sync_wells import sync_wells_func
from well_deduplication import build_spatial_index, nearby_wells


def deduplicate_wells(remote_session, distance_threshold: float = 5.0,
                      name_ratio: float = 0.5) -> None:
    """
    Удаляет дублирующиеся скважины и переносит их зависимости в «каноническую» запись.

    Параметры:
        distance_threshold – максимально допустимое расстояние между координатами (метры);
        name_ratio         – минимальная доля совпадения названий (0..1).
    """

    wells = (remote_session.query(WellRDB).options(
        selectinload(WellRDB.boundaries),
        selectinload(WellRDB.well_optionally),
        selectinload(WellRDB.well_logs),
        selectinload(WellRDB.markups_mlp),
        selectinload(WellRDB.markups_reg)
    ).order_by(WellRDB.id).all())  # все скважины по возрастанию id

    removed_ids = set()  # id скважин, помеченных на удаление
    spatial_index, cell_size = build_spatial_index(wells, distance_threshold)

    # Счётчики для итоговой статистики
    processed = duplicates = 0
    boundary_moved = optional_moved = logs_moved = 0
    mlp_moved = reg_moved = 0

    # Основной цикл с прогресс‑баром
    for well in tqdm(wells, desc='Обработка скважин'):
        processed += 1
        if well.id in removed_ids:  # пропускаем уже удалённые
            continue

        # Проверяем только текущую и соседние ячейки пространственного индекса,
        # а точное расстояние по-прежнему проверяем ниже.
        for other in nearby_wells(well, spatial_index, cell_size, distance_threshold):
            if other.id in removed_ids:
                continue

            # Расстояние между координатами
            dist = math.hypot((well.x_coord or 0) - (other.x_coord or 0),
                              (well.y_coord or 0) - (other.y_coord or 0))
            # Сходство названий
            name_sim = SequenceMatcher(None, well.name or "",
                                       other.name or "").ratio()

            if dist <= distance_threshold and name_sim >= name_ratio:
                duplicates += 1

                # --- Boundary ---
                for b in list(other.boundaries):
                    if not any(abs(b.depth - bb.depth) < 1e-6 and b.title == bb.title
                               for bb in well.boundaries):
                        b.well = well
                        boundary_moved += 1

                # --- WellOptionally ---
                optional_keys = {(o.option, o.value) for o in well.well_optionally}
                for opt in list(other.well_optionally):
                    key = (opt.option, opt.value)
                    if key not in optional_keys:
                        opt.well = well
                        optional_keys.add(key)
                        optional_moved += 1

                # --- WellLog ---
                log_keys = {log.curve_name for log in well.well_logs}
                for log in list(other.well_logs):
                    if log.curve_name not in log_keys:
                        log.well = well
                        log_keys.add(log.curve_name)
                        logs_moved += 1

                # --- MarkupMLP ---
                mlp_keys = {(m.analysis_id, m.profile_id, m.formation_id, m.type_markup)
                            for m in well.markups_mlp}
                for m in list(other.markups_mlp):
                    key = (m.analysis_id, m.profile_id, m.formation_id, m.type_markup)
                    if key not in mlp_keys:
                        m.well = well
                        mlp_keys.add(key)
                        mlp_moved += 1

                # --- MarkupReg ---
                reg_keys = {(m.analysis_id, m.profile_id, m.formation_id, m.type_markup)
                            for m in well.markups_reg}
                for m in list(other.markups_reg):
                    key = (m.analysis_id, m.profile_id, m.formation_id, m.type_markup)
                    if key not in reg_keys:
                        m.well = well
                        reg_keys.add(key)
                        reg_moved += 1

                removed_ids.add(other.id)  # помечаем дубликат
                remote_session.delete(other)  # удаляем из сессии

        # Промежуточный коммит и отчёт каждые 100 скважин
        if processed % 100 == 0:
            remote_session.flush()
            summary = (
                f"Обработано {processed} скважин, найдено {duplicates} дублей; "
                f"Boundary: {boundary_moved}, WellOptionally: {optional_moved}, "
                f"WellLog: {logs_moved}, MarkupMLP: {mlp_moved}, MarkupReg: {reg_moved}"
            )
            print(summary)
            set_info(summary, "blue")

    # Финальный коммит и итоговая статистика
    remote_session.commit()
    summary = (
        f"Обработано {processed} скважин, найдено {duplicates} дублей; "
        f"Boundary: {boundary_moved}, WellOptionally: {optional_moved}, "
        f"WellLog: {logs_moved}, MarkupMLP: {mlp_moved}, MarkupReg: {reg_moved}"
    )
    print(summary)
    set_info(summary, "blue")
