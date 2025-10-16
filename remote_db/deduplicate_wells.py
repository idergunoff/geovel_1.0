from remote_db.model_remote_db import *
from models_db.model import *
from qt.rem_db_window import *
from func import *
from sqlalchemy.orm import selectinload
from remote_db.sync_wells import sync_wells_func


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

    # Счётчики для итоговой статистики
    processed = duplicates = 0
    boundary_moved = optional_moved = logs_moved = 0
    mlp_moved = reg_moved = 0

    # Основной цикл с прогресс‑баром
    for well in tqdm(wells, desc='Обработка скважин'):
        processed += 1
        if well.id in removed_ids:  # пропускаем уже удалённые
            continue

        for other in wells:
            if other.id in removed_ids or other.id <= well.id:
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
                for opt in list(other.well_optionally):
                    if not any(opt.option == o.option and opt.value == o.value
                               for o in well.well_optionally):
                        opt.well = well
                        optional_moved += 1

                # --- WellLog ---
                for log in list(other.well_logs):
                    if not any(log.curve_name == l.curve_name for l in well.well_logs):
                        log.well = well
                        logs_moved += 1

                # --- MarkupMLP ---
                for m in list(other.markups_mlp):
                    if not any(
                            m.analysis_id == ex.analysis_id and
                            m.profile_id == ex.profile_id and
                            m.formation_id == ex.formation_id and
                            m.type_markup == ex.type_markup
                            for ex in well.markups_mlp
                    ):
                        m.well = well
                        mlp_moved += 1

                # --- MarkupReg ---
                for m in list(other.markups_reg):
                    if not any(
                            m.analysis_id == ex.analysis_id and
                            m.profile_id == ex.profile_id and
                            m.formation_id == ex.formation_id and
                            m.type_markup == ex.type_markup
                            for ex in well.markups_reg
                    ):
                        m.well = well
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
