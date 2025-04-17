from func import *
from sqlalchemy.orm import selectinload, lazyload

def sync_objects_direction(source_session, target_session, source_obj_model, source_res_model, source_prof_model,
                           target_obj_model, target_res_model, target_prof_model):
    """
        Универсальная функция для синхронизации данных между двумя базами данных.
        :param source_session: Исходная сессия (откуда берём данные).
        :param target_session: Целевая сессия (куда добавляем или обновляем данные).
        :param source_obj_model: Модель таблицы GeoradarObject источника данных.
        :param source_res_model: Модель таблицы Reserach источника данных.
        :param source_prof_model: Модель таблицы Profile источника данных.
        :param target_obj_model: Модель таблицы GeoradarObject целевой базы данных.
        :param target_res_model: Модель таблицы Research целевой базы данных.
        :param target_prof_model: Модель таблицы Profile целевой базы данных.
        """

    # Получаем все объекты из исходной базы
    source_objects = source_session.query(source_obj_model).all()

    added_objects_count = 0
    added_researches_count = 0
    added_profiles_count = 0

    ui.progressBar.setMaximum(len(source_objects))

    for n, source_object in tqdm(enumerate(source_objects), desc="Синхронизация объектов"):
        ui.progressBar.setValue(n + 1)

        # Проверяем существование объекта в целевой базе
        target_object = target_session.query(target_obj_model).filter_by(title=source_object.title).first()

        if not target_object:
            # Добавляем отсутствующий объект
            new_object = target_obj_model(title=source_object.title)
            target_session.add(new_object)
            target_session.flush()
            added_objects_count += 1
            target_object = new_object

        # Получаем все исследования для текущего объекта из исходной базы
        source_researches = source_session.query(source_res_model).filter_by(object_id=source_object.id).all()

        for source_research in tqdm(source_researches, desc="Синхронизация исследований"):
            # Проверяем существование исследования в целевой базе
            target_research = target_session.query(target_res_model)\
                .filter_by(
                object_id=target_object.id,
                date_research=source_research.date_research
            ).first()

            if not target_research:
                # Добавляем отсутствующее исследование
                new_research = target_res_model(
                    object_id=target_object.id,
                    date_research=source_research.date_research
                )
                target_session.add(new_research)
                target_session.flush()
                added_researches_count += 1
                target_research = new_research

            # Получаем все id и signal_hash профилей для текущего исследования из исходной базы
            source_profiles = source_session.query(source_prof_model.id, source_prof_model.signal_hash_md5).filter_by(
                research_id=source_research.id).all()

            for source_profile in tqdm(source_profiles, desc="Синхронизация профилей"):
                # Проверяем существование профиля в целевой базе по signal_hash
                target_profile = target_session.query(target_prof_model.id).filter_by(
                    research_id=target_research.id,
                    signal_hash_md5=source_profile[1]
                ).first()

                if not target_profile:
                    source_profile_all = source_session.query(source_prof_model).filter_by(id=source_profile[0]).first()
                    # Добавляем отсутствующий профиль
                    new_profile = target_prof_model(
                        research_id=target_research.id,
                        title=source_profile_all.title,
                        signal=source_profile_all.signal,
                        signal_hash_md5=source_profile_all.signal_hash_md5,
                        x_wgs=source_profile_all.x_wgs,
                        y_wgs=source_profile_all.y_wgs,
                        x_pulc=source_profile_all.x_pulc,
                        y_pulc=source_profile_all.y_pulc,
                        abs_relief=source_profile_all.abs_relief,
                        depth_relief=source_profile_all.depth_relief
                    )
                    target_session.add(new_profile)
                    added_profiles_count += 1

                # Периодически коммитим
                if n % 50 == 0:
                    target_session.flush()

            target_session.commit()


    set_info(f'Добавлено: {pluralize(added_objects_count, ["объект", "объекта", "объектов"])}, '
             f'{pluralize(added_researches_count, ["исследование", "исследования", "исследований"])}, '
             f'{pluralize(added_profiles_count, ["профиль", "профиля", "профилей"])}', 'green')

