from remote_db.model_remote_db import *
from func import *


def unload_geochem_func():
    """ Выгрузка геохимических данных с локальной БД на удаленную """

    with (get_session() as remote_session):

        # Получаем выбранный объект из локальной базы
        local_geochems = session.query(Geochem).filter(Geochem.id == get_geochem_id())

        unloaded_geochem_count = 0

        for local_geochem in tqdm(local_geochems, desc='Выгрузка объекта'):
            set_info(f'Выгрузка объекта геохимических данных "{local_geochem.title}"...', 'blue')

            # Проверяем существование объекта в удаленной базе
            remote_geochem = remote_session.query(GeochemRDB).filter_by(title=local_geochem.title).first()

            if not remote_geochem:
                # Добавляем отсутствующий объект
                new_geochem = GeochemRDB(title=local_geochem.title)
                remote_session.add(new_geochem)
                remote_session.commit()
                remote_geochem = new_geochem
                set_info(f'Объект геохимических данных "{local_geochem.title}" выгружен на удаленную БД', 'green')
                unloaded_geochem_count += 1
            else:
                set_info(f'Объект геохимических данных "{local_geochem.title}" есть в удаленной БД', 'red')

            set_info('Выгрузка связанных геохимических данных...', 'blue')

            # Получаем все параметры для текущего объекта из локальной базы
            local_parameters = session.query(GeochemParameter).filter_by(geochem_id=local_geochem.id).all()

            unloaded_params_count = 0

            ui.progressBar.setMaximum(len(local_parameters))
            for n, local_parameter in enumerate(local_parameters):
                ui.progressBar.setValue(n+1)

                # Проверяем существование параметра в удаленной базе
                remote_parameter = remote_session.query(GeochemParameterRDB).filter_by(
                    geochem_id=remote_geochem.id,
                    title=local_parameter.title
                ).first()

                if not remote_parameter:
                    # Добавляем отсутствующий параметр
                    new_parameter = GeochemParameterRDB(
                        geochem_id=remote_geochem.id,
                        title=local_parameter.title
                    )
                    remote_session.add(new_parameter)
                    unloaded_params_count += 1
            remote_session.commit()
            if unloaded_params_count > 0:
                added_word = "Выгружена" if unloaded_params_count == 1 else "Выгружено"
                set_info(f'{added_word} {pluralize((unloaded_params_count), ["параметр", "параметра", "параметров"])}', 'green')
            else:
                set_info('Все параметры объекта уже есть в локальной БД', 'red')

            # Словарь параметров удаленной БД
            remote_params_dict = {
                p.title: p.id
                for p in remote_session.query(GeochemParameterRDB.id, GeochemParameterRDB.title)
                .filter_by(geochem_id=remote_geochem.id)
                .all()
            }

            local_params_dict = {p.id: p.title for p in local_parameters}

            # Получаем все точки для текущего объекта из локальной базы
            local_points = session.query(GeochemPoint).filter_by(geochem_id=local_geochem.id).all()

            unloaded_points_count = 0
            unloaded_p_values_count = 0

            ui.progressBar.setMaximum(len(local_points))
            for n, local_point in enumerate(local_points):
                ui.progressBar.setValue(n+1)

                # Проверяем существование точки в удаленной базе
                remote_point = remote_session.query(GeochemPointRDB).filter_by(
                    geochem_id=remote_geochem.id,
                    title=local_point.title
                ).first()

                if not remote_point:
                    # Добавляем отсутствующую точку
                    new_point = GeochemPointRDB(
                        geochem_id=remote_geochem.id,
                        title=local_point.title,
                        x_coord=local_point.x_coord,
                        y_coord=local_point.y_coord,
                        fake=local_point.fake
                    )
                    remote_session.add(new_point)
                    remote_point = new_point
                    unloaded_points_count += 1
                    remote_session.flush()

                # Получаем все значения геохимических параметров в точке
                local_p_values = session.query(GeochemPointValue).filter_by(g_point_id=local_point.id).all()

                # Создаем множество существующих значений для проверки дубликатов
                existing_p_values_set = set()
                if remote_point:
                    existing_p_values = remote_session.query(
                        GeochemPointValueRDB.g_point_id,
                        GeochemPointValueRDB.g_param_id
                    ).filter_by(g_point_id=remote_point.id).all()
                    existing_p_values_set = {(p_v.g_point_id, p_v.g_param_id) for p_v in existing_p_values}

                for local_p_value in local_p_values:
                    # Получаем заголовок параметра через словарь
                    param_title = local_params_dict.get(local_p_value.g_param_id)

                    if param_title:
                        remote_param_id = remote_params_dict.get(param_title)

                        if remote_param_id:
                            # Проверяем существование значения в удаленной базе
                            key = (remote_point.id, remote_param_id)
                            if key not in existing_p_values_set:
                                # Добавляем отсутствующее значение
                                new_p_value = GeochemPointValueRDB(
                                    g_point_id=remote_point.id,
                                    g_param_id=remote_param_id,
                                    value=local_p_value.value
                                )
                                remote_session.add(new_p_value)
                                unloaded_p_values_count += 1
                                existing_p_values_set.add(key)

            remote_session.commit()
            if unloaded_points_count > 0:
                added_word_point = "Выгружена" if unloaded_points_count == 1 else "Выгружено"
                set_info(f'{added_word_point} {pluralize((unloaded_points_count), ["точка", "точки", "точек"])}',
                         'green')
            else:
                set_info('Все точки объекта уже есть в удаленной БД', 'red')

            if unloaded_p_values_count > 0:
                added_word_p_value = "Выгружена" if unloaded_p_values_count == 1 else "Выгружено"
                set_info(
                    f'{added_word_p_value} '
                    f'{pluralize((unloaded_p_values_count), ["значение параметров в точках", "значения параметров в точках", "значений параметров в точках"])}',
                    'green')
            else:
                set_info('Все значения параметров в точках объекта уже есть в удаленной БД', 'red')

            # Получаем все маски для текущего объекта из локальной базы
            local_masks = session.query(GeochemMask).filter_by(
                geochem_id=local_geochem.id).all()

            unloaded_masks_count = 0

            ui.progressBar.setMaximum(len(local_masks))
            for n, local_mask in enumerate(local_masks):
                ui.progressBar.setValue(n+1)

                # Проверяем существование маски в удаленной базе
                remote_mask = remote_session.query(GeochemMaskRDB).filter_by(
                    geochem_id=remote_geochem.id,
                    count_param=local_mask.count_param,
                    count_points=local_mask.count_points
                ).first()

                if not remote_mask:
                    # Добавляем отсутсвующую маску
                    new_mask = GeochemMaskRDB(
                        geochem_id=remote_geochem.id,
                        count_param=local_mask.count_param,
                        count_points=local_mask.count_points,
                        mask_param=local_mask.mask_param,
                        mask_point=local_mask.mask_point,
                        mask_info=local_mask.mask_info
                    )
                    remote_session.add(new_mask)
                    unloaded_masks_count += 1

            remote_session.commit()
            if unloaded_masks_count > 0:
                added_word = "Выгружена" if unloaded_masks_count == 1 else "Выгружено"
                set_info(f'{added_word} {pluralize((unloaded_masks_count), ["маска", "маски", "масок"])}', 'green')
            else:
                set_info('Все маски объекта уже есть в удаленной БД', 'red')

            # Получаем все скважины для текущего объекта из локальной базы
            local_wells = session.query(GeochemWell).filter_by(geochem_id=local_geochem.id).all()

            unloaded_wells_count = 0

            ui.progressBar.setMaximum(len(local_wells))
            for n, local_well in enumerate(local_wells):
                ui.progressBar.setValue(n+1)

                # Проверяем существование скважины в удаленной базе
                remote_well = remote_session.query(GeochemWellRDB).filter_by(
                    geochem_id=remote_geochem.id,
                    title=local_well.title
                ).first()

                if not remote_well:
                    # Добавляем отсутствующую скважину
                    new_well = GeochemWellRDB(
                        geochem_id=remote_geochem.id,
                        title=local_well.title,
                        color=local_well.color
                    )
                    remote_session.add(new_well)
                    remote_well = new_well
                    unloaded_wells_count += 1
                    remote_session.flush()

                # Получаем все точки скважины из локальной базы
                local_well_points = session.query(GeochemWellPoint).filter_by(g_well_id=local_well.id).all()

                unloaded_well_points_count = 0
                unloaded_w_p_values_count = 0

                for local_well_point in local_well_points:

                    # Проверяем существование точек скважины в удаленной базе
                    remote_well_point = remote_session.query(GeochemWellPointRDB).filter_by(
                        g_well_id=remote_well.id,
                        title=local_well_point.title
                    ).first()

                    if not remote_well_point:
                        # Добавляем отсутствующие точки скважины
                        new_well_point = GeochemWellPointRDB(
                            g_well_id=remote_well.id,
                            title=local_well_point.title,
                            x_coord=local_well_point.x_coord,
                            y_coord=local_well_point.y_coord
                        )
                        remote_session.add(new_well_point)
                        remote_well_point = new_well_point
                        unloaded_well_points_count += 1
                        remote_session.flush()

                    # Получаем все значения геохимических параметров в точке скважины
                    local_w_p_values = session.query(GeochemWellPointValue).filter_by(
                        g_well_point_id=local_well_point.id).all()

                    # Создаем множество существующих значений для проверки дубликатов
                    existing_w_p_values_set = set()
                    if remote_well_point:
                        existing_w_p_values = remote_session.query(
                            GeochemWellPointValueRDB.g_well_point_id,
                            GeochemWellPointValueRDB.g_param_id
                        ).filter_by(g_well_point_id=remote_well_point.id).all()
                        existing_w_p_values_set = {(w_p_v.g_well_point_id, w_p_v.g_param_id) for w_p_v in
                                                   existing_w_p_values}

                    for local_w_p_value in local_w_p_values:
                        # Получаем заголовок параметра через словарь
                        param_title = local_params_dict.get(local_w_p_value.g_param_id)

                        if param_title:
                            remote_param_id = remote_params_dict.get(param_title)

                            if remote_param_id:
                                key = (remote_well_point.id, remote_param_id)
                                if key not in existing_w_p_values_set:
                                    # Добавляем отсутствующее значение
                                    new_w_p_value = GeochemWellPointValueRDB(
                                        g_well_point_id=remote_well_point.id,
                                        g_param_id=remote_param_id,
                                        value=local_w_p_value.value
                                    )
                                    remote_session.add(new_w_p_value)
                                    unloaded_w_p_values_count += 1
                                    existing_w_p_values_set.add(key)

            remote_session.commit()
            if unloaded_wells_count > 0:
                added_word_well = "Выгружена" if unloaded_wells_count == 1 else "Выгружено"
                set_info(
                    f'{added_word_well} '
                    f'{pluralize((unloaded_wells_count), ["скважина", "скважины", "скважин"])}',
                    'green')
            else:
                set_info('Все скважины объекта уже есть в удаленной БД', 'red')
            if unloaded_well_points_count > 0:
                added_word = "Выгружена" if unloaded_well_points_count == 1 else "Выгружено"
                set_info \
                    (f'{added_word} {pluralize((unloaded_well_points_count), ["точка скважин", "точки скважин", "точек скважин"])}', 'green')
            else:
                set_info(f'Все точки скважин объекта уже есть в удаленной БД', 'red')

            if unloaded_w_p_values_count > 0:
                added_word_w_p_value = "Выгружена" if unloaded_w_p_values_count == 1 else "Выгружено"
                set_info(
                    f'{added_word_w_p_value} '
                    f'{pluralize((unloaded_w_p_values_count), ["значение параметров в точках скважин", "значения параметров в точках скважин", "значений параметров в точках скважин"])}'
                    f'"{remote_well.title}"', 'green')
            else:
                set_info(f'Все значения параметров в точках скважин объекта уже есть в '
                         f'удаленной БД', 'red')

            # Получаем все макеты для текущего объекта из локальной базы
            local_makets = session.query(GeochemMaket).filter_by(geochem_id=local_geochem.id).all()

            unloaded_makets_count = 0

            ui.progressBar.setMaximum(len(local_makets))
            for n, local_maket in enumerate(local_makets):
                ui.progressBar.setValue(n+1)

                # Проверяем существование макета в удаленной базе
                remote_maket = remote_session.query(GeochemMaketRDB).filter_by(
                    geochem_id=remote_geochem.id,
                    title=local_maket.title
                ).first()

                if not remote_maket:
                    # Добавляем отсутствующий макет
                    new_maket = GeochemMaketRDB(
                        geochem_id=remote_geochem.id,
                        title=local_maket.title
                    )
                    remote_session.add(new_maket)
                    unloaded_makets_count += 1
            remote_session.commit()
            if unloaded_makets_count > 0:
                added_word = "Выгружен" if unloaded_makets_count == 1 else "Выгружено"
                set_info(
                    f'{added_word} '
                    f'{pluralize((unloaded_makets_count), ["макет", "макета", "макетов"])}',
                    'green')
            else:
                set_info('Все макеты объета уже есть в удаленной БД', 'red')


    set_info(f'Выгрузка данных с локальной БД на удаленную завершена', 'blue')