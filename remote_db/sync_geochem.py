from func import *
from collections import defaultdict


def sync_geochem_direction(source_session, target_session,
                           source_geochem_model, source_g_param_model, source_g_point_model, source_g_mask_model,
                           source_g_well_model, source_g_well_point_model, source_g_point_value_model,
                           source_g_well_point_value_model, source_g_maket_model, target_geochem_model,
                           target_g_param_model, target_g_point_model, target_g_mask_model, target_g_well_model,
                           target_g_well_point_model, target_g_point_value_model, target_g_well_point_value_model,
                           target_g_maket_model):
    # Загружаем все данные из исходной БД один раз
    print("Загрузка данных из исходной БД...")
    source_geochems = source_session.query(source_geochem_model).all()

    # Создаем словари для быстрого доступа
    source_params_by_geochem = defaultdict(list)
    for param in source_session.query(source_g_param_model).all():
        source_params_by_geochem[param.geochem_id].append(param)

    source_points_by_geochem = defaultdict(list)
    for point in source_session.query(source_g_point_model).all():
        source_points_by_geochem[point.geochem_id].append(point)

    source_point_values_by_point = defaultdict(list)
    for val in source_session.query(source_g_point_value_model).all():
        source_point_values_by_point[val.g_point_id].append(val)

    source_masks_by_geochem = defaultdict(list)
    for mask in source_session.query(source_g_mask_model).all():
        source_masks_by_geochem[mask.geochem_id].append(mask)

    source_wells_by_geochem = defaultdict(list)
    for well in source_session.query(source_g_well_model).all():
        source_wells_by_geochem[well.geochem_id].append(well)

    source_well_points_by_well = defaultdict(list)
    for wp in source_session.query(source_g_well_point_model).all():
        source_well_points_by_well[wp.g_well_id].append(wp)

    source_well_point_values_by_point = defaultdict(list)
    for wpv in source_session.query(source_g_well_point_value_model).all():
        source_well_point_values_by_point[wpv.g_well_point_id].append(wpv)

    source_makets_by_geochem = defaultdict(list)
    for maket in source_session.query(source_g_maket_model).all():
        source_makets_by_geochem[maket.geochem_id].append(maket)

    # Загружаем существующие данные из целевой БД
    print("Загрузка существующих данных из целевой БД...")
    target_geochems = {g.title: g for g in target_session.query(target_geochem_model).all()}

    target_params_by_geochem = defaultdict(dict)
    for p in target_session.query(target_g_param_model).all():
        target_params_by_geochem[p.geochem_id][p.title] = p.id

    target_points_by_geochem = {}
    for p in target_session.query(target_g_point_model).all():
        target_points_by_geochem[(p.geochem_id, p.title)] = p

    target_point_values_set = set()
    for v in target_session.query(target_g_point_value_model.g_point_id, target_g_point_value_model.g_param_id).all():
        target_point_values_set.add((v.g_point_id, v.g_param_id))

    target_masks_by_geochem = {}
    for m in target_session.query(target_g_mask_model).all():
        target_masks_by_geochem[(m.geochem_id, m.count_param, m.count_points)] = m

    target_wells_by_geochem = {}
    for w in target_session.query(target_g_well_model).all():
        target_wells_by_geochem[(w.geochem_id, w.title)] = w

    target_well_points_by_well = {}
    for wp in target_session.query(target_g_well_point_model).all():
        target_well_points_by_well[(wp.g_well_id, wp.title)] = wp

    target_well_point_values_set = set()
    for wpv in target_session.query(target_g_well_point_value_model.g_well_point_id,
                                    target_g_well_point_value_model.g_param_id).all():
        target_well_point_values_set.add((wpv.g_well_point_id, wpv.g_param_id))

    target_makets_by_geochem = {}
    for m in target_session.query(target_g_maket_model).all():
        target_makets_by_geochem[(m.geochem_id, m.title)] = m

    # Счетчики
    counts = {
        'geochem': 0, 'param': 0, 'point': 0, 'point_value': 0,
        'mask': 0, 'well': 0, 'well_point': 0, 'well_point_value': 0, 'maket': 0
    }

    ui.progressBar.setMaximum(len(source_geochems))

    # Синхронизация
    print("Синхронизация данных...")
    for n, source_geochem in enumerate(source_geochems):
        ui.progressBar.setValue(n + 1)

        # Geochem
        target_geochem = target_geochems.get(source_geochem.title)
        if not target_geochem:
            target_geochem = target_geochem_model(title=source_geochem.title)
            target_session.add(target_geochem)
            target_session.flush()
            target_geochems[source_geochem.title] = target_geochem
            counts['geochem'] += 1

        # Parameters
        source_params = source_params_by_geochem.get(source_geochem.id, [])
        for source_param in source_params:
            if source_param.title not in target_params_by_geochem.get(target_geochem.id, {}):
                new_param = target_g_param_model(
                    geochem_id=target_geochem.id,
                    title=source_param.title
                )
                target_session.add(new_param)
                target_session.flush()
                target_params_by_geochem[target_geochem.id][source_param.title] = new_param.id
                counts['param'] += 1

        # Маппинг параметров
        source_param_title_by_id = {p.id: p.title for p in source_params}
        target_param_id_by_title = target_params_by_geochem.get(target_geochem.id, {})

        # Points
        source_points = source_points_by_geochem.get(source_geochem.id, [])
        for source_point in source_points:
            key = (target_geochem.id, source_point.title)
            target_point = target_points_by_geochem.get(key)

            if not target_point:
                target_point = target_g_point_model(
                    geochem_id=target_geochem.id,
                    title=source_point.title,
                    x_coord=source_point.x_coord,
                    y_coord=source_point.y_coord,
                    fake=source_point.fake
                )
                target_session.add(target_point)
                target_session.flush()
                target_points_by_geochem[key] = target_point
                counts['point'] += 1

            # Point values
            source_values = source_point_values_by_point.get(source_point.id, [])
            for source_val in source_values:
                param_title = source_param_title_by_id.get(source_val.g_param_id)
                if param_title:
                    target_param_id = target_param_id_by_title.get(param_title)
                    if target_param_id and (target_point.id, target_param_id) not in target_point_values_set:
                        new_value = target_g_point_value_model(
                            g_point_id=target_point.id,
                            g_param_id=target_param_id,
                            value=source_val.value
                        )
                        target_session.add(new_value)
                        target_point_values_set.add((target_point.id, target_param_id))
                        counts['point_value'] += 1

        # Masks
        source_masks = source_masks_by_geochem.get(source_geochem.id, [])
        for source_mask in source_masks:
            key = (target_geochem.id, source_mask.count_param, source_mask.count_points)
            target_mask = target_masks_by_geochem.get(key)
        # if source_masks and target_geochem.id not in target_masks_by_geochem:
        #     source_mask = source_masks[0]  # Берем первую маску (обычно одна)
            if not target_mask:
                new_mask = target_g_mask_model(
                    geochem_id=target_geochem.id,
                    count_param=source_mask.count_param,
                    count_points=source_mask.count_points,
                    mask_param=source_mask.mask_param,
                    mask_point=source_mask.mask_point,
                    mask_info=source_mask.mask_info
                )
                target_session.add(new_mask)
                target_masks_by_geochem[target_geochem.id] = new_mask
                counts['mask'] += 1

        # Wells
        source_wells = source_wells_by_geochem.get(source_geochem.id, [])
        for source_well in source_wells:
            well_key = (target_geochem.id, source_well.title)
            target_well = target_wells_by_geochem.get(well_key)

            if not target_well:
                target_well = target_g_well_model(
                    geochem_id=target_geochem.id,
                    title=source_well.title,
                    color=source_well.color
                )
                target_session.add(target_well)
                target_session.flush()
                target_wells_by_geochem[well_key] = target_well
                counts['well'] += 1

            # Well points
            source_well_points = source_well_points_by_well.get(source_well.id, [])
            for source_wp in source_well_points:
                wp_key = (target_well.id, source_wp.title)
                target_wp = target_well_points_by_well.get(wp_key)

                if not target_wp:
                    target_wp = target_g_well_point_model(
                        g_well_id=target_well.id,
                        title=source_wp.title,
                        x_coord=source_wp.x_coord,
                        y_coord=source_wp.y_coord
                    )
                    target_session.add(target_wp)
                    target_session.flush()
                    target_well_points_by_well[wp_key] = target_wp
                    counts['well_point'] += 1

                # Well point values
                source_wp_values = source_well_point_values_by_point.get(source_wp.id, [])
                for source_wpv in source_wp_values:
                    param_title = source_param_title_by_id.get(source_wpv.g_param_id)
                    if param_title:
                        target_param_id = target_param_id_by_title.get(param_title)
                        if target_param_id and (target_wp.id, target_param_id) not in target_well_point_values_set:
                            new_value = target_g_well_point_value_model(
                                g_well_point_id=target_wp.id,
                                g_param_id=target_param_id,
                                value=source_wpv.value
                            )
                            target_session.add(new_value)
                            target_well_point_values_set.add((target_wp.id, target_param_id))
                            counts['well_point_value'] += 1

        # Makets
        source_makets = source_makets_by_geochem.get(source_geochem.id, [])
        for source_maket in source_makets:
            maket_key = (target_geochem.id, source_maket.title)
            if maket_key not in target_makets_by_geochem:
                new_maket = target_g_maket_model(
                    geochem_id=target_geochem.id,
                    title=source_maket.title
                )
                target_session.add(new_maket)
                target_makets_by_geochem[maket_key] = new_maket
                counts['maket'] += 1

    # Один коммит в конце
    target_session.commit()

    set_info(f'Добавлено: '
             f'{pluralize(counts["geochem"], ["объект", "объекта", "объектов"])}, '
             f'{pluralize(counts["param"], ["параметр", "параметра", "параметров"])}, '
             f'{pluralize(counts["point"], ["точка", "точки", "точек"])}, '
             f'{pluralize(counts["point_value"], ["значение параметра в точке", "значения параметра в точке", "значений параметра в точке"])}, '
             f'{pluralize(counts["mask"], ["маска", "маски", "масок"])}, '
             f'{pluralize(counts["well"], ["скважина", "скважины", "скважин"])}, '
             f'{pluralize(counts["well_point"], ["точка скважины", "точки скважины", "точек скважин"])}, '
             f'{pluralize(counts["well_point_value"], ["значение параметра в точке скважины", "значения параметра в точке скважины", "значений параметра в точке скважины"])}, '
             f'{pluralize(counts["maket"], ["макет", "макета", "макетов"])}', 'green')


