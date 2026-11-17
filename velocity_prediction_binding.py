"""Copy layer/prediction associations using one profile as a template."""

from collections import defaultdict

from sqlalchemy.orm import selectinload

from models_db.model import BindingLayerPrediction, Layers, Profile, ProfileModelPrediction, Research


def _profile_label(profile):
    return f'Профиль «{profile.title}» (id {profile.id})'


def _layers_by_title(profile):
    layers = {}
    for layer in profile.layers:
        if not layer.layer_title or layer.layer_title in layers:
            raise ValueError(
                f'{_profile_label(profile)}: пустые или повторяющиеся имена слоёв; '
                'невозможно однозначно сопоставить границы.'
            )
        layers[layer.layer_title] = layer
    return layers


def _plan_object_bindings(db, object_id, source_profile_id):
    profiles = db.query(Profile).join(Research).filter(Research.object_id == object_id).options(
        selectinload(Profile.layers).load_only(Layers.id, Layers.profile_id, Layers.layer_title)
        .selectinload(Layers.binding_layer_predictions),
        selectinload(Profile.predictions).load_only(
            ProfileModelPrediction.id, ProfileModelPrediction.profile_id,
            ProfileModelPrediction.model_id, ProfileModelPrediction.type_model,
        ),
    ).order_by(Profile.id).all()
    source = next((profile for profile in profiles if profile.id == source_profile_id), None)
    if source is None:
        raise ValueError('Выберите текущий профиль в выбранном объекте.')
    source_layers = _layers_by_title(source)
    if not source_layers:
        raise ValueError('В текущем профиле нет слоёв для привязки.')
    source_predictions = {prediction.id: prediction for prediction in source.predictions}
    template = []
    for title, layer in source_layers.items():
        if len(layer.binding_layer_predictions) != 1:
            raise ValueError(
                f'Текущий профиль, слой «{title}»: нужна ровно одна привязка к prediction. '
                'Настройте привязки через add bind / rmv bind.'
            )
        binding = layer.binding_layer_predictions[0]
        prediction = source_predictions.get(binding.prediction_id)
        if prediction is None or prediction.model_id is None or not prediction.type_model:
            raise ValueError(f'Текущий профиль, слой «{title}»: нет корректного prediction этого профиля.')
        template.append((binding.id, title, (prediction.type_model, prediction.model_id)))
    # Consumers of the velocity bindings use their insertion order.
    template.sort()
    targets = [profile for profile in profiles if profile.id != source_profile_id]
    if not targets:
        raise ValueError('В объекте нет других профилей для привязки.')

    plans, errors = [], []
    for profile in targets:
        try:
            layers = _layers_by_title(profile)
        except ValueError as error:
            errors.append(str(error))
            continue
        if layers.keys() != source_layers.keys():
            details = []
            missing = source_layers.keys() - layers.keys()
            extra = layers.keys() - source_layers.keys()
            if missing:
                details.append('нет слоёв: ' + ', '.join(sorted(missing)))
            if extra:
                details.append('лишние слои: ' + ', '.join(sorted(extra)))
            errors.append(f'{_profile_label(profile)}: состав слоёв отличается ({"; ".join(details)}).')
            continue
        predictions = defaultdict(list)
        for prediction in profile.predictions:
            predictions[(prediction.type_model, prediction.model_id)].append(prediction.id)
        desired = []
        for _, title, key in template:
            matches = predictions[key]
            if len(matches) != 1:
                reason = 'нет нужного prediction' if not matches else 'найдено несколько prediction'
                errors.append(
                    f'{_profile_label(profile)}, слой «{title}»: {reason} '
                    f'для модели {key[0]} id{key[1]}.'
                )
            else:
                desired.append((layers[title].id, matches[0]))
        existing = sorted(
            (binding for layer in profile.layers for binding in layer.binding_layer_predictions),
            key=lambda binding: binding.id,
        )
        plans.append((existing, desired))
    if errors:
        raise ValueError('Привязки не изменены.\n' + '\n'.join(errors))
    return plans


def bind_object_predictions(db, object_id, source_profile_id):
    """Validate every profile, then stage replacements; the caller commits once.

    Layer names must match exactly. Predictions match by model type and ID,
    while each target keeps its own prediction data. Repeated calls are idempotent.
    """
    with db.no_autoflush:
        plans = _plan_object_bindings(db, object_id, source_profile_id)
    for existing, desired in plans:
        if [(binding.layer_id, binding.prediction_id) for binding in existing] == desired:
            continue
        for binding in existing:
            db.delete(binding)
        db.add_all([
            BindingLayerPrediction(layer_id=layer_id, prediction_id=prediction_id)
            for layer_id, prediction_id in desired
        ])
    return len(plans)
