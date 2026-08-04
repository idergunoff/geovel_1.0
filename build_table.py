from collections import Counter

from calc_profile_features import calc_wavelet_features_profile, calc_fractal_features_profile, \
    calc_entropy_features_profile, calc_nonlinear_features_profile, calc_morphology_features_profile, \
    calc_frequency_features_profile, calc_envelope_feature_profile, calc_autocorr_feature_profile, \
    calc_emd_feature_profile, calc_hht_features_profile
from func import *
from calc_additional_features import (calc_hht_features, calc_emd_feature, calc_autocorr_feature, calc_nonlinear_features,
                                      calc_envelope_feature, calc_frequency_features, calc_morphology_features,
                                      calc_entropy_features, calc_fractal_features,
                                      calc_wavelet_features)


# Ограничивает количество тяжёлых Python-словарей, одновременно находящихся в памяти.
TRAIN_TABLE_ROW_CHUNK_SIZE = 1000


def _stored_feature_source(param):
    """Return model, id column and SQL value column for a stored parameter."""
    profile_sources = (
        (list_wavelet_features, WaveletFeatureProfile, 'wavelet_feature_profile'),
        (list_fractal_features, FractalFeatureProfile, 'fractal_feature_profile'),
        (list_entropy_features, EntropyFeatureProfile, 'entropy_feature_profile'),
        (list_nonlinear_features, NonlinearFeatureProfile, 'nonlinear_feature_profile'),
        (list_morphology_feature, MorphologyFeatureProfile, 'morphology_feature_profile'),
        (list_frequency_feature, FrequencyFeatureProfile, 'frequency_feature_profile'),
        (list_envelope_feature, EnvelopeFeatureProfile, 'envelope_feature_profile'),
        (list_autocorr_feature, AutocorrFeatureProfile, 'autocorr_feature_profile'),
        (list_emd_feature, EMDFeatureProfile, 'emd_feature_profile'),
        (list_hht_feature, HHTFeatureProfile, 'hht_feature_profile'),
    )
    formation_sources = (
        (list_wavelet_features, WaveletFeature, 'wavelet_feature'),
        (list_fractal_features, FractalFeature, 'fractal_feature'),
        (list_entropy_features, EntropyFeature, 'entropy_feature'),
        (list_nonlinear_features, NonlinearFeature, 'nonlinear_feature'),
        (list_morphology_feature, MorphologyFeature, 'morphology_feature'),
        (list_frequency_feature, FrequencyFeature, 'frequency_feature'),
        (list_envelope_feature, EnvelopeFeature, 'envelope_feature'),
        (list_autocorr_feature, AutocorrFeature, 'autocorr_feature'),
        (list_emd_feature, EMDFeature, 'emd_feature'),
        (list_hht_feature, HHTFeature, 'hht_feature'),
    )

    if param.startswith('prof'):
        feature_name = param[5:]
        for feature_names, model, table_name in profile_sources:
            if feature_name in feature_names:
                return model, model.profile_id, f'{table_name}.{feature_name}', True
        return None

    for feature_names, model, table_name in formation_sources:
        if param in feature_names:
            return model, model.formation_id, f'{table_name}.{param}', False
    return Formation, Formation.id, f'Formation.{param}', False


def _prefetch_stored_features(markups, list_param):
    """Load requested parameters in one query per feature storage table."""
    formation_ids = {markup.formation_id for markup in markups}
    profile_ids = {markup.profile_id for markup in markups}
    cache = {}
    grouped_columns = {}
    calculated_prefixes = ('Signal', 'distr', 'sep', 'mfcc', 'model_')

    for param in list_param:
        if param in ('CRL', 'CRL_NF', 'X', 'Y') or param.startswith(calculated_prefixes):
            continue
        source = _stored_feature_source(param)
        if source is None:
            continue
        model, id_column, value_column, is_profile = source
        group_key = (model, id_column, is_profile)
        grouped_columns.setdefault(group_key, []).append((param, value_column))

    for (_, id_column, is_profile), param_columns in grouped_columns.items():
        entity_ids = profile_ids if is_profile else formation_ids
        if not entity_ids:
            continue
        columns = [id_column]
        columns.extend(literal_column(value_column) for _, value_column in param_columns)
        rows = session.query(*columns).filter(id_column.in_(entity_ids)).all()
        for row in rows:
            entity_id = row[0]
            for (param, _), value in zip(param_columns, row[1:]):
                if value is not None:
                    cache[(param, entity_id)] = json.loads(value)
    return cache


def _prefetch_profile_data(markups, list_param):
    """Batch-load raw profile arrays used directly while assembling rows."""
    profile_ids = {markup.profile_id for markup in markups}
    cache = {}
    if not profile_ids:
        return cache

    needs_signal = any(
        param in ('CRL', 'CRL_NF')
        or param.startswith(('Signal', 'distr', 'sep', 'mfcc'))
        for param in list_param
    )
    if needs_signal:
        for profile_id, signal in session.query(Profile.id, Profile.signal).filter(Profile.id.in_(profile_ids)).all():
            if signal is not None:
                cache[(profile_id, 'signal')] = json.loads(signal)

    direct_columns = [param for param in ('X', 'Y') if param in list_param]
    if direct_columns:
        columns = [Profile.id]
        columns.extend(Profile.x_pulc if param == 'X' else Profile.y_pulc for param in direct_columns)
        for row in session.query(*columns).filter(Profile.id.in_(profile_ids)).all():
            for param, value in zip(direct_columns, row[1:]):
                if value is not None:
                    cache[(row[0], param)] = json.loads(value)

    for param in (item for item in list_param if item.startswith('model_')):
        model_id = int(param.split('_id')[-1])
        rows = session.query(
            ProfileModelPrediction.profile_id,
            ProfileModelPrediction.prediction,
        ).filter(
            ProfileModelPrediction.profile_id.in_(profile_ids),
            ProfileModelPrediction.model_id == model_id,
        ).all()
        for profile_id, prediction in rows:
            if prediction is not None:
                cache[(profile_id, param)] = json.loads(prediction)
    return cache


def _prefetch_formation_layers(markups):
    """Load formation boundary arrays without lazy-loading ORM relationships."""
    formation_ids = {markup.formation_id for markup in markups}
    formation_layers = session.query(Formation.id, Formation.up, Formation.down).filter(
        Formation.id.in_(formation_ids)
    ).all()
    layer_ids = {layer_id for row in formation_layers for layer_id in row[1:] if layer_id is not None}
    layer_lines = {
        layer_id: json.loads(layer_line)
        for layer_id, layer_line in session.query(Layers.id, Layers.layer_line).filter(Layers.id.in_(layer_ids)).all()
    }
    return {
        formation_id: (layer_lines[up_id], layer_lines[down_id])
        for formation_id, up_id, down_id in formation_layers
    }


def build_table_train(db=False, analisis='lda'):
    # Получение списка параметров
    if analisis == 'mlp':
        list_param = get_list_param_mlp()
        analisis_id = get_MLP_id()
        analis = session.query(AnalysisMLP).filter_by(id=get_MLP_id()).first()
    elif analisis == 'regmod':
        list_param = get_list_param_regmod()
        analisis_id = get_regmod_id()
        analis = session.query(AnalysisReg).filter_by(id=get_regmod_id()).first()
    # Если в базе есть сохранённая обучающая выборка, забираем ее оттуда
    if db or analis.up_data:
        if analisis == 'mlp':
            data = session.query(AnalysisMLP.data).filter_by(id=get_MLP_id()).first()
        elif analisis == 'regmod':
            data = session.query(AnalysisReg.data).filter_by(id=get_regmod_id()).first()

        if data[0]:
            try:
                data_train = pd.read_parquet(data[0])
                return data_train, list_param
            except OSError:
                try:
                    return pd.DataFrame(json.loads(data[0])), list_param
                except JSONDecodeError:
                    pass
            except ImportError:
                 return None, None

    data_train, _ = build_table_train_no_db(analisis, analisis_id, list_param)
    return data_train, list_param


def build_table_train_no_db(analisis: str, analisis_id: int, list_param: list) -> (pd.DataFrame, list):

    # Если в базе нет сохранённой обучающей выборки. Создание таблицы
    if analisis == 'regmod':
        data_train = pd.DataFrame(columns=['prof_well_index', 'target_value'])
    else:
        data_train = pd.DataFrame(columns=['prof_well_index', 'mark'])
    # Не расширяем DataFrame по одной строке: каждый pd.concat копирует уже
    # собранную таблицу, из-за чего время сборки квадратично растёт с числом
    # измерений. Записи преобразуются в DataFrame ограниченными порциями: так
    # ускорение сохраняется без неограниченного роста списка Python-словарей.
    data_train_rows = []
    data_train_chunks = []
    except_param = False
    # Получаем размеченные участки
    if analisis == 'mlp':
        markups = session.query(MarkupMLP).filter_by(analysis_id=analisis_id).all()
        except_param = session.query(ExceptionMLP).filter_by(analysis_id=analisis_id).first()
    elif analisis == 'regmod':
        markups = session.query(MarkupReg).filter_by(analysis_id=analisis_id).all()
        except_param = session.query(ExceptionReg).filter_by(analysis_id=analisis_id).first()

    stored_feature_cache = _prefetch_stored_features(markups, list_param)
    remaining_formations = Counter(markup.formation_id for markup in markups)
    remaining_profiles = Counter(markup.profile_id for markup in markups)
    profile_data_cache = _prefetch_profile_data(markups, list_param)
    formation_layer_cache = _prefetch_formation_layers(markups)

    list_except_signal, list_except_crl = [], []
    if except_param:
        if except_param.except_signal:
            list_except_signal = parse_range_exception(except_param.except_signal)
        if except_param.except_crl:
            list_except_crl = parse_range_exception(except_param.except_crl)

    ui.progressBar.setMaximum(len(markups))
    skipped_measurements = []

    for nm, markup in enumerate(tqdm(markups)):
        # Держим только рабочие массивы текущей разметки. Раньше они добавлялись
        # в locals() под динамическими именами и оставались в памяти до конца
        # всей сборки, включая особенно объёмные signal/CRL массивы.
        runtime_values = {}
        # Получение списка фиктивных меток и границ слоев из разметки
        list_fake = json.loads(markup.list_fake) if markup.list_fake else []
        list_up, list_down = formation_layer_cache[markup.formation_id]
        cached_signal = profile_data_cache.get((markup.profile_id, 'signal'))
        if cached_signal is not None:
            runtime_values[str(markup.profile_id) + '_signal'] = cached_signal
        for param in (item for item in list_param if item.startswith('model_')):
            cached_prediction = profile_data_cache.get((markup.profile_id, param))
            if cached_prediction is not None:
                runtime_values[str(markup.profile_id) + '_' + param] = cached_prediction

        # Загрузка сигналов из профилей, необходимых для параметров 'distr', 'sep' и 'mfcc'
        for param in list_param:
            entity_id = markup.profile_id if param.startswith('prof') else markup.formation_id
            cache_key = (param, entity_id)
            if cache_key in stored_feature_cache:
                runtime_values[f'list_{param}'] = stored_feature_cache[cache_key]
                continue
            # Если параметр является расчётным
            if param.startswith('Signal') or param.startswith('distr') or param.startswith('sep') or param.startswith('mfcc'):
                # Проверка, есть ли уже загруженный сигнал в локальных переменных
                if not str(markup.profile_id) + '_signal' in runtime_values:
                    # Загрузка сигнала из профиля
                    runtime_values[str(markup.profile_id) + '_signal'] = profile_data_cache[(markup.profile_id, 'signal')]
                if param.split('_')[1] == 'SigCRL':
                    if not str(markup.profile_id) + '_CRL' in runtime_values:
                        runtime_values[str(markup.profile_id) + '_CRL'] = calc_CRL_filter(
                            runtime_values[str(markup.profile_id) + '_signal']
                        )
            elif param.startswith('model_'):
                if not str(markup.profile_id) + '_' + param in runtime_values:
                    model_id = int(param.split('_id')[-1])
                    predict = session.query(ProfileModelPrediction).filter_by(profile_id=markup.profile_id,
                                                                              model_id=model_id).first()
                    if predict:
                        runtime_values[str(markup.profile_id) + '_' + param] = json.loads(predict.prediction)
                    else:
                        calc_profile_model_predict(param, markup.formation)
                        runtime_values[str(markup.profile_id) + '_' + param] = json.loads(
                            session.query(ProfileModelPrediction.prediction).filter_by(profile_id=markup.profile_id,
                                                                              model_id=model_id).first()[0])
            elif param == 'CRL':
                if not str(markup.profile_id) + '_CRL' in runtime_values:
                    runtime_values[str(markup.profile_id) + '_CRL'] = calc_CRL_filter(
                        runtime_values[str(markup.profile_id) + '_signal']
                    )
            elif param == 'CRL_NF':
                if not str(markup.profile_id) + '_CRL_NF' in runtime_values:
                    runtime_values[str(markup.profile_id) + '_CRL_NF'] = calc_CRL(
                        runtime_values[str(markup.profile_id) + '_signal']
                    )
            elif param == 'X':
                runtime_values['list_X'] = profile_data_cache[(markup.profile_id, 'X')]
            elif param == 'Y':
                runtime_values['list_Y'] = profile_data_cache[(markup.profile_id, 'Y')]
            elif param.startswith('prof'):
                if param[5:] in list_wavelet_features:
                    calc_wavelet_features_profile(markup.profile_id)
                    runtime_values[f'list_{param}'] = json.loads(session.query(literal_column(f'wavelet_feature_profile.{param[5:]}')).filter(
                        WaveletFeatureProfile.profile_id == markup.profile_id).first()[0])
                elif param[5:] in list_fractal_features:
                    calc_fractal_features_profile(markup.profile_id)
                    runtime_values[f'list_{param}'] = json.loads(session.query(literal_column(f'fractal_feature_profile.{param[5:]}')).filter(
                        FractalFeatureProfile.profile_id == markup.profile_id).first()[0])
                elif param[5:] in list_entropy_features:
                    calc_entropy_features_profile(markup.profile_id)
                    runtime_values[f'list_{param}'] = json.loads(session.query(literal_column(f'entropy_feature_profile.{param[5:]}')).filter(
                        EntropyFeatureProfile.profile_id == markup.profile_id).first()[0])
                elif param[5:] in list_nonlinear_features:
                    calc_nonlinear_features_profile(markup.profile_id)
                    runtime_values[f'list_{param}'] = json.loads(
                        session.query(literal_column(f'nonlinear_feature_profile.{param[5:]}')).filter(
                            NonlinearFeatureProfile.profile_id == markup.profile_id).first()[0])
                elif param[5:] in list_morphology_feature:
                    calc_morphology_features_profile(markup.profile_id)
                    runtime_values[f'list_{param}'] = json.loads(
                        session.query(literal_column(f'morphology_feature_profile.{param[5:]}')).filter(
                            MorphologyFeatureProfile.profile_id == markup.profile_id).first()[0])
                elif param[5:] in list_frequency_feature:
                    calc_frequency_features_profile(markup.profile_id)
                    runtime_values[f'list_{param}'] = json.loads(
                        session.query(literal_column(f'frequency_feature_profile.{param[5:]}')).filter(
                            FrequencyFeatureProfile.profile_id == markup.profile_id).first()[0])
                elif param[5:] in list_envelope_feature:
                    calc_envelope_feature_profile(markup.profile_id)
                    runtime_values[f'list_{param}'] = json.loads(
                        session.query(literal_column(f'envelope_feature_profile.{param[5:]}')).filter(
                            EnvelopeFeatureProfile.profile_id == markup.profile_id).first()[0])
                elif param[5:] in list_autocorr_feature:
                    calc_autocorr_feature_profile(markup.profile_id)
                    runtime_values[f'list_{param}'] = json.loads(
                        session.query(literal_column(f'autocorr_feature_profile.{param[5:]}')).filter(
                            AutocorrFeatureProfile.profile_id == markup.profile_id).first()[0])
                elif param[5:] in list_emd_feature:
                    calc_emd_feature_profile(markup.profile_id)
                    runtime_values[f'list_{param}'] = json.loads(session.query(literal_column(f'emd_feature_profile.{param[5:]}')).filter(
                        EMDFeatureProfile.profile_id == markup.profile_id).first()[0])
                elif param[5:] in list_hht_feature:
                    calc_hht_features_profile(markup.profile_id)
                    runtime_values[f'list_{param}'] = json.loads(session.query(literal_column(f'hht_feature_profile.{param[5:]}')).filter(
                        HHTFeatureProfile.profile_id == markup.profile_id).first()[0])
                else:
                    pass
            # Если параметр сохранён в базе
            else:
                if param in list_wavelet_features:
                    calc_wavelet_features(markup.formation_id)
                    runtime_values[f'list_{param}'] = json.loads(session.query(literal_column(f'wavelet_feature.{param}')).filter(
                        WaveletFeature.formation_id == markup.formation_id).first()[0])
                elif param in list_fractal_features:
                    calc_fractal_features(markup.formation_id)
                    runtime_values[f'list_{param}'] = json.loads(session.query(literal_column(f'fractal_feature.{param}')).filter(
                        FractalFeature.formation_id == markup.formation_id).first()[0])
                elif param in list_entropy_features:
                    calc_entropy_features(markup.formation_id)
                    runtime_values[f'list_{param}'] = json.loads(session.query(literal_column(f'entropy_feature.{param}')).filter(
                        EntropyFeature.formation_id == markup.formation_id).first()[0])
                elif param in list_nonlinear_features:
                    calc_nonlinear_features(markup.formation_id)
                    runtime_values[f'list_{param}'] = json.loads(session.query(literal_column(f'nonlinear_feature.{param}')).filter(
                        NonlinearFeature.formation_id == markup.formation_id).first()[0])
                elif param in list_morphology_feature:
                    calc_morphology_features(markup.formation_id)
                    runtime_values[f'list_{param}'] = json.loads(session.query(literal_column(f'morphology_feature.{param}')).filter(
                        MorphologyFeature.formation_id == markup.formation_id).first()[0])
                elif param in list_frequency_feature:
                    calc_frequency_features(markup.formation_id)
                    runtime_values[f'list_{param}'] = json.loads(session.query(literal_column(f'frequency_feature.{param}')).filter(
                        FrequencyFeature.formation_id == markup.formation_id).first()[0])
                elif param in list_envelope_feature:
                    calc_envelope_feature(markup.formation_id)
                    runtime_values[f'list_{param}'] = json.loads(session.query(literal_column(f'envelope_feature.{param}')).filter(
                        EnvelopeFeature.formation_id == markup.formation_id).first()[0])
                elif param in list_autocorr_feature:
                    calc_autocorr_feature(markup.formation_id)
                    runtime_values[f'list_{param}'] = json.loads(session.query(literal_column(f'autocorr_feature.{param}')).filter(
                        AutocorrFeature.formation_id == markup.formation_id).first()[0])
                elif param in list_emd_feature:
                    calc_emd_feature(markup.formation_id)
                    runtime_values[f'list_{param}'] = json.loads(session.query(literal_column(f'emd_feature.{param}')).filter(
                        EMDFeature.formation_id == markup.formation_id).first()[0])
                elif param in list_hht_feature:
                    calc_hht_features(markup.formation_id)
                    runtime_values[f'list_{param}'] = json.loads(session.query(literal_column(f'hht_feature.{param}')).filter(
                        HHTFeature.formation_id == markup.formation_id).first()[0])
                else:
                    # Загрузка значений параметра из формации
                    runtime_values[f'list_{param}'] = json.loads(session.query(literal_column(f'Formation.{param}')).filter(
                        Formation.id == markup.formation_id).first()[0])


        # Обработка каждого измерения в разметке
        for measure in json.loads(markup.list_measure):
            # Пропустить измерение, если оно является фиктивным
            if measure in list_fake:
                continue

            # Отрицательный индекс в Python обращается к данным с конца списка и
            # поэтому особенно опасен: таблица была бы собрана без ошибки, но с
            # данными другого измерения.
            if not isinstance(measure, int) or measure < 0:
                skipped_measurements.append((markup, measure, 'индекс измерения'))
                continue

            dict_value = {}
            dict_value['prof_well_index'] = f'{markup.profile_id}_{markup.well_id}_{measure}'
            if analisis == 'regmod':
                dict_value['target_value'] = markup.target_value
            else:
                dict_value['mark'] = markup.marker.title

            # Обработка каждого параметра в списке параметров. Некоторые старые
            # или не полностью рассчитанные наборы признаков могут быть короче
            # списка измерений разметки. Не прерываем из-за этого длительный
            # сбор всей таблицы: неконсистентное измерение будет пропущено, а
            # пользователю ниже будет показано, где именно не хватило данных.
            failed_param = 'границы пласта'
            try:
                for param in list_param:
                    failed_param = param
                    if param.startswith('Signal'):
                        # Обработка параметра 'Signal'
                        p, atr = param.split('_')[0], param.split('_')[1]
                        sig_measure = calc_atrib_measure(runtime_values[str(markup.profile_id) + '_signal'][measure], atr)
                        for i_sig in range(len(sig_measure)):
                            if i_sig + 1 not in list_except_signal:
                                dict_value[f'{p}_{atr}_{i_sig + 1}'] = sig_measure[i_sig]
                    elif param == 'CRL':
                        sig_measure = runtime_values[str(markup.profile_id) + '_CRL'][measure]
                        for i_sig in range(len(sig_measure)):
                            if i_sig + 1 not in list_except_crl:
                                dict_value[f'{param}_{i_sig + 1}'] = sig_measure[i_sig]
                    elif param == 'CRL_NF':
                        sig_measure = runtime_values[str(markup.profile_id) + '_CRL_NF'][measure]
                        for i_sig in range(len(sig_measure)):
                            if i_sig + 1 not in list_except_crl:
                                dict_value[f'{param}_{i_sig + 1}'] = sig_measure[i_sig]
                    elif param.startswith('distr'):
                        # Обработка параметра 'distr'
                        p, atr, n = param.split('_')[0], param.split('_')[1], int(param.split('_')[2])
                        if atr == 'SigCRL':
                            sig_measure = runtime_values[str(markup.profile_id) + '_CRL'][measure]
                        else:
                            sig_measure = calc_atrib_measure(runtime_values[str(markup.profile_id) + '_signal'][measure], atr)
                        distr = get_distribution(sig_measure[list_up[measure]: list_down[measure]], n)
                        for num in range(n):
                            dict_value[f'{p}_{atr}_{num + 1}'] = distr[num]
                    elif param.startswith('sep'):
                        # Обработка параметра 'sep'
                        p, atr, n = param.split('_')[0], param.split('_')[1], int(param.split('_')[2])
                        if atr == 'SigCRL':
                            sig_measure = runtime_values[str(markup.profile_id) + '_CRL'][measure]
                        else:
                            sig_measure = calc_atrib_measure(runtime_values[str(markup.profile_id) + '_signal'][measure], atr)
                        sep = get_interpolate_list(sig_measure[list_up[measure]: list_down[measure]], n)
                        for num in range(n):
                            dict_value[f'{p}_{atr}_{num + 1}'] = sep[num]
                    elif param.startswith('mfcc'):
                        # Обработка параметра 'mfcc'
                        p, atr, n = param.split('_')[0], param.split('_')[1], int(param.split('_')[2])
                        if atr == 'SigCRL':
                            sig_measure = runtime_values[str(markup.profile_id) + '_CRL'][measure]
                        else:
                            sig_measure = calc_atrib_measure(runtime_values[str(markup.profile_id) + '_signal'][measure], atr)
                        mfcc = get_mfcc(sig_measure[list_up[measure]: list_down[measure]], n)
                        for num in range(n):
                            dict_value[f'{p}_{atr}_{num + 1}'] = mfcc[num]
                    elif param.startswith('model_'):
                        dict_value[param] = runtime_values[str(markup.profile_id) + '_' + param][measure]
                    else:
                        # Загрузка значения параметра из списка значений
                        dict_value[param] = runtime_values[f'list_{param}'][measure]

            except IndexError:
                skipped_measurements.append((markup, measure, failed_param))
                continue

            data_train_rows.append(dict_value)
            if len(data_train_rows) >= TRAIN_TABLE_ROW_CHUNK_SIZE:
                data_train_chunks.append(pd.DataFrame.from_records(data_train_rows))
                data_train_rows.clear()

        # Как только профиль/пласт больше не понадобится следующим разметкам,
        # освобождаем его предварительно загруженные массивы. Это не даёт кэшу
        # суммироваться с уже собранной таблицей до самого конца операции.
        remaining_formations[markup.formation_id] -= 1
        if remaining_formations[markup.formation_id] == 0:
            formation_layer_cache.pop(markup.formation_id, None)
            for key in [key for key in stored_feature_cache if key[1] == markup.formation_id and not key[0].startswith('prof')]:
                stored_feature_cache.pop(key, None)
        remaining_profiles[markup.profile_id] -= 1
        if remaining_profiles[markup.profile_id] == 0:
            for key in [key for key in profile_data_cache if key[0] == markup.profile_id]:
                profile_data_cache.pop(key, None)
            for key in [key for key in stored_feature_cache if key[1] == markup.profile_id and key[0].startswith('prof')]:
                stored_feature_cache.pop(key, None)

        ui.progressBar.setValue(nm + 1)

    if skipped_measurements:
        examples = []
        for markup, measure, param in skipped_measurements[:10]:
            examples.append(
                f'профиль {markup.profile_id}, скважина {markup.well_id}, '
                f'измерение {measure}, параметр {param}'
            )
        extra = len(skipped_measurements) - len(examples)
        details = '\n'.join(examples)
        if extra:
            details += f'\n... и ещё {extra}'
        message = (
            f'Пропущено измерений: {len(skipped_measurements)}. Для них отсутствуют '
            f'рассчитанные данные или индекс выходит за границы массива.\n\n{details}'
        )
        set_info(message.replace('\n', ' '), 'orange')
        QMessageBox.warning(MainWindow, 'Неполные данные обучающей выборки', message)

    if data_train_rows:
        data_train_chunks.append(pd.DataFrame.from_records(data_train_rows))
        data_train_rows.clear()
    if data_train_chunks:
        data_train = pd.concat(data_train_chunks, ignore_index=True, copy=False)
        data_train_chunks.clear()
    # data_train_to_db = json.dumps(data_train.to_dict())
    p_sep = os.path.sep
    if analisis == 'mlp':
        analysis_mlp = session.query(AnalysisMLP).filter_by(id=analisis_id).first()
        name = f'{analysis_mlp.title}_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
        filepath = f'data_tables{p_sep}cls{p_sep}{name}.parquet'
        try:
            data_train.to_parquet(filepath)
        except OSError:
            pass

        session.query(AnalysisMLP).filter_by(id=analisis_id).update({'data': str(filepath), 'up_data': True}, synchronize_session='fetch')
    elif analisis == 'regmod':
        analysis_reg = session.query(AnalysisReg).filter_by(id=analisis_id).first()
        name = f'{analysis_reg.title}_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
        filepath = f'data_tables{p_sep}reg{p_sep}{name}.parquet'
        try:
            data_train.to_parquet(filepath)
        except OSError:
            pass

        session.query(AnalysisReg).filter_by(id=analisis_id).update({'data': str(filepath), 'up_data': True}, synchronize_session='fetch')
    session.commit()
    return data_train, list_param


def build_table_test(analisis='mlp', model=False, curr_form=False):
    list_except_signal, list_except_crl = [], []
    if analisis == 'mlp':
        if not model:
            model = session.query(TrainedModelClass).filter_by(id=ui.listWidget_trained_model_class.currentItem().data(Qt.UserRole)).first()
        list_param, analysis_title, except_signal, except_crl = (json.loads(model.list_params), model.title,
                                                                 model.except_signal, model.except_crl)
        list_except_signal, list_except_crl = parse_range_exception(except_signal), parse_range_exception(except_crl)
        list_except_signal = [] if list_except_signal == -1 else list_except_signal
        list_except_crl = [] if list_except_crl == -1 else list_except_crl
    elif analisis == 'regmod':
        if not model:
            model = session.query(TrainedModelReg).filter_by(id=ui.listWidget_trained_model_reg.currentItem().data(Qt.UserRole)).first()
        list_param, analysis_title, except_signal, except_crl = (json.loads(model.list_params), model.title,
                                                                 model.except_signal, model.except_crl)
        list_except_signal, list_except_crl = parse_range_exception(except_signal), parse_range_exception(except_crl)
        list_except_signal = [] if list_except_signal == -1 else list_except_signal
        list_except_crl = [] if list_except_crl == -1 else list_except_crl
    elif analisis == 'cluster':

        cluster_analysis = session.query(AnalysisCluster).filter_by(id=ui.comboBox_clust_set.currentText().split(' id')[-1]).first()
        list_param = json.loads(cluster_analysis.parameter)
        analysis_title = f'Cluster {cluster_analysis.title}'
    test_data = pd.DataFrame(columns=['prof_index', 'x_pulc', 'y_pulc'])
    if not curr_form:
        curr_form = session.query(Formation).filter(Formation.id == get_formation_id()).first()
    try:
        list_up = json.loads(curr_form.layer_up.layer_line)
        list_down = json.loads(curr_form.layer_down.layer_line)
    except AttributeError:
        set_info('Не выбран пласт', 'red')
        QMessageBox.critical(MainWindow, 'Ошибка', 'Не выбран пласт')
        return
    x_pulc = json.loads(curr_form.profile.x_pulc)
    y_pulc = json.loads(curr_form.profile.y_pulc)
    for param in list_param:
        if param.startswith('distr') or param.startswith('sep') or param.startswith('mfcc') or param.startswith('Signal'):
            if not str(curr_form.profile.id) + '_signal' in locals():
                locals()[str(curr_form.profile.id) + '_signal'] = json.loads(
                    session.query(Profile.signal).filter(Profile.id == curr_form.profile_id).first()[0])
            if param.split('_')[1] == 'SigCRL':
                if not str(curr_form.profile.id) + '_CRL' in locals():
                    locals()[str(curr_form.profile.id) + '_CRL'] = calc_CRL_filter(json.loads(
                        session.query(Profile.signal).filter(Profile.id == curr_form.profile_id).first()[0]))

        elif param.startswith('model_'):
            if not str(curr_form.profile.id) + '_' + param in locals():
                model_id = int(param.split('_id')[-1])
                predict = session.query(ProfileModelPrediction).filter_by(profile_id=curr_form.profile_id,
                                                                          model_id=model_id).first()
                if predict:
                    locals()[str(curr_form.profile.id) + '_' + param] = json.loads(predict.prediction)
                else:
                    calc_profile_model_predict(param, curr_form)
                    locals()[str(curr_form.profile.id) + '_' + param] = json.loads(
                        session.query(ProfileModelPrediction.prediction).filter_by(profile_id=curr_form.profile_id,
                                                                                   model_id=model_id).first()[0])

        elif param.startswith('CRL') and not param.startswith('CRL_NF') and param not in list_param_geovel:
            if not str(curr_form.profile.id) + '_CRL' in locals():
                locals()[str(curr_form.profile.id) + '_CRL'] = calc_CRL_filter(json.loads(
                    session.query(Profile.signal).filter(Profile.id == curr_form.profile_id).first()[0]))
        elif param.startswith('CRL_NF'):
            if not str(curr_form.profile.id) + '_CRL_NF' in locals():
                locals()[str(curr_form.profile.id) + '_CRL_NF'] = calc_CRL_filter(json.loads(
                    session.query(Profile.signal).filter(Profile.id == curr_form.profile_id).first()[0]))
        elif param == 'X':
            locals()['list_X'] = json.loads(session.query(Profile.x_pulc).filter(Profile.id == curr_form.profile_id).first()[0])
        elif param == 'Y':
            locals()['list_Y'] = json.loads(session.query(Profile.y_pulc).filter(Profile.id == curr_form.profile_id).first()[0])
        elif param.startswith('prof'):
            if param[5:] in list_wavelet_features:
                calc_wavelet_features_profile(curr_form.profile_id)
                locals()[f'list_{param}'] = json.loads(session.query(literal_column(f'wavelet_feature_profile.{param[5:]}')).filter(
                    WaveletFeatureProfile.profile_id == curr_form.profile_id).first()[0])
            elif param[5:] in list_fractal_features:
                calc_fractal_features_profile(curr_form.profile_id)
                locals()[f'list_{param}'] = json.loads(session.query(literal_column(f'fractal_feature_profile.{param[5:]}')).filter(
                    FractalFeatureProfile.profile_id == curr_form.profile_id).first()[0])
            elif param[5:] in list_entropy_features:
                calc_entropy_features_profile(curr_form.profile_id)
                locals()[f'list_{param}'] = json.loads(session.query(literal_column(f'entropy_feature_profile.{param[5:]}')).filter(
                    EntropyFeatureProfile.profile_id == curr_form.profile_id).first()[0])
            elif param[5:] in list_nonlinear_features:
                calc_nonlinear_features_profile(curr_form.profile_id)
                locals()[f'list_{param}'] = json.loads(
                    session.query(literal_column(f'nonlinear_feature_profile.{param[5:]}')).filter(
                        NonlinearFeatureProfile.profile_id == curr_form.profile_id).first()[0])
            elif param[5:] in list_morphology_feature:
                calc_morphology_features_profile(curr_form.profile_id)
                locals()[f'list_{param}'] = json.loads(
                    session.query(literal_column(f'morphology_feature_profile.{param[5:]}')).filter(
                        MorphologyFeatureProfile.profile_id == curr_form.profile_id).first()[0])
            elif param[5:] in list_frequency_feature:
                calc_frequency_features_profile(curr_form.profile_id)
                locals()[f'list_{param}'] = json.loads(
                    session.query(literal_column(f'frequency_feature_profile.{param[5:]}')).filter(
                        FrequencyFeatureProfile.profile_id == curr_form.profile_id).first()[0])
            elif param[5:] in list_envelope_feature:
                calc_envelope_feature_profile(curr_form.profile_id)
                locals()[f'list_{param}'] = json.loads(
                    session.query(literal_column(f'envelope_feature_profile.{param[5:]}')).filter(
                        EnvelopeFeatureProfile.profile_id == curr_form.profile_id).first()[0])
            elif param[5:] in list_autocorr_feature:
                calc_autocorr_feature_profile(curr_form.profile_id)
                locals()[f'list_{param}'] = json.loads(
                    session.query(literal_column(f'autocorr_feature_profile.{param[5:]}')).filter(
                        AutocorrFeatureProfile.profile_id == curr_form.profile_id).first()[0])
            elif param[5:] in list_emd_feature:
                calc_emd_feature_profile(curr_form.profile_id)
                locals()[f'list_{param}'] = json.loads(session.query(literal_column(f'emd_feature_profile.{param[5:]}')).filter(
                    EMDFeatureProfile.profile_id == curr_form.profile_id).first()[0])
            elif param[5:] in list_hht_feature:
                calc_hht_features_profile(curr_form.profile_id)
                locals()[f'list_{param}'] = json.loads(session.query(literal_column(f'hht_feature_profile.{param[5:]}')).filter(
                    HHTFeatureProfile.profile_id == curr_form.profile_id).first()[0])
            else:
                pass
        else:
            if param in list_wavelet_features:
                calc_wavelet_features(curr_form.id)
                locals()[f'list_{param}'] = json.loads(session.query(literal_column(f'wavelet_feature.{param}')).filter(
                    WaveletFeature.formation_id == curr_form.id).first()[0])
            elif param in list_fractal_features:
                calc_fractal_features(curr_form.id)
                locals()[f'list_{param}'] = json.loads(session.query(literal_column(f'fractal_feature.{param}')).filter(
                    FractalFeature.formation_id == curr_form.id).first()[0])
            elif param in list_entropy_features:
                calc_entropy_features(curr_form.id)
                locals()[f'list_{param}'] = json.loads(session.query(literal_column(f'entropy_feature.{param}')).filter(
                    EntropyFeature.formation_id == curr_form.id).first()[0])
            elif param in list_nonlinear_features:
                calc_nonlinear_features(curr_form.id)
                locals()[f'list_{param}'] = json.loads(
                    session.query(literal_column(f'nonlinear_feature.{param}')).filter(
                        NonlinearFeature.formation_id == curr_form.id).first()[0])
            elif param in list_morphology_feature:
                calc_morphology_features(curr_form.id)
                locals()[f'list_{param}'] = json.loads(
                    session.query(literal_column(f'morphology_feature.{param}')).filter(
                        MorphologyFeature.formation_id == curr_form.id).first()[0])
            elif param in list_frequency_feature:
                calc_frequency_features(curr_form.id)
                locals()[f'list_{param}'] = json.loads(
                    session.query(literal_column(f'frequency_feature.{param}')).filter(
                        FrequencyFeature.formation_id == curr_form.id).first()[0])
            elif param in list_envelope_feature:
                calc_envelope_feature(curr_form.id)
                locals()[f'list_{param}'] = json.loads(
                    session.query(literal_column(f'envelope_feature.{param}')).filter(
                        EnvelopeFeature.formation_id == curr_form.id).first()[0])
            elif param in list_autocorr_feature:
                calc_autocorr_feature(curr_form.id)
                locals()[f'list_{param}'] = json.loads(
                    session.query(literal_column(f'autocorr_feature.{param}')).filter(
                        AutocorrFeature.formation_id == curr_form.id).first()[0])
            elif param in list_emd_feature:
                calc_emd_feature(curr_form.id)
                locals()[f'list_{param}'] = json.loads(session.query(literal_column(f'emd_feature.{param}')).filter(
                    EMDFeature.formation_id == curr_form.id).first()[0])
            elif param in list_hht_feature:
                calc_hht_features(curr_form.id)
                locals()[f'list_{param}'] = json.loads(session.query(literal_column(f'hht_feature.{param}')).filter(
                    HHTFeature.formation_id == curr_form.id).first()[0])
            else:
                locals()[f'list_{param}'] = json.loads(getattr(curr_form, param))

    ui.progressBar.setMaximum(len(list_up))
    set_info(f'Процесс сбора параметров {analysis_title} по профилю {curr_form.profile.title}',
             'blue')
    for i in tqdm(range(len(list_up))):
        dict_value = {}
        for param in list_param:
            if param.startswith('Signal'):
                # Обработка параметра 'Signal'
                p, atr = param.split('_')[0], param.split('_')[1]
                sig_measure = calc_atrib_measure(locals()[str(curr_form.profile.id) + '_signal'][i], atr)
                for i_sig in range(len(sig_measure)):
                    if i_sig + 1 not in list_except_signal:
                        dict_value[f'{p}_{atr}_{i_sig + 1}'] = sig_measure[i_sig]
            elif param.startswith('CRL') and not param.startswith('CRL_NF') and param not in list_param_geovel:
                sig_measure = locals()[str(curr_form.profile.id) + '_CRL'][i]
                for i_sig in range(len(sig_measure)):
                    if i_sig + 1 not in list_except_crl:
                        dict_value[f'{param}_{i_sig + 1}'] = sig_measure[i_sig]
            elif param.startswith('CRL_NF'):
                sig_measure = locals()[str(curr_form.profile.id) + '_CRL_NF'][i]
                for i_sig in range(len(sig_measure)):
                    if i_sig + 1 not in list_except_crl:
                        dict_value[f'{param}_{i_sig + 1}'] = sig_measure[i_sig]
            elif param.startswith('distr'):
                p, atr, n = param.split('_')[0], param.split('_')[1], int(param.split('_')[2])
                if atr == 'SigCRL':
                    sig_measure = locals()[str(curr_form.profile.id) + '_CRL'][i]
                else:
                    sig_measure = calc_atrib_measure(locals()[str(curr_form.profile.id) + '_signal'][i], atr)
                distr = get_distribution(sig_measure[list_up[i]: list_down[i]], n)
                for num in range(n):
                    dict_value[f'{p}_{atr}_{num + 1}'] = distr[num]
            elif param.startswith('sep'):
                p, atr, n = param.split('_')[0], param.split('_')[1], int(param.split('_')[2])
                if atr == 'SigCRL':
                    sig_measure = locals()[str(curr_form.profile.id) + '_CRL'][i]
                else:
                    sig_measure = calc_atrib_measure(locals()[str(curr_form.profile.id) + '_signal'][i], atr)
                sep = get_interpolate_list(sig_measure[list_up[i]: list_down[i]], n)
                for num in range(n):
                    dict_value[f'{p}_{atr}_{num + 1}'] = sep[num]
            elif param.startswith('mfcc'):
                p, atr, n = param.split('_')[0], param.split('_')[1], int(param.split('_')[2])
                if atr == 'SigCRL':
                    sig_measure = locals()[str(curr_form.profile.id) + '_CRL'][i]
                else:
                    sig_measure = calc_atrib_measure(locals()[str(curr_form.profile.id) + '_signal'][i], atr)
                mfcc = get_mfcc(sig_measure[list_up[i]: list_down[i]], n)
                for num in range(n):
                    dict_value[f'{p}_{atr}_{num + 1}'] = mfcc[num]
            elif param.startswith('model_'):
                dict_value[param] = locals()[str(curr_form.profile.id) + '_' + param][i]
            else:
                dict_value[param] = locals()[f'list_{param}'][i]
        dict_value['prof_index'] = f'{curr_form.profile_id}_{i}'
        test_data = pd.concat([test_data, pd.DataFrame([dict_value])], ignore_index=True)
        ui.progressBar.setValue(i + 1)
    test_data['x_pulc'] = x_pulc
    test_data['y_pulc'] = y_pulc
    return test_data, curr_form


def set_marks():
    list_cat = [i.title for i in session.query(MarkerMLP).filter(MarkerMLP.analysis_id == get_MLP_id()).all()]
    labels = {}
    labels[list_cat[0]] = 0
    labels[list_cat[1]] = 1
    if len(list_cat) > 2:
        for index, i in enumerate(list_cat[2:]):
            labels[i] = index
    return labels


def calc_profile_model_predict(param, formation):
    set_info(f'Вычисление предсказания {param} для профиля {formation.profile.title}', 'blue')
    print(f'Вычисление предсказания {param} для профиля {formation.profile.title}')

    type_predict = param.split('_')[1]
    model_id = param.split('_id')[-1]

    if type_predict == 'cls':
        model = session.query(TrainedModelClass).filter_by(id=model_id).first()
    else:
        model = session.query(TrainedModelReg).filter_by(id=model_id).first()

    type_table = 'mlp' if type_predict == 'cls' else 'regmod'

    working_data, curr_form = build_table_test(type_table, model, formation)

    # labels = set_marks()
    # labels_dict = {value: key for key, value in labels.items()}

    with open(model.path_model, 'rb') as f:
        profile_model = pickle.load(f)

    # list_cat = list(profile_model.classes_)

    list_param_num = get_list_param_numerical(json.loads(model.list_params), model)
    if type_predict == 'cls':
        model_mask = session.query(TrainedModelClassMask).filter_by(model_id=model.id).first()
    else:
        model_mask = session.query(TrainedModelRegMask).filter_by(model_id=model.id).first()
    if model_mask:
        list_param_num = sorted(json.loads(session.query(ParameterMask).filter_by(id=model_mask.mask_id).first().mask))

    working_sample = working_data[list_param_num].values.tolist()

    try:
        if type_predict == 'cls':
            probability = profile_model.predict_proba(working_sample)
        else:
            probability = profile_model.predict(working_sample)
    except ValueError:
        working_sample = [[np.nan if np.isinf(x) else x for x in y] for y in working_sample]
        data = imputer.fit_transform(working_sample)
        if type_predict == 'cls':
            probability = profile_model.predict_proba(data)
        else:
            probability = profile_model.predict(data)


        set_info(f'Внимание! Возможно значения одного из параметров отсутствуют в интервале обучающей выборки. '
                 f'{model.title} - {formation.profile.title}', 'red')


    list_result = [round(p[0], 6) for p in probability] if type_predict == 'cls' else probability.tolist()
    new_prof_model_pred = ProfileModelPrediction(
        profile_id=formation.profile_id,
        type_model=type_predict,
        model_id=model.id,
        prediction=json.dumps(list_result)
    )

    session.add(new_prof_model_pred)
    session.commit()
    set_info(f'Результат расчета модели "{model.title}" для профиля {formation.profile.title} сохранен', 'green')
