import json

from calc_profile_features import calc_wavelet_features_profile, calc_fractal_features_profile, \
    calc_entropy_features_profile, calc_nonlinear_features_profile, calc_morphology_features_profile, \
    calc_frequency_features_profile, calc_envelope_feature_profile, calc_autocorr_feature_profile, \
    calc_emd_feature_profile, calc_hht_features_profile
from func import *
from calc_additional_features import (calc_hht_features, calc_emd_feature, calc_autocorr_feature, calc_nonlinear_features,
                                      calc_envelope_feature, calc_frequency_features, calc_morphology_features,
                                      calc_entropy_features, calc_fractal_features,
                                      calc_wavelet_features)


def build_table_train(db=False, analisis='lda'):
    # Получение списка параметров
    if analisis == 'lda':
        list_param = get_list_param_lda()
        analisis_id = get_LDA_id()
        analis = session.query(AnalysisLDA).filter_by(id=get_LDA_id()).first()
    elif analisis == 'mlp':
        list_param = get_list_param_mlp()
        analisis_id = get_MLP_id()
        analis = session.query(AnalysisMLP).filter_by(id=get_MLP_id()).first()
    elif analisis == 'regmod':
        list_param = get_list_param_regmod()
        analisis_id = get_regmod_id()
        analis = session.query(AnalysisReg).filter_by(id=get_regmod_id()).first()
    # Если в базе есть сохранённая обучающая выборка, забираем ее оттуда
    if db or analis.up_data:
        if analisis == 'lda':
            data = session.query(AnalysisLDA.data).filter_by(id=get_LDA_id()).first()
        elif analisis == 'mlp':
            data = session.query(AnalysisMLP.data).filter_by(id=get_MLP_id()).first()
        elif analisis == 'regmod':
            data = session.query(AnalysisReg.data).filter_by(id=get_regmod_id()).first()

        if data[0]:
            data_train = pd.DataFrame(json.loads(data[0]))
            return data_train, list_param

    data_train, _ = build_table_train_no_db(analisis, analisis_id, list_param)
    return data_train, list_param


def build_table_train_no_db(analisis: str, analisis_id: int, list_param: list) -> (pd.DataFrame, list):

    # Если в базе нет сохранённой обучающей выборки. Создание таблицы
    if analisis == 'regmod':
        data_train = pd.DataFrame(columns=['prof_well_index', 'target_value'])
    else:
        data_train = pd.DataFrame(columns=['prof_well_index', 'mark'])
    except_param = False
    # Получаем размеченные участки
    if analisis == 'lda':
        markups = session.query(MarkupLDA).filter_by(analysis_id=analisis_id).all()
    elif analisis == 'mlp':
        markups = session.query(MarkupMLP).filter_by(analysis_id=analisis_id).all()
        except_param = session.query(ExceptionMLP).filter_by(analysis_id=analisis_id).first()
    elif analisis == 'regmod':
        markups = session.query(MarkupReg).filter_by(analysis_id=analisis_id).all()
        except_param = session.query(ExceptionReg).filter_by(analysis_id=analisis_id).first()

    list_except_signal, list_except_crl = [], []
    if except_param:
        if except_param.except_signal:
            list_except_signal = parse_range_exception(except_param.except_signal)
        if except_param.except_crl:
            list_except_crl = parse_range_exception(except_param.except_crl)

    ui.progressBar.setMaximum(len(markups))

    for nm, markup in enumerate(tqdm(markups)):
        # Получение списка фиктивных меток и границ слоев из разметки
        list_fake = json.loads(markup.list_fake) if markup.list_fake else []
        list_up = json.loads(markup.formation.layer_up.layer_line)
        list_down = json.loads(markup.formation.layer_down.layer_line)

        # Загрузка сигналов из профилей, необходимых для параметров 'distr', 'sep' и 'mfcc'
        for param in list_param:
            # Если параметр является расчётным
            if param.startswith('Signal') or param.startswith('distr') or param.startswith('sep') or param.startswith('mfcc'):
                # Проверка, есть ли уже загруженный сигнал в локальных переменных
                if not str(markup.profile.id) + '_signal' in locals():
                    # Загрузка сигнала из профиля
                    locals()[str(markup.profile.id) + '_signal'] = json.loads(
                        session.query(Profile.signal).filter(Profile.id == markup.profile_id).first()[0])
                if param.split('_')[1] == 'SigCRL':
                    if not str(markup.profile.id) + '_CRL' in locals():
                        locals()[str(markup.profile.id) + '_CRL'] = calc_CRL_filter(json.loads(
                            session.query(Profile.signal).filter(Profile.id == markup.profile_id).first()[0]))
            elif param == 'CRL':
                if not str(markup.profile.id) + '_CRL' in locals():
                    locals()[str(markup.profile.id) + '_CRL'] = calc_CRL_filter(json.loads(
                        session.query(Profile.signal).filter(Profile.id == markup.profile_id).first()[0]))
            elif param == 'CRL_NF':
                if not str(markup.profile.id) + '_CRL_NF' in locals():
                    locals()[str(markup.profile.id) + '_CRL_NF'] = calc_CRL(json.loads(
                        session.query(Profile.signal).filter(Profile.id == markup.profile_id).first()[0]))
            elif param == 'X':
                locals()['list_X'] = json.loads(session.query(Profile.x_pulc).filter(
                        Profile.id == markup.profile_id).first()[0])
            elif param == 'Y':
                locals()['list_Y'] = json.loads(session.query(Profile.y_pulc).filter(
                        Profile.id == markup.profile_id).first()[0])
            elif param.startswith('prof'):
                if param[5:] in list_wavelet_features:
                    calc_wavelet_features_profile(markup.profile_id)
                    locals()[f'list_{param}'] = json.loads(session.query(literal_column(f'wavelet_feature_profile.{param[5:]}')).filter(
                        WaveletFeatureProfile.profile_id == markup.profile_id).first()[0])
                elif param[5:] in list_fractal_features:
                    calc_fractal_features_profile(markup.profile_id)
                    locals()[f'list_{param}'] = json.loads(session.query(literal_column(f'fractal_feature_profile.{param[5:]}')).filter(
                        FractalFeatureProfile.profile_id == markup.profile_id).first()[0])
                elif param[5:] in list_entropy_features:
                    calc_entropy_features_profile(markup.profile_id)
                    locals()[f'list_{param}'] = json.loads(session.query(literal_column(f'entropy_feature_profile.{param[5:]}')).filter(
                        EntropyFeatureProfile.profile_id == markup.profile_id).first()[0])
                elif param[5:] in list_nonlinear_features:
                    calc_nonlinear_features_profile(markup.profile_id)
                    locals()[f'list_{param}'] = json.loads(
                        session.query(literal_column(f'nonlinear_feature_profile.{param[5:]}')).filter(
                            NonlinearFeatureProfile.profile_id == markup.profile_id).first()[0])
                elif param[5:] in list_morphology_feature:
                    calc_morphology_features_profile(markup.profile_id)
                    locals()[f'list_{param}'] = json.loads(
                        session.query(literal_column(f'morphology_feature_profile.{param[5:]}')).filter(
                            MorphologyFeatureProfile.profile_id == markup.profile_id).first()[0])
                elif param[5:] in list_frequency_feature:
                    calc_frequency_features_profile(markup.profile_id)
                    locals()[f'list_{param}'] = json.loads(
                        session.query(literal_column(f'frequency_feature_profile.{param[5:]}')).filter(
                            FrequencyFeatureProfile.profile_id == markup.profile_id).first()[0])
                elif param[5:] in list_envelope_feature:
                    calc_envelope_feature_profile(markup.profile_id)
                    locals()[f'list_{param}'] = json.loads(
                        session.query(literal_column(f'envelope_feature_profile.{param[5:]}')).filter(
                            EnvelopeFeatureProfile.profile_id == markup.profile_id).first()[0])
                elif param[5:] in list_autocorr_feature:
                    calc_autocorr_feature_profile(markup.profile_id)
                    locals()[f'list_{param}'] = json.loads(
                        session.query(literal_column(f'autocorr_feature_profile.{param[5:]}')).filter(
                            AutocorrFeatureProfile.profile_id == markup.profile_id).first()[0])
                elif param[5:] in list_emd_feature:
                    calc_emd_feature_profile(markup.profile_id)
                    locals()[f'list_{param}'] = json.loads(session.query(literal_column(f'emd_feature_profile.{param[5:]}')).filter(
                        EMDFeatureProfile.profile_id == markup.profile_id).first()[0])
                elif param[5:] in list_hht_feature:
                    calc_hht_features_profile(markup.profile_id)
                    locals()[f'list_{param}'] = json.loads(session.query(literal_column(f'hht_feature_profile.{param[5:]}')).filter(
                        HHTFeatureProfile.profile_id == markup.profile_id).first()[0])
                else:
                    pass
            # Если параметр сохранён в базе
            else:
                if param in list_wavelet_features:
                    calc_wavelet_features(markup.formation_id)
                    locals()[f'list_{param}'] = json.loads(session.query(literal_column(f'wavelet_feature.{param}')).filter(
                        WaveletFeature.formation_id == markup.formation_id).first()[0])
                elif param in list_fractal_features:
                    calc_fractal_features(markup.formation_id)
                    locals()[f'list_{param}'] = json.loads(session.query(literal_column(f'fractal_feature.{param}')).filter(
                        FractalFeature.formation_id == markup.formation_id).first()[0])
                elif param in list_entropy_features:
                    calc_entropy_features(markup.formation_id)
                    locals()[f'list_{param}'] = json.loads(session.query(literal_column(f'entropy_feature.{param}')).filter(
                        EntropyFeature.formation_id == markup.formation_id).first()[0])
                elif param in list_nonlinear_features:
                    calc_nonlinear_features(markup.formation_id)
                    locals()[f'list_{param}'] = json.loads(session.query(literal_column(f'nonlinear_feature.{param}')).filter(
                        NonlinearFeature.formation_id == markup.formation_id).first()[0])
                elif param in list_morphology_feature:
                    calc_morphology_features(markup.formation_id)
                    locals()[f'list_{param}'] = json.loads(session.query(literal_column(f'morphology_feature.{param}')).filter(
                        MorphologyFeature.formation_id == markup.formation_id).first()[0])
                elif param in list_frequency_feature:
                    calc_frequency_features(markup.formation_id)
                    locals()[f'list_{param}'] = json.loads(session.query(literal_column(f'frequency_feature.{param}')).filter(
                        FrequencyFeature.formation_id == markup.formation_id).first()[0])
                elif param in list_envelope_feature:
                    calc_envelope_feature(markup.formation_id)
                    locals()[f'list_{param}'] = json.loads(session.query(literal_column(f'envelope_feature.{param}')).filter(
                        EnvelopeFeature.formation_id == markup.formation_id).first()[0])
                elif param in list_autocorr_feature:
                    calc_autocorr_feature(markup.formation_id)
                    locals()[f'list_{param}'] = json.loads(session.query(literal_column(f'autocorr_feature.{param}')).filter(
                        AutocorrFeature.formation_id == markup.formation_id).first()[0])
                elif param in list_emd_feature:
                    calc_emd_feature(markup.formation_id)
                    locals()[f'list_{param}'] = json.loads(session.query(literal_column(f'emd_feature.{param}')).filter(
                        EMDFeature.formation_id == markup.formation_id).first()[0])
                elif param in list_hht_feature:
                    calc_hht_features(markup.formation_id)
                    locals()[f'list_{param}'] = json.loads(session.query(literal_column(f'hht_feature.{param}')).filter(
                        HHTFeature.formation_id == markup.formation_id).first()[0])
                else:
                    # Загрузка значений параметра из формации
                    locals()[f'list_{param}'] = json.loads(session.query(literal_column(f'Formation.{param}')).filter(
                        Formation.id == markup.formation_id).first()[0])


        # Обработка каждого измерения в разметке
        for measure in json.loads(markup.list_measure):
            # Пропустить измерение, если оно является фиктивным
            if measure in list_fake:
                continue

            dict_value = {}
            dict_value['prof_well_index'] = f'{markup.profile_id}_{markup.well_id}_{measure}'
            if analisis == 'regmod':
                dict_value['target_value'] = markup.target_value
            else:
                dict_value['mark'] = markup.marker.title

            # Обработка каждого параметра в списке параметров
            for param in list_param:
                if param.startswith('Signal'):
                    # Обработка параметра 'Signal'
                    p, atr = param.split('_')[0], param.split('_')[1]
                    sig_measure = calc_atrib_measure(locals()[str(markup.profile.id) + '_signal'][measure], atr)
                    for i_sig in range(len(sig_measure)):
                        if i_sig + 1 not in list_except_signal:
                            dict_value[f'{p}_{atr}_{i_sig + 1}'] = sig_measure[i_sig]
                elif param == 'CRL':
                    sig_measure = locals()[str(markup.profile.id) + '_CRL'][measure]
                    for i_sig in range(len(sig_measure)):
                        if i_sig + 1 not in list_except_crl:
                            dict_value[f'{param}_{i_sig + 1}'] = sig_measure[i_sig]
                elif param == 'CRL_NF':
                    sig_measure = locals()[str(markup.profile.id) + '_CRL_NF'][measure]
                    for i_sig in range(len(sig_measure)):
                        if i_sig + 1 not in list_except_crl:
                            dict_value[f'{param}_{i_sig + 1}'] = sig_measure[i_sig]
                elif param.startswith('distr'):
                    # Обработка параметра 'distr'
                    p, atr, n = param.split('_')[0], param.split('_')[1], int(param.split('_')[2])
                    if atr == 'SigCRL':
                        sig_measure = locals()[str(markup.profile.id) + '_CRL'][measure]
                    else:
                        sig_measure = calc_atrib_measure(locals()[str(markup.profile.id) + '_signal'][measure], atr)
                    distr = get_distribution(sig_measure[list_up[measure]: list_down[measure]], n)
                    for num in range(n):
                        dict_value[f'{p}_{atr}_{num + 1}'] = distr[num]
                elif param.startswith('sep'):
                    # Обработка параметра 'sep'
                    p, atr, n = param.split('_')[0], param.split('_')[1], int(param.split('_')[2])
                    if atr == 'SigCRL':
                        sig_measure = locals()[str(markup.profile.id) + '_CRL'][measure]
                    else:
                        sig_measure = calc_atrib_measure(locals()[str(markup.profile.id) + '_signal'][measure], atr)
                    sep = get_interpolate_list(sig_measure[list_up[measure]: list_down[measure]], n)
                    for num in range(n):
                        dict_value[f'{p}_{atr}_{num + 1}'] = sep[num]
                elif param.startswith('mfcc'):
                    # Обработка параметра 'mfcc'
                    p, atr, n = param.split('_')[0], param.split('_')[1], int(param.split('_')[2])
                    if atr == 'SigCRL':
                        sig_measure = locals()[str(markup.profile.id) + '_CRL'][measure]
                    else:
                        sig_measure = calc_atrib_measure(locals()[str(markup.profile.id) + '_signal'][measure], atr)
                    mfcc = get_mfcc(sig_measure[list_up[measure]: list_down[measure]], n)
                    for num in range(n):
                        dict_value[f'{p}_{atr}_{num + 1}'] = mfcc[num]
                else:
                    # Загрузка значения параметра из списка значений
                    dict_value[param] = locals()[f'list_{param}'][measure]

            # Добавление данных в обучающую выборку
            data_train = pd.concat([data_train, pd.DataFrame([dict_value])], ignore_index=True)

        ui.progressBar.setValue(nm + 1)
    data_train_to_db = json.dumps(data_train.to_dict())
    if analisis == 'lda':
        session.query(AnalysisLDA).filter_by(id=analisis_id).update({'data': data_train_to_db, 'up_data': True}, synchronize_session='fetch')
    elif analisis == 'mlp':
        session.query(AnalysisMLP).filter_by(id=analisis_id).update({'data': data_train_to_db, 'up_data': True}, synchronize_session='fetch')
    elif analisis == 'regmod':
        session.query(AnalysisReg).filter_by(id=analisis_id).update({'data': data_train_to_db, 'up_data': True}, synchronize_session='fetch')
    session.commit()
    return data_train, list_param


def build_table_test(analisis='lda'):
    list_except_signal, list_except_crl = [], []
    if analisis == 'lda':
        list_param, analisis_title = get_list_param_lda(), ui.comboBox_lda_analysis.currentText()
    elif analisis == 'mlp':
        model = session.query(TrainedModelClass).filter_by(id=ui.listWidget_trained_model_class.currentItem().data(Qt.UserRole)).first()
        list_param, analisis_title, except_signal, except_crl = (json.loads(model.list_params), model.title,
                                                                 model.except_signal, model.except_crl)
        list_except_signal, list_except_crl = parse_range_exception(except_signal), parse_range_exception(except_crl)
        list_except_signal = [] if list_except_signal == -1 else list_except_signal
        list_except_crl = [] if list_except_crl == -1 else list_except_crl
    elif analisis == 'regmod':
        model = session.query(TrainedModelReg).filter_by(id=ui.listWidget_trained_model_reg.currentItem().data(Qt.UserRole)).first()
        list_param, analisis_title = json.loads(model.list_params), model.title
    test_data = pd.DataFrame(columns=['prof_index', 'x_pulc', 'y_pulc'])
    curr_form = session.query(Formation).filter(Formation.id == get_formation_id()).first()
    list_up = json.loads(curr_form.layer_up.layer_line)
    list_down = json.loads(curr_form.layer_down.layer_line)
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
    set_info(f'Процесс сбора параметров {analisis_title} по профилю {curr_form.profile.title}',
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
            else:
                dict_value[param] = locals()[f'list_{param}'][i]
        dict_value['prof_index'] = f'{curr_form.profile_id}_{i}'
        test_data = pd.concat([test_data, pd.DataFrame([dict_value])], ignore_index=True)
        ui.progressBar.setValue(i + 1)
    test_data['x_pulc'] = x_pulc
    test_data['y_pulc'] = y_pulc
    return test_data, curr_form


def get_working_data_lda():
    data_train, list_param = build_table_train(True)
    list_param_lda = data_train.columns.tolist()[2:]
    training_sample = data_train[list_param_lda].values.tolist()
    markup = sum(data_train[['mark']].values.tolist(), [])
    clf = LinearDiscriminantAnalysis()
    try:
        trans_coef = clf.fit(training_sample, markup).transform(training_sample)
    except ValueError:
        set_info(f'Ошибка в расчетах LDA! Возможно значения одного из параметров отсутствуют в интервале обучающей '
                 f'выборки.', 'red')
        return
    data_trans_coef = pd.DataFrame(trans_coef)
    data_trans_coef['mark'] = data_train['mark'].values.tolist()
    data_trans_coef['shape'] = ['train']*len(data_trans_coef)
    list_cat = list(clf.classes_)
    working_data, curr_form = build_table_test()
    profile_title = session.query(Profile.title).filter_by(id=working_data['prof_index'][0].split('_')[0]).first()[0][0]
    set_info(f'Процесс расчёта LDA. {ui.comboBox_lda_analysis.currentText()} по профилю {profile_title}', 'blue')
    try:
        new_trans_coef = clf.transform(working_data.iloc[:, 3:])
        new_mark = clf.predict(working_data.iloc[:, 3:])
        probability = clf.predict_proba(working_data.iloc[:, 3:])
    except ValueError:
        data = imputer.fit_transform(working_data.iloc[:, 3:])
        new_trans_coef = clf.transform(data)
        new_mark = clf.predict(data)
        probability = clf.predict_proba(data)
        for i in working_data.index:
            p_nan = [working_data.columns[ic + 3] for ic, v in enumerate(working_data.iloc[i, 3:].tolist()) if
                     np.isnan(v)]
            if len(p_nan) > 0:
                set_info(f'Внимание для измерения "{i}" отсутствуют параметры "{", ".join(p_nan)}", поэтому расчёт для'
                         f' этого измерения может быть не корректен', 'red')
    working_data = pd.concat([working_data, pd.DataFrame(probability, columns=list_cat)], axis=1)
    working_data['mark'] = new_mark
    test_data_trans_coef = pd.DataFrame(new_trans_coef)
    test_data_trans_coef['mark'] = new_mark
    test_data_trans_coef['shape'] = ['test'] * len(new_trans_coef)
    data_trans_coef = pd.concat([data_trans_coef, test_data_trans_coef], ignore_index=True)
    return working_data, data_trans_coef, curr_form
