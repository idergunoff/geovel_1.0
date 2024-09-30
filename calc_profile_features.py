from calc_additional_features import *

def calc_profile_features():
    if not ui.checkBox_calc_all_features.isChecked():
        return
    p_id = get_profile_id()
    calc_wavelet_features_profile(p_id)
    calc_fractal_features_profile(p_id)
    calc_entropy_features_profile(p_id)
    calc_nonlinear_features_profile(p_id)
    calc_morphology_features_profile(p_id)
    calc_frequency_features_profile(p_id)
    calc_envelope_feature_profile(p_id)
    calc_autocorr_feature_profile(p_id)
    calc_emd_feature_profile(p_id)
    calc_hht_features_profile(p_id)


# Вейвлет преобразования

def calc_wavelet_features_profile(p_id, wavelet='db4', level=5):
    if session.query(WaveletFeatureProfile).filter_by(profile_id=p_id).count() != 0:
        return
    profile = session.query(Profile).filter_by(id=p_id).first()
    set_info(f'Расчет вейвлет параметров для профиля {profile.title}.'
             f' {profile.research.object.title}', 'blue')
    signal = json.loads(profile.signal)
    dict_wvt_ftr_list = {f'{wvt}_l': [] for wvt in list_wavelet_features}
    ui.progressBar.setMaximum(len(signal))
    for meas, s in enumerate(tqdm(signal)):
        ui.progressBar.setValue(meas)
        coeffs = pywt.wavedec(s, wavelet, level=level)
        list_name_lebel = ['A5', 'D5', 'D4', 'D3', 'D2', 'D1']
        low_freq_energy = np.sum(coeffs[0] ** 2)
        high_freq_energy = sum(np.sum(coeff ** 2) for coeff in coeffs[1:])
        dict_wvt_ftr_list['wvt_HfLf_Ratio_l'].append(high_freq_energy / low_freq_energy)
        ratios = [np.sum(coeff ** 2) / low_freq_energy for coeff in coeffs[1:]]
        for i in range(6):
            dict_wvt_ftr_list[f'wvt_energ_{list_name_lebel[i]}_l'].append(np.sum(coeffs[i] ** 2))
            dict_wvt_ftr_list[f'wvt_mean_{list_name_lebel[i]}_l'].append(np.mean(coeffs[i]))
            dict_wvt_ftr_list[f'wvt_std_{list_name_lebel[i]}_l'].append(np.std(coeffs[i]))
            dict_wvt_ftr_list[f'wvt_skew_{list_name_lebel[i]}_l'].append(skew(coeffs[i]))
            dict_wvt_ftr_list[f'wvt_kurt_{list_name_lebel[i]}_l'].append(kurtosis(coeffs[i]))
            dict_wvt_ftr_list[f'wvt_max_{list_name_lebel[i]}_l'].append(np.max(coeffs[i]))
            dict_wvt_ftr_list[f'wvt_min_{list_name_lebel[i]}_l'].append(np.min(coeffs[i]))
            dict_wvt_ftr_list[f'wvt_entr_{list_name_lebel[i]}_l'].append(entropy(np.abs(coeffs[i]) / np.sum(np.abs(coeffs[i]))))
            if i > 0:
                dict_wvt_ftr_list[f'wvt_HfLf_{list_name_lebel[i]}_l'].append(ratios[i-1])
        dict_wvt_ftr_list['wvt_energ_D1D2_l'].append(np.sum(coeffs[5] ** 2) / np.sum(coeffs[4] ** 2))
        dict_wvt_ftr_list['wvt_energ_D2D3_l'].append(np.sum(coeffs[4] ** 2) / np.sum(coeffs[3] ** 2))
        dict_wvt_ftr_list['wvt_energ_D3D4_l'].append(np.sum(coeffs[3] ** 2) / np.sum(coeffs[2] ** 2))
        dict_wvt_ftr_list['wvt_energ_D4D5_l'].append(np.sum(coeffs[2] ** 2) / np.sum(coeffs[1] ** 2))
        dict_wvt_ftr_list['wvt_energ_D5A5_l'].append(np.sum(coeffs[1] ** 2) / np.sum(coeffs[0] ** 2))


    dict_wvt_ftr_json = {key[:-2]: json.dumps(value) for key, value in dict_wvt_ftr_list.items()}

    new_wavelet_formation = WaveletFeatureProfile(profile_id=p_id, **dict_wvt_ftr_json)
    session.add(new_wavelet_formation)
    session.commit()


def calc_fractal_features_profile(p_id):
    if session.query(FractalFeatureProfile).filter_by(profile_id=p_id).count() != 0:
        return
    profile = session.query(Profile).filter_by(id=p_id).first()
    set_info(f'Расчет фрактальных параметров для профиля {profile.title}.'
             f' {profile.research.object.title}', 'blue')
    signal = json.loads(profile.signal)
    dict_frl_ftr_list = {f'{frl}_l': [] for frl in list_fractal_features}
    ui.progressBar.setMaximum(len(signal))
    for meas, s in enumerate(tqdm(signal)):
        ui.progressBar.setValue(meas)
        s = np.array(s)
        dict_frl_ftr_list['fractal_dim_l'].append(box_counting_dim(s))
        dict_frl_ftr_list['hurst_exp_l'].append(hurst_rs(s))
        dict_frl_ftr_list['lacunarity_l'].append(lacunarity(s))
        try:
            alpha, f_alpha = mfdfa(s)
            (width_alpha, max_position_alpha, asymmetry_alpha, max_height_alpha, mean_alpha, mean_f_alpha, std_alpha,
             std_f_alpha) = calc_character_mfdfa(alpha, f_alpha)
            dict_frl_ftr_list['mf_width_l'].append(width_alpha)
            dict_frl_ftr_list['mf_max_position_l'].append(max_position_alpha)
            dict_frl_ftr_list['mf_asymmetry_l'].append(asymmetry_alpha)
            dict_frl_ftr_list['mf_max_height_l'].append(max_height_alpha)
            dict_frl_ftr_list['mf_mean_alpha_l'].append(mean_alpha)
            dict_frl_ftr_list['mf_mean_f_alpha_l'].append(mean_f_alpha)
            dict_frl_ftr_list['mf_std_alpha_l'].append(std_alpha)
            dict_frl_ftr_list['mf_std_f_alpha_l'].append(std_f_alpha)
        except (ValueError, LinAlgError):

            dict_frl_ftr_list['mf_width_l'].append(None)
            dict_frl_ftr_list['mf_max_position_l'].append(None)
            dict_frl_ftr_list['mf_asymmetry_l'].append(None)
            dict_frl_ftr_list['mf_max_height_l'].append(None)
            dict_frl_ftr_list['mf_mean_alpha_l'].append(None)
            dict_frl_ftr_list['mf_mean_f_alpha_l'].append(None)
            dict_frl_ftr_list['mf_std_alpha_l'].append(None)
            dict_frl_ftr_list['mf_std_f_alpha_l'].append(None)

    dict_frl_ftr_json = {key[:-2]: json.dumps(value) for key, value in dict_frl_ftr_list.items()}

    new_fractal_formation = (FractalFeatureProfile(profile_id=p_id, **dict_frl_ftr_json))
    session.add(new_fractal_formation)
    session.commit()


def calc_entropy_features_profile(p_id):
    if session.query(EntropyFeatureProfile).filter_by(profile_id=p_id).count() != 0:
        return
    profile = session.query(Profile).filter_by(id=p_id).first()
    set_info(f'Расчет параметров энтропии для профиля {profile.title}.'
             f' {profile.research.object.title}', 'blue')
    signal = json.loads(profile.signal)
    dict_ent_ftr_list = {f'{ent}_l': [] for ent in list_entropy_features}
    ui.progressBar.setMaximum(len(signal))
    for meas, s in enumerate(tqdm(signal)):
        ui.progressBar.setValue(meas)
        form_signal = np.array(s)
        dict_ent_ftr_list['ent_sh_l'].append(shannon_entropy(form_signal))
        dict_ent_ftr_list['ent_perm_l'].append(permutation_entropy(form_signal))
        try:
            dict_ent_ftr_list['ent_appr_l'].append(approx_entropy(form_signal))
        except ZeroDivisionError:
            dict_ent_ftr_list['ent_appr_l'].append(None)
        for n_se, i_se in enumerate(sample_ent(form_signal)):
            dict_ent_ftr_list[f'ent_sample{n_se + 1}_l'].append(i_se)
        for n_me, i_me in enumerate(multiscale_entropy(form_signal)):
            dict_ent_ftr_list[f'ent_ms{n_me + 1}_l'].append(i_me)
        dict_ent_ftr_list['ent_fft_l'].append(fourier_entropy(form_signal))

    dict_ent_ftr_json = {key[:-2]: json.dumps(value) for key, value in dict_ent_ftr_list.items()}

    new_entropy_formation = (EntropyFeatureProfile(profile_id=p_id, **dict_ent_ftr_json))
    session.add(new_entropy_formation)
    session.commit()


def calc_nonlinear_features_profile(p_id):
    if session.query(NonlinearFeatureProfile).filter_by(profile_id=p_id).count() != 0:
        return
    profile = session.query(Profile).filter_by(id=p_id).first()
    set_info(f'Расчет нелинейныхпараметров для профиля {profile.title}. {profile.research.object.title}', 'blue')
    signal = json.loads(profile.signal)
    dict_nln_ftr_list = {f'{ent}_l': [] for ent in list_nonlinear_features}
    ui.progressBar.setMaximum(len(signal))
    for meas, s in enumerate(tqdm(signal)):
        ui.progressBar.setValue(meas)
        form_signal = np.array(s)
        dict_nln_ftr_list['nln_corr_dim_l'].append(correlation_dimension(form_signal))
        try:
            rec_plot = recurrence_plot_features(form_signal)
            dict_nln_ftr_list['nln_rec_rate_l'].append(rec_plot['recurrence_rate'])
            dict_nln_ftr_list['nln_determin_l'].append(rec_plot['determinism'])
            dict_nln_ftr_list['nln_avg_diag_l'].append(rec_plot['avg_diagonal_line'])
        except ValueError:
            dict_nln_ftr_list['nln_rec_rate_l'].append(None)
            dict_nln_ftr_list['nln_determin_l'].append(None)
            dict_nln_ftr_list['nln_avg_diag_l'].append(None)

        dict_nln_ftr_list['nln_hirsh_l'].append(hirschman_index(form_signal))

    dict_nln_ftr_json = {key[:-2]: json.dumps(value) for key, value in dict_nln_ftr_list.items()}
    session.add(NonlinearFeatureProfile(profile_id=p_id, **dict_nln_ftr_json))
    session.commit()


def calc_morphology_features_profile(p_id):
    if session.query(MorphologyFeatureProfile).filter_by(profile_id=p_id).count() != 0:
        return
    profile = session.query(Profile).filter_by(id=p_id).first()
    set_info(f'Расчет морфологических параметров для профиля {profile.title}.'
             f' {profile.research.object.title}', 'blue')
    signal = json.loads(profile.signal)
    dict_mph_ftr_list = {f'{mph}_l': [] for mph in list_morphology_feature}
    ui.progressBar.setMaximum(len(signal))
    for meas, s in enumerate(tqdm(signal)):
        ui.progressBar.setValue(meas)
        form_signal = np.array(s)
        peaks, _ = find_peaks(form_signal)
        dict_mph_ftr_list['mph_peak_num_l'].append(count_peaks(peaks))
        dict_mph_ftr_list['mph_peak_width_l'].append(main_peak_width(form_signal, peaks))
        dict_mph_ftr_list['mph_peak_amp_ratio_l'].append(peak_amplitude_ratio(form_signal, peaks))
        dict_mph_ftr_list['mph_peak_asymm_l'].append(peak_asymmetry(form_signal, peaks))
        dict_mph_ftr_list['mph_peak_steep_l'].append(slope_steepness(form_signal))
        morph_feature = morphological_features(form_signal)
        dict_mph_ftr_list['mph_erosion_l'].append(morph_feature['erosion_ratio'])
        dict_mph_ftr_list['mph_dilation_l'].append(morph_feature['dilation_ratio'])

    dict_mph_ftr_json = {key[:-2]: json.dumps(value) for key, value in dict_mph_ftr_list.items()}
    session.add(MorphologyFeatureProfile(profile_id=p_id, **dict_mph_ftr_json))
    session.commit()


def calc_frequency_features_profile(p_id):
    if session.query(FrequencyFeatureProfile).filter_by(profile_id=p_id).count() > 0:
        return
    profile = session.query(Profile).filter_by(id=p_id).first()
    set_info(f'Расчет частотных характеристик для профиля {profile.title}.'
             f' {profile.research.object.title}', 'blue')
    signal = json.loads(profile.signal)
    dict_freq_ftr_list = {f'{freq}_l': [] for freq in list_frequency_feature}
    ui.progressBar.setMaximum(len(signal))
    for meas, s in enumerate(tqdm(signal)):
        ui.progressBar.setValue(meas)
        form_signal = np.array(s)
        freqs, ps = power_spectrum(form_signal)
        dict_freq_ftr_list['frq_central_l'].append(central_frequency(freqs, ps))
        try:
            dict_freq_ftr_list['frq_bandwidth_l'].append(bandwidth(freqs, ps))
        except ValueError:
            dict_freq_ftr_list['frq_bandwidth_l'].append(np.nan)

        dict_freq_ftr_list['frq_hl_ratio_l'].append(high_low_frequency_ratio(freqs, ps))
        dict_freq_ftr_list['frq_spec_centroid_l'].append(spectral_centroid(freqs, ps))
        try:
            dict_freq_ftr_list['frq_spec_slope_l'].append(spectral_slope(freqs, ps))
        except ValueError:
            dict_freq_ftr_list['frq_spec_slope_l'].append(np.nan)
        dict_freq_ftr_list['frq_spec_entr_l'].append(spectral_entropy(ps))
        for n_freq, f in enumerate(dominant_frequencies(freqs, ps)):
            dict_freq_ftr_list[f'frq_dom{n_freq+1}_l'].append(f)
        for n_freq, f in enumerate(spectral_moments(freqs, ps)):
            dict_freq_ftr_list[f'frq_mmt{n_freq+1}_l'].append(f)
        dict_freq_ftr_list['frq_attn_coef_l'].append(attenuation_coefficient(form_signal, depth=range(len(form_signal)), freqs_rate=freqs))

    dict_freq_ftr_json = {key[:-2]: json.dumps(value) for key, value in dict_freq_ftr_list.items()}
    session.add(FrequencyFeatureProfile(profile_id=p_id, **dict_freq_ftr_json))
    session.commit()


def calc_envelope_feature_profile(p_id):
    if session.query(EnvelopeFeatureProfile).filter_by(profile_id=p_id).count() > 0:
        return
    profile = session.query(Profile).filter_by(id=p_id).first()
    set_info(f'Расчет характеристик огибающей для профиля {profile.title}.'
             f' {profile.research.object.title}', 'blue')
    signal = json.loads(profile.signal)
    dict_env_ftr_list = {f'{env}_l': [] for env in list_envelope_feature}
    ui.progressBar.setMaximum(len(signal))
    for meas, s in enumerate(tqdm(signal)):
        ui.progressBar.setValue(meas)
        form_signal = np.array(s)
        envelope = calculate_envelope(form_signal)
        dict_env_ftr_list['env_area_l'].append(area_under_envelope(envelope))
        dict_env_ftr_list['env_max_l'].append(envelope_max(envelope))
        dict_env_ftr_list['env_t_max_l'].append(float(time_to_max_envelope(envelope)))
        dict_env_ftr_list['env_mean_l'].append(envelope_mean(envelope))
        dict_env_ftr_list['env_std_l'].append(envelope_std(envelope))
        dict_env_ftr_list['env_skew_l'].append(envelope_skewness(envelope))
        dict_env_ftr_list['env_kurt_l'].append(envelope_kurtosis(envelope))
        dict_env_ftr_list['env_max_mean_ratio_l'].append(max_to_mean_ratio(envelope))
        dict_env_ftr_list['env_peak_width_l'].append(float(main_peak_width_env(envelope)))
        for n_inv, i_env in enumerate(envelope_energy_windows(envelope)):
            dict_env_ftr_list[f'env_energy_win{n_inv+1}_l'].append(i_env)
    dict_env_ftr_json = {key[:-2]: json.dumps(value) for key, value in dict_env_ftr_list.items()}
    session.add(EnvelopeFeatureProfile(profile_id=p_id, **dict_env_ftr_json))
    session.commit()


def calc_autocorr_feature_profile(p_id):
    if session.query(AutocorrFeatureProfile).filter_by(profile_id=p_id).count() > 0:
        return
    profile = session.query(Profile).filter_by(id=p_id).first()
    set_info(f'Расчет характеристик автокорреляции для профиля {profile.title}.'
             f' {profile.research.object.title}', 'blue')
    signal = json.loads(profile.signal)
    dict_acf_list = {f'{acf}_l': [] for acf in list_autocorr_feature}
    ui.progressBar.setMaximum(len(signal))
    for meas, s in enumerate(tqdm(signal)):
        ui.progressBar.setValue(meas)
        form_signal = np.array(s)
        acf = autocorrelation(form_signal)
        try:
            dict_acf_list['acf_first_min_l'].append(float(first_minimum(acf)))
            dict_acf_list['acf_lag_10_l'].append(float(autocorrelation_at_lag(acf)))
            dict_acf_list['acf_decay_l'].append(autocorrelation_decay(acf))
            dict_acf_list['acf_integral_l'].append(acf_integral(acf))
            dict_acf_list['acf_peak_width_l'].append(float(acf_main_peak_width(acf)))
            dict_acf_list['acf_ratio_l'].append(acf_ratio(acf))
        except TypeError:
            dict_acf_list['acf_first_min_l'].append(None)
            dict_acf_list['acf_lag_10_l'].append(None)
            dict_acf_list['acf_decay_l'].append(None)
            dict_acf_list['acf_integral_l'].append(None)
            dict_acf_list['acf_peak_width_l'].append(None)
            dict_acf_list['acf_ratio_l'].append(None)

    dict_acf_json = {key[:-2]: json.dumps(value) for key, value in dict_acf_list.items()}
    session.add(AutocorrFeatureProfile(profile_id=p_id, **dict_acf_json))
    session.commit()


def calc_emd_feature_profile(p_id):
    if session.query(EMDFeatureProfile).filter_by(profile_id=p_id).count() > 0:
        return
    profile = session.query(Profile).filter_by(id=p_id).first()
    set_info(f'Расчет характеристик EMD для профиля {profile.title}. {profile.research.object.title}', 'blue')
    signal = json.loads(profile.signal)
    dict_emd_ftr_list = {f'{emd}_l': [] for emd in list_emd_feature}
    ui.progressBar.setMaximum(len(signal))
    for meas, s in enumerate(tqdm(signal)):
        ui.progressBar.setValue(meas)
        form_signal = np.array(s)
        try:
            imfs = perform_emd(form_signal)
            dict_emd_ftr_list['emd_num_imfs_l'].append(float(count_imfs(imfs)))
            emd_energ = imf_energies(imfs)
            dict_emd_ftr_list['emd_energ_mean_l'].append(emd_energ['mean_energy'])
            dict_emd_ftr_list['emd_energ_med_l'].append(emd_energ['median_energy'])
            dict_emd_ftr_list['emd_energ_max_l'].append(emd_energ['max_energy'])
            dict_emd_ftr_list['emd_energ_min_l'].append(emd_energ['min_energy'])
            dict_emd_ftr_list['emd_energ_std_l'].append(emd_energ['std_energy'])
            emd_rel_energ = relative_imf_energies(imfs)
            dict_emd_ftr_list['emd_rel_energ_mean_l'].append(emd_rel_energ['mean_energy'])
            dict_emd_ftr_list['emd_rel_energ_med_l'].append(emd_rel_energ['median_energy'])
            dict_emd_ftr_list['emd_rel_energ_max_l'].append(emd_rel_energ['max_energy'])
            dict_emd_ftr_list['emd_rel_energ_min_l'].append(emd_rel_energ['min_energy'])
            dict_emd_ftr_list['emd_rel_energ_std_l'].append(emd_rel_energ['std_energy'])
            emd_dom_freqs = emd_dominant_frequencies(imfs)
            dict_emd_ftr_list['emd_dom_freqs_mean_l'].append(float(emd_dom_freqs['mean_freq']))
            dict_emd_ftr_list['emd_dom_freqs_med_l'].append(float(emd_dom_freqs['median_freq']))
            dict_emd_ftr_list['emd_dom_freqs_max_l'].append(float(emd_dom_freqs['max_freq']))
            dict_emd_ftr_list['emd_dom_freqs_min_l'].append(float(emd_dom_freqs['min_freq']))
            dict_emd_ftr_list['emd_dom_freqs_std_l'].append(float(emd_dom_freqs['std_freq']))
            emd_corr = imf_correlations(imfs)
            dict_emd_ftr_list['emd_mean_corr_l'].append(float(emd_corr['mean_corr']))
            dict_emd_ftr_list['emd_median_corr_l'].append(float(emd_corr['median_corr']))
            dict_emd_ftr_list['emd_max_corr_l'].append(float(emd_corr['max_corr']))
            dict_emd_ftr_list['emd_min_corr_l'].append(float(emd_corr['min_corr']))
            dict_emd_ftr_list['emd_std_corr_l'].append(float(emd_corr['std_corr']))
            dict_emd_ftr_list['emd_corr_25_l'].append(float(emd_corr['corr_25']))
            dict_emd_ftr_list['emd_corr_50_l'].append(float(emd_corr['corr_50']))
            dict_emd_ftr_list['emd_corr_75_l'].append(float(emd_corr['corr_75']))
            dict_emd_ftr_list['emd_energ_entropy_l'].append(float(imf_energy_entropy(imfs)))
            dict_emd_ftr_list['emd_oi_l'].append(float(orthogonality_index(imfs)))
            dict_emd_ftr_list['emd_hi_l'].append(float(emd_hilbert_index(imfs)))
        except ValueError:
            for key in dict_emd_ftr_list.keys():
                dict_emd_ftr_list[key].append(None)

    dict_emd_ftr_json = {key[:-2]: json.dumps(value) for key, value in dict_emd_ftr_list.items()}
    session.add(EMDFeatureProfile(profile_id=p_id, **dict_emd_ftr_json))
    session.commit()


def calc_hht_features_profile(p_id):
    if session.query(HHTFeatureProfile).filter_by(profile_id=p_id).count() > 0:
        return
    profile = session.query(Profile).filter(Profile.id == p_id).first()
    set_info(f'Расчет характеристик HHT для профиля {profile.title}.'
             f' {profile.research.object.title}', 'blue')
    signal = json.loads(profile.signal)
    dict_hht_ftr_list = {f'{hht}_l': [] for hht in list_hht_feature}
    ui.progressBar.setMaximum(len(signal))
    for meas, s in enumerate(tqdm(signal)):
        ui.progressBar.setValue(meas)
        form_signal = np.array(s)
        try:
            imfs, hht = perform_hht(form_signal)
        except ValueError:
            for key in dict_hht_ftr_list.keys():
                dict_hht_ftr_list[key].append(None)
            continue

        inst_freq, inst_amp = instantaneous_frequency_amplitude(hht)
        hht_inst_freq = calc_stat_list(inst_freq, 'hht_inst_freq')
        hht_inst_amp = calc_stat_list(inst_amp, 'hht_inst_amp')
        dict_hht_ftr_list['hht_inst_freq_mean_l'].append(hht_inst_freq['hht_inst_freq_mean'])
        dict_hht_ftr_list['hht_inst_freq_med_l'].append(hht_inst_freq['hht_inst_freq_med'])
        dict_hht_ftr_list['hht_inst_freq_max_l'].append(hht_inst_freq['hht_inst_freq_max'])
        dict_hht_ftr_list['hht_inst_freq_min_l'].append(hht_inst_freq['hht_inst_freq_min'])
        dict_hht_ftr_list['hht_inst_freq_std_l'].append(hht_inst_freq['hht_inst_freq_std'])

        dict_hht_ftr_list['hht_inst_amp_mean_l'].append(hht_inst_amp['hht_inst_amp_mean'])
        dict_hht_ftr_list['hht_inst_amp_med_l'].append(hht_inst_amp['hht_inst_amp_med'])
        dict_hht_ftr_list['hht_inst_amp_max_l'].append(hht_inst_amp['hht_inst_amp_max'])
        dict_hht_ftr_list['hht_inst_amp_min_l'].append(hht_inst_amp['hht_inst_amp_min'])
        dict_hht_ftr_list['hht_inst_amp_std_l'].append(hht_inst_amp['hht_inst_amp_std'])

        mean_freq, mean_amp = mean_inst_freq_amp(hht)
        hht_mean_freq = calc_stat_list(mean_freq, 'hht_mean_freq')
        hht_mean_amp = calc_stat_list(mean_amp, 'hht_mean_amp')
        dict_hht_ftr_list['hht_mean_freq_mean_l'].append(hht_mean_freq['hht_mean_freq_mean'])
        dict_hht_ftr_list['hht_mean_freq_med_l'].append(hht_mean_freq['hht_mean_freq_med'])
        dict_hht_ftr_list['hht_mean_freq_max_l'].append(hht_mean_freq['hht_mean_freq_max'])
        dict_hht_ftr_list['hht_mean_freq_min_l'].append(hht_mean_freq['hht_mean_freq_min'])
        dict_hht_ftr_list['hht_mean_freq_std_l'].append(hht_mean_freq['hht_mean_freq_std'])

        dict_hht_ftr_list['hht_mean_amp_mean_l'].append(hht_mean_amp['hht_mean_amp_mean'])
        dict_hht_ftr_list['hht_mean_amp_med_l'].append(hht_mean_amp['hht_mean_amp_med'])
        dict_hht_ftr_list['hht_mean_amp_max_l'].append(hht_mean_amp['hht_mean_amp_max'])
        dict_hht_ftr_list['hht_mean_amp_min_l'].append(hht_mean_amp['hht_mean_amp_min'])
        dict_hht_ftr_list['hht_mean_amp_std_l'].append(hht_mean_amp['hht_mean_amp_std'])

        freq_bins, marg_spec = marginal_spectrum(hht)
        hht_marg_spec = calc_stat_list(marg_spec.tolist(), 'hht_marg_spec')
        dict_hht_ftr_list['hht_marg_spec_mean_l'].append(hht_marg_spec['hht_marg_spec_mean'])
        dict_hht_ftr_list['hht_marg_spec_med_l'].append(hht_marg_spec['hht_marg_spec_med'])
        dict_hht_ftr_list['hht_marg_spec_max_l'].append(hht_marg_spec['hht_marg_spec_max'])
        dict_hht_ftr_list['hht_marg_spec_min_l'].append(hht_marg_spec['hht_marg_spec_min'])
        dict_hht_ftr_list['hht_marg_spec_std_l'].append(hht_marg_spec['hht_marg_spec_std'])

        teager_energ = teager_energies(imfs)
        hht_teager_energ = calc_stat_list(teager_energ, 'hht_teager_energ')
        dict_hht_ftr_list['hht_teager_energ_mean_l'].append(hht_teager_energ['hht_teager_energ_mean'])
        dict_hht_ftr_list['hht_teager_energ_med_l'].append(hht_teager_energ['hht_teager_energ_med'])
        dict_hht_ftr_list['hht_teager_energ_max_l'].append(hht_teager_energ['hht_teager_energ_max'])
        dict_hht_ftr_list['hht_teager_energ_min_l'].append(hht_teager_energ['hht_teager_energ_min'])
        dict_hht_ftr_list['hht_teager_energ_std_l'].append(hht_teager_energ['hht_teager_energ_std'])

        hi = hilbert_index(hht)
        dict_hht_ftr_list['hht_hi_l'].append(hi)

        dos = degree_of_stationarity(hht)
        hht_dos = calc_stat_list(dos, 'hht_dos')
        dict_hht_ftr_list['hht_dos_mean_l'].append(hht_dos['hht_dos_mean'])
        dict_hht_ftr_list['hht_dos_med_l'].append(hht_dos['hht_dos_med'])
        dict_hht_ftr_list['hht_dos_max_l'].append(hht_dos['hht_dos_max'])
        dict_hht_ftr_list['hht_dos_min_l'].append(hht_dos['hht_dos_min'])
        dict_hht_ftr_list['hht_dos_std_l'].append(hht_dos['hht_dos_std'])

        oi = orthogonality_index(imfs)
        dict_hht_ftr_list['hht_oi_l'].append(oi)

        hsd = hilbert_spectral_density(hht, form_signal)
        hht_hsd = calc_stat_list(hsd.tolist(), 'hht_hsd')
        dict_hht_ftr_list['hht_hsd_mean_l'].append(hht_hsd['hht_hsd_mean'])
        dict_hht_ftr_list['hht_hsd_med_l'].append(hht_hsd['hht_hsd_med'])
        dict_hht_ftr_list['hht_hsd_max_l'].append(hht_hsd['hht_hsd_max'])
        dict_hht_ftr_list['hht_hsd_min_l'].append(hht_hsd['hht_hsd_min'])
        dict_hht_ftr_list['hht_hsd_std_l'].append(hht_hsd['hht_hsd_std'])

        ci = complexity_index(hht)
        dict_hht_ftr_list['hht_ci_l'].append(ci)

    dict_hht_ftr_json = {key[:-2]: json.dumps(value) for key, value in dict_hht_ftr_list.items()}
    session.add(HHTFeatureProfile(profile_id=p_id, **dict_hht_ftr_json))
    session.commit()
