import numpy as np

from func import *

def calc_all_params():
    for r in session.query(Profile).filter(Profile.research_id == get_research_id()).all():
        for f in session.query(Formation).filter(Formation.profile_id == r.id).all():
            calc_wavelet_features(f.id)
            calc_fractal_features(f.id)
            calc_entropy_features(f.id)
            calc_nonlinear_features(f.id)
            calc_morphology_features(f.id)
            calc_frequency_features(f.id)
            calc_envelope_feature(f.id)
            calc_autocorr_feature(f.id)
            calc_emd_feature(f.id)
            calc_hht_features(f.id)
            calc_grid_features(f.id)



def calc_add_features_profile():
    for f in session.query(Formation).filter(Formation.profile_id == get_profile_id()).all():
        calc_wavelet_features(f.id)
        calc_fractal_features(f.id)
        calc_entropy_features(f.id)
        calc_nonlinear_features(f.id)
        calc_morphology_features(f.id)
        calc_frequency_features(f.id)
        calc_envelope_feature(f.id)
        calc_autocorr_feature(f.id)
        calc_emd_feature(f.id)
        calc_hht_features(f.id)
        calc_grid_features(f.id)


# Grid features

def calc_grid_features(f_id):
    formation = session.query(Formation).filter(Formation.id == f_id).first()
    set_info(f'Расчет GRID параметров для профиля {formation.profile.title} и пласта {formation.title}.'
             f' {formation.profile.research.object.title}', 'blue')
    layer_up, layer_down = json.loads(formation.layer_up.layer_line), json.loads(formation.layer_down.layer_line)

    x_pulc = json.loads(session.query(Profile.x_pulc).filter(Profile.id == formation.profile_id).first()[0])
    y_pulc = json.loads(session.query(Profile.y_pulc).filter(Profile.id == formation.profile_id).first()[0])

    grid_uf = check_profile_all_grid(formation.profile_id, 'uf')
    grid_m = check_profile_all_grid(formation.profile_id, 'm')
    grid_r = check_profile_all_grid(formation.profile_id, 'r')
    tree_grid_uf, tree_grid_m, tree_grid_r = None, None, None
    if grid_uf:
        tree_grid_uf = cKDTree(np.array(grid_uf)[:, :2])
        grid_val_uf = np.array(grid_uf)[:, 2]
    if grid_m:
        tree_grid_m = cKDTree(np.array(grid_m)[:, :2])
        grid_val_m = np.array(grid_m)[:, 2]
    if grid_r:
        tree_grid_r = cKDTree(np.array(grid_r)[:, :2])
        grid_val_r = np.array(grid_r)[:, 2]
    width_l, top_l, land_l, speed_l, speed_cover_l = [], [], [], [], []

    ui.progressBar.setMaximum(len(layer_up))
    for i in range(len(layer_up)):
        ui.progressBar.setValue(i)
        if grid_uf:
            i_uf = idw_interpolation(tree_grid_uf, x_pulc[i], y_pulc[i], grid_val_uf)
            top_l.append(i_uf)
        if grid_m:
            i_m = idw_interpolation(tree_grid_m, x_pulc[i], y_pulc[i], grid_val_m)
            width_l.append(i_m)
            speed_l.append(i_m * 100 / (layer_down[i] * 8 - layer_up[i] * 8))
        if grid_r:
            i_r = idw_interpolation(tree_grid_r, x_pulc[i], y_pulc[i], grid_val_r)
            land_l.append(i_r)
            if grid_uf:
                speed_cover_l.append((i_r - i_uf) * 100 / (layer_up[i] * 8))
    dict_feature = {}
    if grid_m:
        dict_feature['width'] = json.dumps(width_l)
        dict_feature['speed'] = json.dumps(speed_l)
    if grid_uf:
        dict_feature['top'] = json.dumps(top_l)
    if grid_r:
        dict_feature['land'] = json.dumps(land_l)
        if grid_uf:
            dict_feature['speed_cover'] = json.dumps(speed_cover_l)

        session.query(Formation).filter(Formation.id == f_id).update(dict_feature, synchronize_session="fetch")
        session.commit()

# Вейвлет преобразования

def calc_wavelet_features(f_id, wavelet='db4', level=5):
    if session.query(WaveletFeature).filter_by(formation_id=f_id).count() != 0:
        return
    formation = session.query(Formation).filter(Formation.id == f_id).first()
    set_info(f'Расчет вейвлет параметров для профиля {formation.profile.title} и пласта {formation.title}.'
             f' {formation.profile.research.object.title}', 'blue')
    signal = json.loads(session.query(Profile.signal).filter(Profile.id == formation.profile_id).first()[0])
    layer_up, layer_down = json.loads(formation.layer_up.layer_line), json.loads(formation.layer_down.layer_line)
    dict_wvt_ftr_list = {f'{wvt}_l': [] for wvt in list_wavelet_features}
    ui.progressBar.setMaximum(len(signal))
    for meas, s in enumerate(tqdm(signal)):
        ui.progressBar.setValue(meas)
        form_signal = s[layer_up[meas]:layer_down[meas]]
        coeffs = pywt.wavedec(form_signal, wavelet, level=level)
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

    new_wavelet_formation = WaveletFeature(formation_id=f_id)
    session.add(new_wavelet_formation)
    session.query(WaveletFeature).filter(WaveletFeature.formation_id == f_id).update(dict_wvt_ftr_json, synchronize_session="fetch")
    session.commit()


# Фрактальные характеристики

def box_counting_dim(signal, eps=None):
    if eps is None:
        eps = np.logspace(-1, 1, 20)

    N = []
    for scale in eps:
        bins = np.arange(0, len(signal), scale)
        N.append(len(np.unique(np.digitize(np.arange(len(signal)), bins))))

    coeffs = np.polyfit(np.log(eps), np.log(N), 1)
    return -coeffs[0]


def mfdfa(signal, q_range=(-5, 5), q_steps=20, scales=None):
    signal = np.cumsum(signal - np.mean(signal))
    N = len(signal)

    if scales is None:
        scales = np.logspace(1, np.log10(N // 2), 20, dtype=int)

    q_values = np.linspace(q_range[0], q_range[1], q_steps)

    fluctuations = np.zeros((len(scales), len(q_values)))

    for i, scale in enumerate(scales):
        segments = N // scale
        X = signal[:segments * scale].reshape((segments, scale))

        # Линейная аппроксимация для каждого сегмента
        X_detrended = np.zeros_like(X)
        for j, x in enumerate(X):
            coeffs = np.polyfit(np.arange(scale), x, 1)
            X_detrended[j] = x - (coeffs[0] * np.arange(scale) + coeffs[1])

        # Вычисление флуктуаций
        F2 = np.mean(X_detrended ** 2, axis=1)

        for j, q in enumerate(q_values):
            if q == 0:
                fluctuations[i, j] = np.exp(0.5 * np.mean(np.log(F2)))
            else:
                fluctuations[i, j] = (np.mean(F2 ** (q / 2))) ** (1 / q)

    # Вычисление обобщенного показателя Херста
    H = np.zeros(len(q_values))
    for j in range(len(q_values)):
        H[j], _ = np.polyfit(np.log(scales), np.log(fluctuations[:, j]), 1)

    # Вычисление спектра сингулярности
    tau = H * q_values - 1
    alpha = np.diff(tau) / np.diff(q_values)
    f_alpha = q_values[:-1] * alpha - tau[:-1]

    return alpha, f_alpha


def calc_character_mfdfa(alpha, f_alpha):
    # Вычисляем характеристики мультифрактального спектра
    width_alpha = np.max(alpha) - np.min(alpha)
    max_position_alpha = alpha[np.argmax(f_alpha)]
    max_index = np.argmax(f_alpha)
    left_width = alpha[max_index] - alpha[0]
    right_width = alpha[-1] - alpha[max_index]
    asymmetry_alpha = left_width / right_width
    max_height_alpha = np.max(f_alpha)
    mean_alpha = np.mean(alpha)
    mean_f_alpha = np.mean(f_alpha)
    std_alpha = np.std(alpha)
    std_f_alpha = np.std(f_alpha)

    return width_alpha, max_position_alpha, asymmetry_alpha, max_height_alpha, mean_alpha, mean_f_alpha, std_alpha, std_f_alpha


def lacunarity(signal, box_sizes=None):
    if box_sizes is None:
        box_sizes = [2 ** i for i in range(1, int(np.log2(len(signal))))]

    lac = []
    for box_size in box_sizes:
        strided = as_strided(signal, shape=(len(signal) - box_size + 1, box_size), strides=(signal.strides[0],) * 2)
        box_masses = strided.sum(axis=1)
        lac.append(np.var(box_masses) / np.mean(box_masses) ** 2)

    return np.mean(lac)


def calc_fractal_features(f_id):
    if session.query(FractalFeature).filter_by(formation_id=f_id).count() != 0:
        return
    formation = session.query(Formation).filter(Formation.id == f_id).first()
    set_info(f'Расчет фрактальных параметров для профиля {formation.profile.title} и пласта {formation.title}.'
             f' {formation.profile.research.object.title}', 'blue')
    signal = json.loads(session.query(Profile.signal).filter(Profile.id == formation.profile_id).first()[0])
    layer_up, layer_down = json.loads(formation.layer_up.layer_line), json.loads(formation.layer_down.layer_line)
    dict_frl_ftr_list = {f'{frl}_l': [] for frl in list_fractal_features}
    ui.progressBar.setMaximum(len(signal))
    for meas, s in enumerate(tqdm(signal)):
        ui.progressBar.setValue(meas)
        form_signal = np.array(s[layer_up[meas]:layer_down[meas]])
        dict_frl_ftr_list['fractal_dim_l'].append(box_counting_dim(form_signal))
        dict_frl_ftr_list['hurst_exp_l'].append(hurst_rs(form_signal))
        dict_frl_ftr_list['lacunarity_l'].append(lacunarity(form_signal))
        alpha, f_alpha = mfdfa(form_signal)
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

    dict_frl_ftr_json = {key[:-2]: json.dumps(value) for key, value in dict_frl_ftr_list.items()}

    new_fractal_formation = (FractalFeature(formation_id=f_id))
    session.add(new_fractal_formation)
    session.query(FractalFeature).filter(FractalFeature.formation_id == f_id).update(dict_frl_ftr_json, synchronize_session="fetch")
    session.commit()


# Параметры энтропии

def shannon_entropy(signal, bins=50):
    hist, _ = np.histogram(signal, bins=bins)
    hist = hist / np.sum(hist)
    return entropy(hist, base=2)


def permutation_entropy(signal, order=3, delay=1):
    x = np.array(signal)
    n = len(x)
    n_permutations = np.array(list(permutations(range(order))))
    c = [0] * len(n_permutations)

    for i in range(n - delay * (order - 1)):
        # Extract a window of the time series
        sorted_idx = np.argsort(x[i:i + delay * order:delay])
        for j, perm in enumerate(n_permutations):
            if np.all(sorted_idx == perm):
                c[j] += 1
                break

    c = np.array(c) / float(sum(c))
    return -np.sum(c[c > 0] * np.log2(c[c > 0]))


def approx_entropy(signal, m=2, r=0.2):
    def _maxdist(x_i, x_j):
        return max([abs(ua - va) for ua, va in zip(x_i, x_j)])

    def _phi(m):
        x = [[signal[j] for j in range(i, i + m - 1 + 1)] for i in range(N - m + 1)]
        C = [len([1 for x_j in x if _maxdist(x_i, x_j) <= r]) / (N - m + 1.0) for x_i in x]
        return (N - m + 1.0) ** (-1) * sum(np.log(C))

    N = len(signal)
    r *= np.std(signal)

    return abs(_phi(m + 1) - _phi(m))


def sample_ent(signal, m=2, r=0.2):
    return entr.sample_entropy(signal, m, r)


def multiscale_entropy(signal, scales=10):
    def coarse_grain(data, scale):
        return np.array([np.mean(data[i:i + scale]) for i in range(0, len(data) - scale + 1, scale)])

    return [shannon_entropy(coarse_grain(signal, i + 1), bins=50) for i in range(scales)]


def fourier_entropy(signal):
    f = np.abs(fft(signal))**2
    f = f / np.sum(f)
    return -np.sum(f * np.log2(f + 1e-12))  # добавляем малое число, чтобы избежать log(0)


def calc_entropy_features(f_id):
    if session.query(EntropyFeature).filter_by(formation_id=f_id).count() != 0:
        return
    formation = session.query(Formation).filter(Formation.id == f_id).first()
    set_info(f'Расчет параметров энтропии для профиля {formation.profile.title} и пласта {formation.title}.'
             f' {formation.profile.research.object.title}', 'blue')
    signal = json.loads(session.query(Profile.signal).filter(Profile.id == formation.profile_id).first()[0])
    layer_up, layer_down = json.loads(formation.layer_up.layer_line), json.loads(formation.layer_down.layer_line)
    dict_ent_ftr_list = {f'{ent}_l': [] for ent in list_entropy_features}
    ui.progressBar.setMaximum(len(signal))
    for meas, s in enumerate(tqdm(signal)):
        ui.progressBar.setValue(meas)
        form_signal = np.array(s[layer_up[meas]:layer_down[meas]])
        dict_ent_ftr_list['ent_sh_l'].append(shannon_entropy(form_signal))
        dict_ent_ftr_list['ent_perm_l'].append(permutation_entropy(form_signal))
        dict_ent_ftr_list['ent_appr_l'].append(approx_entropy(form_signal))
        for n_se, i_se in enumerate(sample_ent(form_signal)):
            dict_ent_ftr_list[f'ent_sample{n_se + 1}_l'].append(i_se)
        for n_me, i_me in enumerate(multiscale_entropy(form_signal)):
            dict_ent_ftr_list[f'ent_ms{n_me + 1}_l'].append(i_me)
        dict_ent_ftr_list['ent_fft_l'].append(fourier_entropy(form_signal))

    dict_ent_ftr_json = {key[:-2]: json.dumps(value) for key, value in dict_ent_ftr_list.items()}

    new_entropy_formation = (EntropyFeature(formation_id=f_id))
    session.add(new_entropy_formation)
    session.query(EntropyFeature).filter(EntropyFeature.formation_id == f_id).update(dict_ent_ftr_json, synchronize_session="fetch")
    session.commit()


# Нелинейные характеристики

def correlation_dimension(signal, emb_dim=10, lag=1):
    try:
        # Проверяем, что сигнал не константный
        if np.all(signal == signal[0]):
            return 0  # Возвращаем 0 для константного сигнала

        # Нормализуем сигнал
        signal = (signal - np.mean(signal)) / np.std(signal)

        # Проверяем, достаточно ли точек для вычисления
        if len(signal) < 2 * emb_dim:
            return np.nan

        return corr_dim(signal, emb_dim=emb_dim, lag=lag)
    except Exception as e:
        print(f"Ошибка при вычислении корреляционной размерности: {e}")
        set_info(f"Ошибка при вычислении корреляционной размерности: {e}", 'red')
        return np.nan


def recurrence_plot_features(signal, dimension=3, time_delay=1, threshold='point', percentage=10):
    rp = RecurrencePlot(dimension=dimension, time_delay=time_delay, threshold=threshold, percentage=percentage)
    X = signal.reshape(1, -1)
    rec_plot = rp.fit_transform(X)[0]

    # Рассчитываем некоторые характеристики рекуррентного графика
    recurrence_rate = np.mean(rec_plot)
    determinism = np.sum(np.diag(rec_plot, k=1)) / np.sum(rec_plot)
    avg_diagonal_line = np.mean(np.diag(rec_plot, k=1))

    return {
        'recurrence_rate': recurrence_rate,
        'determinism': determinism,
        'avg_diagonal_line': avg_diagonal_line
    }


def hirschman_index(signal):
    f = np.abs(fft(signal))**2
    f = f / np.sum(f)
    return np.exp(np.sum(f * np.log(f + 1e-12)))  # добавляем малое число, чтобы избежать log(0)


def calc_nonlinear_features(f_id):
    if session.query(NonlinearFeature).filter_by(formation_id=f_id).count() != 0:
        return
    formation = session.query(Formation).filter(Formation.id == f_id).first()
    set_info(f'Расчет нелинейныхпараметров для профиля {formation.profile.title} и пласта {formation.title}. {formation.profile.research.object.title}', 'blue')
    signal = json.loads(session.query(Profile.signal).filter(Profile.id == formation.profile_id).first()[0])
    layer_up, layer_down = json.loads(formation.layer_up.layer_line), json.loads(formation.layer_down.layer_line)
    dict_nln_ftr_list = {f'{ent}_l': [] for ent in list_nonlinear_features}
    ui.progressBar.setMaximum(len(signal))
    for meas, s in enumerate(tqdm(signal)):
        ui.progressBar.setValue(meas)
        form_signal = np.array(s[layer_up[meas]:layer_down[meas]])
        dict_nln_ftr_list['nln_corr_dim_l'].append(correlation_dimension(form_signal))
        rec_plot = recurrence_plot_features(form_signal)
        dict_nln_ftr_list['nln_rec_rate_l'].append(rec_plot['recurrence_rate'])
        dict_nln_ftr_list['nln_determin_l'].append(rec_plot['determinism'])
        dict_nln_ftr_list['nln_avg_diag_l'].append(rec_plot['avg_diagonal_line'])
        dict_nln_ftr_list['nln_hirsh_l'].append(hirschman_index(form_signal))

    dict_nln_ftr_json = {key[:-2]: json.dumps(value) for key, value in dict_nln_ftr_list.items()}
    session.add(NonlinearFeature(formation_id=f_id, **dict_nln_ftr_json))
    session.commit()


# Морофологические параметры

def count_peaks(peaks):
    return len(peaks)


def main_peak_width(signal, peaks, rel_height=0.5):
    if len(peaks) == 0:
        return 0
    main_peak = peaks[np.argmax(signal[peaks])]
    widths = peak_widths(signal, [main_peak], rel_height=rel_height)
    return widths[0][0]


def peak_amplitude_ratio(signal, peaks):
    if len(peaks) < 2:
        return 1
    peak_amplitudes = signal[peaks]
    return np.mean(peak_amplitudes[1:] / peak_amplitudes[:-1])


def peak_asymmetry(signal, peaks, window_size=10):
    if len(peaks) == 0:
        return 0
    asymmetries = []
    for peak in peaks:
        start = max(0, peak - window_size)
        end = min(len(signal), peak + window_size + 1)
        window = signal[start:end]
        asymmetries.append(skew(window))
    return np.mean(asymmetries)


def slope_steepness(signal, window_size=5):
    slopes = np.abs(np.diff(signal))
    return np.mean([np.max(slopes[i:i+window_size]) for i in range(len(slopes)-window_size)])



def morphological_features(signal, threshold=0.5):
    binary_signal = signal > (np.max(signal) * threshold)
    eroded = binary_erosion(binary_signal)
    dilated = binary_dilation(binary_signal)
    return {
        'erosion_ratio': np.sum(eroded) / np.sum(binary_signal),
        'dilation_ratio': np.sum(dilated) / np.sum(binary_signal)
    }


def calc_morphology_features(f_id):
    if session.query(MorphologyFeature).filter_by(formation_id=f_id).count() != 0:
        return
    formation = session.query(Formation).filter(Formation.id == f_id).first()
    set_info(f'Расчет морфологических параметров для профиля {formation.profile.title} и пласта {formation.title}.'
             f' {formation.profile.research.object.title}', 'blue')
    signal = json.loads(session.query(Profile.signal).filter(Profile.id == formation.profile_id).first()[0])
    layer_up, layer_down = json.loads(formation.layer_up.layer_line), json.loads(formation.layer_down.layer_line)
    dict_mph_ftr_list = {f'{mph}_l': [] for mph in list_morphology_feature}
    ui.progressBar.setMaximum(len(signal))
    for meas, s in enumerate(tqdm(signal)):
        ui.progressBar.setValue(meas)
        form_signal = np.array(s[layer_up[meas]:layer_down[meas]])
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
    session.add(MorphologyFeature(formation_id=f_id, **dict_mph_ftr_json))
    session.commit()


# Частотные характеристики


def power_spectrum(signal, fs=15e6):
    n = len(signal)
    freqs = np.fft.fftfreq(n, 1/fs)
    ps = np.abs(fft(signal))**2
    return freqs[:n//2], ps[:n//2]


def central_frequency(freqs, ps):
    return np.sum(freqs * ps) / np.sum(ps)


def bandwidth(freqs, ps, threshold=0.5):
    max_power = np.max(ps[1:])
    mask = ps > (max_power * threshold)
    return np.max(freqs[mask]) - np.min(freqs[mask])


def high_low_frequency_ratio(freqs, ps):
    threshold_freq = np.median(freqs)
    low_freq_energy = np.sum(ps[freqs < threshold_freq])
    high_freq_energy = np.sum(ps[freqs >= threshold_freq])
    return high_freq_energy / low_freq_energy if low_freq_energy != 0 else np.inf


def spectral_centroid(freqs, ps):
    return np.sum(freqs * ps) / np.sum(ps)


def spectral_slope(freqs, ps):
    log_freqs = np.log10(freqs[1:])  # исключаем нулевую частоту
    log_ps = np.log10(ps[1:])
    return linregress(log_freqs, log_ps).slope


def spectral_entropy(ps):
    ps_norm = ps / np.sum(ps)
    return entropy(ps_norm)

def dominant_frequencies(freqs, ps, num_peaks=4):
    peak_indices = np.argsort(ps)[-num_peaks:][::-1]
    return freqs[peak_indices[1:]]

def spectral_moments(freqs, ps, order=4):
    moments = []
    for i in range(order):
        moments.append(np.sum((freqs**i) * ps) / np.sum(ps))
    return moments[1:]


def attenuation_coefficient(signal, depth, freqs_rate):
    n = len(signal)

    # Проверка на минимальную длину сигнала
    if n < 4:
        return np.nan  # Сигнал слишком короткий для анализа

    # Разделение сигнала на две равные части
    half = n // 2
    signal1 = signal[:half]
    signal2 = signal[half:]

    # Выравнивание длин сигналов, если они различаются
    if len(signal1) != len(signal2):
        signal2 = signal2[:len(signal1)]

    # Вычисление FFT
    fft1 = fft(signal1)
    fft2 = fft(signal2)

    # Вычисление частот
    freqs = fftfreq(len(signal1), d=(depth[1] - depth[0]))

    # Вычисление спектров мощности
    ps1 = np.abs(fft1) ** 2
    ps2 = np.abs(fft2) ** 2

    # Создание маски для нужного диапазона частот
    mask = (freqs_rate >= freqs[0]) & (freqs_rate <= freqs[1])

    # Проверка, есть ли частоты в заданном диапазоне
    if not np.any(mask):
        return np.nan

    # Вычисление отношения спектров и логарифма
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(ps2[mask] != 0, ps1[mask] / ps2[mask], 1)
        log_ratio = np.where(ratio > 0, np.log(ratio), 0)

    # Вычисление коэффициента затухания
    return np.mean(log_ratio) / (depth[half] - depth[0])



def calc_frequency_features(f_id):
    if session.query(FrequencyFeature).filter(FrequencyFeature.formation_id == f_id).count() > 0:
        return
    formation = session.query(Formation).filter(Formation.id == f_id).first()
    set_info(f'Расчет частотных характеристик для профиля {formation.profile.title} пласт {formation.title}.'
             f' {formation.profile.research.object.title}', 'blue')
    signal = json.loads(session.query(Profile.signal).filter(Profile.id == formation.profile_id).first()[0])
    layer_up, layer_down = json.loads(formation.layer_up.layer_line), json.loads(formation.layer_down.layer_line)
    dict_freq_ftr_list = {f'{freq}_l': [] for freq in list_frequency_feature}
    ui.progressBar.setMaximum(len(signal))
    for meas, s in enumerate(tqdm(signal)):
        ui.progressBar.setValue(meas)
        form_signal = np.array(s[layer_up[meas]:layer_down[meas]])
        freqs, ps = power_spectrum(form_signal)
        dict_freq_ftr_list['frq_central_l'].append(central_frequency(freqs, ps))
        dict_freq_ftr_list['frq_bandwidth_l'].append(bandwidth(freqs, ps))
        dict_freq_ftr_list['frq_hl_ratio_l'].append(high_low_frequency_ratio(freqs, ps))
        dict_freq_ftr_list['frq_spec_centroid_l'].append(spectral_centroid(freqs, ps))
        dict_freq_ftr_list['frq_spec_slope_l'].append(spectral_slope(freqs, ps))
        dict_freq_ftr_list['frq_spec_entr_l'].append(spectral_entropy(ps))
        for n_freq, f in enumerate(dominant_frequencies(freqs, ps)):
            dict_freq_ftr_list[f'frq_dom{n_freq+1}_l'].append(f)
        for n_freq, f in enumerate(spectral_moments(freqs, ps)):
            dict_freq_ftr_list[f'frq_mmt{n_freq+1}_l'].append(f)
        dict_freq_ftr_list['frq_attn_coef_l'].append(attenuation_coefficient(form_signal, depth=range(len(form_signal)), freqs_rate=freqs))

    dict_freq_ftr_json = {key[:-2]: json.dumps(value) for key, value in dict_freq_ftr_list.items()}
    session.add(FrequencyFeature(formation_id=f_id, **dict_freq_ftr_json))
    session.commit()


# характеристики огибающей

def calculate_envelope(signal):
    analytic_signal = hilbert(signal)
    envelope = np.abs(analytic_signal)
    return envelope


def area_under_envelope(envelope):
    return np.trapz(envelope)


def envelope_max(envelope):
    return np.max(envelope)


def time_to_max_envelope(envelope):
    return np.argmax(envelope) # if calculated for formation - + Ttop


def envelope_mean(envelope):
    return np.mean(envelope)


def envelope_std(envelope):
    return np.std(envelope)


def envelope_skewness(envelope):
    return skew(envelope)


def envelope_kurtosis(envelope):
    return kurtosis(envelope)


def max_to_mean_ratio(envelope):
    return np.max(envelope) / np.mean(envelope)


def main_peak_width_env(envelope, height_ratio=0.5):
    peaks, _ = find_peaks(envelope)
    if len(peaks) == 0:
        return 0
    main_peak = peaks[np.argmax(envelope[peaks])]
    left = main_peak
    right = main_peak
    threshold = envelope[main_peak] * height_ratio
    while left > 0 and envelope[left] > threshold:
        left -= 1
    while right < len(envelope) - 1 and envelope[right] > threshold:
        right += 1
    return right - left


def envelope_energy_windows(envelope, num_windows=3):
    window_size = len(envelope) // num_windows
    return [np.sum(envelope[i*window_size:(i+1)*window_size]**2) for i in range(num_windows)]


def calc_envelope_feature(f_id):
    if session.query(EnvelopeFeature).filter(EnvelopeFeature.formation_id == f_id).count() > 0:
        return
    formation = session.query(Formation).filter(Formation.id == f_id).first()
    set_info(f'Расчет характеристик огибающей для профиля {formation.profile.title} пласт {formation.title}.'
             f' {formation.profile.research.object.title}', 'blue')
    signal = json.loads(session.query(Profile.signal).filter(Profile.id == formation.profile_id).first()[0])
    layer_up, layer_down = json.loads(formation.layer_up.layer_line), json.loads(formation.layer_down.layer_line)
    dict_env_ftr_list = {f'{env}_l': [] for env in list_envelope_feature}
    ui.progressBar.setMaximum(len(signal))
    for meas, s in enumerate(tqdm(signal)):
        ui.progressBar.setValue(meas)
        form_signal = np.array(s[layer_up[meas]:layer_down[meas]])
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
    session.add(EnvelopeFeature(formation_id=f_id, **dict_env_ftr_json))
    session.commit()

# параметры автокорреляции

def autocorrelation(signal):
    result = correlate(signal, signal, mode='full')
    return result[result.size // 2:]


def first_minimum(acf):
    for i in range(1, len(acf)-1):
        if acf[i] < acf[i-1] and acf[i] <= acf[i+1]:
            return i
    # Проверка последнего элемента
    if len(acf) > 1 and acf[-1] < acf[-2]:
        return len(acf) - 1
    return None


def autocorrelation_at_lag(acf, lag=10):
    if lag < len(acf):
        return acf[lag]
    return None


def autocorrelation_decay(acf, num_points=10):
    return np.polyfit(range(num_points), acf[:num_points], 1)[0]


def acf_integral(acf):
    return np.trapz(acf)


def acf_main_peak_width(acf, height_ratio=0.5):
    peak_height = acf[0]
    threshold = peak_height * height_ratio
    right = 0
    while right < len(acf) - 1 and acf[right] > threshold:
        right += 1
    return right


def acf_ratio(acf, lag1=10, lag2=20):
    if lag1 < len(acf) and lag2 < len(acf):
        return acf[lag1] / acf[lag2] if acf[lag2] != 0 else np.inf
    return None


def calc_autocorr_feature(f_id):
    if session.query(AutocorrFeature).filter(AutocorrFeature.formation_id == f_id).count() > 0:
        return
    formation = session.query(Formation).filter(Formation.id == f_id).first()
    set_info(f'Расчет характеристик автокорреляции для профиля {formation.profile.title} пласт {formation.title}.'
             f' {formation.profile.research.object.title}', 'blue')
    signal = json.loads(session.query(Profile.signal).filter(Profile.id == formation.profile_id).first()[0])
    layer_up, layer_down = json.loads(formation.layer_up.layer_line), json.loads(formation.layer_down.layer_line)
    dict_acf_list = {f'{acf}_l': [] for acf in list_autocorr_feature}
    ui.progressBar.setMaximum(len(signal))
    for meas, s in enumerate(tqdm(signal)):
        ui.progressBar.setValue(meas)
        form_signal = np.array(s[layer_up[meas]:layer_down[meas]])
        acf = autocorrelation(form_signal)
        dict_acf_list['acf_first_min_l'].append(float(first_minimum(acf)))
        dict_acf_list['acf_lag_10_l'].append(float(autocorrelation_at_lag(acf)))
        dict_acf_list['acf_decay_l'].append(autocorrelation_decay(acf))
        dict_acf_list['acf_integral_l'].append(acf_integral(acf))
        dict_acf_list['acf_peak_width_l'].append(float(acf_main_peak_width(acf)))
        dict_acf_list['acf_ratio_l'].append(acf_ratio(acf))

    dict_acf_json = {key[:-2]: json.dumps(value) for key, value in dict_acf_list.items()}
    session.add(AutocorrFeature(formation_id=f_id, **dict_acf_json))
    session.commit()


# параметры эмпирической модовой декомпозиции EMD

def perform_emd(signal):
    emd = EMD()
    imfs = emd(signal)
    return imfs


def count_imfs(imfs):
    return len(imfs)


def imf_energies(imfs):
    energies =[np.sum(imf**2) for imf in imfs]
    return {'mean_energy': np.mean(energies),
            'median_energy': np.median(energies),
            'max_energy': np.max(energies),
            'min_energy': np.min(energies),
            'std_energy': np.std(energies)}


def relative_imf_energies(imfs):
    energies =[np.sum(imf**2) for imf in imfs]
    total_energy = sum(energies)
    enrs = [enr / total_energy for enr in energies]
    return {
        'mean_energy': np.mean(enrs),
        'median_energy': np.median(enrs),
        'max_energy': np.max(enrs),
        'min_energy': np.min(enrs),
        'std_energy': np.std(enrs)
    }


def emd_dominant_frequencies(imfs, fs=15e6):
    frequencies = []
    for imf in imfs:
        peaks, _ = find_peaks(imf)
        if len(peaks) > 1:
            mean_period = np.mean(np.diff(peaks))
            frequencies.append(fs / mean_period)
        else:
            frequencies.append(0)
    return {
        'mean_freq': np.mean(frequencies),
        'median_freq': np.median(frequencies),
        'max_freq': np.max(frequencies),
        'min_freq': np.min(frequencies),
        'std_freq': np.std(frequencies)
    }


def imf_correlations(imfs):
    corr_matrix = np.corrcoef(imfs)
    try:
        corr_values = corr_matrix[np.triu_indices(len(imfs), k=1)]
    except IndexError:
        corr_values = np.array([1.0])
    return {
        'mean_corr': np.mean(corr_values),
        'median_corr': np.median(corr_values),
        'max_corr': np.max(np.abs(corr_values)),
        'min_corr': np.min(np.abs(corr_values)),
        'std_corr': np.std(corr_values),
        'corr_25': np.percentile(corr_values, 25),
        'corr_50': np.percentile(corr_values, 50),
        'corr_75': np.percentile(corr_values, 75)
    }


def imf_energy_entropy(imfs):
    energies =[np.sum(imf**2) for imf in imfs]
    total_energy = sum(energies)
    rel_energies = [enr / total_energy for enr in energies]
    return entropy(rel_energies)


def orthogonality_index(imfs):
    n = len(imfs)
    m = len(imfs[0])
    sum_sq = np.sum(np.sum(imfs**2, axis=1))
    cross_terms = 0
    for i in range(n):
        for j in range(i+1, n):
            cross_terms += np.abs(np.sum(imfs[i] * imfs[j]))
    return cross_terms / sum_sq


def instantaneous_frequency(imf, fs=15e6):
    analytic_signal = hilbert(imf)
    instantaneous_phase = np.unwrap(np.angle(analytic_signal))
    instant_freq = np.diff(instantaneous_phase) / (2.0*np.pi) * fs
    return np.insert(instant_freq, 0, instant_freq[0])


def emd_hilbert_index(imfs):
    n = len(imfs)
    if n == 1:
        return float(0)
    h_index = 0
    for i in range(n-1):
        mean_freq_i = np.mean(instantaneous_frequency(imfs[i]))
        mean_freq_i1 = np.mean(instantaneous_frequency(imfs[i+1]))
        h_index += np.abs(mean_freq_i - mean_freq_i1) / mean_freq_i
    return h_index / (n - 1)


def calc_emd_feature(f_id):
    if session.query(EMDFeature).filter(EMDFeature.formation_id == f_id).count() > 0:
        return
    formation = session.query(Formation).filter(Formation.id == f_id).first()
    set_info(f'Расчет характеристик EMD для профиля {formation.profile.title} пласт {formation.title}. {formation.profile.research.object.title}', 'blue')
    signal = json.loads(session.query(Profile.signal).filter(Profile.id == formation.profile_id).first()[0])
    layer_up, layer_down = json.loads(formation.layer_up.layer_line), json.loads(formation.layer_down.layer_line)
    dict_emd_ftr_list = {f'{emd}_l': [] for emd in list_emd_feature}
    ui.progressBar.setMaximum(len(signal))
    for meas, s in enumerate(tqdm(signal)):
        ui.progressBar.setValue(meas)
        form_signal = np.array(s[layer_up[meas]:layer_down[meas]])
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

    dict_emd_ftr_json = {key[:-2]: json.dumps(value) for key, value in dict_emd_ftr_list.items()}
    session.add(EMDFeature(formation_id=f_id, **dict_emd_ftr_json))
    session.commit()
    
    
# Преобразования Гильберта-Хуанга (HHT)


def calc_stat_list(param: list, param_name: str) -> dict:
    return {
        f'{param_name}_mean': np.mean(param),
        f'{param_name}_med': np.median(param),
        f'{param_name}_max': np.max(param),
        f'{param_name}_min': np.min(param),
        f'{param_name}_std': np.std(param),
        # f'{param_name}_25': np.percentile(param, 25),
        # f'{param_name}_50': np.percentile(param, 50),
        # f'{param_name}_75': np.percentile(param, 75)
    }


def perform_hht(signal, fs=15e6):
    emd = EMD()
    imfs = emd(signal)

    hht = []
    for imf in imfs:
        analytic_signal = hilbert(imf)
        amplitude = np.abs(analytic_signal)
        phase = np.unwrap(np.angle(analytic_signal))
        frequency = np.diff(phase) / (2.0 * np.pi * (1/fs))
        frequency = np.insert(frequency, 0, frequency[0])  # Добавляем первое значение, чтобы длина совпадала
        hht.append((amplitude, frequency))

    return imfs, hht


def instantaneous_frequency_amplitude(hht):
    inst_freq = [h[1] for h in hht]
    inst_amp = [h[0] for h in hht]
    return inst_freq, inst_amp


def mean_inst_freq_amp(hht):
    mean_freq = [np.mean(h[1]) for h in hht]
    mean_amp = [np.mean(h[0]) for h in hht]
    return mean_freq, mean_amp


def marginal_spectrum(hht, fs=15e6):
    n_samples = len(hht[0][0])
    freq_bins = np.linspace(0, fs / 2, n_samples // 2 + 1)
    marginal_spectrum = np.zeros(len(freq_bins) - 1)

    for amp, freq in hht:
        hist, _ = np.histogram(freq, bins=freq_bins, weights=amp ** 2)
        marginal_spectrum += hist

    return freq_bins[:-1], marginal_spectrum


def teager_energy(imf):
    return np.mean(imf[1:-1]**2 - imf[:-2]*imf[2:])


def teager_energies(imfs):
    return [teager_energy(imf) for imf in imfs]


def hilbert_index(hht):
    mean_freqs = [np.mean(h[1]) for h in hht]
    return np.mean(np.abs(np.diff(mean_freqs)) / mean_freqs[:-1])


def degree_of_stationarity(hht):
    dos = []
    for amp, freq in hht:
        amp_mean = np.mean(amp)
        dos.append(np.mean((amp - amp_mean)**2) / amp_mean**2)
    return dos


def hht_orthogonality_index(imfs):
    n = len(imfs)
    m = len(imfs[0])
    sum_sq = np.sum(np.sum(imfs**2, axis=1))
    cross_terms = 0
    for i in range(n):
        for j in range(i+1, n):
            cross_terms += np.abs(np.sum(imfs[i] * imfs[j]))
    return cross_terms / sum_sq

def hilbert_spectral_density(hht, signal):
    time = np.arange(len(signal))
    hsd = np.zeros((len(hht), len(time)))
    for i, (amp, freq) in enumerate(hht):
        hsd[i] = amp**2
    return hsd

def complexity_index(hht):
    return len(hht)  # Количество IMF


def calc_hht_features(f_id):
    if session.query(HHTFeature).filter(HHTFeature.formation_id == f_id).count() > 0:
        return
    formation = session.query(Formation).filter(Formation.id == f_id).first()
    set_info(f'Расчет характеристик HHT для профиля {formation.profile.title} пласт {formation.title}.'
             f' {formation.profile.research.object.title}', 'blue')
    signal = json.loads(session.query(Profile.signal).filter(Profile.id == formation.profile_id).first()[0])
    layer_up, layer_down = json.loads(formation.layer_up.layer_line), json.loads(formation.layer_down.layer_line)
    dict_hht_ftr_list = {f'{hht}_l': [] for hht in list_hht_feature}
    ui.progressBar.setMaximum(len(signal))
    for meas, s in enumerate(tqdm(signal)):
        ui.progressBar.setValue(meas)
        form_signal = np.array(s[layer_up[meas]:layer_down[meas]])
        imfs, hht = perform_hht(form_signal)

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
    session.add(HHTFeature(formation_id=f_id, **dict_hht_ftr_json))
    session.commit()