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



def calc_add_features_profile():
    for f in session.query(Formation).filter(Formation.profile_id == get_profile_id()).all():
        calc_wavelet_features(f.id)
        calc_fractal_features(f.id)
        calc_entropy_features(f.id)
        calc_nonlinear_features(f.id)
        calc_morphology_features(f.id)
        calc_frequency_features(f.id)
        calc_envelope_feature(f.id)



# Вейвлет преобразования

def calc_wavelet_features(f_id, wavelet='db4', level=5):
    if session.query(WaveletFeature).filter_by(formation_id=f_id).count() != 0:
        return
    formation = session.query(Formation).filter(Formation.id == f_id).first()
    set_info(f'Расчет вейвлет параметров для профиля {formation.profile.title} и пласта {formation.title}', 'blue')
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
    set_info(f'Расчет фрактальных параметров для профиля {formation.profile.title} и пласта {formation.title}', 'blue')
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
    set_info(f'Расчет параметров энтропии для профиля {formation.profile.title} и пласта {formation.title}', 'blue')
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
    return corr_dim(signal, emb_dim=emb_dim, lag=lag)


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
    set_info(f'Расчет нелинейныхпараметров для профиля {formation.profile.title} и пласта {formation.title}', 'blue')
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
    set_info(f'Расчет морфологических параметров для профиля {formation.profile.title} и пласта {formation.title}', 'blue')
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


def power_spectrum(signal, fs=15):
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
    set_info(f'Расчет частотных характеристик для профиля {formation.profile.title} пласт {formation.title}', 'blue')
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
    set_info(f'Расчет характеристик огибающей для профиля {formation.profile.title} пласт {formation.title}', 'blue')
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


