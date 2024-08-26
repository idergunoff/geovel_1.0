from func import *
from test_fractal import alpha, f_alpha


# Вейвлет преобразования

def calc_add_futures_profile():
    for f in session.query(Formation).filter(Formation.profile_id == get_profile_id()).all():
        calc_wavelet_futures(f.id)
        calc_fractal_futures(f.id)


def calc_wavelet_futures(f_id, wavelet='db4', level=5):
    if session.query(WaveletFuture).filter_by(formation_id=f_id).count() != 0:
        return
    formation = session.query(Formation).filter(Formation.id == f_id).first()
    signal = json.loads(session.query(Profile.signal).filter(Profile.id == formation.profile_id).first()[0])
    layer_up, layer_down = json.loads(formation.layer_up.layer_line), json.loads(formation.layer_down.layer_line)
    dict_wvt_ftr_list = {f'{wvt}_l': [] for wvt in list_wavelet_futures}
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

    new_wavelet_formation = WaveletFuture(formation_id=f_id)
    session.add(new_wavelet_formation)
    session.query(WaveletFuture).filter(WaveletFuture.formation_id == f_id).update(dict_wvt_ftr_json, synchronize_session="fetch")
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


def calc_fractal_futures(f_id):
    if session.query(FractalFuture).filter_by(formation_id=f_id).count() != 0:
        return
    formation = session.query(Formation).filter(Formation.id == f_id).first()
    signal = json.loads(session.query(Profile.signal).filter(Profile.id == formation.profile_id).first()[0])
    layer_up, layer_down = json.loads(formation.layer_up.layer_line), json.loads(formation.layer_down.layer_line)
    dict_frl_ftr_list = {f'{frl}_l': [] for frl in list_fractal_futures}
    print(dict_frl_ftr_list)
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

    new_fractal_formation = (FractalFuture(formation_id=f_id))
    session.add(new_fractal_formation)
    session.query(FractalFuture).filter(FractalFuture.formation_id == f_id).update(dict_frl_ftr_json, synchronize_session="fetch")
    session.commit()

