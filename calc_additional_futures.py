from func import *


def calc_wavelet_futures_profile():
    for f in session.query(Formation).filter(Formation.profile_id == get_profile_id()).all():
        calc_wavelet_futures(f.id)


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