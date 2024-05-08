from func import *


for f in tqdm(session.query(Formation).all()):
    signals = json.loads(session.query(Profile.signal).filter(Profile.id == f.profile_id).first()[0])
    layer_up, layer_down = json.loads(f.layer_up.layer_line), json.loads(f.layer_down.layer_line)
    CRL_signal = calc_CRL_filter(signals)
    if len(CRL_signal) != len(layer_up) != len(layer_down):
        set_info(f'ВНИМАНИЕ! ОШИБКА!!! Для пласта {f.title} профиля {f.profile.title} не совпадает количество '
                 f'измерений в радарпограмме и в границах кровли/подошвы', 'red')
        continue
    else:
        (CRL_top_l, CRL_bottom_l, dCRL_l, CRL_sum_l, CRL_mean_l, CRL_max_l, CRL_T_max_l, CRL_Sn_l, CRL_wmf_l,
         CRL_Qf_l, CRL_Sn_wmf_l, CRL_skew_l, CRL_kurt_l, CRL_std_l, CRL_k_var_l) = (
            [], [], [], [], [], [], [], [], [], [], [], [], [], [], [])

        for i in range(len(layer_up)):
            CRl_s = CRL_signal[i]
            nt = layer_up[i]
            nb = layer_down[i]

            CRL_top_l.append(CRl_s[nt])
            CRL_bottom_l.append(CRl_s[nb])
            dCRL_l.append(CRl_s[nb] - CRl_s[nt])
            CRL_sum_l.append(float(np.sum(CRl_s[nt:nb])))
            CRL_mean_l.append(float(np.mean(CRl_s[nt:nb])))
            try:
                CRL_max_l.append(max(CRl_s[nt:nb]))
                CRL_T_max_l.append((CRl_s[nt:nb].tolist().index(max(CRl_s[nt:nb])) + nt) * 8)
            except ValueError:
                CRL_max_l.append(None)
                CRL_T_max_l.append(None)

            CRL_Sn, CRL_wmf, CRL_Qf, CRL_Sn_wmf = calc_fft_attributes(CRl_s[nt:nb])

            CRL_Sn_l.append(CRL_Sn)
            CRL_wmf_l.append(CRL_wmf)
            CRL_Qf_l.append(CRL_Qf)
            CRL_Sn_wmf_l.append(CRL_Sn_wmf)

            CRL_skew_l.append(skew(CRl_s[nt:nb]))
            CRL_kurt_l.append(kurtosis(CRl_s[nt:nb]))
            CRL_std_l.append(np.std(CRl_s[nt:nb]))
            CRL_k_var_l.append(np.var(CRl_s[nt:nb]))

        dict_CRL_param = {
            'CRL_top': json.dumps(CRL_top_l),
            'CRL_bottom': json.dumps(CRL_bottom_l),
            'dCRL': json.dumps(dCRL_l),
            'CRL_sum': json.dumps(CRL_sum_l),
            'CRL_mean': json.dumps(CRL_mean_l),
            'CRL_max': json.dumps(CRL_max_l),
            'CRL_T_max': json.dumps(CRL_T_max_l),
            'CRL_Sn': json.dumps(CRL_Sn_l),
            'CRL_wmf': json.dumps(CRL_wmf_l),
            'CRL_Qf': json.dumps(CRL_Qf_l),
            'CRL_Sn_wmf': json.dumps(CRL_Sn_wmf_l),
            'CRL_skew': json.dumps(CRL_skew_l),
            'CRL_kurt': json.dumps(CRL_kurt_l),
            'CRL_std': json.dumps(CRL_std_l),
            'CRL_k_var': json.dumps(CRL_k_var_l)
        }

        session.query(Formation).filter(Formation.id == f.id).update(dict_CRL_param, synchronize_session="fetch")
        session.commit()
