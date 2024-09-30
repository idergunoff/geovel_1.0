from models_db.model import *


class WaveletFeatureProfile(Base):
    __tablename__ = 'wavelet_feature_profile'

    id = Column(Integer, primary_key=True)
    profile_id = Column(Integer, ForeignKey('profile.id'))

    # Энергия вейвлета для каждого уровня декомпозиции
    wvt_energ_D1 = Column(Text)
    wvt_energ_D2 = Column(Text)
    wvt_energ_D3 = Column(Text)
    wvt_energ_D4 = Column(Text)
    wvt_energ_D5 = Column(Text)
    wvt_energ_A5 = Column(Text)

    # Среднее вейвлета для каждого уровня декомпозиции
    wvt_mean_D1 = Column(Text)
    wvt_mean_D2 = Column(Text)
    wvt_mean_D3 = Column(Text)
    wvt_mean_D4 = Column(Text)
    wvt_mean_D5 = Column(Text)
    wvt_mean_A5 = Column(Text)

    # Максимум вейвлета для каждого уровня декомпозиции
    wvt_max_D1 = Column(Text)
    wvt_max_D2 = Column(Text)
    wvt_max_D3 = Column(Text)
    wvt_max_D4 = Column(Text)
    wvt_max_D5 = Column(Text)
    wvt_max_A5 = Column(Text)

    # Минимум вейвлета для каждого уровня декомпозиции
    wvt_min_D1 = Column(Text)
    wvt_min_D2 = Column(Text)
    wvt_min_D3 = Column(Text)
    wvt_min_D4 = Column(Text)
    wvt_min_D5 = Column(Text)
    wvt_min_A5 = Column(Text)

    # Стандартное отклонение вейвлета для каждого уровня декомпозиции
    wvt_std_D1 = Column(Text)
    wvt_std_D2 = Column(Text)
    wvt_std_D3 = Column(Text)
    wvt_std_D4 = Column(Text)
    wvt_std_D5 = Column(Text)
    wvt_std_A5 = Column(Text)

    # Коэффициент асимметрии вейвлета для каждого уровня декомпозиции
    wvt_skew_D1 = Column(Text)
    wvt_skew_D2 = Column(Text)
    wvt_skew_D3 = Column(Text)
    wvt_skew_D4 = Column(Text)
    wvt_skew_D5 = Column(Text)
    wvt_skew_A5 = Column(Text)

    # Коэффициент эксцесса вейвлета для каждого уровня декомпозиции
    wvt_kurt_D1 = Column(Text)
    wvt_kurt_D2 = Column(Text)
    wvt_kurt_D3 = Column(Text)
    wvt_kurt_D4 = Column(Text)
    wvt_kurt_D5 = Column(Text)
    wvt_kurt_A5 = Column(Text)

    # Энтропия вейвлета для каждого уровня декомпозиции
    wvt_entr_D1 = Column(Text)
    wvt_entr_D2 = Column(Text)
    wvt_entr_D3 = Column(Text)
    wvt_entr_D4 = Column(Text)
    wvt_entr_D5 = Column(Text)
    wvt_entr_A5 = Column(Text)

    # Отношение энергий между различными уровнями декомпозиции
    wvt_energ_D1D2 = Column(Text)
    wvt_energ_D2D3 = Column(Text)
    wvt_energ_D3D4 = Column(Text)
    wvt_energ_D4D5 = Column(Text)
    wvt_energ_D5A5 = Column(Text)

    # Отношение энергии высокочастотных компонент к низкочастотным
    wvt_HfLf_Ratio = Column(Text)

    # Соотношение высоких и низких частот на разных масштабах
    wvt_HfLf_D1 = Column(Text)
    wvt_HfLf_D2 = Column(Text)
    wvt_HfLf_D3 = Column(Text)
    wvt_HfLf_D4 = Column(Text)
    wvt_HfLf_D5 = Column(Text)

    profile = relationship('Profile', back_populates='wavelet_feature')


class FractalFeatureProfile(Base):
    __tablename__ = 'fractal_feature_profile'

    id = Column(Integer, primary_key=True)
    profile_id = Column(Integer, ForeignKey('profile.id'))

    fractal_dim = Column(Text)
    hurst_exp = Column(Text)
    lacunarity = Column(Text)
    mf_width = Column(Text)
    mf_max_position = Column(Text)
    mf_asymmetry = Column(Text)
    mf_max_height = Column(Text)
    mf_mean_alpha = Column(Text)
    mf_mean_f_alpha = Column(Text)
    mf_std_alpha = Column(Text)
    mf_std_f_alpha = Column(Text)

    profile = relationship('Profile', back_populates='fractal_feature')


class EntropyFeatureProfile(Base):
    __tablename__ = 'entropy_feature_profile'

    id = Column(Integer, primary_key=True)
    profile_id = Column(Integer, ForeignKey('profile.id'))

    ent_sh = Column(Text)
    ent_perm = Column(Text)
    ent_appr = Column(Text)
    ent_sample1 = Column(Text)
    ent_sample2 = Column(Text)
    ent_ms1 = Column(Text)
    ent_ms2 = Column(Text)
    ent_ms3 = Column(Text)
    ent_ms4 = Column(Text)
    ent_ms5 = Column(Text)
    ent_ms6 = Column(Text)
    ent_ms7 = Column(Text)
    ent_ms8 = Column(Text)
    ent_ms9 = Column(Text)
    ent_ms10 = Column(Text)
    ent_fft = Column(Text)

    profile = relationship('Profile', back_populates='entropy_feature')


class NonlinearFeatureProfile(Base):
    __tablename__ = 'nonlinear_feature_profile'

    id = Column(Integer, primary_key=True)
    profile_id = Column(Integer, ForeignKey('profile.id'))

    nln_corr_dim = Column(Text)
    nln_rec_rate = Column(Text)
    nln_determin = Column(Text)
    nln_avg_diag = Column(Text)
    nln_hirsh = Column(Text)

    profile = relationship('Profile', back_populates='nonlinear_feature')


class MorphologyFeatureProfile(Base):
    __tablename__ = 'morphology_feature_profile'

    id = Column(Integer, primary_key=True)
    profile_id = Column(Integer, ForeignKey('profile.id'))

    mph_peak_num = Column(Text)
    mph_peak_width = Column(Text)
    mph_peak_amp_ratio = Column(Text)
    mph_peak_asymm = Column(Text)
    mph_peak_steep = Column(Text)
    mph_erosion = Column(Text)
    mph_dilation = Column(Text)

    profile = relationship('Profile', back_populates='morphology_feature')


class FrequencyFeatureProfile(Base):
    __tablename__ = 'frequency_feature_profile'

    id = Column(Integer, primary_key=True)
    profile_id = Column(Integer, ForeignKey('profile.id'))

    frq_central = Column(Text)
    frq_bandwidth = Column(Text)
    frq_hl_ratio = Column(Text)
    frq_spec_centroid = Column(Text)
    frq_spec_slope = Column(Text)
    frq_spec_entr = Column(Text)
    frq_dom1 = Column(Text)
    frq_dom2 = Column(Text)
    frq_dom3 = Column(Text)
    frq_mmt1 = Column(Text)
    frq_mmt2 = Column(Text)
    frq_mmt3 = Column(Text)
    frq_attn_coef = Column(Text)

    profile = relationship('Profile', back_populates='frequency_feature')


class EnvelopeFeatureProfile(Base):
    __tablename__ = 'envelope_feature_profile'

    id = Column(Integer, primary_key=True)
    profile_id = Column(Integer, ForeignKey('profile.id'))

    env_area = Column(Text)
    env_max = Column(Text)
    env_t_max = Column(Text)
    env_mean = Column(Text)
    env_std = Column(Text)
    env_skew = Column(Text)
    env_kurt = Column(Text)
    env_max_mean_ratio = Column(Text)
    env_peak_width = Column(Text)
    env_energy_win1 = Column(Text)
    env_energy_win2 = Column(Text)
    env_energy_win3 = Column(Text)

    profile = relationship('Profile', back_populates='envelope_feature')


class AutocorrFeatureProfile(Base):
    __tablename__ = 'autocorr_feature_profile'

    id = Column(Integer, primary_key=True)
    profile_id = Column(Integer, ForeignKey('profile.id'))

    acf_first_min = Column(Text)
    acf_lag_10 = Column(Text)
    acf_decay = Column(Text)
    acf_integral = Column(Text)
    acf_peak_width = Column(Text)
    acf_ratio = Column(Text)

    profile = relationship('Profile', back_populates='autocorr_feature')


class EMDFeatureProfile(Base):
    __tablename__ = 'emd_feature_profile'

    id = Column(Integer, primary_key=True)
    profile_id = Column(Integer, ForeignKey('profile.id'))

    emd_num_imfs = Column(Text)
    emd_energ_mean = Column(Text)
    emd_energ_med = Column(Text)
    emd_energ_max = Column(Text)
    emd_energ_min = Column(Text)
    emd_energ_std = Column(Text)
    emd_rel_energ_mean = Column(Text)
    emd_rel_energ_med = Column(Text)
    emd_rel_energ_max = Column(Text)
    emd_rel_energ_min = Column(Text)
    emd_rel_energ_std = Column(Text)
    emd_dom_freqs_mean = Column(Text)
    emd_dom_freqs_med = Column(Text)
    emd_dom_freqs_max = Column(Text)
    emd_dom_freqs_min = Column(Text)
    emd_dom_freqs_std = Column(Text)
    emd_mean_corr = Column(Text)
    emd_median_corr = Column(Text)
    emd_max_corr = Column(Text)
    emd_min_corr = Column(Text)
    emd_std_corr = Column(Text)
    emd_corr_25 = Column(Text)
    emd_corr_50 = Column(Text)
    emd_corr_75 = Column(Text)
    emd_energ_entropy = Column(Text)
    emd_oi = Column(Text)
    emd_hi = Column(Text)

    profile = relationship('Profile', back_populates='emd_feature')


class HHTFeatureProfile(Base):
    __tablename__ = 'hht_feature_profile'

    id = Column(Integer, primary_key=True)
    profile_id = Column(Integer, ForeignKey('profile.id'))

    hht_inst_freq_mean = Column(Text)
    hht_inst_freq_med = Column(Text)
    hht_inst_freq_max = Column(Text)
    hht_inst_freq_min = Column(Text)
    hht_inst_freq_std = Column(Text)

    hht_inst_amp_mean = Column(Text)
    hht_inst_amp_med = Column(Text)
    hht_inst_amp_max = Column(Text)
    hht_inst_amp_min = Column(Text)
    hht_inst_amp_std = Column(Text)

    hht_mean_freq_mean = Column(Text)
    hht_mean_freq_med = Column(Text)
    hht_mean_freq_max = Column(Text)
    hht_mean_freq_min = Column(Text)
    hht_mean_freq_std = Column(Text)

    hht_mean_amp_mean = Column(Text)
    hht_mean_amp_med = Column(Text)
    hht_mean_amp_max = Column(Text)
    hht_mean_amp_min = Column(Text)
    hht_mean_amp_std = Column(Text)

    hht_marg_spec_mean = Column(Text)
    hht_marg_spec_med = Column(Text)
    hht_marg_spec_max = Column(Text)
    hht_marg_spec_min = Column(Text)
    hht_marg_spec_std = Column(Text)

    hht_teager_energ_mean = Column(Text)
    hht_teager_energ_med = Column(Text)
    hht_teager_energ_max = Column(Text)
    hht_teager_energ_min = Column(Text)
    hht_teager_energ_std = Column(Text)

    hht_hi = Column(Text)

    hht_dos_mean = Column(Text)
    hht_dos_med = Column(Text)
    hht_dos_max = Column(Text)
    hht_dos_min = Column(Text)
    hht_dos_std = Column(Text)

    hht_oi = Column(Text)

    hht_hsd_mean = Column(Text)
    hht_hsd_med = Column(Text)
    hht_hsd_max = Column(Text)
    hht_hsd_min = Column(Text)
    hht_hsd_std = Column(Text)

    hht_ci = Column(Text)

    profile = relationship('Profile', back_populates='hht_feature')