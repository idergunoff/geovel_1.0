import datetime
import json

from sqlalchemy import (create_engine, Column, Integer, String, Float, Boolean, DateTime, LargeBinary, ForeignKey,
                        Date, Text, text, literal_column, or_, func, Index, desc)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

DATABASE_NAME = 'geovel_db.sqlite'

engine = create_engine(f'sqlite:///{DATABASE_NAME}', echo=False)

Session = sessionmaker(bind=engine)
session = Session()

Base = declarative_base()


class GeoradarObject(Base):
    __tablename__ = 'georadar_object'

    id = Column(Integer, primary_key=True)
    title = Column(String)

    researches = relationship('Research', back_populates='object')
    grid = relationship('Grid', back_populates='object')
    h_wells = relationship('HorizontalWell', back_populates='object')
    explorations = relationship('Exploration', back_populates='object')
    set_points = relationship('SetPointsTrain', back_populates='object')



class Research(Base):
    __tablename__ = 'research'

    id = Column(Integer, primary_key=True)
    object_id = Column(Integer, ForeignKey('georadar_object.id'))
    date_research = Column(Date)

    object = relationship('GeoradarObject', back_populates='researches')
    profiles = relationship('Profile', back_populates='research')
    index_productivity = relationship('IndexProductivity', back_populates='research')


class Profile(Base):
    __tablename__ = 'profile'

    id = Column(Integer, primary_key=True)
    research_id = Column(Integer, ForeignKey('research.id'))
    title = Column(String)

    signal = Column(Text)
    signal_hash_md5 = Column(String)

    x_wgs = Column(Text)
    y_wgs = Column(Text)
    x_pulc = Column(Text)
    y_pulc = Column(Text)
    abs_relief = Column(Text)
    depth_relief = Column(Text)

    research = relationship('Research', back_populates='profiles')
    current = relationship('CurrentProfile', back_populates='profile')
    window = relationship('WindowProfile', back_populates='profile')
    min_max = relationship('CurrentProfileMinMax', back_populates='profile')
    layers = relationship('Layers', back_populates='profile')
    formations = relationship('Formation', back_populates='profile')
    bindings = relationship('Binding', back_populates='profile')
    velocity_formations = relationship('VelocityFormation', back_populates='profile')
    velocity_models = relationship('VelocityModel', back_populates='profile')
    deep_profiles = relationship('DeepProfile', back_populates='profile')
    markups_mlp = relationship('MarkupMLP', back_populates='profile')
    markups_reg = relationship('MarkupReg', back_populates='profile')
    intersections = relationship('Intersection', back_populates='profile')
    predictions = relationship('ProfileModelPrediction', back_populates='profile')

    wavelet_feature = relationship('WaveletFeatureProfile', back_populates='profile')
    fractal_feature = relationship('FractalFeatureProfile', back_populates='profile')
    entropy_feature = relationship('EntropyFeatureProfile', back_populates='profile')
    nonlinear_feature = relationship('NonlinearFeatureProfile', back_populates='profile')
    morphology_feature = relationship('MorphologyFeatureProfile', back_populates='profile')
    frequency_feature = relationship('FrequencyFeatureProfile', back_populates='profile')
    envelope_feature = relationship('EnvelopeFeatureProfile', back_populates='profile')
    autocorr_feature = relationship('AutocorrFeatureProfile', back_populates='profile')
    emd_feature = relationship('EMDFeatureProfile', back_populates='profile')
    hht_feature = relationship('HHTFeatureProfile', back_populates='profile')


class ProfileModelPrediction(Base):
    __tablename__ = 'profile_model_prediction'

    id = Column(Integer, primary_key=True)
    profile_id = Column(Integer, ForeignKey('profile.id'))
    model_id = Column(Integer)
    type_model = Column(String)
    prediction = Column(Text)

    profile = relationship('Profile', back_populates='predictions')
    binding_layer_predictions = relationship('BindingLayerPrediction', back_populates='prediction')
    corrected = relationship('PredictionCorrect', back_populates='prediction')
    index_productivity = relationship('IndexProductivity', back_populates='prediction')


class PredictionCorrect(Base):
    __tablename__ = 'prediction_correct'

    id = Column(Integer, primary_key=True)
    prediction_id = Column(Integer, ForeignKey('profile_model_prediction.id'))
    correct = Column(Text)

    prediction = relationship('ProfileModelPrediction', back_populates='corrected')


class IndexProductivity(Base):
    __tablename__ = 'index_productivity'

    id = Column(Integer, primary_key=True)
    research_id = Column(Integer, ForeignKey('research.id'))
    prediction_id = Column(Integer, ForeignKey('profile_model_prediction.id'))

    research = relationship('Research', back_populates='index_productivity')
    prediction = relationship('ProfileModelPrediction', back_populates='index_productivity')


class BindingLayerPrediction(Base):
    __tablename__ = 'binding_layer_prediction'

    id = Column(Integer, primary_key=True)
    prediction_id = Column(Integer, ForeignKey('profile_model_prediction.id'))
    layer_id = Column(Integer, ForeignKey('layers.id'))

    prediction = relationship('ProfileModelPrediction', back_populates='binding_layer_predictions')
    layer = relationship('Layers', back_populates='binding_layer_predictions')


class DeepProfile(Base):
    __tablename__ = 'deep_profile'

    id = Column(Integer, primary_key=True)
    profile_id = Column(Integer, ForeignKey('profile.id'))
    vel_model_id = Column(Integer, ForeignKey('velocity_model.id'))
    signal = Column(Text)

    profile = relationship('Profile', back_populates='deep_profiles')
    vel_model = relationship('VelocityModel', back_populates='deep_profile')
    deep_layers = relationship('DeepLayer', back_populates='deep_profile')


class DeepLayer(Base):
    __tablename__ = 'deep_layer'

    id = Column(Integer, primary_key=True)
    deep_profile_id = Column(Integer, ForeignKey('deep_profile.id'))
    vel_form_id = Column(Integer, ForeignKey('velocity_formation.id'))
    layer_line_thick = Column(Text)
    index = Column(Integer)     # номер по порядку сверху вниз

    deep_profile = relationship('DeepProfile', back_populates='deep_layers')
    vel_form = relationship('VelocityFormation', back_populates='deep_layer')


class Binding(Base):
    __tablename__ = 'binding'

    id = Column(Integer, primary_key=True)
    profile_id = Column(Integer, ForeignKey('profile.id'))
    boundary_id = Column(Integer, ForeignKey('boundary.id'))
    layer_id = Column(Integer, ForeignKey('layers.id'))
    index_measure = Column(Integer)

    profile = relationship("Profile", back_populates="bindings")
    boundary = relationship("Boundary", back_populates="bindings")
    layer = relationship('Layers', back_populates="bindings")


class VelocityFormation(Base):
    __tablename__ = 'velocity_formation'

    id = Column(Integer, primary_key=True)
    profile_id = Column(Integer, ForeignKey('profile.id'))
    vel_model_id = Column(Integer, ForeignKey('velocity_model.id'))
    layer_top = Column(String)
    layer_bottom = Column(String)
    color = Column(String)
    velocity = Column(String)
    index = Column(Integer)

    profile = relationship("Profile", back_populates="velocity_formations")
    velocity_model = relationship("VelocityModel", back_populates="velocity_formations")
    deep_layer = relationship('DeepLayer', back_populates='vel_form')


class VelocityModel(Base):
    __tablename__ = 'velocity_model'

    id = Column(Integer, primary_key=True)
    profile_id = Column(Integer, ForeignKey('profile.id'))
    title = Column(String)

    profile = relationship("Profile", back_populates="velocity_models")
    velocity_formations = relationship("VelocityFormation", back_populates="velocity_model")
    deep_profile = relationship('DeepProfile', back_populates='vel_model')
    current_velocity_model = relationship('CurrentVelocityModel', back_populates='vel_model')


class CurrentProfile(Base):
    __tablename__ = 'current_profile'

    id = Column(Integer, primary_key=True)
    profile_id = Column(Integer, ForeignKey('profile.id'))
    signal = Column(Text)

    profile = relationship('Profile', back_populates='current')


class CurrentProfileMinMax(Base):
    __tablename__ = 'current_profile_min_max'

    id = Column(Integer, primary_key=True)
    profile_id = Column(Integer, ForeignKey('profile.id'))
    signal = Column(Text)

    profile = relationship('Profile', back_populates='min_max')


class CurrentVelocityModel(Base):
    __tablename__ = 'current_velocity_model'

    id = Column(Integer, primary_key=True)
    active = Column(Boolean, default=False)
    vel_model_id = Column(Integer, ForeignKey('velocity_model.id'))
    scale = Column(Float)

    vel_model = relationship('VelocityModel', back_populates='current_velocity_model')


class Grid(Base):
    __tablename__ = 'grid'

    id = Column(Integer, primary_key=True)
    object_id = Column(Integer, ForeignKey('georadar_object.id'))
    grid_table_uf = Column(Text)
    grid_table_m = Column(Text)
    grid_table_r = Column(Text)

    object = relationship('GeoradarObject', back_populates='grid')


class CommonGrid(Base):
    __tablename__ = 'common_grid'

    id = Column(Integer, primary_key=True)
    title = Column(String)
    type = Column(String)
    grid_table = Column(Text)


class FFTSpectr(Base):
    __tablename__ = 'fft_spectr'

    id = Column(Integer, primary_key=True)
    spectr = Column(Text)


class WindowProfile(Base):
    __tablename__ = 'window_profile'

    id = Column(Integer, primary_key=True)
    profile_id = Column(Integer, ForeignKey('profile.id'))
    signal = Column(Text)

    profile = relationship('Profile', back_populates='window')


class Layers(Base):
    __tablename__ = 'layers'

    id = Column(Integer, primary_key=True)
    profile_id = Column(Integer, ForeignKey('profile.id'))
    layer_title = Column(String)
    layer_line = Column(Text)

    profile = relationship('Profile', back_populates='layers')
    points = relationship('PointsOfLayer', back_populates='layer')
    formation_up = relationship('Formation', back_populates='layer_up', foreign_keys='Formation.up')
    formation_down = relationship('Formation', back_populates='layer_down', foreign_keys='Formation.down')
    bindings = relationship('Binding', back_populates='layer')
    binding_layer_predictions = relationship('BindingLayerPrediction', back_populates='layer')


class PointsOfLayer(Base):
    __tablename__ = 'points_of_layer'

    id = Column(Integer, primary_key=True)
    layer_id = Column(Integer, ForeignKey('layers.id'))
    point_x = Column(Float)
    point_y = Column(Float)

    layer = relationship('Layers', back_populates='points')


class Formation(Base):
    __tablename__ = 'formation'

    id = Column(Integer, primary_key=True)
    profile_id = Column(Integer, ForeignKey('profile.id'))
    title = Column(String)
    up = Column(Integer, ForeignKey('layers.id'))
    down = Column(Integer, ForeignKey('layers.id'))

    T_top = Column(Text)
    T_bottom = Column(Text)
    dT = Column(Text)

    A_top = Column(Text)
    A_bottom = Column(Text)
    dA = Column(Text)
    A_sum = Column(Text)
    A_mean = Column(Text)

    dVt = Column(Text)
    Vt_top = Column(Text)
    Vt_sum = Column(Text)
    Vt_mean = Column(Text)

    dAt = Column(Text)
    At_top = Column(Text)
    At_sum = Column(Text)
    At_mean = Column(Text)

    dPht = Column(Text)
    Pht_top = Column(Text)
    Pht_sum = Column(Text)
    Pht_mean = Column(Text)

    Wt_top = Column(Text)
    Wt_mean = Column(Text)
    Wt_sum = Column(Text)

    width = Column(Text)
    top = Column(Text)
    land = Column(Text)
    speed = Column(Text)
    speed_cover = Column(Text)

    skew = Column(Text)
    kurt = Column(Text)
    std = Column(Text)
    k_var = Column(Text)

    A_max = Column(Text)
    Vt_max = Column(Text)
    At_max = Column(Text)
    Pht_max = Column(Text)
    Wt_max = Column(Text)

    A_T_max = Column(Text)
    Vt_T_max = Column(Text)
    At_T_max = Column(Text)
    Pht_T_max = Column(Text)
    Wt_T_max = Column(Text)

    A_Sn = Column(Text)
    Vt_Sn = Column(Text)
    At_Sn = Column(Text)
    Pht_Sn = Column(Text)
    Wt_Sn = Column(Text)

    A_wmf = Column(Text)
    Vt_wmf = Column(Text)
    At_wmf = Column(Text)
    Pht_wmf = Column(Text)
    Wt_wmf = Column(Text)

    A_Qf = Column(Text)
    Vt_Qf = Column(Text)
    At_Qf = Column(Text)
    Pht_Qf = Column(Text)
    Wt_Qf = Column(Text)

    A_Sn_wmf = Column(Text)
    Vt_Sn_wmf = Column(Text)
    At_Sn_wmf = Column(Text)
    Pht_Sn_wmf = Column(Text)
    Wt_Sn_wmf = Column(Text)

    CRL_top = Column(Text)
    CRL_bottom = Column(Text)
    dCRL = Column(Text)
    CRL_sum = Column(Text)
    CRL_mean = Column(Text)
    CRL_max = Column(Text)
    CRL_T_max = Column(Text)
    CRL_Sn = Column(Text)
    CRL_wmf = Column(Text)
    CRL_Qf = Column(Text)
    CRL_Sn_wmf = Column(Text)

    CRL_skew = Column(Text)
    CRL_kurt = Column(Text)
    CRL_std = Column(Text)
    CRL_k_var = Column(Text)

    k_r = Column(Text)

    profile = relationship('Profile', back_populates='formations')
    layer_up = relationship('Layers', back_populates='formation_up', foreign_keys=[up])
    layer_down = relationship('Layers', back_populates='formation_down', foreign_keys=[down])
    markups_mlp = relationship('MarkupMLP', back_populates='formation')
    markups_reg = relationship('MarkupReg', back_populates='formation')
    model = relationship('FormationAI', back_populates='formation')
    wavelet_feature = relationship('WaveletFeature', back_populates='formation')
    fractal_feature = relationship('FractalFeature', back_populates='formation')
    entropy_feature = relationship('EntropyFeature', back_populates='formation')
    nonlinear_feature = relationship('NonlinearFeature', back_populates='formation')
    morphology_feature = relationship('MorphologyFeature', back_populates='formation')
    frequency_feature = relationship('FrequencyFeature', back_populates='formation')
    envelope_feature = relationship('EnvelopeFeature', back_populates='formation')
    autocorr_feature = relationship('AutocorrFeature', back_populates='formation')
    emd_feature = relationship('EMDFeature', back_populates='formation')
    hht_feature = relationship('HHTFeature', back_populates='formation')


class WaveletFeature(Base):
    __tablename__ = 'wavelet_feature'
    # __table_args__ = {'info': dict(is_view=True)}dd

    id = Column(Integer, primary_key=True)
    formation_id = Column(Integer, ForeignKey('formation.id'))

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

    formation = relationship('Formation', back_populates='wavelet_feature')

class FractalFeature(Base):
    __tablename__ = 'fractal_feature'

    id = Column(Integer, primary_key=True)
    formation_id = Column(Integer, ForeignKey('formation.id'))
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

    formation = relationship('Formation', back_populates='fractal_feature')


class EntropyFeature(Base):
    __tablename__ = 'entropy_feature'

    id = Column(Integer, primary_key=True)
    formation_id = Column(Integer, ForeignKey('formation.id'))
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

    formation = relationship('Formation', back_populates='entropy_feature')


class NonlinearFeature(Base):
    __tablename__ = 'nonlinear_feature'

    id = Column(Integer, primary_key=True)
    formation_id = Column(Integer, ForeignKey('formation.id'))
    nln_corr_dim = Column(Text)
    nln_rec_rate = Column(Text)
    nln_determin = Column(Text)
    nln_avg_diag = Column(Text)
    nln_hirsh = Column(Text)

    formation = relationship('Formation', back_populates='nonlinear_feature')


class MorphologyFeature(Base):
    __tablename__ = 'morphology_feature'

    id = Column(Integer, primary_key=True)
    formation_id = Column(Integer, ForeignKey('formation.id'))
    mph_peak_num = Column(Text)
    mph_peak_width = Column(Text)
    mph_peak_amp_ratio = Column(Text)
    mph_peak_asymm = Column(Text)
    mph_peak_steep = Column(Text)
    mph_erosion = Column(Text)
    mph_dilation = Column(Text)

    formation = relationship('Formation', back_populates='morphology_feature')


class FrequencyFeature(Base):
    __tablename__ = 'frequency_feature'

    id = Column(Integer, primary_key=True)
    formation_id = Column(Integer, ForeignKey('formation.id'))
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

    formation = relationship('Formation', back_populates='frequency_feature')


class EnvelopeFeature(Base):
    __tablename__ = 'envelope_feature'

    id = Column(Integer, primary_key=True)
    formation_id = Column(Integer, ForeignKey('formation.id'))
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

    formation = relationship('Formation', back_populates='envelope_feature')
    
    
class AutocorrFeature(Base):
    __tablename__ = 'autocorr_feature'

    id = Column(Integer, primary_key=True)
    formation_id = Column(Integer, ForeignKey('formation.id'))
    acf_first_min = Column(Text)
    acf_lag_10 = Column(Text)
    acf_decay = Column(Text)
    acf_integral = Column(Text)
    acf_peak_width = Column(Text)
    acf_ratio = Column(Text)

    formation = relationship('Formation', back_populates='autocorr_feature')


class EMDFeature(Base):
    __tablename__ = 'emd_feature'

    id = Column(Integer, primary_key=True)
    formation_id = Column(Integer, ForeignKey('formation.id'))
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

    formation = relationship('Formation', back_populates='emd_feature')


class HHTFeature(Base):
    __tablename__ = 'hht_feature'

    id = Column(Integer, primary_key=True)
    formation_id = Column(Integer, ForeignKey('formation.id'))

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

    formation = relationship('Formation', back_populates='hht_feature')


class Well(Base):
    __tablename__ = 'well'

    id = Column(Integer, primary_key=True)
    name = Column(String)
    x_coord = Column(Float)
    y_coord = Column(Float)
    alt = Column(Float)

    boundaries = relationship("Boundary", back_populates="well")
    well_optionally = relationship("WellOptionally", back_populates="well")
    well_logs = relationship("WellLog", back_populates="well")
    markups_mlp = relationship('MarkupMLP', back_populates='well')
    markups_reg = relationship('MarkupReg', back_populates='well')


class Boundary(Base):
    __tablename__ = 'boundary'

    id = Column(Integer, primary_key=True)
    well_id = Column(Integer, ForeignKey('well.id'))
    depth = Column(Float)
    title = Column(String)

    well = relationship("Well", back_populates="boundaries")
    bindings = relationship("Binding", back_populates='boundary')


class WellOptionally(Base):
    __tablename__ = 'well_optionally'

    id = Column(Integer, primary_key=True)
    well_id = Column(Integer, ForeignKey('well.id'))
    option = Column(String)
    value = Column(String)

    well = relationship("Well", back_populates="well_optionally")


class WellLog(Base):
    __tablename__ = 'well_log'

    id = Column(Integer, primary_key=True)
    well_id = Column(Integer, ForeignKey('well.id'))
    curve_name = Column(String)
    curve_data = Column(Text)
    begin = Column(Float)
    end = Column(Float)
    step = Column(Float)
    description = Column(Text)

    well = relationship("Well", back_populates="well_logs")


#####################################################
######################  MLP  ########################
#####################################################


class AnalysisMLP(Base):
    __tablename__ = 'analysis_mlp'

    id = Column(Integer, primary_key=True)
    title = Column(String)
    data = Column(Text)
    up_data = Column(Boolean, default=False)

    parameters = relationship('ParameterMLP', back_populates='analysis')
    markers = relationship('MarkerMLP', back_populates='analysis')
    markups = relationship('MarkupMLP', back_populates='analysis')
    trained_models = relationship('TrainedModelClass', back_populates='analysis')
    exceptions = relationship('ExceptionMLP', back_populates='analysis')
    pareto_analysis = relationship('ParetoAnalysis', back_populates='analysis_mlp')



class ParameterMLP(Base):
    __tablename__ = 'parameter_mlp'

    id = Column(Integer, primary_key=True)
    analysis_id = Column(Integer, ForeignKey('analysis_mlp.id'))
    parameter = Column(String)

    analysis = relationship('AnalysisMLP', back_populates='parameters')


class ExceptionMLP(Base):
    __tablename__ = 'exception_mlp'

    id = Column(Integer, primary_key=True)
    analysis_id = Column(Integer, ForeignKey('analysis_mlp.id'))
    except_signal = Column(String, default="")
    except_crl = Column(String, default="")

    analysis = relationship('AnalysisMLP', back_populates='exceptions')


class MarkerMLP(Base):
    __tablename__ = 'marker_mlp'

    id = Column(Integer, primary_key=True)
    analysis_id = Column(Integer, ForeignKey('analysis_mlp.id'))
    title = Column(String)
    color = Column(String)

    analysis = relationship('AnalysisMLP', back_populates='markers')
    markups = relationship('MarkupMLP', back_populates='marker')


class MarkupMLP(Base):
    __tablename__ = 'markup_mlp'

    id = Column(Integer, primary_key=True)
    analysis_id = Column(Integer, ForeignKey('analysis_mlp.id'))
    well_id = Column(Integer, ForeignKey('well.id'))    # возможно не нужно
    profile_id = Column(Integer, ForeignKey('profile.id'))
    formation_id = Column(Integer, ForeignKey('formation.id'))
    marker_id = Column(Integer, ForeignKey('marker_mlp.id'))
    list_measure = Column(Text)
    list_fake = Column(Text)
    type_markup = Column(String)

    analysis = relationship('AnalysisMLP', back_populates='markups')
    well = relationship("Well", back_populates="markups_mlp")
    profile = relationship("Profile", back_populates="markups_mlp")
    formation = relationship("Formation", back_populates="markups_mlp")
    marker = relationship("MarkerMLP", back_populates="markups")


class TrainedModelClass(Base):
    __tablename__ = 'trained_model_class'

    id = Column(Integer, primary_key=True)
    analysis_id = Column(Integer, ForeignKey('analysis_mlp.id'))
    title = Column(String)
    path_model = Column(String)
    list_params = Column(Text)
    except_signal = Column(String, default="")
    except_crl = Column(String, default="")
    comment = Column(Text)

    analysis = relationship('AnalysisMLP', back_populates='trained_models')
    model_mask = relationship('TrainedModelClassMask', back_populates='model')


class TrainedModelClassMask(Base):
    __tablename__ = 'trained_model_class_mask'

    id = Column(Integer, primary_key=True)
    model_id = Column(Integer, ForeignKey('trained_model_class.id'))
    mask_id = Column(Integer, ForeignKey('parameter_mask.id'))

    model = relationship('TrainedModelClass', back_populates='model_mask')
    mask = relationship('ParameterMask', back_populates='model_mask')



#####################################################
################## Regression #######################
#####################################################


class AnalysisReg(Base):
    __tablename__ = 'analysis_reg'

    id = Column(Integer, primary_key=True)
    title = Column(String)
    data = Column(Text)
    up_data = Column(Boolean, default=False)

    parameters = relationship('ParameterReg', back_populates='analysis')
    markups = relationship('MarkupReg', back_populates='analysis')
    trained_models = relationship('TrainedModelReg', back_populates='analysis')
    exceptions = relationship('ExceptionReg', back_populates='analysis')


class ParameterReg(Base):
    __tablename__ = 'parameter_reg'

    id = Column(Integer, primary_key=True)
    analysis_id = Column(Integer, ForeignKey('analysis_reg.id'))
    parameter = Column(String)

    analysis = relationship('AnalysisReg', back_populates='parameters')


class ExceptionReg(Base):
    __tablename__ = 'exception_reg'

    id = Column(Integer, primary_key=True)
    analysis_id = Column(Integer, ForeignKey('analysis_reg.id'))
    except_signal = Column(String, default="")
    except_crl = Column(String, default="")

    analysis = relationship('AnalysisReg', back_populates='exceptions')

class MarkupReg(Base):
    __tablename__ = 'markup_reg'

    id = Column(Integer, primary_key=True)
    analysis_id = Column(Integer, ForeignKey('analysis_reg.id'))
    well_id = Column(Integer, ForeignKey('well.id'))    # возможно не нужно
    profile_id = Column(Integer, ForeignKey('profile.id'))
    formation_id = Column(Integer, ForeignKey('formation.id'))
    target_value = Column(Float)
    list_measure = Column(Text)
    list_fake = Column(Text)
    type_markup = Column(String)

    analysis = relationship('AnalysisReg', back_populates='markups')
    well = relationship("Well", back_populates="markups_reg")
    profile = relationship("Profile", back_populates="markups_reg")
    formation = relationship("Formation", back_populates="markups_reg")


class TrainedModelReg(Base):
    __tablename__ = 'trained_model_reg'

    id = Column(Integer, primary_key=True)
    analysis_id = Column(Integer, ForeignKey('analysis_reg.id'))
    title = Column(String)
    path_model = Column(String)
    path_scaler = Column(String)
    list_params = Column(Text)
    except_signal = Column(String, default="")
    except_crl = Column(String, default="")
    comment = Column(Text)

    analysis = relationship('AnalysisReg', back_populates='trained_models')


class LineupTrain(Base):
    __tablename__ = 'lineup_train'

    id = Column(Integer, primary_key=True)
    type_ml = Column(String)
    analysis_id = Column(Integer, ForeignKey('analysis_mlp.id'))
    list_param = Column(Text)
    list_param_short = Column(Text)
    except_signal = Column(String)
    except_crl = Column(String)
    text_model = Column(Text)
    model_name = Column(String)
    pipe = Column(LargeBinary)
    over_sampling = Column(String)
    random_seed = Column(Integer, default=0)
    cvw = Column(Boolean, default=False)


#####################################################
###################  Monitoring  ####################
#####################################################


class HorizontalWell(Base):
    __tablename__ = 'horizontal_well'

    id = Column(Integer, primary_key=True)
    object_id = Column(Integer, ForeignKey('georadar_object.id'))
    title = Column(String)
    x_coord = Column(Float)
    y_coord = Column(Float)
    alt = Column(Float)

    object = relationship("GeoradarObject", back_populates="h_wells")
    parameters = relationship('ParameterHWell', back_populates='h_well')
    thermograms = relationship('Thermogram', back_populates='h_well')


class ParameterHWell(Base):
    __tablename__ = 'parameter_h_well'

    id = Column(Integer, primary_key=True)
    h_well_id = Column(Integer, ForeignKey('horizontal_well.id'))
    parameter = Column(String)
    data = Column(Text)

    h_well = relationship("HorizontalWell", back_populates="parameters")


class Thermogram(Base):
    __tablename__ = 'thermogram'

    id = Column(Integer, primary_key=True)
    h_well_id = Column(Integer, ForeignKey('horizontal_well.id'))
    date_time = Column(DateTime)
    therm_data = Column(Text)

    h_well = relationship("HorizontalWell", back_populates="thermograms")
    intersections = relationship('Intersection', back_populates='thermogram')

# Создаем индекс на столбец "id"
# index_id = Index('idx_id', Thermogram.id)
# index_id.create(engine)

class Intersection(Base):
    """ Точки пересечения термограмм с профилями """
    __tablename__ = 'intersection'

    id = Column(Integer, primary_key=True)
    therm_id = Column(Integer, ForeignKey('thermogram.id'))
    profile_id = Column(Integer, ForeignKey('profile.id'))
    name = Column(String)
    x_coord = Column(Float)
    y_coord = Column(Float)
    temperature = Column(Float)
    i_therm = Column(Integer)
    i_profile = Column(Integer)

    profile = relationship("Profile", back_populates="intersections")
    thermogram = relationship("Thermogram", back_populates="intersections")


#####################################################
##################  Formation AI  ###################
#####################################################


class ModelFormationAI(Base):
    __tablename__ = 'model_formation_ai'

    id = Column(Integer, primary_key=True)
    title = Column(String)

    formations = relationship('FormationAI', back_populates='model')


class FormationAI(Base):
    __tablename__ = 'formation_ai'

    id = Column(Integer, primary_key=True)
    model_id = Column(Integer, ForeignKey('model_formation_ai.id'))
    formation_id = Column(Integer, ForeignKey('formation.id'))

    model = relationship('ModelFormationAI', back_populates='formations')
    formation = relationship('Formation', back_populates='model')


class TrainedModel(Base):
    __tablename__ = 'trained_model'

    id = Column(Integer, primary_key=True)
    title = Column(String)
    path_top = Column(String)
    path_bottom = Column(String)



###################  EXPLORATION  ##################



class Exploration(Base):
    __tablename__ = 'exploration'

    id = Column(Integer, primary_key=True)
    title = Column(String)
    date_explore = Column(Date)
    object_id = Column(Integer, ForeignKey('georadar_object.id'))

    object = relationship("GeoradarObject", back_populates="explorations")
    parameters = relationship('ParameterExploration', back_populates='exploration')
    points = relationship('SetPoints', back_populates='exploration')
    # analysis = relationship('AnalysisExploration', back_populates='exploration')


class ParameterExploration(Base):
    __tablename__ = 'parameter_exploration'

    id = Column(Integer, primary_key=True)
    exploration_id = Column(Integer, ForeignKey('exploration.id'))
    parameter = Column(String)

    exploration = relationship("Exploration", back_populates="parameters")
    grids = relationship('GridExploration', back_populates='param')
    points = relationship('ParameterPoint', back_populates='param')
    analysis = relationship('ParameterAnalysisExploration', back_populates='param')


class SetPoints(Base):
    __tablename__ = 'set_points'

    id = Column(Integer, primary_key=True)
    exploration_id = Column(Integer, ForeignKey('exploration.id'))
    title = Column(String)

    exploration = relationship("Exploration", back_populates="points")
    points = relationship('PointExploration', back_populates='set_point')
    # analysis = relationship('AnalysisExploration', back_populates='set_point')


class PointExploration(Base):
    __tablename__ = 'point_exploration'

    id = Column(Integer, primary_key=True)
    set_points_id = Column(Integer, ForeignKey('set_points.id'))
    x_coord = Column(Float)
    y_coord = Column(Float)
    title = Column(String)

    set_point = relationship("SetPoints", back_populates="points")
    parameters = relationship('ParameterPoint', back_populates='point')


class SetPointsTrain(Base):
    __tablename__ = 'set_points_train'

    id = Column(Integer, primary_key=True)
    object_id = Column(Integer, ForeignKey('georadar_object.id'))
    title = Column(String)

    object = relationship("GeoradarObject", back_populates="set_points")
    points = relationship('PointTrain', back_populates='set_points_train')


class PointTrain(Base):
    __tablename__ = 'point_train'

    id = Column(Integer, primary_key=True)
    set_points_train_id = Column(Integer, ForeignKey('set_points_train.id'))
    title = Column(String)
    x_coord = Column(Float)
    y_coord = Column(Float)
    target = Column(Float)

    set_points_train = relationship("SetPointsTrain", back_populates="points")


class ParameterPoint(Base):
    __tablename__ = 'parameter_point'
    id = Column(Integer, primary_key=True)
    point_id = Column(Integer, ForeignKey('point_exploration.id'))
    param_id = Column(Integer, ForeignKey('parameter_exploration.id'))
    value = Column(Float)

    point = relationship("PointExploration", back_populates="parameters")
    param = relationship("ParameterExploration", back_populates="points")


class GridExploration(Base):
    __tablename__ = 'grid_exploration'

    id = Column(Integer, primary_key=True)
    param_id = Column(Integer, ForeignKey('parameter_exploration.id'))
    grid = Column(Text)
    title = Column(String)

    param = relationship("ParameterExploration", back_populates="grids")


class AnalysisExploration(Base):
    __tablename__ = 'analysis_exploration'

    id = Column(Integer, primary_key=True)
    # train_points_id = Column(Integer, ForeignKey('set_points.id'))
    title = Column(String)
    type_analysis = Column(String)
    data = Column(Text)
    up_data = Column(Boolean, default=False)

    # exploration = relationship("Exploration", back_populates="analysis")
    parameters = relationship('ParameterAnalysisExploration', back_populates='analysis')
    # set_point = relationship("SetPoints", back_populates="analysis")
    geo_parameters = relationship('GeoParameterAnalysisExploration', back_populates='analysis')
    trained_models = relationship('TrainedModelExploration', back_populates='analysis')


class ParameterAnalysisExploration(Base):
    __tablename__ = 'parameter_analysis_exploration'

    id = Column(Integer, primary_key=True)
    title = Column(String)
    analysis_id = Column(Integer, ForeignKey('analysis_exploration.id'))
    parameter_id = Column(Integer, ForeignKey('parameter_exploration.id'))

    analysis = relationship("AnalysisExploration", back_populates="parameters")
    param = relationship("ParameterExploration", back_populates="analysis")


class GeoParameterAnalysisExploration(Base):
    __tablename__ = 'geo_parameter_analysis_exploration'

    id = Column(Integer, primary_key=True)
    param = Column(String)
    analysis_id = Column(Integer, ForeignKey('analysis_exploration.id'))

    analysis = relationship("AnalysisExploration", back_populates="geo_parameters")


class TrainedModelExploration(Base):
    __tablename__ = 'trained_model_exploration'

    id = Column(Integer, primary_key=True)
    analysis_id = Column(Integer, ForeignKey('analysis_exploration.id'))
    title = Column(String)
    path_model = Column(String)
    list_params = Column(Text)
    comment = Column(Text)

    analysis = relationship('AnalysisExploration', back_populates='trained_models')


############## GEOCHEM ################


class Geochem(Base):
    __tablename__ = 'geochem'

    id = Column(Integer, primary_key=True)
    title = Column(String)

    g_parameters = relationship("GeochemParameter", back_populates="geochem")
    g_points = relationship("GeochemPoint", back_populates="geochem")
    g_wells = relationship("GeochemWell", back_populates="geochem")
    makets = relationship("GeochemMaket", back_populates="geochem")


class GeochemParameter(Base):
    __tablename__ = 'geochem_parameter'

    id = Column(Integer, primary_key=True)
    geochem_id = Column(Integer, ForeignKey('geochem.id'))
    title = Column(String)

    geochem = relationship("Geochem", back_populates="g_parameters")
    g_point_values = relationship("GeochemPointValue", back_populates="g_param")
    g_well_point_values = relationship("GeochemWellPointValue", back_populates="g_param")
    train_params = relationship("GeochemTrainParameter", back_populates="param")


class GeochemPoint(Base):
    __tablename__ = 'geochem_point'

    id = Column(Integer, primary_key=True)
    geochem_id = Column(Integer, ForeignKey('geochem.id'))
    title = Column(String)
    x_coord = Column(Float)
    y_coord = Column(Float)
    fake = Column(Boolean, default=False)

    geochem = relationship("Geochem", back_populates="g_points")
    g_point_values = relationship("GeochemPointValue", back_populates="g_point")
    train_points = relationship("GeochemTrainPoint", back_populates="point")



class GeochemWell(Base):
    __tablename__ = 'geochem_well'

    id = Column(Integer, primary_key=True)
    geochem_id = Column(Integer, ForeignKey('geochem.id'))
    title = Column(String)
    color = Column(String)

    geochem = relationship("Geochem", back_populates="g_wells")
    g_w_points = relationship("GeochemWellPoint", back_populates="g_well")


class GeochemWellPoint(Base):
    __tablename__ = 'geochem_well_point'

    id = Column(Integer, primary_key=True)
    g_well_id = Column(Integer, ForeignKey('geochem_well.id'))
    title = Column(String)
    x_coord = Column(Float)
    y_coord = Column(Float)

    g_well = relationship("GeochemWell", back_populates="g_w_points")
    g_well_point_values = relationship("GeochemWellPointValue", back_populates="g_well_point")
    train_points = relationship("GeochemTrainPoint", back_populates="well_point")


class GeochemPointValue(Base):
    __tablename__ = 'geochem_point_value'

    id = Column(Integer, primary_key=True)
    g_point_id = Column(Integer, ForeignKey('geochem_point.id'))
    g_param_id = Column(Integer, ForeignKey('geochem_parameter.id'))
    value = Column(Float)

    g_point = relationship("GeochemPoint", back_populates="g_point_values")
    g_param = relationship("GeochemParameter", back_populates="g_point_values")


class GeochemWellPointValue(Base):
    __tablename__ = 'geochem_well_point_value'

    id = Column(Integer, primary_key=True)
    g_well_point_id = Column(Integer, ForeignKey('geochem_well_point.id'))
    g_param_id = Column(Integer, ForeignKey('geochem_parameter.id'))
    value = Column(Float)


    g_well_point = relationship("GeochemWellPoint", back_populates="g_well_point_values")
    g_param = relationship("GeochemParameter", back_populates="g_well_point_values")


class GeochemMaket(Base):
    __tablename__ = 'geochem_maket'

    id = Column(Integer, primary_key=True)
    geochem_id = Column(Integer, ForeignKey('geochem.id'))
    title = Column(String)

    geochem = relationship("Geochem", back_populates="makets")
    categories = relationship("GeochemCategory", back_populates="maket")
    train_params = relationship("GeochemTrainParameter", back_populates="maket")
    g_trained_models = relationship("GeochemTrainedModel", back_populates="maket")


class GeochemCategory(Base):
    __tablename__ = 'geochem_category'

    id = Column(Integer, primary_key=True)
    maket_id = Column(Integer, ForeignKey('geochem_maket.id'))
    title = Column(String)
    color = Column(String)

    maket = relationship("GeochemMaket", back_populates="categories")
    train_points = relationship("GeochemTrainPoint", back_populates="category")


class GeochemTrainPoint(Base):
    __tablename__ = 'geochem_train_point'

    id = Column(Integer, primary_key=True)

    cat_id = Column(Integer, ForeignKey('geochem_category.id'))
    title = Column(String)
    type_point = Column(String, default='well')
    point_well_id = Column(Integer, ForeignKey('geochem_well_point.id'))
    point_id = Column(Integer, ForeignKey('geochem_point.id'))
    fake = Column(Boolean, default=False)

    category = relationship("GeochemCategory", back_populates="train_points")
    point = relationship("GeochemPoint", back_populates="train_points")
    well_point = relationship("GeochemWellPoint", back_populates="train_points")


class GeochemTrainParameter(Base):
    __tablename__ = 'geochem_train_parameter'

    id = Column(Integer, primary_key=True)
    maket_id = Column(Integer, ForeignKey('geochem_maket.id'))
    param_id = Column(Integer, ForeignKey('geochem_parameter.id'))

    maket = relationship("GeochemMaket", back_populates="train_params")
    param = relationship("GeochemParameter", back_populates="train_params")


class GeochemTrainedModel(Base):
    __tablename__ = 'geochem_trained_model'

    id = Column(Integer, primary_key=True)
    maket_id = Column(Integer, ForeignKey('geochem_maket.id'))
    title = Column(String)
    path_model = Column(String)
    list_params = Column(Text)
    comment = Column(Text)

    maket = relationship('GeochemMaket', back_populates='g_trained_models')


######### PARETO #########

class ParetoAnalysis(Base):
    __tablename__ = 'pareto_analysis'

    id = Column(Integer, primary_key=True)
    analysis_mlp_id = Column(Integer, ForeignKey('analysis_mlp.id'))
    n_iter = Column(Integer)
    problem_type = Column(String) # 'MINIMIZE' or 'MAXIMIZE' or 'NO'
    start_params = Column(Text)

    analysis_mlp = relationship("AnalysisMLP", back_populates="pareto_analysis")
    pareto_results = relationship("ParetoResult", back_populates="pareto_analysis")


class ParetoResult(Base):
    __tablename__ = 'pareto_result'

    id = Column(Integer, primary_key=True)
    pareto_analysis_id = Column(Integer, ForeignKey('pareto_analysis.id'))
    pareto_data = Column(Text)
    distance = Column(Float)

    pareto_analysis = relationship("ParetoAnalysis", back_populates="pareto_results")


class ParameterMask(Base):
    __tablename__ = 'parameter_mask'

    id = Column(Integer, primary_key=True)
    count_param = Column(Integer)
    mask = Column(Text)
    mask_info = Column(Text)

    model_mask = relationship("TrainedModelClassMask", back_populates="mask")


