import datetime
import json
from contextlib import contextmanager

from sqlalchemy import (create_engine, Column, Integer, String, Float, Boolean, DateTime, LargeBinary, ForeignKey,
                        Date, Text, text, literal_column, or_, func, Index, desc, select, update, bindparam, literal)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship, selectinload

DATABASE_NAME = 'geovel_local:123qaz456wsx@ovz2.j56960636.0n03n.vps.myjino.ru:49221/geovel_remote'

engine_remote = create_engine(f'postgresql+psycopg2://{DATABASE_NAME}', echo=False)
Session_remote = sessionmaker(bind=engine_remote)


@contextmanager
def get_session():
    session_remote = Session_remote()
    try:
        yield session_remote
        session_remote.commit()
    except Exception:
        session_remote.rollback()
        raise
    finally:
        session_remote.close()


BaseRDB = declarative_base()


class GeoradarObjectRDB(BaseRDB):
    __tablename__ = 'georadar_object_rdb'

    id = Column(Integer, primary_key=True)
    title = Column(String)

    researches = relationship('ResearchRDB', back_populates='object')


class ResearchRDB(BaseRDB):
    __tablename__ = 'research_rdb'

    id = Column(Integer, primary_key=True)
    object_id = Column(Integer, ForeignKey('georadar_object_rdb.id'))
    date_research = Column(Date)

    object = relationship('GeoradarObjectRDB', back_populates='researches')
    profiles = relationship('ProfileRDB', back_populates='research')



class ProfileRDB(BaseRDB):
    __tablename__ = 'profile_rdb'

    id = Column(Integer, primary_key=True)
    research_id = Column(Integer, ForeignKey('research_rdb.id'))
    title = Column(String)

    signal = Column(Text)
    signal_hash_md5 = Column(String)

    x_wgs = Column(Text)
    y_wgs = Column(Text)
    x_pulc = Column(Text)
    y_pulc = Column(Text)
    abs_relief = Column(Text)
    depth_relief = Column(Text)

    research = relationship('ResearchRDB', back_populates='profiles')
    formations = relationship('FormationRDB', back_populates='profile')
    markups_mlp = relationship('MarkupMLPRDB', back_populates='profile')
    markups_reg = relationship('MarkupRegRDB', back_populates='profile')
    entropy_feature = relationship('EntropyFeatureProfileRDB', back_populates='profile')


class WellRDB(BaseRDB):
    __tablename__ = 'well_rdb'

    id = Column(Integer, primary_key=True)
    name = Column(String)
    x_coord = Column(Float)
    y_coord = Column(Float)
    alt = Column(Float)
    well_hash = Column(String)
    ignore = Column(Boolean)

    boundaries = relationship("BoundaryRDB", back_populates="well")
    well_optionally = relationship("WellOptionallyRDB", back_populates="well")
    well_logs = relationship("WellLogRDB", back_populates="well")
    markups_mlp = relationship('MarkupMLPRDB', back_populates='well')
    markups_reg = relationship('MarkupRegRDB', back_populates='well')

class BoundaryRDB(BaseRDB):
    __tablename__ = 'boundary_rdb'

    id = Column(Integer, primary_key=True)
    well_id = Column(Integer, ForeignKey('well_rdb.id'))
    depth = Column(Float)
    title = Column(String)

    well = relationship("WellRDB", back_populates="boundaries")

class WellOptionallyRDB(BaseRDB):
    __tablename__ = 'well_optionally_rdb'

    id = Column(Integer, primary_key=True)
    well_id = Column(Integer, ForeignKey('well_rdb.id'))
    option = Column(String)
    value = Column(String)

    well = relationship("WellRDB", back_populates="well_optionally")

class WellLogRDB(BaseRDB):
    __tablename__ = 'well_log_rdb'

    id = Column(Integer, primary_key=True)
    well_id = Column(Integer, ForeignKey('well_rdb.id'))
    curve_name = Column(String)
    curve_data = Column(Text)
    begin = Column(Float)
    end = Column(Float)
    step = Column(Float)
    description = Column(Text)

    well = relationship("WellRDB", back_populates="well_logs")

class FormationRDB(BaseRDB):
    __tablename__ = 'formation_rdb'

    id = Column(Integer, primary_key=True)
    profile_id = Column(Integer, ForeignKey('profile_rdb.id'))
    title = Column(String)
    up = Column(Text)
    down = Column(Text)
    up_hash = Column(String)
    down_hash = Column(String)

    profile = relationship('ProfileRDB', back_populates='formations')
    markups_mlp = relationship('MarkupMLPRDB', back_populates='formation')
    markups_reg = relationship('MarkupRegRDB', back_populates='formation')
    formation_feature = relationship('FormationFeatureRDB', back_populates='formation')
    wavelet_feature = relationship('WaveletFeatureRDB', back_populates='formation')
    fractal_feature = relationship('FractalFeatureRDB', back_populates='formation')
    entropy_feature = relationship('EntropyFeatureRDB', back_populates='formation')
    nonlinear_feature = relationship('NonlinearFeatureRDB', back_populates='formation')
    morphology_feature = relationship('MorphologyFeatureRDB', back_populates='formation')
    frequency_feature = relationship('FrequencyFeatureRDB', back_populates='formation')
    envelope_feature = relationship('EnvelopeFeatureRDB', back_populates='formation')
    autocorr_feature = relationship('AutocorrFeatureRDB', back_populates='formation')
    emd_feature = relationship('EMDFeatureRDB', back_populates='formation')
    hht_feature = relationship('HHTFeatureRDB', back_populates='formation')

class FormationFeatureRDB(BaseRDB):
    __tablename__ = 'formation_feature_rdb'

    id = Column(Integer, primary_key=True)
    formation_id = Column(Integer, ForeignKey('formation_rdb.id'))
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

    formation = relationship('FormationRDB', back_populates='formation_feature')

class WaveletFeatureRDB(BaseRDB):
    __tablename__ = 'wavelet_feature_rdb'
    # __table_args__ = {'info': dict(is_view=True)}dd

    id = Column(Integer, primary_key=True)
    formation_id = Column(Integer, ForeignKey('formation_rdb.id'))

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

    formation = relationship('FormationRDB', back_populates='wavelet_feature')

class FractalFeatureRDB(BaseRDB):
    __tablename__ = 'fractal_feature_rdb'

    id = Column(Integer, primary_key=True)
    formation_id = Column(Integer, ForeignKey('formation_rdb.id'))
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

    formation = relationship('FormationRDB', back_populates='fractal_feature')

class EntropyFeatureRDB(BaseRDB):
    __tablename__ = 'entropy_feature_rdb'

    id = Column(Integer, primary_key=True)
    formation_id = Column(Integer, ForeignKey('formation_rdb.id'))
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

    formation = relationship('FormationRDB', back_populates='entropy_feature')

class NonlinearFeatureRDB(BaseRDB):
    __tablename__ = 'nonlinear_feature_rdb'

    id = Column(Integer, primary_key=True)
    formation_id = Column(Integer, ForeignKey('formation_rdb.id'))
    nln_corr_dim = Column(Text)
    nln_rec_rate = Column(Text)
    nln_determin = Column(Text)
    nln_avg_diag = Column(Text)
    nln_hirsh = Column(Text)

    formation = relationship('FormationRDB', back_populates='nonlinear_feature')

class MorphologyFeatureRDB(BaseRDB):
    __tablename__ = 'morphology_feature_rdb'

    id = Column(Integer, primary_key=True)
    formation_id = Column(Integer, ForeignKey('formation_rdb.id'))
    mph_peak_num = Column(Text)
    mph_peak_width = Column(Text)
    mph_peak_amp_ratio = Column(Text)
    mph_peak_asymm = Column(Text)
    mph_peak_steep = Column(Text)
    mph_erosion = Column(Text)
    mph_dilation = Column(Text)

    formation = relationship('FormationRDB', back_populates='morphology_feature')

class FrequencyFeatureRDB(BaseRDB):
    __tablename__ = 'frequency_feature_rdb'

    id = Column(Integer, primary_key=True)
    formation_id = Column(Integer, ForeignKey('formation_rdb.id'))
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

    formation = relationship('FormationRDB', back_populates='frequency_feature')

class EnvelopeFeatureRDB(BaseRDB):
    __tablename__ = 'envelope_feature_rdb'

    id = Column(Integer, primary_key=True)
    formation_id = Column(Integer, ForeignKey('formation_rdb.id'))
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

    formation = relationship('FormationRDB', back_populates='envelope_feature')

class AutocorrFeatureRDB(BaseRDB):
    __tablename__ = 'autocorr_feature_rdb'

    id = Column(Integer, primary_key=True)
    formation_id = Column(Integer, ForeignKey('formation_rdb.id'))
    acf_first_min = Column(Text)
    acf_lag_10 = Column(Text)
    acf_decay = Column(Text)
    acf_integral = Column(Text)
    acf_peak_width = Column(Text)
    acf_ratio = Column(Text)

    formation = relationship('FormationRDB', back_populates='autocorr_feature')

class EMDFeatureRDB(BaseRDB):
    __tablename__ = 'emd_feature_rdb'

    id = Column(Integer, primary_key=True)
    formation_id = Column(Integer, ForeignKey('formation_rdb.id'))
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

    formation = relationship('FormationRDB', back_populates='emd_feature')

class HHTFeatureRDB(BaseRDB):
    __tablename__ = 'hht_feature_rdb'

    id = Column(Integer, primary_key=True)
    formation_id = Column(Integer, ForeignKey('formation_rdb.id'))

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

    formation = relationship('FormationRDB', back_populates='hht_feature')

class EntropyFeatureProfileRDB(BaseRDB):
    __tablename__ = 'entropy_feature_profile_rdb'

    id = Column(Integer, primary_key=True)
    profile_id = Column(Integer, ForeignKey('profile_rdb.id'))

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

    profile = relationship('ProfileRDB', back_populates='entropy_feature')


#####################################################
######################  MLP  ########################
#####################################################

class AnalysisMLPRDB(BaseRDB):
    __tablename__ = 'analysis_mlp_rdb'

    id = Column(Integer, primary_key=True)
    title = Column(String)

    markers = relationship('MarkerMLPRDB', back_populates='analysis')
    markups = relationship('MarkupMLPRDB', back_populates='analysis')
    genetic_algorithms = relationship('GeneticAlgorithmCLSRDB', back_populates='analysis_mlp')
    trained_models = relationship('TrainedModelClassRDB', back_populates='analysis')


class MarkerMLPRDB(BaseRDB):
    __tablename__ = 'marker_mlp_rdb'

    id = Column(Integer, primary_key=True)
    analysis_id = Column(Integer, ForeignKey('analysis_mlp_rdb.id'))
    title = Column(String)
    color = Column(String)

    analysis = relationship('AnalysisMLPRDB', back_populates='markers')
    markups = relationship('MarkupMLPRDB', back_populates='marker')


class MarkupMLPRDB(BaseRDB):
    __tablename__ = 'markup_mlp_rdb'

    id = Column(Integer, primary_key=True)
    analysis_id = Column(Integer, ForeignKey('analysis_mlp_rdb.id'))
    well_id = Column(Integer, ForeignKey('well_rdb.id'))
    profile_id = Column(Integer, ForeignKey('profile_rdb.id'))
    formation_id = Column(Integer, ForeignKey('formation_rdb.id'))
    marker_id = Column(Integer, ForeignKey('marker_mlp_rdb.id'))
    list_measure = Column(Text)
    type_markup = Column(String)

    analysis = relationship('AnalysisMLPRDB', back_populates='markups')
    well = relationship("WellRDB", back_populates="markups_mlp")
    profile = relationship("ProfileRDB", back_populates="markups_mlp")
    formation = relationship("FormationRDB", back_populates="markups_mlp")
    marker = relationship("MarkerMLPRDB", back_populates="markups")


class GeneticAlgorithmCLSRDB(BaseRDB):
    __tablename__ = 'genetic_algorithm_cls_rdb'

    id = Column(Integer, primary_key=True)
    analysis_id = Column(Integer, ForeignKey('analysis_mlp_rdb.id'))
    title = Column(String)
    pipeline = Column(Text)
    checkfile_path = Column(LargeBinary)
    list_params = Column(Text)
    population_size = Column(Integer)
    comment = Column(Text)
    type_problem = Column(String)

    analysis_mlp = relationship('AnalysisMLPRDB', back_populates='genetic_algorithms')


class TrainedModelClassRDB(BaseRDB):
    __tablename__ = 'trained_model_class_rdb'

    id = Column(Integer, primary_key=True)
    analysis_id = Column(Integer, ForeignKey('analysis_mlp_rdb.id'))
    title = Column(String)
    file_model = Column(LargeBinary)
    list_params = Column(Text)
    except_signal = Column(String, default="")
    except_crl = Column(String, default="")
    comment = Column(Text)
    mask = Column(Text)

    analysis = relationship('AnalysisMLPRDB', back_populates='trained_models')


#####################################################
###################  Regression  ####################
#####################################################

class AnalysisRegRDB(BaseRDB):
    __tablename__ = 'analysis_reg_rdb'

    id = Column(Integer, primary_key=True)
    title = Column(String)

    markups = relationship('MarkupRegRDB', back_populates='analysis')
    trained_models = relationship('TrainedModelRegRDB', back_populates='analysis')
    genetic_algorithms = relationship('GeneticAlgorithmRegRDB', back_populates='analysis_reg')


class MarkupRegRDB(BaseRDB):
    __tablename__ = 'markup_reg_rdb'

    id = Column(Integer, primary_key=True)
    analysis_id = Column(Integer, ForeignKey('analysis_reg_rdb.id'))
    well_id = Column(Integer, ForeignKey('well_rdb.id'))    # возможно не нужно
    profile_id = Column(Integer, ForeignKey('profile_rdb.id'))
    formation_id = Column(Integer, ForeignKey('formation_rdb.id'))
    target_value = Column(Float)
    list_measure = Column(Text)
    type_markup = Column(String)

    analysis = relationship('AnalysisRegRDB', back_populates='markups')
    well = relationship("WellRDB", back_populates="markups_reg")
    profile = relationship("ProfileRDB", back_populates="markups_reg")
    formation = relationship("FormationRDB", back_populates="markups_reg")

class GeneticAlgorithmRegRDB(BaseRDB):
    __tablename__ = 'genetic_algorithm_reg_rdb'

    id = Column(Integer, primary_key=True)
    analysis_id = Column(Integer, ForeignKey('analysis_reg_rdb.id'))
    title = Column(String)
    pipeline = Column(Text)
    checkfile_path = Column(LargeBinary)
    list_params = Column(Text)
    population_size = Column(Integer)
    comment = Column(Text)
    type_problem = Column(String)

    analysis_reg = relationship('AnalysisRegRDB', back_populates='genetic_algorithms')


class TrainedModelRegRDB(BaseRDB):
    __tablename__ = 'trained_model_reg_rdb'

    id = Column(Integer, primary_key=True)
    analysis_id = Column(Integer, ForeignKey('analysis_reg_rdb.id'))
    title = Column(String)
    file_model = Column(LargeBinary)
    list_params = Column(Text)
    except_signal = Column(String, default="")
    except_crl = Column(String, default="")
    comment = Column(Text)
    mask = Column(Text)

    analysis = relationship('AnalysisRegRDB', back_populates='trained_models')



