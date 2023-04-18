import datetime
import json

from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, ForeignKey, Date, Text, text, literal_column, or_
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

DATABASE_NAME = 'geovel_db.sqlite'

engine = create_engine(f'sqlite:///{DATABASE_NAME}', echo=False)

Session = sessionmaker(bind=engine)

Base = declarative_base()


class GeoradarObject(Base):
    __tablename__ = 'georadar_object'

    id = Column(Integer, primary_key=True)
    title = Column(String)

    researches = relationship('Research', back_populates='object')
    grid = relationship('Grid', back_populates='object')


class Research(Base):
    __tablename__ = 'research'

    id = Column(Integer, primary_key=True)
    object_id = Column(Integer, ForeignKey('georadar_object.id'))
    date_research = Column(Date)

    object = relationship('GeoradarObject', back_populates='researches')
    profiles = relationship('Profile', back_populates='research')


class Profile(Base):
    __tablename__ = 'profile'

    id = Column(Integer, primary_key=True)
    research_id = Column(Integer, ForeignKey('research.id'))
    title = Column(String)

    signal = Column(Text)

    x_wgs = Column(Text)
    y_wgs = Column(Text)
    x_pulc = Column(Text)
    y_pulc = Column(Text)

    research = relationship('Research', back_populates='profiles')
    current = relationship('CurrentProfile', back_populates='profile')
    window = relationship('WindowProfile', back_populates='profile')
    min_max = relationship('CurrentProfileMinMax', back_populates='profile')
    layers = relationship('Layers', back_populates='profile')
    formations = relationship('Formation', back_populates='profile')
    markups_lda = relationship('MarkupLDA', back_populates='profile')
    markups_mlp = relationship('MarkupMLP', back_populates='profile')
    # дальше всё убираем

    # T_top = Column(Text)
    # T_bottom = Column(Text)
    # dT = Column(Text)
    #
    # A_top = Column(Text)
    # A_bottom = Column(Text)
    # dA = Column(Text)
    # A_sum = Column(Text)
    # A_mean = Column(Text)
    # dVt = Column(Text)
    # Vt_top = Column(Text)
    # Vt_sum = Column(Text)
    # Vt_mean = Column(Text)
    # dAt = Column(Text)
    # At_top = Column(Text)
    # At_sum = Column(Text)
    # At_mean = Column(Text)
    # dPht = Column(Text)
    # Pht_top = Column(Text)
    # Pht_sum = Column(Text)
    # Pht_mean = Column(Text)
    # Wt_top = Column(Text)
    # Wt_mean = Column(Text)
    # Wt_sum = Column(Text)
    #
    # width = Column(Text)
    # top = Column(Text)
    # land = Column(Text)
    # speed = Column(Text)
    # speed_cover = Column(Text)
    #
    # skew = Column(Text)
    # kurt = Column(Text)
    # std = Column(Text)
    # k_var = Column(Text)


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


class Grid(Base):
    __tablename__ = 'grid'

    id = Column(Integer, primary_key=True)
    object_id = Column(Integer, ForeignKey('georadar_object.id'))
    grid_table_uf = Column(Text)
    grid_table_m = Column(Text)
    grid_table_r = Column(Text)

    object = relationship('GeoradarObject', back_populates='grid')


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
    boundary_to_layers = relationship('BoundaryToLayer', back_populates='layer')


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

    profile = relationship('Profile', back_populates='formations')
    layer_up = relationship('Layers', back_populates='formation_up', foreign_keys=[up])
    layer_down = relationship('Layers', back_populates='formation_down', foreign_keys=[down])
    markups_lda = relationship('MarkupLDA', back_populates='formation')
    markups_mlp = relationship('MarkupMLP', back_populates='formation')


class Well(Base):
    __tablename__ = 'well'

    id = Column(Integer, primary_key=True)
    name = Column(String)
    x_coord = Column(Float)
    y_coord = Column(Float)
    alt = Column(Float)

    boundaries = relationship("Boundary", back_populates="well")
    well_logs = relationship("WellLog", back_populates="well")
    markups_lda = relationship("MarkupLDA", back_populates="well")
    markups_mlp = relationship('MarkupMLP', back_populates='well')


class Boundary(Base):
    __tablename__ = 'boundary'

    id = Column(Integer, primary_key=True)
    well_id = Column(Integer, ForeignKey('well.id'))
    depth = Column(Float)
    title = Column(String)

    well = relationship("Well", back_populates="boundaries")
    boundary_to_layers = relationship('BoundaryToLayer', back_populates='boundary')


class BoundaryToLayer(Base):
    __tablename__ = 'boundary_to_layer'

    id = Column(Integer, primary_key=True)
    boundary_id = Column(Integer, ForeignKey('boundary.id'))
    layer_id = Column(Integer, ForeignKey('layers.id'))
    index = Column(Float)

    boundary = relationship("Boundary", back_populates="boundary_to_layers")
    layer = relationship('Layers', back_populates="boundary_to_layers")


class WellLog(Base):
    __tablename__ = 'well_log'

    id = Column(Integer, primary_key=True)
    well_id = Column(Integer, ForeignKey('well.id'))
    curve_name = Column(String)
    depth_data = Column(Text)
    curve_data = Column(Text)

    well = relationship("Well", back_populates="well_logs")


#####################################################
######################  LDA  ########################
#####################################################


class AnalysisLDA(Base):
    __tablename__ = 'analysis_lda'

    id = Column(Integer, primary_key=True)
    title = Column(String)
    data = Column(Text)

    parameters = relationship('ParameterLDA', back_populates='analysis')
    markers = relationship('MarkerLDA', back_populates='analysis')
    markups = relationship('MarkupLDA', back_populates='analysis')


class ParameterLDA(Base):
    __tablename__ = 'parameter_lda'

    id = Column(Integer, primary_key=True)
    analysis_id = Column(Integer, ForeignKey('analysis_lda.id'))
    parameter = Column(String)

    analysis = relationship('AnalysisLDA', back_populates='parameters')


class MarkerLDA(Base):
    __tablename__ = 'marker_lda'

    id = Column(Integer, primary_key=True)
    analysis_id = Column(Integer, ForeignKey('analysis_lda.id'))
    title = Column(String)
    color = Column(String)

    analysis = relationship('AnalysisLDA', back_populates='markers')
    markups = relationship('MarkupLDA', back_populates='marker')


class MarkupLDA(Base):
    __tablename__ = 'markup_lda'

    id = Column(Integer, primary_key=True)
    analysis_id = Column(Integer, ForeignKey('analysis_lda.id'))
    well_id = Column(Integer, ForeignKey('well.id'))    # возможно не нужно
    profile_id = Column(Integer, ForeignKey('profile.id'))
    formation_id = Column(Integer, ForeignKey('formation.id'))
    marker_id = Column(Integer, ForeignKey('marker_lda.id'))
    list_measure = Column(Text)
    list_fake = Column(Text)

    analysis = relationship('AnalysisLDA', back_populates='markups')
    well = relationship("Well", back_populates="markups_lda")
    profile = relationship("Profile", back_populates="markups_lda")
    formation = relationship("Formation", back_populates="markups_lda")
    marker = relationship("MarkerLDA", back_populates="markups")


#####################################################
######################  MLP  ########################
#####################################################


class AnalysisMLP(Base):
    __tablename__ = 'analysis_mlp'

    id = Column(Integer, primary_key=True)
    title = Column(String)
    data = Column(Text)

    parameters = relationship('ParameterMLP', back_populates='analysis')
    markers = relationship('MarkerMLP', back_populates='analysis')
    markups = relationship('MarkupMLP', back_populates='analysis')


class ParameterMLP(Base):
    __tablename__ = 'parameter_mlp'

    id = Column(Integer, primary_key=True)
    analysis_id = Column(Integer, ForeignKey('analysis_mlp.id'))
    parameter = Column(String)

    analysis = relationship('AnalysisMLP', back_populates='parameters')


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

    analysis = relationship('AnalysisMLP', back_populates='markups')
    well = relationship("Well", back_populates="markups_mlp")
    profile = relationship("Profile", back_populates="markups_mlp")
    formation = relationship("Formation", back_populates="markups_mlp")
    marker = relationship("MarkerMLP", back_populates="markups")


Base.metadata.create_all(engine)