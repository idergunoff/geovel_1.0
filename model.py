import datetime

from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, ForeignKey, Date, Text, text, literal_column
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

DATABASE_NAME = 'geovel_db.sqlite'

engine = create_engine(f'sqlite:///{DATABASE_NAME}', echo=False)

Session = sessionmaker(bind=engine)

Base = declarative_base()


class GeoradarObject(Base):
    __tablename__ = 'georadar_object'

    id = Column(Integer, primary_key=True)
    title = Column(String)
    date_exam = Column(Date, default=datetime.date.today())

    profiles = relationship('Profile', back_populates='georadar_object')
    grid = relationship('Grid', back_populates='object')


class Profile(Base):
    __tablename__ = 'profile'

    id = Column(Integer, primary_key=True)
    object_id = Column(Integer, ForeignKey('georadar_object.id'))
    title = Column(String)

    signal = Column(Text)

    x_wgs = Column(Text)
    y_wgs = Column(Text)
    x_pulc = Column(Text)
    y_pulc = Column(Text)

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

    georadar_object = relationship('GeoradarObject', back_populates='profiles')
    current = relationship('CurrentProfile', back_populates='profile')
    window = relationship('WindowProfile', back_populates='profile')
    min_max = relationship('CurrentProfileMinMax', back_populates='profile')
    layers = relationship('Layers', back_populates='profile')


# class Measure(Base):
#     __tablename__ = 'measure'
#
#     id = Column(Integer, primary_key=True)
#     profile_id = Column(Integer, ForeignKey('profile.id'))
#     number = Column(Integer)
#
#     signal = Column(Text)
#
#     x_wgs = Column(Float)
#     y_wgs = Column(Float)
#     x_pulc = Column(Float)
#     y_pulc = Column(Float)
#
#     T_top = Column(Float)
#     T_bottom = Column(Float)
#     dT = Column(Float)
#
#     A_top = Column(Float)
#     A_bottom = Column(Float)
#     dA = Column(Float)
#     A_sum = Column(Float)
#     A_mean = Column(Float)
#     dVt = Column(Float)
#     Vt_top = Column(Float)
#     Vt_sum = Column(Float)
#     Vt_mean = Column(Float)
#     dAt = Column(Float)
#     At_top = Column(Float)
#     At_sum = Column(Float)
#     At_mean = Column(Float)
#     dPht = Column(Float)
#     Pht_top = Column(Float)
#     Pht_sum = Column(Float)
#     Pht_mean = Column(Float)
#     Wt_top = Column(Float)
#     Wt_mean = Column(Float)
#     Wt_sum = Column(Float)
#
#     width = Column(Float)
#     top = Column(Float)
#     land = Column(Float)
#     speed = Column(Float)
#     speed_cover = Column(Float)
#
#     skew = Column(Float)
#     kurt = Column(Float)
#     std = Column(Float)
#     k_var = Column(Float)
#
#     profile = relationship('Profile', back_populates='measures')
#     # signals = relationship('Signal', back_populates='measure')


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
    layer_title = Column(Text)
    layer_line = Column(Text)

    profile = relationship('Profile', back_populates='layers')
    points = relationship('PointsOfLayer', back_populates='layer')
    formation_up = relationship('Formation', back_populates='layer_up', foreign_keys='Formation.up')
    formation_down = relationship('Formation', back_populates='layer_down', foreign_keys='Formation.down')


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

    layer_up = relationship('Layers', back_populates='formation_up', foreign_keys=[up])
    layer_down = relationship('Layers', back_populates='formation_down', foreign_keys=[down])




Base.metadata.create_all(engine)