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

    georadar_object = relationship('GeoradarObject', back_populates='profiles')
    measures = relationship('Measure', back_populates='profile')
    current = relationship('CurrentProfile', back_populates='profile')


class Measure(Base):
    __tablename__ = 'measure'

    id = Column(Integer, primary_key=True)
    profile_id = Column(Integer, ForeignKey('profile.id'))
    number = Column(Integer)

    signal = Column(Text)
    # diff = Column(Text)
    # max_min = Column(Float)
    # At = Column(Text)
    # Vt = Column(Text)
    # Pht = Column(Text)
    # Wt = Column(Text)

    x_wgs = Column(Float)
    y_wgs = Column(Float)
    x_pulc = Column(Float)
    y_pulc = Column(Float)

    T_top = Column(Float)
    T_bottom = Column(Float)
    dT = Column(Float)

    A_top = Column(Float)
    A_bottom = Column(Float)
    dA = Column(Float)
    A_sum = Column(Float)
    A_mean = Column(Float)
    dVt = Column(Float)
    Vt_top = Column(Float)
    Vt_sum = Column(Float)
    Vt_mean = Column(Float)
    dAt = Column(Float)
    At_top = Column(Float)
    At_sum = Column(Float)
    At_mean = Column(Float)
    dPht = Column(Float)
    Pht_top = Column(Float)
    Pht_sum = Column(Float)
    Pht_mean = Column(Float)
    Wt_top = Column(Float)
    Wt_mean = Column(Float)
    Wt_sum = Column(Float)

    width = Column(Float)
    top = Column(Float)
    land = Column(Float)
    speed = Column(Float)
    speed_cover = Column(Float)

    skew = Column(Float)
    kurt = Column(Float)
    std = Column(Float)
    k_var = Column(Float)

    profile = relationship('Profile', back_populates='measures')
    # signals = relationship('Signal', back_populates='measure')


class CurrentProfile(Base):
    __tablename__ = 'current_profile'

    id = Column(Integer, primary_key=True)
    profile_id = Column(Integer, ForeignKey('profile.id'))
    signal = Column(Text)

    profile = relationship('Profile', back_populates='current')


class Grid(Base):
    __tablename__ = 'grid'

    id = Column(Integer, primary_key=True)
    object_id = Column(Integer, ForeignKey('georadar_object.id'))
    grid_table_uf = Column(Text)
    grid_table_m = Column(Text)
    grid_table_r = Column(Text)

    object = relationship('GeoradarObject', back_populates='grid')



# class Signal(Base):
#     __tablename__ = 'signal'
#
#     id = Column(Integer, primary_key=True)
#     measure_id = Column(Integer, ForeignKey('measure.id'))
#     time_ns = Column(Integer)
#     A = Column(Float)
#     diff = Column(Float)
#     diff2 = Column(Float)
#     max_min = Column(Float)
#     At = Column(Float)
#     Vt = Column(Float)
#     Pht = Column(Float)
#     Wt = Column(Float)
#
#     measure = relationship('Measure', back_populates='signals')


Base.metadata.create_all(engine)