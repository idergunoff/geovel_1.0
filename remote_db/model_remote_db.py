import datetime
import json
from contextlib import contextmanager

from sqlalchemy import (create_engine, Column, Integer, String, Float, Boolean, DateTime, LargeBinary, ForeignKey,
                        Date, Text, text, literal_column, or_, func, Index, desc, MetaData, Table)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

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


Base = declarative_base()


class GeoradarObject(Base):
    __tablename__ = 'georadar_object'

    id = Column(Integer, primary_key=True)
    title = Column(String)

    researches = relationship('Research', back_populates='object')


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
    signal_hash = Column(String)

    x_wgs = Column(Text)
    y_wgs = Column(Text)
    x_pulc = Column(Text)
    y_pulc = Column(Text)
    abs_relief = Column(Text)
    depth_relief = Column(Text)

    research = relationship('Research', back_populates='profiles')


Base.metadata.create_all(engine_remote)