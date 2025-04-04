import datetime
import json
from contextlib import contextmanager

from sqlalchemy import (create_engine, Column, Integer, String, Float, Boolean, DateTime, LargeBinary, ForeignKey,
                        Date, Text, text, literal_column, or_, func, Index, desc, select, update, bindparam, literal)
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


class WellRDB(BaseRDB):
    __tablename__ = 'well_rdb'

    id = Column(Integer, primary_key=True)
    name = Column(String)
    x_coord = Column(Float)
    y_coord = Column(Float)
    alt = Column(Float)
    well_hash = Column(String)
