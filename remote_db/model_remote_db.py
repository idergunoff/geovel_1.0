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
    formations = relationship('FormationRDB', back_populates='profile')
    markups_mlp = relationship('MarkupMLPRDB', back_populates='profile')
    markups_reg = relationship('MarkupRegRDB', back_populates='profile')


class WellRDB(BaseRDB):
    __tablename__ = 'well_rdb'

    id = Column(Integer, primary_key=True)
    name = Column(String)
    x_coord = Column(Float)
    y_coord = Column(Float)
    alt = Column(Float)
    well_hash = Column(String)

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



