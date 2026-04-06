from models_db.model import *

class AnalysisCluster(Base):
    __tablename__ = 'analysis_cluster'

    id = Column(Integer, primary_key=True)
    title = Column(String)
    parameter = Column(String)

    object_set = relationship('ObjectSet', back_populates='analysis')


class ObjectSet(Base):
    __tablename__ = 'object_set'

    id = Column(Integer, primary_key=True)
    research_id = Column(Integer, ForeignKey('research.id'))
    analysis_id = Column(Integer, ForeignKey('analysis_cluster.id'))
    data = Column(String)

    research = relationship('Research', back_populates='cluster_set')
    analysis = relationship('AnalysisCluster', back_populates='object_set')