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
    report = Column(String)

    research = relationship('Research', back_populates='cluster_set')
    analysis = relationship('AnalysisCluster', back_populates='object_set')
    auto_tuning_cache = relationship('ClusterAutoTuningCache', back_populates='object_set')


class ClusterAutoTuningCache(Base):
    __tablename__ = 'cluster_auto_tuning_cache'

    id = Column(Integer, primary_key=True)
    object_set_id = Column(Integer, ForeignKey('object_set.id'), nullable=False, index=True)
    cache_key = Column(String, nullable=False, unique=True, index=True)
    created_at = Column(DateTime, nullable=False, default=datetime.datetime.utcnow)
    top_results = Column(Text, nullable=False)

    object_set = relationship('ObjectSet', back_populates='auto_tuning_cache')
