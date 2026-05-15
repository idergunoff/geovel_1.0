from models_db.model import *

class AnalysisCluster(Base):
    __tablename__ = 'analysis_cluster'

    id = Column(Integer, primary_key=True)
    title = Column(String)
    parameter = Column(String)

    object_set = relationship('ObjectSet', back_populates='analysis', cascade='all, delete-orphan')


class ObjectSet(Base):
    __tablename__ = 'object_set'

    id = Column(Integer, primary_key=True)
    research_id = Column(Integer, ForeignKey('research.id'))
    analysis_id = Column(Integer, ForeignKey('analysis_cluster.id'))
    data = Column(Text)
    report = Column(Text)

    research = relationship('Research', back_populates='cluster_set')
    analysis = relationship('AnalysisCluster', back_populates='object_set')
    auto_tuning_cache = relationship('ClusterAutoTuningCache', back_populates='object_set', cascade='all, delete-orphan')
    auto_tuning_runs = relationship('ClusterAutoTuningRunState', back_populates='object_set', cascade='all, delete-orphan')


class ClusterAutoTuningCache(Base):
    __tablename__ = 'cluster_auto_tuning_cache'

    id = Column(Integer, primary_key=True)
    object_set_id = Column(Integer, ForeignKey('object_set.id'), nullable=False, index=True)
    cache_key = Column(String, nullable=False, unique=True, index=True)
    created_at = Column(DateTime, nullable=False, default=datetime.datetime.utcnow)
    top_results = Column(Text, nullable=False)

    object_set = relationship('ObjectSet', back_populates='auto_tuning_cache')


class ClusterAutoTuningRunState(Base):
    __tablename__ = 'cluster_auto_tuning_run_state'

    id = Column(Integer, primary_key=True)
    object_set_id = Column(Integer, ForeignKey('object_set.id'), nullable=False, index=True)
    run_key = Column(String, nullable=False, unique=True, index=True)
    random_seed = Column(Integer, nullable=False)
    sampled_indices_json = Column(Text, nullable=False)
    completed_candidate_ids_json = Column(Text, nullable=False, default='[]')
    raw_results_json = Column(Text, nullable=False, default='[]')
    coarse_count = Column(Integer, nullable=False, default=0)
    fine_count = Column(Integer, nullable=False, default=0)
    updated_at = Column(DateTime, nullable=False, default=datetime.datetime.utcnow)

    object_set = relationship('ObjectSet', back_populates='auto_tuning_runs')
