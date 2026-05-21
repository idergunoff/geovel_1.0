import datetime

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


########################################################################################
###########################       WELL LOG CLUSTERING       ############################
########################################################################################


class WellLogClusterDataset(Base):
    __tablename__ = 'well_log_cluster_dataset'

    id = Column(Integer, primary_key=True)
    name = Column(String)
    created_by = Column(DateTime, default=datetime.datetime.now())

    well_for_cluster = relationship('WellForCluster', back_populates='well_cluster_set')
    cluster_well_log_param = relationship('ClusterWellLogParameter', back_populates='well_cluster_set')
    cluster_well_log_param_from_calculator = relationship('ClusterWellLogParameterFromCalculator', back_populates='well_cluster_set')
    data = relationship('WellLogClusterDatasetData', back_populates='well_cluster_set')


class WellForCluster(Base):
    __tablename__ = 'well_for_cluster'

    id = Column(Integer, primary_key=True)
    dataset_id = Column(Integer, ForeignKey('well_log_cluster_dataset.id'))
    well_id = Column(Integer, ForeignKey('well.id'))
    top_md = Column(Float)
    bottom_md = Column(Float)

    well_cluster_set = relationship('WellLogClusterDataset', back_populates='well_for_cluster')
    well = relationship('Well', back_populates='well_for_cluster')


class CanonicalWellLog(Base):
    __tablename__ = 'canonical_well_log'

    id = Column(Integer, primary_key=True)
    canonical_name = Column(String)
    description = Column(String)

    alias_name = relationship('AliasWellLog', back_populates='canonical_name')
    cluster_well_log_param = relationship('ClusterWellLogParameter', back_populates='canonical_name')


class AliasWellLog(Base):
    __tablename__ = 'alias_well_log'

    id = Column(Integer, primary_key=True)
    alias_name = Column(String)
    canonical_id = Column(Integer, ForeignKey('canonical_well_log.id'))

    canonical_name = relationship('CanonicalWellLog', back_populates='alias_name')


class FeatureCalculator(Base):
    __tablename__ = 'feature_calculator'

    id = Column(Integer, primary_key=True)
    feature_name = Column(String)
    expression = Column(Text)
    used_canonical_well_log = Column(Text)
    transform_type = Column(String)
    params_json = Column(Text)
    created_at = Column(DateTime)

    cluster_well_log_param_from_calculator = relationship('ClusterWellLogParameterFromCalculator', back_populates='calculator')


class ClusterWellLogParameter(Base):
    __tablename__ = 'cluster_well_log_parameter'

    id = Column(Integer, primary_key=True)
    dataset_id = Column(Integer, ForeignKey('well_log_cluster_dataset.id'))
    canonical_id = Column(Integer, ForeignKey('canonical_well_log.id'))

    well_cluster_set = relationship('WellLogClusterDataset', back_populates='cluster_well_log_param')
    canonical_name = relationship('CanonicalWellLog', back_populates='cluster_well_log_param')


class ClusterWellLogParameterFromCalculator(Base):
    __tablename__ = 'cluster_well_log_parameter_from_calculator'

    id = Column(Integer, primary_key=True)
    dataset_id = Column(Integer, ForeignKey('well_log_cluster_dataset.id'))
    calculator_id = Column(Integer, ForeignKey('feature_calculator.id'))

    well_cluster_set = relationship('WellLogClusterDataset', back_populates='cluster_well_log_param_from_calculator')
    calculator = relationship('FeatureCalculator', back_populates='cluster_well_log_param_from_calculator')


class WellLogClusterDatasetData(Base):
    __tablename__ = 'cluster_well_log_dataset_data'

    id = Column(Integer, primary_key=True)
    dataset_id = Column(Integer, ForeignKey('well_log_cluster_dataset.id'))
    data = Column(Text)

    well_cluster_set = relationship('WellLogClusterDataset', back_populates='data')













