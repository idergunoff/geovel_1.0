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
    calculation_cache = relationship('ClusterCalculationCache', back_populates='object_set', cascade='all, delete-orphan')


class ClusterAutoTuningCache(Base):
    __tablename__ = 'cluster_auto_tuning_cache'

    id = Column(Integer, primary_key=True)
    object_set_id = Column(Integer, ForeignKey('object_set.id'), nullable=False, index=True)
    cache_key = Column(String, nullable=False, unique=True, index=True)
    created_at = Column(DateTime, nullable=False, default=datetime.datetime.utcnow)
    top_results = Column(Text, nullable=False)

    object_set = relationship('ObjectSet', back_populates='auto_tuning_cache')


class WellLogClusterAutoTuningCache(Base):
    __tablename__ = 'well_log_cluster_auto_tuning_cache'

    id = Column(Integer, primary_key=True)
    dataset_id = Column(Integer, ForeignKey('well_log_cluster_dataset.id'), nullable=False, index=True)
    cache_key = Column(String, nullable=False, unique=True, index=True)
    created_at = Column(DateTime, nullable=False, default=datetime.datetime.utcnow)
    top_results = Column(Text, nullable=False)

    well_cluster_set = relationship('WellLogClusterDataset', back_populates='auto_tuning_cache')


class ClusterCalculationCache(Base):
    __tablename__ = 'cluster_calculation_cache'

    id = Column(Integer, primary_key=True)
    object_set_id = Column(Integer, ForeignKey('object_set.id'), nullable=False, index=True)
    cache_key = Column(String, nullable=False, unique=True, index=True)
    created_at = Column(DateTime, nullable=False, default=datetime.datetime.utcnow)
    data_hash = Column(String, nullable=False)
    config_json = Column(Text, nullable=False)
    result_payload = Column(Text, nullable=False)

    object_set = relationship('ObjectSet', back_populates='calculation_cache')




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
    name = Column(String, nullable=False)
    created_at = Column(DateTime, nullable=False, default=datetime.datetime.utcnow)

    well_for_cluster = relationship('WellForCluster', back_populates='well_cluster_set', cascade='all, delete-orphan')
    cluster_well_log_param = relationship('ClusterWellLogParameter', back_populates='well_cluster_set', cascade='all, delete-orphan')
    cluster_well_log_param_from_calculator = relationship('ClusterWellLogParameterFromCalculator', back_populates='well_cluster_set', cascade='all, delete-orphan')
    data = relationship('WellLogClusterDatasetData', back_populates='well_cluster_set', cascade='all, delete-orphan')
    auto_tuning_cache = relationship('WellLogClusterAutoTuningCache', back_populates='well_cluster_set', cascade='all, delete-orphan')
    calculation_cache = relationship('WellLogClusterCalculationCache', back_populates='well_cluster_set', cascade='all, delete-orphan')


class WellLogClusterCalculationCache(Base):
    __tablename__ = 'well_log_cluster_calculation_cache'

    id = Column(Integer, primary_key=True)
    dataset_id = Column(Integer, ForeignKey('well_log_cluster_dataset.id'), nullable=False, index=True)
    cache_key = Column(String, nullable=False, unique=True, index=True)
    created_at = Column(DateTime, nullable=False, default=datetime.datetime.utcnow)
    data_hash = Column(String, nullable=False)
    config_json = Column(Text, nullable=False)
    result_payload = Column(Text, nullable=False)

    well_cluster_set = relationship('WellLogClusterDataset', back_populates='calculation_cache')




class WellForCluster(Base):
    __tablename__ = 'well_for_cluster'

    id = Column(Integer, primary_key=True)
    dataset_id = Column(Integer, ForeignKey('well_log_cluster_dataset.id'), nullable=False, index=True)
    well_id = Column(Integer, ForeignKey('well.id'), nullable=False, index=True)
    top_md = Column(Float, nullable=False)
    bottom_md = Column(Float, nullable=False)

    __table_args__ = (
        UniqueConstraint('dataset_id', 'well_id', name='uq_well_for_cluster_dataset_well'),
    )

    well_cluster_set = relationship('WellLogClusterDataset', back_populates='well_for_cluster')
    well = relationship('Well', back_populates='well_for_cluster')


class CanonicalWellLog(Base):
    __tablename__ = 'canonical_well_log'

    id = Column(Integer, primary_key=True)
    canonical_name = Column(String, nullable=False, unique=True, index=True)
    canonical_name_norm = Column(String, nullable=False, unique=True, index=True)
    description = Column(String)

    aliases = relationship('AliasWellLog', back_populates='canonical_log', cascade='all, delete-orphan')
    cluster_well_log_param = relationship('ClusterWellLogParameter', back_populates='canonical_name')


class AliasWellLog(Base):
    __tablename__ = 'alias_well_log'

    id = Column(Integer, primary_key=True)
    alias_name = Column(String, nullable=False, unique=True, index=True)
    alias_name_norm = Column(String, nullable=False, unique=True, index=True)
    canonical_id = Column(Integer, ForeignKey('canonical_well_log.id'), nullable=False, index=True)

    canonical_log = relationship('CanonicalWellLog', back_populates='aliases')




def _normalize_well_log_name(value):
    if value is None:
        return None
    normalized = value.strip()
    return normalized.casefold()


@event.listens_for(CanonicalWellLog, 'before_insert')
@event.listens_for(CanonicalWellLog, 'before_update')
def _sync_canonical_name_norm(mapper, connection, target):
    if target.canonical_name is None:
        raise ValueError('canonical_name cannot be empty')
    target.canonical_name = target.canonical_name.strip()
    if not target.canonical_name:
        raise ValueError('canonical_name cannot be empty')
    target.canonical_name_norm = _normalize_well_log_name(target.canonical_name)


@event.listens_for(AliasWellLog, 'before_insert')
@event.listens_for(AliasWellLog, 'before_update')
def _sync_alias_name_norm(mapper, connection, target):
    if target.alias_name is None:
        raise ValueError('alias_name cannot be empty')
    target.alias_name = target.alias_name.strip()
    if not target.alias_name:
        raise ValueError('alias_name cannot be empty')
    target.alias_name_norm = _normalize_well_log_name(target.alias_name)

class FeatureCalculator(Base):
    __tablename__ = 'feature_calculator'

    id = Column(Integer, primary_key=True)
    feature_name = Column(String, nullable=False, unique=True, index=True)
    expression = Column(Text)
    used_canonical_well_log = Column(Text, nullable=False)
    transform_type = Column(String, nullable=False)
    params_json = Column(Text, nullable=False)
    created_at = Column(DateTime, nullable=False, default=datetime.datetime.utcnow)

    cluster_well_log_param_from_calculator = relationship('ClusterWellLogParameterFromCalculator', back_populates='calculator')


class ClusterWellLogParameter(Base):
    __tablename__ = 'cluster_well_log_parameter'

    id = Column(Integer, primary_key=True)
    dataset_id = Column(Integer, ForeignKey('well_log_cluster_dataset.id'), nullable=False, index=True)
    canonical_id = Column(Integer, ForeignKey('canonical_well_log.id'), nullable=False, index=True)

    __table_args__ = (
        UniqueConstraint('dataset_id', 'canonical_id', name='uq_cluster_well_log_param_dataset_canonical'),
    )

    well_cluster_set = relationship('WellLogClusterDataset', back_populates='cluster_well_log_param')
    canonical_name = relationship('CanonicalWellLog', back_populates='cluster_well_log_param')


class ClusterWellLogParameterFromCalculator(Base):
    __tablename__ = 'cluster_well_log_parameter_from_calculator'

    id = Column(Integer, primary_key=True)
    dataset_id = Column(Integer, ForeignKey('well_log_cluster_dataset.id'), nullable=False, index=True)
    calculator_id = Column(Integer, ForeignKey('feature_calculator.id'), nullable=False, index=True)

    __table_args__ = (
        UniqueConstraint('dataset_id', 'calculator_id', name='uq_cluster_well_log_param_calc_dataset_calculator'),
    )

    well_cluster_set = relationship('WellLogClusterDataset', back_populates='cluster_well_log_param_from_calculator')
    calculator = relationship('FeatureCalculator', back_populates='cluster_well_log_param_from_calculator')


class WellLogClusterDatasetData(Base):
    __tablename__ = 'cluster_well_log_dataset_data'

    id = Column(Integer, primary_key=True)
    dataset_id = Column(Integer, ForeignKey('well_log_cluster_dataset.id'), nullable=False, index=True)
    data = Column(Text, nullable=False)

    well_cluster_set = relationship('WellLogClusterDataset', back_populates='data')












