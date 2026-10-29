"""Resume upgrades after the application created newer tables on an old database."""

from pathlib import Path
import sqlite3

import pytest
import sqlalchemy as sa
from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory

from models_db.model import Base
from models_db import model_cluster  # Register tables created by object.py at startup.


ROOT = Path(__file__).resolve().parents[1]
TARGET_COLUMNS = (
    'target_source_type', 'target_source_config', 'target_source_details',
    'target_is_manual_override',
)
HARDENED_TABLES = (
    'well_log_cluster_dataset', 'well_for_cluster', 'canonical_well_log',
    'alias_well_log', 'feature_calculator', 'cluster_well_log_parameter',
    'cluster_well_log_parameter_from_calculator', 'cluster_well_log_dataset_data',
)
CACHE_TABLES = (
    'well_log_cluster_auto_tuning_cache', 'cluster_calculation_cache',
    'well_log_cluster_calculation_cache',
)
ALIAS_TABLES = ('canonical_boundary', 'alias_boundary', 'canonical_well_option', 'alias_well_option')


def _config(database):
    config = Config(str(ROOT / 'alembic.ini'))
    config.set_main_option('script_location', str(ROOT / 'alembic'))
    config.set_main_option('sqlalchemy.url', f'sqlite:///{database}')
    return config


def _legacy_cluster_tables(connection):
    # These tables originally had no Alembic CREATE TABLE revision.
    definitions = {
        'well_log_cluster_dataset': 'name VARCHAR, created_by DATETIME',
        'well_for_cluster': 'dataset_id INTEGER, well_id INTEGER, top_md FLOAT, bottom_md FLOAT',
        'canonical_well_log': 'canonical_name VARCHAR, description VARCHAR',
        'alias_well_log': 'alias_name VARCHAR, canonical_id INTEGER',
        'feature_calculator': ('feature_name VARCHAR, expression TEXT, used_canonical_well_log TEXT, '
                               'transform_type VARCHAR, params_json TEXT, created_at DATETIME'),
        'cluster_well_log_parameter': 'dataset_id INTEGER, canonical_id INTEGER',
        'cluster_well_log_parameter_from_calculator': 'dataset_id INTEGER, calculator_id INTEGER',
        'cluster_well_log_dataset_data': 'dataset_id INTEGER, data TEXT',
    }
    for table, columns in definitions.items():
        connection.exec_driver_sql(f'CREATE TABLE {table} (id INTEGER NOT NULL PRIMARY KEY, {columns})')


@pytest.mark.parametrize('cluster_schema', ['legacy', 'current', 'partial'])
@pytest.mark.parametrize('existing_targets', [0, 2, 4])
def test_old_database_upgrades_to_head_without_losing_data(tmp_path, cluster_schema, existing_targets):
    config = _config(tmp_path / 'old.sqlite')
    engine = sa.create_engine(config.get_main_option('sqlalchemy.url'))
    with engine.begin() as connection:
        if cluster_schema == 'legacy':
            _legacy_cluster_tables(connection)
        connection.exec_driver_sql(
            'CREATE TABLE markup_reg (id INTEGER PRIMARY KEY, analysis_id INTEGER, well_id INTEGER, '
            'profile_id INTEGER, formation_id INTEGER, target_value FLOAT, '
            'list_measure TEXT, list_fake TEXT, type_markup VARCHAR)'
        )
        for name in TARGET_COLUMNS[:existing_targets]:
            column_type = 'BOOLEAN' if name == 'target_is_manual_override' else 'TEXT'
            connection.exec_driver_sql(f'ALTER TABLE markup_reg ADD COLUMN {name} {column_type}')
        connection.exec_driver_sql(
            "INSERT INTO markup_reg (id, analysis_id, target_value, list_measure) "
            "VALUES (1, 15, 42.5, '[1, 2]')"
        )
        if existing_targets:
            connection.exec_driver_sql(
                "UPDATE markup_reg SET target_source_type = 'well_log', target_source_config = '{}'"
            )
        connection.exec_driver_sql('CREATE TABLE lineup_train (id INTEGER PRIMARY KEY)')

    # Simulate the updated application's create_all before the failed ORM query.
    Base.metadata.create_all(engine)
    with engine.begin() as connection:
        timestamp = 'created_by' if cluster_schema == 'legacy' else 'created_at'
        connection.exec_driver_sql(
            f"INSERT INTO well_log_cluster_dataset (id, name, {timestamp}) "
            "VALUES (1, 'dataset', '2025-01-02 03:04:05')"
        )
        normalized = '' if cluster_schema == 'legacy' else ', canonical_name_norm'
        value = '' if cluster_schema == 'legacy' else ", 'гамма'"
        connection.exec_driver_sql(
            f"INSERT INTO canonical_well_log (id, canonical_name{normalized}) VALUES (1, 'ГАММА'{value})"
        )
        connection.exec_driver_sql(
            'INSERT INTO well_for_cluster (id, dataset_id, well_id, top_md, bottom_md) '
            'VALUES (1, 1, 1, 10, 20)'
        )
        connection.exec_driver_sql(
            "INSERT INTO well_log_cluster_calculation_cache "
            "(id, dataset_id, cache_key, created_at, data_hash, config_json, result_payload, "
            "labels_json, kept_row_indices_json, assignments_json, postprocess_results_json) "
            "VALUES (1, 1, 'key', '2025-01-01', 'hash', '{}', '{\"value\":42}', '[7]', '[0]', '[]', '{}')"
        )
        if cluster_schema == 'legacy':
            for table in (*CACHE_TABLES, *ALIAS_TABLES):
                connection.exec_driver_sql(f'DROP TABLE {table}')
        elif cluster_schema == 'partial':
            # A previous SQLite migration may have committed some DDL already.
            for table in (*HARDENED_TABLES, *CACHE_TABLES, *ALIAS_TABLES):
                for index in sa.inspect(connection).get_indexes(table):
                    connection.exec_driver_sql(f'DROP INDEX "{index["name"]}"')
            for table in CACHE_TABLES[1:]:
                for column in ('kept_row_indices_json', 'assignments_json', 'postprocess_results_json'):
                    connection.exec_driver_sql(f'ALTER TABLE {table} DROP COLUMN {column}')

    command.stamp(config, '436670216df8')
    command.upgrade(config, 'head')
    with sqlite3.connect(tmp_path / 'old.sqlite') as database:
        before_repeat = list(database.iterdump())
    command.upgrade(config, 'head')
    with sqlite3.connect(tmp_path / 'old.sqlite') as database:
        assert list(database.iterdump()) == before_repeat

    with engine.begin() as connection:
        assert connection.exec_driver_sql('SELECT version_num FROM alembic_version').scalar_one() == (
            ScriptDirectory.from_config(config).get_current_head()
        )
        assert not sa.inspect(connection).has_table('lineup_train')
        assert connection.exec_driver_sql(
            'SELECT analysis_id, target_value, list_measure FROM markup_reg'
        ).one() == (15, 42.5, '[1, 2]')
        assert connection.exec_driver_sql('SELECT created_at FROM well_log_cluster_dataset').scalar_one() == (
            '2025-01-02 03:04:05'
        )
        expected_norm = 'ГАММА'.lower() if cluster_schema != 'legacy' else 'ГАММА'
        assert connection.exec_driver_sql('SELECT canonical_name_norm FROM canonical_well_log').scalar_one() == expected_norm
        assert set(TARGET_COLUMNS) <= {column['name'] for column in sa.inspect(connection).get_columns('markup_reg')}
        if existing_targets:
            assert connection.exec_driver_sql('SELECT target_source_type FROM markup_reg').scalar_one() == 'well_log'
        else:
            assert connection.exec_driver_sql('SELECT target_is_manual_override FROM markup_reg').scalar_one() == 0
        if cluster_schema != 'legacy':
            assert connection.exec_driver_sql(
                'SELECT labels_json, result_payload FROM well_log_cluster_calculation_cache'
            ).one() == ('[7]', '{"value":42}')
        inspector = sa.inspect(connection)
        for table in (*HARDENED_TABLES, *CACHE_TABLES, *ALIAS_TABLES):
            expected = {index.name for index in Base.metadata.tables[table].indexes}
            assert expected <= {index['name'] for index in inspector.get_indexes(table)}
            expected_constraints = {
                constraint.name for constraint in Base.metadata.tables[table].constraints
                if isinstance(constraint, sa.UniqueConstraint)
            }
            assert expected_constraints <= {constraint['name'] for constraint in inspector.get_unique_constraints(table)}
        for table in HARDENED_TABLES:
            expected_required = {column.name for column in Base.metadata.tables[table].columns if not column.nullable}
            assert expected_required <= {column['name'] for column in inspector.get_columns(table) if not column['nullable']}
        with pytest.raises(sa.exc.IntegrityError):
            connection.exec_driver_sql(
                'INSERT INTO well_for_cluster (dataset_id, well_id, top_md, bottom_md) VALUES (1, 1, 20, 30)'
            )
    engine.dispose()
