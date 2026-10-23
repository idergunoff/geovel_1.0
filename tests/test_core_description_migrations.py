"""Upgrade databases whose tables were created before Alembic recorded them."""

from pathlib import Path

import pytest
import sqlalchemy as sa
from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory

from models_db.model import Base


ROOT = Path(__file__).resolve().parents[1]
CORE_TABLES = ('core_description_document', 'core_description_interval', 'core_description_rock')


def _config(database):
    config = Config(str(ROOT / 'alembic.ini'))
    config.set_main_option('script_location', str(ROOT / 'alembic'))
    config.set_main_option('sqlalchemy.url', f'sqlite:///{database}')
    return config


@pytest.mark.parametrize('existing', ['none', 'choices', 'document', 'legacy_core', 'thickness', 'index', 'current'])
@pytest.mark.parametrize('missing_indexes', [False, True])
def test_upgrade_head_preserves_preexisting_tables(tmp_path, existing, missing_indexes):
    config = _config(tmp_path / 'legacy.sqlite')
    engine = sa.create_engine(config.get_main_option('sqlalchemy.url'))
    tables = [Base.metadata.tables['well']]
    if existing != 'none':
        tables.append(Base.metadata.tables['well_parameter_choice'])
    if existing == 'document':
        tables.append(Base.metadata.tables[CORE_TABLES[0]])
    elif existing in ('legacy_core', 'thickness', 'index', 'current'):
        tables.extend(Base.metadata.tables[name] for name in CORE_TABLES)
    Base.metadata.create_all(engine, tables=tables)
    with engine.begin() as connection:
        connection.exec_driver_sql('INSERT INTO well (id, name) VALUES (1, \'12\')')
        if existing != 'none':
            connection.exec_driver_sql(
                "INSERT INTO well_parameter_choice VALUES (1, 1, 'boundary', 2, 3, '2026-01-01')"
            )
        if existing in ('legacy_core', 'thickness', 'index', 'current'):
            connection.exec_driver_sql(
                "INSERT INTO core_description_document "
                "(id, well_id, source_file_name, source_file_hash, source_format, parser_version, "
                "match_method, match_confidence, imported_at, status, diagnostics) "
                "VALUES (1, 1, 'core.docx', 'hash', 'docx', '1', 'manual', 1, '2026-01-01', 'imported', '{}')"
            )
            connection.exec_driver_sql(
                "INSERT INTO core_description_interval "
                "(id, document_id, well_id, top_depth, bottom_depth, raw_description, "
                "normalized_description, oil_saturation, confidence, needs_review, "
                "source_table_index, source_row_index, manually_edited, recognition_details, thickness, geological_index_raw) "
                "VALUES (1, 1, 1, 45, 56.5, 'Глина', 'глина', 'unknown', 1, 0, 3, 3, 1, '{}', 11.5, 'J₂')"
            )
            connection.exec_driver_sql(
                "INSERT INTO core_description_rock "
                "(id, interval_id, rock_name, role, source_text, confidence) "
                "VALUES (1, 1, 'clay', 'main', 'Глина', 1)"
            )
            if existing in ('legacy_core', 'thickness'):
                connection.exec_driver_sql('ALTER TABLE core_description_interval DROP COLUMN geological_index_raw')
            if existing in ('legacy_core', 'index'):
                connection.exec_driver_sql('ALTER TABLE core_description_interval DROP COLUMN thickness')
        if missing_indexes:
            for table in tables[1:]:
                for index in sa.inspect(connection).get_indexes(table.name):
                    connection.exec_driver_sql(f'DROP INDEX "{index["name"]}"')

    # Both earlier branches are applied; only their version markers are present.
    command.stamp(config, ['b8c9d0e1f2a3', '0d253a325606'])
    command.upgrade(config, 'head')
    command.upgrade(config, 'head')

    with engine.begin() as connection:
        assert connection.exec_driver_sql('SELECT version_num FROM alembic_version').scalars().all() == ['2f3a4b5c6d7e']
        inspector = sa.inspect(connection)
        for name in ('well_parameter_choice', *CORE_TABLES):
            assert inspector.has_table(name)
            expected = {index.name for index in Base.metadata.tables[name].indexes}
            assert expected <= {index['name'] for index in inspector.get_indexes(name)}
        if existing != 'none':
            assert connection.exec_driver_sql('SELECT source_id FROM well_parameter_choice').scalar_one() == 3
        if existing in ('legacy_core', 'thickness', 'index', 'current'):
            row = connection.exec_driver_sql(
                'SELECT raw_description, thickness, geological_index_raw FROM core_description_interval'
            ).one()
            assert row == (
                'Глина', None if existing in ('legacy_core', 'index') else 11.5,
                'J₂' if existing in ('index', 'current') else None,
            )
            assert connection.exec_driver_sql('SELECT rock_name FROM core_description_rock').scalar_one() == 'clay'
            connection.exec_driver_sql(
                "UPDATE core_description_interval SET thickness = 11.5, geological_index_raw = 'Р2kz11' WHERE id = 1"
            )
            assert connection.exec_driver_sql('SELECT thickness FROM core_description_interval').scalar_one() == 11.5
            with pytest.raises(sa.exc.IntegrityError):
                connection.exec_driver_sql('UPDATE core_description_interval SET bottom_depth = top_depth WHERE id = 1')
        with pytest.raises(sa.exc.IntegrityError):
            connection.exec_driver_sql(
                "INSERT INTO well_parameter_choice (well_id, source_type, canonical_id, source_id, updated_at) "
                "VALUES (1, 'test', 5, 1, '2026-01-01'), (1, 'test', 5, 2, '2026-01-01')"
            )
        document = Base.metadata.tables['core_description_document']
        document_id = connection.execute(document.insert().values(
            well_id=1, source_file_name='new.docx', source_file_hash='new_hash',
            source_format='docx', parser_version='1', match_method='manual', match_confidence=1,
        )).inserted_primary_key[0]
        interval = Base.metadata.tables['core_description_interval']
        interval_id = connection.execute(interval.insert().values(
            document_id=document_id, well_id=1, top_depth=45, bottom_depth=56.5,
            raw_description='Глина', normalized_description='глина', thickness=11.5,
            geological_index_raw='Р2kz11', confidence=1, source_table_index=3, source_row_index=3,
        )).inserted_primary_key[0]
        assert connection.execute(sa.select(interval.c.thickness, interval.c.geological_index_raw)
                                  .where(interval.c.id == interval_id)).one() == (11.5, 'Р2kz11')
    engine.dispose()


def test_core_upgrade_can_generate_offline_sql(tmp_path):
    import io
    from alembic.migration import MigrationContext
    from alembic.operations import Operations

    config = _config(tmp_path / 'unused.sqlite')
    config.output_buffer = io.StringIO()
    # Start after the merge so unrelated legacy batch migrations are excluded.
    command.upgrade(config, '1e2f3a4b5c6d:head', sql=True)
    sql = config.output_buffer.getvalue()
    assert 'ADD COLUMN thickness FLOAT' in sql
    assert 'ADD COLUMN geological_index_raw TEXT' in sql
    scripts = ScriptDirectory.from_config(config)
    assert scripts.get_heads() == ['2f3a4b5c6d7e']
    output = io.StringIO()
    context = MigrationContext.configure(dialect_name='sqlite', opts={'as_sql': True, 'output_buffer': output})
    with Operations.context(context):
        for revision in ('c9d0e1f2a3b4', '1e2f3a4b5c6d', '2f3a4b5c6d7e'):
            scripts.get_revision(revision).module.upgrade()
    sql = output.getvalue()
    assert 'CREATE TABLE well_parameter_choice' in sql
    assert 'CREATE TABLE core_description_interval' in sql
    assert 'ADD COLUMN thickness FLOAT' in sql
