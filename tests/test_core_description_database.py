"""Database contract tests for stage 4 of the core-description import."""

from __future__ import annotations

import json
import sqlite3

import pytest

sqlalchemy = pytest.importorskip('sqlalchemy')
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from core_description.database import (
    DuplicateCoreDocumentError, check_duplicates, match_well, save_core_import,
)
from core_description.models import ParsedCoreDocument, ParsedCoreInterval, RockMention
from core_description.semantics import enrich_intervals
from models_db.model import (
    Base, CoreDescriptionDocument, CoreDescriptionInterval, CoreDescriptionRock,
    Well, WellOptionally,
)
# Register models referenced by string relationships in the legacy root model.
import models_db.model_cluster  # noqa: E402,F401
import models_db.model_profile_features  # noqa: E402,F401


@pytest.fixture()
def db_session():
    engine = create_engine('sqlite:///:memory:')

    @event.listens_for(engine, 'connect')
    def enable_foreign_keys(connection, _record):
        connection.execute('PRAGMA foreign_keys=ON')

    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    try:
        yield session
    finally:
        session.close()


def _document(file_hash='a' * 64):
    interval_one = ParsedCoreInterval(
        100.0, 102.0, 'Песчаник интенсивно нефтенасыщенный', 0, 3,
        rocks=[RockMention('sandstone', 'Песчаник')],
        oil_saturation='intense', confidence=1.0,
    )
    interval_two = ParsedCoreInterval(
        102.0, 103.5, 'Аргиллит', 0, 4,
        rocks=[RockMention('claystone', 'Аргиллит')], confidence=1.0,
    )
    return ParsedCoreDocument(
        source_path='/incoming/core.docx', source_format='docx', file_hash=file_hash,
        well_name_raw=' № 12 ', well_name='12', area_name_raw='Северная',
        area_name='Северная', described_by_raw='Описал: Иванов И.И.',
        described_by='Иванов И.И.', intervals=[interval_one, interval_two],
        dictionary_version='1.0',
    )


def test_match_well_uses_area_to_resolve_duplicate_names(db_session):
    first = Well(name='12')
    second = Well(name='12')
    db_session.add_all([first, second])
    db_session.flush()
    db_session.add_all([
        WellOptionally(well_id=first.id, option='Площадь', value='Южная'),
        WellOptionally(well_id=second.id, option='пл.', value='Северная'),
    ])
    db_session.commit()

    result = match_well(db_session, '12', ' северная ')

    assert result.status == 'matched'
    assert result.selected.well_id == second.id
    assert result.selected.method == 'exact_name_area'


@pytest.mark.parametrize('area_name', ['Северная', None])
def test_match_well_with_many_duplicate_names_respects_sqlite_limit(db_session, area_name):
    wells = [Well(name='12') for _ in range(1205)]
    db_session.add_all(wells)
    db_session.flush()
    well_ids = [well.id for well in wells]
    db_session.add_all([
        WellOptionally(well_id=well_id, option='Площадь',
                       value='Северная' if index == len(well_ids) - 1 else 'Южная')
        for index, well_id in enumerate(well_ids)
    ])
    db_session.commit()
    connection = db_session.connection().connection.driver_connection
    connection.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, 999)

    result = match_well(db_session, '12', area_name)

    if area_name:
        assert result.status == 'matched'
        assert result.selected.well_id == well_ids[-1]
        assert result.selected.area_name == 'Северная'
        assert result.selected.method == 'exact_name_area'
    else:
        assert result.status == 'ambiguous'
        assert [item.well_id for item in result.candidates] == well_ids
        assert result.candidates[-1].area_name == 'Северная'


@pytest.mark.parametrize('well_name, expected_area_ids', [('1234', [1206]), ('120', list(range(1, 11)))])
def test_match_well_loads_areas_only_for_returned_candidates(db_session, well_name, expected_area_ids):
    db_session.add_all([Well(id=index, name='12') for index in range(1, 1206)])
    db_session.add(Well(id=1206, name='1234'))
    db_session.flush()
    db_session.add_all([
        WellOptionally(well_id=index, option='пл.', value=f'Площадь {index}')
        for index in range(1, 1207)
    ])
    db_session.commit()
    connection = db_session.connection().connection.driver_connection
    connection.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, 999)
    queried_area_ids = []

    def record_area_query(_connection, _cursor, statement, parameters, _context, _executemany):
        if 'FROM well_optionally' in statement:
            queried_area_ids.extend(parameters)

    engine = db_session.get_bind()
    event.listen(engine, 'before_cursor_execute', record_area_query)
    try:
        result = match_well(db_session, well_name)
    finally:
        event.remove(engine, 'before_cursor_execute', record_area_query)

    assert queried_area_ids == expected_area_ids
    if result.selected:
        assert result.status == 'matched'
        assert result.selected.well_id == 1206
        assert result.selected.area_name == 'Площадь 1206'
    else:
        assert result.status == 'manual_required'
        assert [item.well_id for item in result.candidates] == expected_area_ids
        assert [item.area_name for item in result.candidates] == [
            f'Площадь {index}' for index in expected_area_ids
        ]


def test_extended_dictionary_results_and_evidence_survive_save(db_session):
    well = Well(name='12')
    db_session.add(well)
    db_session.commit()
    parsed = _document()
    interval = parsed.intervals[0]
    interval.raw_description = (
        'Алевролиты с прослойками мергелей, слабо, средне и интенсивно битумонасыщенные.'
    )
    parsed.dictionary_version = enrich_intervals(parsed.intervals)

    save_core_import(db_session, parsed, well_id=well.id, selected_interval_ids={'0:3'})

    saved = db_session.query(CoreDescriptionInterval).one()
    assert saved.raw_description == interval.raw_description
    assert saved.oil_saturation == 'weak_to_intense'
    assert db_session.query(CoreDescriptionDocument).one().dictionary_version == '1.2.0'
    assert {(rock.rock_name, rock.role) for rock in db_session.query(CoreDescriptionRock)} == {
        ('siltstone', 'primary'), ('marl', 'interlayer'),
    }
    details = json.loads(saved.recognition_details)
    assert any(
        match['canonical_value'] == 'weak_to_intense'
        and match['matched_text'] == 'слабо, средне и интенсивно битумонасыщенные'
        for match in details['matches']
    )


@pytest.mark.parametrize('edited', [False, True])
def test_thickness_and_geological_index_survive_save(db_session, edited):
    well = Well(name='12')
    db_session.add(well)
    db_session.commit()
    parsed = _document()
    parsed.intervals[0].thickness = 1.5
    parsed.intervals[0].geological_index_raw = 'J₂'
    edits = {'0:3': {'thickness': 1.25, 'geological_index_raw': 'K₁'}} if edited else {}
    save_core_import(db_session, parsed, well_id=well.id, selected_interval_ids={'0:3'}, edits=edits)
    saved = db_session.query(CoreDescriptionInterval).one()
    assert saved.thickness == (1.25 if edited else 1.5)
    assert saved.geological_index_raw == ('K₁' if edited else 'J₂')


def test_thickness_index_migration_preserves_existing_rows():
    import importlib.util
    from pathlib import Path
    from alembic.migration import MigrationContext
    from alembic.operations import Operations

    path = Path(__file__).resolve().parents[1] / 'alembic/versions/2f3a4b5c6d7e_add_core_interval_thickness_index.py'
    spec = importlib.util.spec_from_file_location('core_interval_fields_migration', path)
    migration = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(migration)
    engine = create_engine('sqlite:///:memory:')
    with engine.begin() as connection:
        connection.exec_driver_sql('CREATE TABLE core_description_interval (id INTEGER PRIMARY KEY, raw_description TEXT)')
        connection.exec_driver_sql("INSERT INTO core_description_interval VALUES (1, 'Песчаник')")
        with Operations.context(MigrationContext.configure(connection)):
            migration.upgrade()
            assert connection.exec_driver_sql('SELECT * FROM core_description_interval').one() == (1, 'Песчаник', None, None)
            migration.downgrade()
            assert connection.exec_driver_sql('SELECT * FROM core_description_interval').one() == (1, 'Песчаник')


def test_selective_save_duplicate_rejection_and_replace(db_session):
    well = Well(name='12')
    db_session.add(well)
    db_session.commit()
    parsed = _document()

    result = save_core_import(
        db_session, parsed, well_id=well.id, selected_interval_ids={'0:3'},
        edits={'0:3': {'oil_saturation': 'medium'}}, match_method='exact_name',
    )

    assert result.intervals_saved == 1
    assert result.rocks_saved == 1
    assert db_session.query(CoreDescriptionInterval).one().oil_saturation == 'medium'
    assert db_session.query(CoreDescriptionInterval).one().manually_edited is True
    assert check_duplicates(db_session, parsed, well.id).exact_document_id == result.document_id
    with pytest.raises(DuplicateCoreDocumentError):
        save_core_import(db_session, parsed, well_id=well.id, selected_interval_ids={'0:4'})
    assert db_session.query(CoreDescriptionDocument).count() == 1

    replaced = save_core_import(
        db_session, parsed, well_id=well.id, selected_interval_ids={'0:4'},
        duplicate_policy='replace',
    )
    assert replaced.replaced_document_id == result.document_id
    assert db_session.query(CoreDescriptionDocument).count() == 1
    assert db_session.query(CoreDescriptionInterval).one().source_row_index == 4
    assert db_session.query(CoreDescriptionRock).count() == 1


def test_failed_save_rolls_back_the_whole_document(db_session):
    well = Well(name='12')
    db_session.add(well)
    db_session.commit()
    parsed = _document('b' * 64)

    with pytest.raises(Exception, match='Invalid depths'):
        save_core_import(
            db_session, parsed, well_id=well.id, selected_interval_ids={'0:3', '0:4'},
            edits={'0:4': {'top_depth': 200, 'bottom_depth': 100}},
        )

    assert db_session.query(CoreDescriptionDocument).count() == 0
    assert db_session.query(CoreDescriptionInterval).count() == 0
