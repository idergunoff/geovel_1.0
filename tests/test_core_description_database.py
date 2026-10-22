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
from core_description.areas import area_names_match
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


@pytest.mark.parametrize('file_area, database_area', [
    ('Сарабикуловское поднятие', 'Сарабикуловский'),
    ('Ягодное поднятие', 'Ягодный'),
    ('Мельничное поднятие', 'Мельничный'),
    ('Площадь: Сарабикуловское', 'Сарабикуловская'),
    ('Нефтеперспективный объект №58', 'Объект 58'),
    ('Газоперспективный объект 58', '58'),
    ('Нефтеперспективный объект №8094', 'Перспективный объект № 8094'),
])
def test_area_variants_offer_correct_well_for_confirmation(db_session, file_area, database_area):
    other = Well(name='12')
    wanted = Well(name='12')
    db_session.add_all([other, wanted])
    db_session.flush()
    db_session.add_all([
        WellOptionally(well_id=other.id, option='Площадь', value='Южная'),
        WellOptionally(well_id=wanted.id, option='Площадь', value=database_area),
    ])
    db_session.commit()

    result = match_well(db_session, '12', file_area)
    assert result.status == 'manual_required'
    assert result.selected is None
    assert [candidate.well_id for candidate in result.candidates] == [wanted.id]
    assert result.candidates[0].method == 'area_variant_suggestion'


@pytest.mark.parametrize('first, second', [
    ('Объект 58', 'Объект 580'), ('Объект 58', 'Объект 59'),
    ('Мельничное поднятие', 'Ягодный'), ('', ''),
])
def test_area_lookup_preserves_distinct_names_and_numbers(first, second):
    assert not area_names_match(first, second)


def test_area_ranking_is_applied_before_suggestion_limit(db_session):
    wells = [Well(name='120') for _ in range(12)]
    db_session.add_all(wells)
    db_session.flush()
    db_session.add(WellOptionally(well_id=wells[-1].id, option='Площадь', value='Сарабикуловский'))
    db_session.commit()
    result = match_well(db_session, '12', 'Сарабикуловское поднятие')
    assert len(result.candidates) == 10
    assert result.candidates[0].well_id == wells[-1].id


def test_single_well_area_variant_still_requires_manual_choice(db_session):
    well = Well(name='12')
    db_session.add(well)
    db_session.flush()
    db_session.add(WellOptionally(well_id=well.id, option='Площадь', value='Сарабикуловский'))
    db_session.commit()
    result = match_well(db_session, '12', 'Сарабикуловское поднятие')
    assert result.selected is None
    assert [candidate.well_id for candidate in result.candidates] == [well.id]


@pytest.mark.parametrize('category', ['Залежь', 'Месторождение', 'Участок', 'Произвольная категория'])
def test_area_can_match_any_data_of_well_category(db_session, category):
    first, second = Well(name='12'), Well(name='12')
    db_session.add_all([first, second])
    db_session.flush()
    db_session.add(WellOptionally(well_id=second.id, option=category, value='Северная'))
    db_session.commit()
    result = match_well(db_session, '12', 'Северная')
    assert result.selected.well_id == second.id


@pytest.mark.parametrize('field, value, category, stored', [
    ('drilling_start', '2019-08-28', 'Дата начала бурения', '28.08.19'),
    ('drilling_end', '2019-09-02', 'Скважина окончена', '2019-09-02'),
    ('wellhead_elevation', 132.432, 'Альтитуда устья', '132,43 м'),
])
def test_one_identity_match_is_only_a_suggestion(db_session, field, value, category, stored):
    first, second = Well(name='12'), Well(name='12')
    db_session.add_all([first, second])
    db_session.flush()
    db_session.add(WellOptionally(well_id=second.id, option=category, value=stored))
    db_session.commit()
    source = _document()
    setattr(source, field, value)
    result = match_well(db_session, '12', source.area_name, document=source)
    assert result.selected is None
    candidate = next(item for item in result.candidates if item.well_id == second.id)
    assert any(item.field == field and item.status == 'match' for item in candidate.comparisons)


def test_altitude_alone_does_not_select_a_well(db_session):
    first, second = Well(name='12', alt=133), Well(name='12', alt=132.43)
    db_session.add_all([first, second])
    db_session.commit()
    source = _document()
    source.wellhead_elevation = 132.432
    result = match_well(db_session, '12', document=source)
    assert result.selected is None


@pytest.mark.parametrize('count', [0, 1, 2, 3, 4])
def test_automatic_selection_requires_two_distinct_categories(db_session, count):
    well = Well(name='12', alt=134 if count >= 4 else None)
    db_session.add(well)
    db_session.flush()
    entries = [('Участок', 'мельничный уч'), ('Начало бурения', '28.08.2019'),
               ('Окончание бурения', '02.09.2019')]
    db_session.add_all([WellOptionally(well_id=well.id, option=label, value=value)
                       for label, value in entries[:count]])
    db_session.commit()
    source = _document()
    source.area_name = 'Мельничное поднятие'
    source.drilling_start = '2019-08-28'
    source.drilling_end = '2019-09-02'
    source.wellhead_elevation = 132.432
    result = match_well(db_session, '12', source.area_name, document=source)
    if count >= 2:
        assert result.selected.well_id == well.id
        assert result.selected.method == 'exact_name_identity'
        assert sum(item.status == 'match' for item in result.selected.comparisons) == count
    else:
        assert result.selected is None


def test_two_metre_tolerance_can_supply_second_category(db_session):
    well = Well(name='12', alt=134.432)
    db_session.add(well)
    db_session.flush()
    db_session.add_all([
        WellOptionally(well_id=well.id, option='Участок', value='мельничный уч'),
    ])
    db_session.commit()
    source = _document()
    source.area_name = 'Мельничное поднятие'
    source.wellhead_elevation = 132.432
    result = match_well(db_session, '12', source.area_name, document=source)
    assert result.selected.well_id == well.id
    well.alt = 134.433
    db_session.commit()
    assert match_well(db_session, '12', source.area_name, document=source).selected is None


def test_multiple_qualifying_wells_still_require_manual_selection(db_session):
    wells = [Well(name='12'), Well(name='12')]
    db_session.add_all(wells)
    db_session.flush()
    for well in wells:
        db_session.add_all([WellOptionally(well_id=well.id, option=label, value=value) for label, value in (
            ('Залежь', 'Мельничный уч'), ('Начало бурения', '28.08.2019'), ('Окончание бурения', '02.09.2019'))])
    db_session.commit()
    source = _document()
    source.area_name = 'Мельничное поднятие'
    source.drilling_start = '2019-08-28'
    result = match_well(db_session, '12', source.area_name, document=source)
    assert result.selected is None
    assert 'multiple_identity_matches' in result.warnings


def test_duplicate_area_values_count_as_one_category(db_session):
    well = Well(name='12')
    db_session.add(well)
    db_session.flush()
    db_session.add_all([WellOptionally(well_id=well.id, option=label, value='мельничный уч')
                       for label in ('Площадь', 'Залежь', 'Участок')])
    db_session.commit()
    source = _document()
    source.area_name = 'Мельничное поднятие'
    assert match_well(db_session, '12', source.area_name, document=source).selected is None


@pytest.mark.parametrize('different_field', ['area', 'drilling_start', 'drilling_end', 'wellhead_elevation'])
def test_three_matches_can_select_despite_a_fourth_difference(db_session, different_field):
    well = Well(name='12', alt=140 if different_field == 'wellhead_elevation' else 134)
    db_session.add(well)
    db_session.flush()
    db_session.add_all([WellOptionally(well_id=well.id, option=label, value=value) for label, value in (
        ('Участок', 'Ягодный' if different_field == 'area' else 'мельничный уч'),
        ('Начало бурения', '29.08.2019' if different_field == 'drilling_start' else '28.08.2019'),
        ('Окончание бурения', '03.09.2019' if different_field == 'drilling_end' else '02.09.2019'))])
    db_session.commit()
    source = _document()
    source.area_name = 'Мельничное поднятие'
    source.drilling_start, source.drilling_end = '2019-08-28', '2019-09-02'
    source.wellhead_elevation = 132.432
    result = match_well(db_session, '12', source.area_name, document=source)
    assert result.selected.well_id == well.id
    assert sum(item.status == 'match' for item in result.selected.comparisons) == 3
    assert next(item.status for item in result.selected.comparisons if item.field == different_field) == 'different'


@pytest.mark.parametrize('problem', ['multiple_matches', 'partial_date', 'conflicting_date', 'similar_number'])
def test_ambiguous_or_conflicting_evidence_needs_manual_selection(db_session, problem):
    name = '120' if problem == 'similar_number' else '12'
    first, second = Well(name=name), Well(name=name)
    db_session.add_all([first, second])
    db_session.flush()
    stored = '08.2019' if problem == 'partial_date' else '28.08.2019'
    db_session.add(WellOptionally(well_id=first.id, option='Начало бурения', value=stored))
    if problem == 'multiple_matches':
        db_session.add(WellOptionally(well_id=second.id, option='Начало бурения', value=stored))
    if problem == 'conflicting_date':
        db_session.add(WellOptionally(well_id=first.id, option='Окончание бурения', value='05.09.2019'))
    db_session.commit()
    source = _document()
    source.drilling_start = '2019-08-28'
    if problem == 'conflicting_date':
        source.drilling_end = '2019-09-02'
    result = match_well(db_session, '12', document=source)
    assert result.selected is None
    assert result.candidates


def test_conflicting_metadata_blocks_an_otherwise_exact_label_match(db_session):
    well = Well(name='12', alt=130)
    db_session.add(well)
    db_session.commit()
    source = _document()
    source.wellhead_elevation = 132.432
    result = match_well(db_session, '12', document=source)
    assert result.selected is None
    assert 'identity_conflict' in result.warnings


def test_identity_evidence_is_retained_when_saving(db_session):
    well = Well(name='12', alt=132.43)
    db_session.add(well)
    db_session.commit()
    source = _document()
    source.wellhead_elevation = 132.432
    source.wellhead_elevation_raw = '132,432 м'
    save_core_import(db_session, source, well_id=well.id, selected_interval_ids={'0:3'})
    diagnostics = json.loads(db_session.query(CoreDescriptionDocument).one().diagnostics)
    assert diagnostics['well_identity']['wellhead_elevation_raw'] == '132,432 м'
    assert diagnostics['well_comparison'][3]['status'] == 'match'
    assert diagnostics['well_match_policy'] == {
        'elevation_tolerance_m': 2.0, 'area_similarity_above': 0.5, 'minimum_categories': 2}


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
    assert db_session.query(CoreDescriptionDocument).one().dictionary_version == '1.3.0'
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


@pytest.mark.parametrize('text, expected', [
    ('Песчаник остаточно нефтенасыщенный', 'residual'),
    ('Песчаник с запахом нефти', 'odor'),
    ('Песчаник, пятна нефти', 'patchy'),
])
def test_new_saturation_categories_survive_save(db_session, text, expected):
    well = Well(name='12')
    db_session.add(well)
    db_session.commit()
    parsed = _document()
    parsed.intervals[0].raw_description = text
    parsed.dictionary_version = enrich_intervals(parsed.intervals)
    save_core_import(db_session, parsed, well_id=well.id, selected_interval_ids={'0:3'})
    saved = db_session.query(CoreDescriptionInterval).one()
    assert saved.oil_saturation == expected
    assert saved.raw_description == text
    assert any(match['canonical_value'] == expected
               for match in json.loads(saved.recognition_details)['matches'])
