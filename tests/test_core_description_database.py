"""Database contract tests for stage 4 of the core-description import."""

from __future__ import annotations

import pytest

sqlalchemy = pytest.importorskip('sqlalchemy')
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from core_description.database import (
    DuplicateCoreDocumentError, check_duplicates, match_well, save_core_import,
)
from core_description.models import ParsedCoreDocument, ParsedCoreInterval, RockMention
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
