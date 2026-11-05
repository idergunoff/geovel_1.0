"""Real SQLite transactions, semantic previews, protected edits and undo."""

import json

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from core_description.database import save_core_import
from core_description.models import ParsedCoreDocument, ParsedCoreInterval
from core_description.reparse import (
    ReparseCancelled, ReparseError, apply_reparse, interval_token, preview_reparse, processing_metadata,
    recognition_details, undo_reparse,
)
from core_description.semantics import load_dictionary_payload
from models_db.model import Base, CoreDescriptionDocument, CoreDescriptionInterval, Well
import models_db.model_cluster
import models_db.model_profile_features


@pytest.fixture()
def dataset(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'core.sqlite'}")
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine)()
    db.add_all([Well(id=1, name="12"), Well(id=2, name="13")])
    db.commit()
    for well_id in (1, 2):
        document = ParsedCoreDocument(
            source_path="core.docx", source_format="docx", file_hash=str(well_id) * 64,
            well_name_raw="12", well_name="12", area_name_raw=None, area_name=None,
            described_by_raw=None, described_by=None, dictionary_version="old-dictionary",
            intervals=[ParsedCoreInterval(100, 102, "Глауконитит, пропитан нефтью", 0, 1, thickness=1.5,
                                          geological_index_raw="J2"),
                       ParsedCoreInterval(102, 105, "Песчаник", 0, 2)])
        save_core_import(db, document, well_id=well_id, selected_interval_ids={"0:1", "0:2"})
    yield db
    db.close()
    engine.dispose()


@pytest.fixture()
def payload():
    draft = load_dictionary_payload()
    draft["version"] = "test.new"
    draft["rules"].append({"id": "test.glauconite", "category": "rock", "canonical_value": "Глауконитит",
                           "patterns": ["глауконитит"], "pattern_type": "word", "priority": 200, "enabled": True})
    draft["rules"].append({"id": "test.oil", "category": "oil_indicator", "canonical_value": "present",
                           "patterns": ["пропитан нефтью"], "pattern_type": "phrase", "priority": 200, "enabled": True})
    return draft


def test_preview_is_readonly_and_apply_changes_only_reviewed_semantics(dataset, payload):
    db = dataset
    row = db.query(CoreDescriptionInterval).filter_by(well_id=1, top_depth=100).one()
    original = (row.id, row.document_id, row.well_id, row.top_depth, row.bottom_depth,
                row.raw_description, row.thickness, row.geological_index_raw)
    document_version = row.document.dictionary_version
    preview = preview_reparse(db, well_ids=[1], dictionary_payload=payload)
    assert len(preview.proposals) == 2
    assert row.oil_saturation == "unknown" and not row.rocks
    assert recognition_details(row).get("history") is None
    proposal = next(p for p in preview.proposals if p.interval_id == row.id)
    assert proposal.changed
    assert apply_reparse(db, preview, {row.id}) == 1
    assert row.oil_saturation == "present"
    assert [(rock.rock_name, rock.source_text, rock.match_start, rock.match_end) for rock in row.rocks] == [
        ("Глауконитит", "Глауконитит", 0, len("Глауконитит"))]
    assert original == (row.id, row.document_id, row.well_id, row.top_depth, row.bottom_depth,
                        row.raw_description, row.thickness, row.geological_index_raw)
    assert row.document.dictionary_version == document_version
    assert processing_metadata(row)["dictionary_version"] == "test.new"
    assert processing_metadata(row)["processed_at"]
    assert json.loads(row.document.diagnostics)["semantic_dictionaries"][preview.dictionary_hash] == payload
    assert db.query(CoreDescriptionDocument).count() == 2
    assert db.query(CoreDescriptionInterval).count() == 4
    assert db.query(CoreDescriptionInterval).filter_by(well_id=2, top_depth=100).one().oil_saturation == "unknown"
    assert db.query(CoreDescriptionInterval).filter_by(well_id=1, top_depth=102).one().rocks == []


def test_manual_rows_require_explicit_override_and_undo_preserves_protection(dataset, payload):
    db = dataset
    row = db.query(CoreDescriptionInterval).filter_by(well_id=1, top_depth=100).one()
    row.manually_edited = True
    row.oil_saturation = "intense"
    db.commit()
    preview = preview_reparse(db, well_ids=[1], dictionary_payload=payload)
    assert next(p for p in preview.proposals if p.interval_id == row.id).protected
    with pytest.raises(ReparseError, match="защищён"):
        apply_reparse(db, preview, {row.id})
    assert row.oil_saturation == "intense"
    apply_reparse(db, preview, {row.id}, allow_manual=True)
    assert row.manually_edited and row.oil_saturation == "present"
    undo_reparse(db, row.id, expected_token=interval_token(row))
    assert row.manually_edited and row.oil_saturation == "intense"


@pytest.mark.parametrize("mutation", ["text", "semantic", "deleted", "manual", "depth", "rock"])
def test_external_edit_invalidates_preview_without_overwriting_any_rows(dataset, payload, mutation):
    db = dataset
    preview = preview_reparse(db, well_ids=[1], dictionary_payload=payload)
    first, second = preview.proposals
    other = sessionmaker(bind=db.get_bind())()
    row = other.get(CoreDescriptionInterval, second.interval_id)
    if mutation == "text":
        row.raw_description = "Исправленный текст"
    elif mutation == "semantic":
        row.oil_saturation = "intense"
    elif mutation == "deleted":
        other.delete(row)
    elif mutation == "manual":
        row.manually_edited = True
    elif mutation == "depth":
        row.bottom_depth += 1
    else:
        from models_db.model import CoreDescriptionRock
        row.rocks.append(CoreDescriptionRock(rock_name="clay", role="primary", source_text="глина", confidence=1))
    other.commit()
    other.close()
    with pytest.raises(ReparseError, match="изменилось|удалён"):
        apply_reparse(db, preview, {first.interval_id, second.interval_id})
    assert db.get(CoreDescriptionInterval, first.interval_id).rocks == []
    assert recognition_details(db.get(CoreDescriptionInterval, first.interval_id)).get("history") is None


def test_failure_while_writing_rolls_back_entire_batch(dataset, payload):
    db = dataset
    preview = preview_reparse(db, well_ids=[1], dictionary_payload=payload)

    def fail(_mapper, _connection, target):
        if target.top_depth == 102:
            raise RuntimeError("write failed")

    event.listen(CoreDescriptionInterval, "before_update", fail)
    try:
        with pytest.raises(RuntimeError, match="write failed"):
            apply_reparse(db, preview, {p.interval_id for p in preview.proposals})
    finally:
        event.remove(CoreDescriptionInterval, "before_update", fail)
    assert all(not row.rocks and row.oil_saturation == "unknown" for row in db.query(CoreDescriptionInterval))
    assert all("semantic_dictionaries" not in json.loads(row.diagnostics) for row in db.query(CoreDescriptionDocument))


def test_history_can_undo_multiple_runs_without_recursive_snapshots(dataset, payload):
    db = dataset
    row = db.query(CoreDescriptionInterval).filter_by(well_id=1, top_depth=100).one()
    original_token = interval_token(row)
    for version in ("new.1", "new.2"):
        payload["version"] = version
        preview = preview_reparse(db, interval_ids=[row.id], dictionary_payload=payload)
        apply_reparse(db, preview, {row.id})
    assert len(recognition_details(row)["history"]) == 2
    with pytest.raises(ReparseError):
        undo_reparse(db, row.id, expected_token=original_token)
    undo_reparse(db, row.id)
    assert processing_metadata(row)["dictionary_version"] == "new.1"
    undo_reparse(db, row.id)
    assert row.oil_saturation == "unknown" and row.rocks == []
    assert processing_metadata(row)["dictionary_version"] == "old-dictionary"
    history = recognition_details(row)["history"]
    assert len(history) == 4
    assert all("history" not in entry["before"]["recognition_details"] for entry in history)
    with pytest.raises(ReparseError, match="Нет переразбора"):
        undo_reparse(db, row.id)


def test_reparse_replaces_obsolete_semantic_warnings_and_keeps_structural_diagnostics(dataset, payload):
    db = dataset
    row = db.query(CoreDescriptionInterval).first()
    row.recognition_details = json.dumps({"warnings": ["uncertain_wording", "locality_marker", "source_warning"],
                                          "errors": ["source_error"]})
    db.commit()
    preview = preview_reparse(db, interval_ids=[row.id], dictionary_payload=payload)
    apply_reparse(db, preview, {row.id})
    assert recognition_details(row)["warnings"] == ["source_warning"]
    assert recognition_details(row)["errors"] == ["source_error"]
    assert row.needs_review


def test_empty_scope_and_unknown_selections_never_expand_to_entire_database(dataset, payload):
    with pytest.raises(ReparseError):
        preview_reparse(dataset, dictionary_payload=payload)
    assert preview_reparse(dataset, interval_ids=[], dictionary_payload=payload).proposals == ()
    preview = preview_reparse(dataset, well_ids=[1], dictionary_payload=payload)
    with pytest.raises(ReparseError):
        apply_reparse(dataset, preview, {999999})


def test_repeated_parse_has_no_semantic_changes_and_deduplicates_dictionary_snapshot(dataset, payload):
    db = dataset
    first = preview_reparse(db, well_ids=[1], dictionary_payload=payload)
    apply_reparse(db, first, {p.interval_id for p in first.proposals})
    second = preview_reparse(db, well_ids=[1], dictionary_payload=payload)
    assert not any(p.changed for p in second.proposals)
    apply_reparse(db, second, {p.interval_id for p in second.proposals})
    document = db.query(CoreDescriptionDocument).filter_by(well_id=1).one()
    assert len(json.loads(document.diagnostics)["semantic_dictionaries"]) == 1


def test_cancelling_preview_never_changes_saved_results(dataset, payload):
    with pytest.raises(ReparseCancelled):
        preview_reparse(dataset, well_ids=[1, 2], dictionary_payload=payload,
                        progress=lambda current, total: current < 1)
    assert all(not row.rocks and not recognition_details(row).get("history")
               for row in dataset.query(CoreDescriptionInterval))


@pytest.mark.parametrize("field,value", [("raw_description", "Новый текст"), ("oil_saturation", "intense")])
def test_undo_refuses_results_edited_after_reparse(dataset, payload, field, value):
    row = dataset.query(CoreDescriptionInterval).first()
    preview = preview_reparse(dataset, interval_ids=[row.id], dictionary_payload=payload)
    apply_reparse(dataset, preview, {row.id})
    setattr(row, field, value)
    dataset.commit()
    with pytest.raises(ReparseError, match="Откат заблокирован"):
        undo_reparse(dataset, row.id, expected_token=interval_token(row))
    assert getattr(row, field) == value
