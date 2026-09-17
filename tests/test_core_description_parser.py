"""Contract tests for stage 2 of the core-description import."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from core_description import (
    CoreDescriptionParseError,
    DictionaryValidationError,
    analyze_description,
    load_dictionary,
    parse_core_document,
)
from core_description.converter import DocConversionError, docx_source

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "core_description"


@pytest.fixture(scope="module")
def generated(tmp_path_factory):
    output = tmp_path_factory.mktemp("core-docx")
    subprocess.run(
        [sys.executable, str(FIXTURE_DIR / "generate_fixtures.py"), str(output)],
        check=True,
        capture_output=True,
        text=True,
    )
    return output


def test_supported_fixtures_have_expected_structural_result(generated):
    expected = json.loads((FIXTURE_DIR / "fixture_expectations.json").read_text(encoding="utf-8"))
    for filename, contract in expected["fixtures"].items():
        result = parse_core_document(generated / filename)
        assert result.well_name == contract["well_name"]
        assert result.area_name == contract["area_name"]
        assert result.described_by_raw == contract["described_by_raw"]
        assert result.described_by == contract["described_by"]
        assert len(result.intervals) == len(contract["intervals"])
        assert result.file_hash
        for actual, wanted in zip(result.intervals, contract["intervals"]):
            assert actual.top_depth == wanted["top_depth"]
            assert actual.bottom_depth == wanted["bottom_depth"]
            assert actual.warnings == wanted.get("warnings", [])
            assert actual.errors == wanted.get("errors", [])
            assert actual.selected_by_default == wanted.get("selected_by_default", True)
            assert actual.raw_description
            assert [rock.canonical_value for rock in actual.rocks] == wanted["rocks"]
            assert actual.oil_saturation == wanted["oil_saturation"]
            assert 0.0 <= actual.confidence <= 1.0
            assert all(match.rule_id for match in actual.semantic_matches)
        assert set(result.warnings) == set(contract.get("document_warnings", []))
        assert result.dictionary_version == "1.0.0"


def test_source_coordinates_and_optional_columns(generated):
    result = parse_core_document(generated / "standard_separate_columns.docx")
    assert [(row.source_table, row.source_row) for row in result.intervals] == [(0, 1), (0, 2)]
    assert result.intervals[0].manifestation_raw == "битум"
    assert result.intervals[1].manifestation_raw is None


def test_cli_emits_json_and_is_independent_of_gui_or_database(generated):
    path = generated / "single_range_column.docx"
    process = subprocess.run(
        [sys.executable, "-m", "core_description", str(path), "--indent", "0"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert process.returncode == 0, process.stderr
    payload = json.loads(process.stdout)
    assert payload[0]["well_name"] == "TEST-202"
    assert payload[0]["intervals"][0]["top_depth"] == 205.0


def test_unknown_extension_is_rejected(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("not a Word document", encoding="utf-8")
    with pytest.raises(CoreDescriptionParseError, match="Unsupported file extension"):
        parse_core_document(path)


def test_broken_docx_is_reported_as_parser_error(tmp_path):
    path = tmp_path / "broken.docx"
    path.write_bytes(b"not a zip")
    with pytest.raises(CoreDescriptionParseError, match="Cannot open DOCX"):
        parse_core_document(path)


def test_doc_conversion_requires_libreoffice(tmp_path, monkeypatch):
    source = tmp_path / "legacy.doc"
    source.write_bytes(b"legacy")
    monkeypatch.setattr("core_description.converter.shutil.which", lambda _name: None)
    with pytest.raises(DocConversionError, match="LibreOffice is required"):
        with docx_source(source):
            pass


def test_rock_relations_and_explanations_are_preserved():
    rocks, oil, confidence, matches, warnings = analyze_description(
        "Глина с прослоями известняка, без признаков нефти."
    )
    assert [(rock.canonical_value, rock.relation) for rock in rocks] == [
        ("clay", "primary"), ("limestone", "interlayer")
    ]
    assert oil == "none"
    assert confidence == 1.0
    assert warnings == []
    assert {match.category for match in matches} >= {"rock", "rock_relation", "negation"}


def test_negation_has_priority_and_uncertainty_marks_disputed_result():
    assert analyze_description("Песчаник не нефтенасыщен.")[1] == "none"
    rocks, oil, confidence, _, warnings = analyze_description(
        "Известняк, предположительно нефтенасыщенный."
    )
    assert [rock.canonical_value for rock in rocks] == ["limestone"]
    assert oil == "uncertain"
    assert confidence < 1.0
    assert warnings == ["uncertain_wording"]


def test_dictionary_validation_rejects_duplicate_ids(tmp_path):
    path = tmp_path / "dictionary.json"
    rule = {
        "id": "duplicate", "category": "rock", "canonical_value": "x",
        "patterns": ["x"], "pattern_type": "word", "priority": 1, "enabled": True,
    }
    path.write_text(json.dumps({
        "schema_version": 1, "version": "test", "rules": [rule, rule]
    }), encoding="utf-8")
    with pytest.raises(DictionaryValidationError, match="Duplicate rule id"):
        load_dictionary(path)
