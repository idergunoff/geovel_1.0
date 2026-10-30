"""User dictionary persistence and its effect on actual document parsing."""

from copy import deepcopy
from pathlib import Path

import pytest

from core_description import semantics
from core_description.batch import parse_batch
from core_description.parser import parse_core_document


@pytest.fixture()
def dictionary_file(tmp_path, monkeypatch):
    target = tmp_path / "config" / "dictionary.json"
    monkeypatch.setattr(semantics, "user_dictionary_path", lambda: target)
    return target


def _rule(**changes):
    rule = {"id": "custom.rock", "category": "rock", "canonical_value": "Глауконитит",
            "patterns": ["глауконитит"], "pattern_type": "word", "priority": 100, "enabled": True}
    rule.update(changes)
    return rule


def test_save_is_used_by_default_and_preserves_disabled_rules_and_comments(dictionary_file):
    payload = semantics.load_dictionary_payload()
    original = deepcopy(payload)
    payload["rules"].extend([_rule(), _rule(id="disabled", enabled=False, comment="Оставить для справки")])
    saved = semantics.save_dictionary(payload)

    assert dictionary_file.is_file()
    assert payload["version"] == original["version"]
    assert saved["version"].startswith(original["version"] + "+user.")
    assert semantics.load_dictionary_payload() == saved
    assert semantics.analyze_description("Глауконитит")[0][0].canonical_value == "Глауконитит"
    assert "disabled" not in {rule.id for rule in semantics.load_dictionary().rules}
    assert saved["rules"][-1]["comment"] == "Оставить для справки"
    assert semantics.load_dictionary_payload(Path(semantics.__file__).with_name("dictionary.json")) == original


def test_edit_remove_and_disable_change_recognition(dictionary_file):
    payload = {"schema_version": 1, "version": "1.0.0", "rules": [_rule()]}
    first = semantics.save_dictionary(payload)
    assert semantics.analyze_description("Глауконитит")[0]
    payload["rules"][0]["patterns"] = ["глауконитовая порода"]
    payload["rules"][0]["pattern_type"] = "phrase"
    second = semantics.save_dictionary(payload)
    assert first["version"] != second["version"]
    assert not semantics.analyze_description("Глауконитит")[0]
    assert semantics.analyze_description("Глауконитовая порода")[0]
    payload["rules"][0]["enabled"] = False
    semantics.save_dictionary(payload)
    assert not semantics.analyze_description("Глауконитовая порода")[0]
    payload["rules"].clear()
    semantics.save_dictionary(payload)
    assert semantics.load_dictionary().rules == ()


@pytest.mark.parametrize("changes", [
    {"id": []}, {"id": " "}, {"canonical_value": " "}, {"canonical_value": 12},
    {"category": []}, {"category": "invalid"}, {"pattern_type": []},
    {"patterns": [" "]}, {"patterns": []}, {"priority": True}, {"enabled": "yes"},
    {"pattern_type": "regex", "patterns": ["["]},
    {"pattern_type": "regex", "patterns": ["["], "enabled": False},
])
def test_invalid_drafts_never_overwrite_saved_dictionary(dictionary_file, changes):
    payload = {"schema_version": 1, "version": "1.0.0", "rules": [_rule()]}
    semantics.save_dictionary(payload)
    before = dictionary_file.read_bytes()
    payload["rules"][0].update(changes)
    with pytest.raises(semantics.DictionaryValidationError):
        semantics.save_dictionary(payload)
    assert dictionary_file.read_bytes() == before


def test_atomic_replace_failure_leaves_previous_dictionary_usable(dictionary_file, monkeypatch):
    payload = {"schema_version": 1, "version": "1.0.0", "rules": [_rule()]}
    semantics.save_dictionary(payload)
    before = dictionary_file.read_bytes()

    def fail_replace(*_args):
        raise OSError("Disk failure")

    monkeypatch.setattr(semantics.os, "replace", fail_replace)
    payload["rules"].clear()
    with pytest.raises(OSError, match="Disk failure"):
        semantics.save_dictionary(payload)
    assert dictionary_file.read_bytes() == before
    assert list(dictionary_file.parent.iterdir()) == [dictionary_file]


def test_corrupt_custom_dictionary_is_reported_instead_of_silently_ignored(dictionary_file):
    dictionary_file.parent.mkdir(parents=True)
    dictionary_file.write_bytes(b"\xff")
    with pytest.raises(semantics.DictionaryValidationError):
        semantics.load_dictionary()


def test_document_and_batch_parser_use_saved_revision(dictionary_file, tmp_path):
    docx = pytest.importorskip("docx")
    source = tmp_path / "core.docx"
    document = docx.Document()
    document.add_paragraph("Скважина 12")
    table = document.add_table(rows=2, cols=3)
    for cell, text in zip(table.rows[0].cells, ("Верх", "Низ", "Описание")):
        cell.text = text
    for cell, text in zip(table.rows[1].cells, ("100", "102", "Глауконитит")):
        cell.text = text
    document.save(source)
    saved = semantics.save_dictionary({"schema_version": 1, "version": "1.0.0", "rules": [_rule()]})

    parsed = parse_core_document(source)
    assert parsed.dictionary_version == saved["version"]
    assert parsed.intervals[0].rocks[0].canonical_value == "Глауконитит"
    result = parse_batch([source])
    assert result.succeeded == 1
    assert result.items[0].document.dictionary_version == saved["version"]
    assert result.items[0].document.intervals[0].rocks[0].canonical_value == "Глауконитит"
