"""User dictionary persistence and its effect on actual document parsing."""

from copy import deepcopy
import json
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


def _legacy_user_dictionary():
    suffix = r"(?:нефте|битумо)насыщен(?:н(?:ый|ая|ое|ые|ого|ой|ых|ую|ым|ыми|ом)|[аоы])?(?!\w)"
    return {"schema_version": 1, "version": "1.3.0+user.20261030070000", "rules": [
        _rule(id="oil.weak.01", category="oil_saturation", canonical_value="weak", pattern_type="regex",
              patterns=[r"(?<!\w)слабо[\s-]*" + suffix], priority=234, comment="Проверено пользователем"),
        _rule(id="oil.present.01", category="oil_indicator", canonical_value="present", pattern_type="regex",
              patterns=[r"(?<!\w)" + suffix]),
        _rule(id="oil.none.01", category="negation", canonical_value="none", pattern_type="regex",
              patterns=[r"(?<!\w)не\s+" + suffix]),
        _rule(),
    ]}


def test_existing_user_dictionary_receives_new_forms_without_writing_on_read(dictionary_file):
    legacy = _legacy_user_dictionary()
    dictionary_file.parent.mkdir(parents=True)
    dictionary_file.write_text(json.dumps(legacy, ensure_ascii=False), encoding="utf-8")
    before = dictionary_file.read_bytes()

    upgraded = semantics.load_dictionary_payload()
    assert upgraded["version"] == "1.4.0+user.20261030070000"
    assert {rule["id"] for rule in upgraded["rules"]} == {rule["id"] for rule in legacy["rules"]}
    assert upgraded["rules"][-1] == legacy["rules"][-1]
    assert upgraded["rules"][0]["priority"] == 234
    assert upgraded["rules"][0]["comment"] == "Проверено пользователем"
    assert semantics.analyze_description("Со слабым битумонасыщением")[1] == "weak"
    assert semantics.analyze_description("Слабо битуминонасыщеные")[1] == "weak"
    assert semantics.analyze_description("Битуминонасыщеные")[1] == "present"
    assert semantics.analyze_description("Без слабого битумонасыщения")[1] == "none"
    assert semantics.analyze_description("Глауконитит")[0][0].canonical_value == "Глауконитит"
    assert dictionary_file.read_bytes() == before
    # Explicit paths remain an exact view of the requested historical file.
    assert semantics.load_dictionary_payload(dictionary_file) == legacy


def test_dictionary_upgrade_preserves_disabled_removed_and_moved_rules(dictionary_file):
    legacy = _legacy_user_dictionary()
    legacy["rules"][0]["enabled"] = False
    legacy["rules"] = [rule for rule in legacy["rules"] if rule["id"] != "oil.none.01"]
    moved = deepcopy(legacy["rules"][1])
    moved["category"] = "ignore"
    legacy["rules"][1] = moved
    dictionary_file.parent.mkdir(parents=True)
    dictionary_file.write_text(json.dumps(legacy, ensure_ascii=False), encoding="utf-8")

    upgraded = semantics.load_dictionary_payload()
    assert not upgraded["rules"][0]["enabled"]
    assert not any(rule["id"] == "oil.none.01" for rule in upgraded["rules"])
    assert upgraded["rules"][1] == moved
    assert semantics.analyze_description("Со слабым битумонасыщением")[1] == "unknown"


def test_upgraded_dictionary_can_be_customized_without_readding_new_forms(dictionary_file):
    dictionary_file.parent.mkdir(parents=True)
    dictionary_file.write_text(json.dumps(_legacy_user_dictionary(), ensure_ascii=False), encoding="utf-8")
    upgraded = semantics.load_dictionary_payload()
    upgraded["rules"][0]["patterns"] = [r"слабый пользовательский признак"]
    upgraded["rules"] = [rule for rule in upgraded["rules"] if rule["id"] != "oil.none.01"]
    saved = semantics.save_dictionary(upgraded)
    assert semantics.load_dictionary_payload() == saved
    assert semantics.analyze_description("Со слабым битумонасыщением")[1] == "present"
