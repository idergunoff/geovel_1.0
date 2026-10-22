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
        assert result.dictionary_version == "1.2.0"


def test_source_coordinates_and_optional_columns(generated):
    result = parse_core_document(generated / "standard_separate_columns.docx")
    assert [(row.source_table, row.source_row) for row in result.intervals] == [(0, 1), (0, 2)]
    assert result.intervals[0].manifestation_raw == "битум"
    assert result.intervals[1].manifestation_raw is None


@pytest.mark.parametrize("number_position", ["before", "after", "both"])
def test_column_number_rows_are_skipped_without_losing_source_coordinates(number_position):
    from core_description.docx import ExtractedTable
    from core_description.parser import _column_map, _parse_rows

    header = ["От, м", "До, м", "Мощность, м", "Геологический индекс", "Описание керна"]
    numbers = ["1", "2", "3", "4", "5"]
    rows = ([numbers] if number_position in {"before", "both"} else []) + [header]
    if number_position in {"after", "both"}:
        rows.append(numbers)
    data_row = len(rows)
    rows.extend([["100", "102", "1,5", "J₂", "Песчаник"], numbers,
                 ["102", "103", "", "K₁", "Глина"]])
    mapping, start, score = _column_map(rows)
    result = _parse_rows(ExtractedTable(rows), 0, mapping, start)

    assert score == 4
    assert [item.source_row for item in result] == [data_row, data_row + 2]
    assert [item.thickness for item in result] == [1.5, 1.0]
    assert [item.geological_index_raw for item in result] == ["J₂", "K₁"]
    assert not any(item.errors for item in result)


def test_thickness_is_extracted_from_merged_headers(generated):
    document = parse_core_document(generated / "merged_header.docx")
    assert [item.thickness for item in document.intervals] == [1.2, 1.8]


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
    monkeypatch.delenv("LIBREOFFICE_PATH", raising=False)
    monkeypatch.setattr("core_description.converter.shutil.which", lambda _name: None)
    monkeypatch.setattr("core_description.converter._libreoffice_command", lambda: None)
    monkeypatch.setattr("core_description.converter._powershell_executable", lambda: None)
    with pytest.raises(DocConversionError, match="requires LibreOffice, or Microsoft Word"):
        with docx_source(source):
            pass


def test_doc_conversion_finds_libreoffice_outside_path(tmp_path, monkeypatch):
    source = tmp_path / "legacy.doc"
    source.write_bytes(b"legacy")
    program_files = tmp_path / "Program Files"
    executable = program_files / "LibreOffice" / "program" / "soffice.exe"
    executable.parent.mkdir(parents=True)
    executable.write_bytes(b"")
    captured = {}

    class Result:
        returncode = 0
        stdout = ""
        stderr = ""

    def convert(command, **_kwargs):
        captured["executable"] = command[0]
        output_dir = Path(command[command.index("--outdir") + 1])
        (output_dir / "source.docx").write_bytes(b"converted")
        return Result()

    monkeypatch.delenv("LIBREOFFICE_PATH", raising=False)
    monkeypatch.setenv("PROGRAMFILES", str(program_files))
    monkeypatch.delenv("PROGRAMFILES(X86)", raising=False)
    monkeypatch.delenv("LOCALAPPDATA", raising=False)
    monkeypatch.setattr("core_description.converter.shutil.which", lambda _name: None)
    monkeypatch.setattr("core_description.converter.subprocess.run", convert)

    with docx_source(source) as converted:
        assert converted.read_bytes() == b"converted"

    assert captured["executable"] == str(executable)


def test_doc_conversion_honours_configured_libreoffice_path(tmp_path, monkeypatch):
    executable = tmp_path / "portable" / "soffice"
    executable.parent.mkdir()
    executable.write_bytes(b"")
    monkeypatch.setenv("LIBREOFFICE_PATH", str(executable))

    from core_description.converter import _libreoffice_executable

    assert _libreoffice_executable() == str(executable)


def test_doc_conversion_uses_user_flatpak_install(tmp_path, monkeypatch):
    source = tmp_path / "legacy.doc"
    source.write_bytes(b"legacy")
    flatpak_app = tmp_path / ".local/share/flatpak/app/org.libreoffice.LibreOffice"
    flatpak_app.mkdir(parents=True)
    captured = {}

    class Result:
        returncode = 0
        stdout = ""
        stderr = ""

    def which(command):
        return "/usr/bin/flatpak" if command == "flatpak" else None

    def convert(command, **_kwargs):
        captured["command"] = command
        output_dir = Path(command[command.index("--outdir") + 1])
        (output_dir / "source.docx").write_bytes(b"converted")
        return Result()

    monkeypatch.delenv("LIBREOFFICE_PATH", raising=False)
    monkeypatch.setattr("core_description.converter.Path.home", lambda: tmp_path)
    monkeypatch.setattr("core_description.converter.shutil.which", which)
    monkeypatch.setattr("core_description.converter.subprocess.run", convert)

    with docx_source(source) as converted:
        assert converted.read_bytes() == b"converted"

    assert captured["command"][:2] == ["/usr/bin/flatpak", "run"]
    assert captured["command"][2].startswith("--filesystem=")
    assert captured["command"][3] == "org.libreoffice.LibreOffice"


def test_doc_conversion_falls_back_to_microsoft_word(tmp_path, monkeypatch):
    source = tmp_path / "legacy.doc"
    source.write_bytes(b"legacy")
    captured = {}

    class Result:
        returncode = 0
        stdout = ""
        stderr = ""

    def convert(command, **kwargs):
        captured["command"] = command
        captured["environment"] = kwargs["env"]
        Path(kwargs["env"]["GEOVEL_DOCX_TARGET"]).write_bytes(b"converted-by-word")
        return Result()

    monkeypatch.setattr("core_description.converter._libreoffice_command", lambda: None)
    monkeypatch.setattr("core_description.converter._powershell_executable", lambda: "powershell.exe")
    monkeypatch.setattr("core_description.converter.subprocess.run", convert)

    with docx_source(source) as converted:
        assert converted.read_bytes() == b"converted-by-word"

    assert captured["command"][:3] == ["powershell.exe", "-NoProfile", "-NonInteractive"]
    assert Path(captured["environment"]["GEOVEL_DOC_SOURCE"]).name == "source.doc"


def test_doc_conversion_stages_user_path_before_calling_libreoffice(tmp_path, monkeypatch):
    source = tmp_path / "архив керна" / "Скважина № 1.doc"
    source.parent.mkdir()
    source.write_bytes(b"legacy")
    captured = {}

    class Result:
        returncode = 0
        stdout = ""
        stderr = ""

    def convert(command, **_kwargs):
        captured["command"] = command
        input_path = Path(command[-1])
        output_dir = Path(command[command.index("--outdir") + 1])
        assert input_path.name == "source.doc"
        assert input_path.read_bytes() == b"legacy"
        (output_dir / "source.docx").write_bytes(b"converted")
        return Result()

    monkeypatch.setattr("core_description.converter.shutil.which", lambda _name: "soffice")
    monkeypatch.setattr("core_description.converter.subprocess.run", convert)

    with docx_source(source) as converted:
        assert converted.read_bytes() == b"converted"

    assert str(source) not in captured["command"]


@pytest.mark.parametrize("host_bridge", [False, True])
def test_doc_import_uses_isolated_profiles_and_existing_parser(tmp_path, monkeypatch, generated, host_bridge):
    source = tmp_path / "Описание керна.DOC"
    source.write_bytes(b"legacy")
    expected = parse_core_document(generated / "standard_separate_columns.docx")
    profiles = []
    staging_directories = []
    application = "com.jetbrains.PyCharm-Community"
    monkeypatch.setenv("FLATPAK_ID", application)
    monkeypatch.setattr("core_description.converter.Path.home", lambda: tmp_path)

    def convert(command, **kwargs):
        staged = Path(command[-1])
        staging_directories.append(staged.parent)
        if host_bridge:
            assert command[:3] == ["flatpak-spawn", "--host", "/usr/bin/libreoffice"]
            assert staged.parent.parent == tmp_path / ".var/app" / application / "cache"
        profile = next(arg for arg in command if arg.startswith("-env:UserInstallation="))
        assert profile == f"-env:UserInstallation={(staged.parent / 'profile').as_uri()}"
        profiles.append(profile)
        assert kwargs["timeout"] == 120
        output = Path(command[command.index("--outdir") + 1]) / "source.docx"
        output.write_bytes((generated / "standard_separate_columns.docx").read_bytes())
        return subprocess.CompletedProcess(command, 0, "", "")

    prefix = ["flatpak-spawn", "--host", "/usr/bin/libreoffice"] if host_bridge else ["soffice"]
    monkeypatch.setattr("core_description.converter._libreoffice_command", lambda: prefix)
    monkeypatch.setattr("core_description.converter.subprocess.run", convert)
    for _ in range(2):
        result = parse_core_document(source)
        assert result.source_format == "doc"
        assert result.well_name == expected.well_name
        assert result.intervals == expected.intervals
    assert profiles[0] != profiles[1]
    assert all(not directory.exists() for directory in staging_directories)
    assert source.read_bytes() == b"legacy"


@pytest.mark.parametrize("outcome", ["success", "denied", "timeout", "configured"])
def test_flatpak_ide_discovers_host_libreoffice(tmp_path, monkeypatch, outcome):
    from core_description.converter import _libreoffice_command

    monkeypatch.setenv("FLATPAK_ID", "com.jetbrains.PyCharm-Community")
    monkeypatch.delenv("LIBREOFFICE_PATH", raising=False)
    if outcome == "configured":
        monkeypatch.setenv("LIBREOFFICE_PATH", "/opt/Libre Office/soffice")
    monkeypatch.setattr("core_description.converter._libreoffice_executable", lambda: None)
    monkeypatch.setattr("core_description.converter.shutil.which",
                        lambda name: "/usr/bin/flatpak-spawn" if name == "flatpak-spawn" else None)

    def probe(command, **kwargs):
        assert command[:2] == ["/usr/bin/flatpak-spawn", "--host"]
        assert command[-1] == "--version"
        assert kwargs["timeout"] == 10
        if outcome == "timeout":
            raise subprocess.TimeoutExpired(command, 10)
        return subprocess.CompletedProcess(command, 1 if outcome == "denied" else 0, "", "")

    monkeypatch.setattr("core_description.converter.subprocess.run", probe)
    command = _libreoffice_command()
    if outcome in {"denied", "timeout"}:
        assert command is None
    else:
        executable = "/opt/Libre Office/soffice" if outcome == "configured" else "/usr/bin/libreoffice"
        assert command == ["/usr/bin/flatpak-spawn", "--host", executable]


@pytest.mark.parametrize("failure", ["timeout", "missing_output", "exit_error"])
def test_failed_doc_conversion_cleans_temporary_files(tmp_path, monkeypatch, failure):
    source = tmp_path / "legacy.doc"
    source.write_bytes(b"legacy")
    staging_directories = []

    def convert(command, **kwargs):
        staging_directories.append(Path(command[-1]).parent)
        if failure == "timeout":
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        return subprocess.CompletedProcess(command, 1 if failure == "exit_error" else 0, "", "")

    monkeypatch.setattr("core_description.converter._libreoffice_command", lambda: ["soffice"])
    monkeypatch.setattr("core_description.converter.subprocess.run", convert)
    with pytest.raises(CoreDescriptionParseError, match="LibreOffice"):
        parse_core_document(source)
    assert staging_directories
    assert all(not directory.exists() for directory in staging_directories)
    assert source.read_bytes() == b"legacy"


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


def test_plural_sandstone_with_bitumen_saturation_range():
    text = (
        "Песчаники от темно-серых до черных, м/з, массивные, "
        "слабосцементированные, от средне до интенсивно битумонасыщенные."
    )
    rocks, oil, confidence, matches, warnings = analyze_description(text)

    assert [(rock.canonical_value, rock.relation) for rock in rocks] == [
        ("sandstone", "primary")
    ]
    assert oil == "medium_to_intense"
    assert confidence == 1.0
    assert warnings == []
    assert any(
        match.canonical_value == oil
        and match.matched_text == "от средне до интенсивно битумонасыщенные"
        for match in matches
    )
    assert all(text[match.start:match.end] == match.matched_text for match in matches)


@pytest.mark.parametrize("text, expected", [
    ("Песчаники с прослоями глин", [("sandstone", "primary"), ("clay", "interlayer")]),
    ("Глина с прослоями песчаников", [("clay", "primary"), ("sandstone", "interlayer")]),
    ("Известняки с включениями доломитов", [("limestone", "primary"), ("dolomite", "inclusion")]),
    ("Песчаниковый, глинистый, известняковый, доломитовый", []),
])
def test_rock_word_forms_preserve_roles_and_word_boundaries(text, expected):
    rocks = analyze_description(text)[0]
    assert [(rock.canonical_value, rock.relation) for rock in rocks] == expected


@pytest.mark.parametrize("fluid", ["нефте", "битумо"])
@pytest.mark.parametrize("ending", ["енный", "енная", "енное", "енные", "ен", "ена", "ено", "ены"])
@pytest.mark.parametrize("degree, expected", [
    ("слабо", "weak"), ("средне", "medium"),
    ("интенсивно", "intense"), ("сильно", "intense"),
])
def test_oil_and_bitumen_saturation_word_forms(fluid, ending, degree, expected):
    assert analyze_description(f"{degree} {fluid}насыщ{ending}")[1] == expected


@pytest.mark.parametrize("text, expected", [
    ("от слабо до средне битумонасыщенные", "weak_to_medium"),
    ("от средне до сильно нефтенасыщенная", "medium_to_intense"),
    ("от слабо до интенсивно битумонасыщенный", "weak_to_intense"),
    ("средненефтенасыщенный", "medium"),
    ("интенсивно-битумонасыщенные", "intense"),
    ("средне\nбитумонасыщенные", "medium"),
    ("БИТУМОНАСЫЩЕННЫЕ", "present"),
    ("не нефтенасыщенные", "none"),
    ("не битумонасыщенная", "none"),
    ("без признаков битума", "none"),
    ("предположительно средне битумонасыщенные", "uncertain"),
    ("местами от средне до интенсивно битумонасыщенные", "medium_to_intense"),
    ("Песчаники от темно-серых до черных, интенсивно трещиноватые", "unknown"),
    ("Песчаник слабосцементированный, битумонасыщенный", "present"),
])
def test_saturation_ranges_context_and_negation(text, expected):
    assert analyze_description(text)[1] == expected


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
