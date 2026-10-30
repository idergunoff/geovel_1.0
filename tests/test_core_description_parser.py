"""Contract tests for stage 2 of the core-description import."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from xml.sax.saxutils import escape
from zipfile import ZipFile

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
        assert result.dictionary_version == "1.4.0"


def test_source_coordinates_and_optional_columns(generated):
    result = parse_core_document(generated / "standard_separate_columns.docx")
    assert [(row.source_table, row.source_row) for row in result.intervals] == [(0, 1), (0, 2)]
    assert result.intervals[0].manifestation_raw == "битум"
    assert result.intervals[1].manifestation_raw is None


@pytest.mark.parametrize('wrapper', ['sdt', 'customXml'])
@pytest.mark.parametrize('wrapped_part', ['data_row', 'description_cell', 'table'])
def test_word_content_controls_do_not_hide_core_table_data(tmp_path, wrapper, wrapped_part):
    def wrap(value):
        content = f'<w:sdtContent>{value}</w:sdtContent>' if wrapper == 'sdt' else value
        return f'<w:{wrapper}>{content}</w:{wrapper}>'

    def cell(value):
        return f'<w:tc><w:p><w:r><w:t>{value}</w:t></w:r></w:p></w:tc>'

    description_header = cell('Описание породы')
    description_data = cell('Песчаник серый')
    if wrapped_part == 'description_cell':
        description_header, description_data = wrap(description_header), wrap(description_data)
    header = '<w:tr>' + cell('От') + cell('До') + description_header + '</w:tr>'
    data = '<w:tr>' + cell('100') + cell('102') + description_data + '</w:tr>'
    if wrapped_part == 'data_row':
        data = wrap(data)
    table = '<w:tbl>' + header + data + '</w:tbl>'
    if wrapped_part == 'table':
        table = wrap(table)
    source = tmp_path / 'word-content-controls.docx'
    with ZipFile(source, 'w') as archive:
        archive.writestr('word/document.xml', '<w:document xmlns:w="http://schemas.openxmlformats.org/'
                         f'wordprocessingml/2006/main"><w:body>{table}</w:body></w:document>')
    result = parse_core_document(source)
    assert len(result.intervals) == 1
    assert result.intervals[0].top_depth == 100
    assert result.intervals[0].bottom_depth == 102
    assert result.intervals[0].raw_description == 'Песчаник серый'
    assert result.intervals[0].source_row == 1


def test_content_controls_preserve_order_without_flattening_nested_tables(tmp_path):
    from core_description.docx import ExtractedTable, extract_docx

    source = tmp_path / 'nested-controls.docx'
    xml = '''<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
      <w:body><w:p><w:r><w:t>Before</w:t></w:r></w:p>
        <w:sdt><w:sdtContent><w:customXml><w:tbl>
          <w:tr><w:tc><w:p><w:r><w:t>First</w:t></w:r></w:p>
            <w:tbl><w:tr><w:tc><w:p><w:r><w:t>Nested</w:t></w:r></w:p></w:tc></w:tr></w:tbl>
          </w:tc></w:tr>
          <w:sdt><w:sdtContent><w:tr><w:customXml><w:tc>
            <w:p><w:r><w:t>Second</w:t></w:r></w:p>
          </w:tc></w:customXml></w:tr></w:sdtContent></w:sdt>
        </w:tbl></w:customXml></w:sdtContent></w:sdt>
        <w:customXml><w:p><w:r><w:t>After</w:t></w:r></w:p></w:customXml>
      </w:body></w:document>'''
    with ZipFile(source, 'w') as archive:
        archive.writestr('word/document.xml', xml)
    assert extract_docx(source).blocks == ['Before', ExtractedTable([['First\nNested'], ['Second']]), 'After']


@pytest.mark.parametrize("metadata_table", [False, True])
def test_well_identity_is_extracted_before_core_table(generated, tmp_path, metadata_table):
    entries = [('Скважина начата', '28.08.19'), ('Скважина окончена', '02.09.2019'),
               ('Альтитуда устья', '132,432 м'), ('Альтитуда ротора', '133,432 м')]
    def paragraph(text):
        return f'<w:p><w:r><w:t>{escape(text)}</w:t></w:r></w:p>'
    if metadata_table:
        content = '<w:tbl>' + ''.join('<w:tr><w:tc>' + paragraph(label) + '</w:tc><w:tc>'
                                    + paragraph(value) + '</w:tc></w:tr>' for label, value in entries) + '</w:tbl>'
    else:
        content = ''.join(paragraph(f'{label}: {value}') for label, value in entries)
    source = tmp_path / 'identity.docx'
    with ZipFile(generated / 'standard_separate_columns.docx') as original, ZipFile(source, 'w') as output:
        for item in original.infolist():
            data = original.read(item)
            if item.filename == 'word/document.xml':
                data = data.decode().replace('<w:tbl>', content + '<w:tbl>', 1).encode()
            output.writestr(item, data)
    document = parse_core_document(source)
    assert document.well_name == 'TEST-101'
    assert document.drilling_start_raw == '28.08.19'
    assert document.drilling_start == '2019-08-28'
    assert document.drilling_end == '2019-09-02'
    assert document.wellhead_elevation == 132.432
    assert 'well_name_ambiguous' not in document.warnings


@pytest.mark.parametrize('well_name', ['10 Вр', '10Вр', 'TEST-101'])
@pytest.mark.parametrize('footer', ['Скважину документировал:', 'Работы по скважинам завершены',
                                    'Наблюдение за скважиной'])
def test_well_number_preserves_letters_and_ignores_inflected_mentions(generated, tmp_path, well_name, footer):
    source = tmp_path / 'well-with-letters.docx'
    with ZipFile(generated / 'standard_separate_columns.docx') as original, ZipFile(source, 'w') as output:
        for item in original.infolist():
            data = original.read(item)
            if item.filename == 'word/document.xml':
                data = data.replace(b'TEST-101', well_name.encode())
                paragraph = f'<w:p><w:r><w:t>{escape(footer)}</w:t></w:r></w:p>'
                data = data.replace(b'</w:body>', paragraph.encode() + b'</w:body>')
            output.writestr(item, data)
    result = parse_core_document(source)
    assert result.well_name_raw == result.well_name == well_name
    assert 'well_name_ambiguous' not in result.warnings
    assert len(result.intervals) == 2


@pytest.mark.parametrize('label', ['СКВАЖИНА № ', 'Скв. ', 'Скв.', '№ скв. ', '№ скважины: '])
def test_alphanumeric_well_number_supports_existing_labels(label):
    from core_description.parser import _find_metadata

    (well, _area, _author, warnings), _ = _find_metadata([label + '10 Вр'], [])
    assert well == '10 Вр'
    assert 'well_name_missing' not in warnings


def test_different_real_well_numbers_still_report_ambiguity():
    from core_description.parser import _find_metadata

    (well, _area, _author, warnings), _ = _find_metadata(
        ['СКВАЖИНА № 10 Вр', 'СКВАЖИНА № 11 Вр', 'Скважину документировал:'], [])
    assert well is None
    assert 'well_name_ambiguous' in warnings


@pytest.mark.parametrize('label', ['Скважину документировал:', 'Скважину документировала:',
                                  'Документировал:', 'Описал геолог:'])
@pytest.mark.parametrize('separate_paragraph', [False, True])
def test_author_signature_reads_name_after_role_and_organization(generated, tmp_path, label, separate_paragraph):
    source = tmp_path / 'documented-by.docx'
    signature = 'геолог ООО «Пример»                Иванов И.И.'
    lines = [label, signature] if separate_paragraph else [label + ' ' + signature]
    paragraphs = ''.join(f'<w:p><w:r><w:t>{escape(line)}</w:t></w:r></w:p>' for line in lines)
    with ZipFile(generated / 'standard_separate_columns.docx') as original, ZipFile(source, 'w') as output:
        for item in original.infolist():
            data = original.read(item)
            if item.filename == 'word/document.xml':
                data = data.replace('Описал геолог: Иванов И. И.'.encode(), b'')
                data = data.replace(b'</w:body>', paragraphs.encode() + b'</w:body>')
            output.writestr(item, data)
    result = parse_core_document(source)
    assert result.described_by == 'Иванов И.И.'
    assert label in result.described_by_raw
    assert signature in result.described_by_raw
    assert 'description_author_missing' not in result.warnings
    assert result.well_name == 'TEST-101'
    assert 'well_name_ambiguous' not in result.warnings


@pytest.mark.parametrize('following, expected', [
    ('Иванов И. И.', 'Иванов И. И.'),
    ('геолог Иванов Иван Иванович', 'Иванов Иван Иванович'),
    ('СКВАЖИНА № 10 Вр', None), ('Конструкция скважины:', None), ('', None),
])
def test_separate_author_label_requires_person_signature(following, expected):
    from core_description.parser import _find_metadata

    (_well, _area, raw, warnings), author = _find_metadata(
        ['Скважину документировал:', '', following], [])
    assert author == expected
    if expected:
        assert raw == 'Скважину документировал:\n' + following
    else:
        assert raw is None
        assert 'description_author_missing' in warnings


def test_different_documenting_authors_remain_ambiguous():
    from core_description.parser import _find_metadata

    (_well, _area, raw, warnings), author = _find_metadata(
        ['Скважину документировал: Иванов И.И.', 'Скважину документировала: Петрова А.А.'], [])
    assert raw is None and author is None
    assert 'description_author_ambiguous' in warnings


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


@pytest.mark.parametrize("index_header", [
    "<w:p><w:r><w:t>Геологический</w:t></w:r></w:p><w:p><w:r><w:t>индекс</w:t></w:r></w:p>",
    "<w:p><w:r><w:t>Геологи-</w:t><w:br/><w:t>ческий индекс</w:t></w:r></w:p>",
    "<w:p><w:r><w:t>Геологиче\u00adский индекс</w:t></w:r></w:p>",
])
def test_geological_index_header_survives_word_paragraphs_and_wrapping(tmp_path, index_header):
    from zipfile import ZipFile
    from core_description.docx import WORD_NS

    def cell(text):
        return f"<w:tc><w:p><w:r><w:t>{text}</w:t></w:r></w:p></w:tc>"

    xml = (
        f'<w:document xmlns:w="{WORD_NS}"><w:body><w:tbl><w:tr>'
        + cell("От") + cell("До") + f"<w:tc>{index_header}</w:tc>" + cell("Описание керна")
        + "</w:tr><w:tr>" + cell("100") + cell("102")
        + '<w:tc><w:p><w:r><w:t>Р</w:t></w:r><w:r><w:rPr><w:vertAlign w:val="subscript"/></w:rPr><w:t>2</w:t></w:r><w:r><w:t>kz</w:t></w:r></w:p></w:tc>'
        + cell("Песчаник") + "</w:tr></w:tbl></w:body></w:document>"
    )
    path = tmp_path / "wrapped-index.docx"
    with ZipFile(path, "w") as archive:
        archive.writestr("word/document.xml", xml)
    document = parse_core_document(path)
    assert len(document.intervals) == 1
    assert document.intervals[0].geological_index_raw == "Р2kz"


def test_optional_header_on_lower_line_is_not_lost():
    from core_description.docx import ExtractedTable
    from core_description.parser import _column_map, _parse_rows

    rows = [
        ["От", "До", "Геологический", "Описание керна"],
        ["", "", "индекс", ""],
        ["1", "2", "3", "4"],
        ["100", "102", "P1uf2", "Песчаник"],
    ]
    mapping, start, score = _column_map(rows)
    assert score == 4
    intervals = _parse_rows(ExtractedTable(rows), 0, mapping, start)
    assert len(intervals) == 1
    assert intervals[0].source_row == 3
    assert intervals[0].geological_index_raw == "P1uf2"


@pytest.mark.parametrize("header_area", [
    "Сарабикуловское поднятие", "Ягодное поднятие", "Сарабикуловская",
    "Нефтеперспективный объект 58", "Нефтеперспективный объект №58",
    "Перспективный объект № 8094", "Площадь: Сарабикуловское поднятие",
    "поднятие Мельничное", "Поднятие: Мельничное",
    "месторождение Северное", "структура Южная",
])
@pytest.mark.parametrize("single_paragraph", [False, True])
def test_area_is_read_from_unlabelled_header(tmp_path, header_area, single_paragraph):
    lines = ["Геологический журнал", "ТГРУ", header_area, "СКВАЖИНА № 12"]
    paragraphs = ["\n".join(lines)] if single_paragraph else lines
    body = "".join(f"<w:p><w:r><w:t>{escape(text)}</w:t></w:r></w:p>" for text in paragraphs)
    body += ('<w:tbl><w:tr><w:tc><w:p><w:r><w:t>Интервал</w:t></w:r></w:p></w:tc>'
             '<w:tc><w:p><w:r><w:t>Описание</w:t></w:r></w:p></w:tc></w:tr>'
             '<w:tr><w:tc><w:p><w:r><w:t>100-101</w:t></w:r></w:p></w:tc>'
             '<w:tc><w:p><w:r><w:t>Песчаник</w:t></w:r></w:p></w:tc></w:tr></w:tbl>')
    source = tmp_path / "header.docx"
    with ZipFile(source, "w") as archive:
        archive.writestr("word/document.xml", '<w:document xmlns:w="http://schemas.openxmlformats.org/'
                         f'wordprocessingml/2006/main"><w:body>{body}</w:body></w:document>')

    result = parse_core_document(source)
    assert result.area_name_raw == header_area.removeprefix("Площадь: ")
    assert result.well_name == "12"


@pytest.mark.parametrize('header_area', ['Сарабикуловское поднятие', 'поднятие Мельничное'])
def test_area_inference_ignores_text_after_table_and_conflicting_headers(header_area):
    from core_description.parser import _find_metadata

    header = ["Геологический журнал", "ТГРУ", "Описание керна", "Скважина № 12"]
    (well, area, _author, _warnings), _ = _find_metadata(
        header + [header_area], [], header)
    assert well == "12"
    assert area is None
    (_well, area, _author, warnings), _ = _find_metadata(
        ["Ягодное поднятие", header_area], [])
    assert area is None
    assert "area_name_ambiguous" in warnings


@pytest.fixture()
def no_area_document(generated, tmp_path):
    path = tmp_path / 'no-area.docx'
    with ZipFile(generated / 'standard_separate_columns.docx') as original, ZipFile(path, 'w') as output:
        for item in original.infolist():
            data = original.read(item)
            if item.filename == 'word/document.xml':
                data = data.replace('Площадь: Тестовая Северная'.encode(), b'')
            output.writestr(item, data)
    return path


@pytest.mark.parametrize('relative_path', [False, True])
def test_missing_area_uses_immediate_source_directory(no_area_document, tmp_path, monkeypatch, relative_path):
    directory = tmp_path / 'Архив' / 'Мельничное поднятие'
    directory.mkdir(parents=True)
    source = directory / '311.docx'
    shutil.copyfile(no_area_document, source)
    if relative_path:
        monkeypatch.chdir(directory)
        source = Path(source.name)
    result = parse_core_document(source)
    assert result.area_name_raw == result.area_name == 'Мельничное поднятие'
    assert 'area_name_from_directory' in result.warnings
    assert result.well_name == 'TEST-101'
    assert len(result.intervals) == 2


def test_document_area_takes_priority_over_directory(generated, tmp_path):
    directory = tmp_path / 'Другая площадь'
    directory.mkdir()
    source = directory / '311.docx'
    shutil.copyfile(generated / 'standard_separate_columns.docx', source)
    result = parse_core_document(source)
    assert result.area_name == 'Тестовая Северная'
    assert 'area_name_from_directory' not in result.warnings


def test_doc_directory_fallback_uses_original_path_before_conversion(no_area_document, tmp_path, monkeypatch):
    directory = tmp_path / 'Мельничное'
    directory.mkdir()
    source = directory / '311.doc'
    source.write_bytes(b'legacy document staged for conversion')

    @contextmanager
    def converted_document(path):
        assert path == source
        yield no_area_document

    monkeypatch.setattr('core_description.parser.docx_source', converted_document)
    result = parse_core_document(source)
    assert result.area_name == 'Мельничное'
    assert 'area_name_from_directory' in result.warnings
    assert result.source_path == str(source)
    assert result.source_format == 'doc'


def test_directory_fallback_preserves_ambiguous_header_warning(no_area_document, tmp_path):
    directory = tmp_path / 'Мельничное'
    directory.mkdir()
    source = directory / '311.docx'
    conflicting = ''.join(f'<w:p><w:r><w:t>{text}</w:t></w:r></w:p>'
                          for text in ('Ягодное поднятие', 'Северное поднятие'))
    with ZipFile(no_area_document) as original, ZipFile(source, 'w') as output:
        for item in original.infolist():
            data = original.read(item)
            if item.filename == 'word/document.xml':
                data = data.replace(b'<w:tbl>', conflicting.encode() + b'<w:tbl>', 1)
            output.writestr(item, data)
    result = parse_core_document(source)
    assert result.area_name == 'Мельничное'
    assert 'area_name_ambiguous' in result.warnings
    assert 'area_name_from_directory' in result.warnings


def test_recursive_batch_uses_each_files_parent_directory(no_area_document, tmp_path):
    from core_description.batch import parse_batch

    root = tmp_path / 'Архив'
    for name in ('Мельничное', 'Северная'):
        directory = root / name
        directory.mkdir(parents=True)
        shutil.copyfile(no_area_document, directory / '311.docx')
    result = parse_batch([root], recursive=True)
    assert result.failed == 0
    assert {item.document.area_name for item in result.items} == {'Мельничное', 'Северная'}
    assert all('area_name_from_directory' in item.document.warnings for item in result.items)


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
    monkeypatch.setattr("core_description.converter._run_libreoffice", convert)

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
    monkeypatch.setattr("core_description.converter._run_libreoffice", convert)

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
    monkeypatch.setattr("core_description.converter._run_libreoffice", convert)

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
    monkeypatch.setattr("core_description.converter._run_libreoffice", convert)
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
    monkeypatch.setattr("core_description.converter._run_libreoffice", convert)
    with pytest.raises(CoreDescriptionParseError, match="LibreOffice"):
        parse_core_document(source)
    assert staging_directories
    assert all(not directory.exists() for directory in staging_directories)
    assert source.read_bytes() == b"legacy"


@pytest.mark.parametrize('output', ['complete', 'truncated', 'invalid_xml', 'missing'])
def test_conversion_timeout_accepts_only_readable_docx(tmp_path, monkeypatch, generated, output):
    source = tmp_path / 'legacy.doc'
    source.write_bytes(b'original')
    staged = []

    def convert(command, **kwargs):
        staged.append(Path(command[-1]).parent)
        target = Path(command[command.index('--outdir') + 1]) / 'source.docx'
        if output == 'complete':
            shutil.copyfile(generated / 'standard_separate_columns.docx', target)
        elif output == 'truncated':
            target.write_bytes(b'PK\x03\x04partial docx')
        elif output == 'invalid_xml':
            with ZipFile(target, 'w') as archive:
                archive.writestr('word/document.xml', '<broken>')
        raise subprocess.TimeoutExpired(command, kwargs['timeout'])

    monkeypatch.setattr('core_description.converter._libreoffice_command', lambda: ['soffice'])
    monkeypatch.setattr('core_description.converter._run_libreoffice', convert)
    if output == 'complete':
        result = parse_core_document(source)
        assert result.well_name == 'TEST-101'
        assert len(result.intervals) == 2
    else:
        with pytest.raises(CoreDescriptionParseError, match='timed out after 120 seconds.*no readable DOCX'):
            parse_core_document(source)
    assert all(not directory.exists() for directory in staged)
    assert source.read_bytes() == b'original'


@pytest.mark.skipif(sys.platform != 'linux', reason='Linux process-group cleanup')
def test_conversion_timeout_stops_only_its_own_process_group(tmp_path):
    from core_description.converter import _run_libreoffice

    child_pid_path = tmp_path / 'child.pid'
    script = '''import pathlib, subprocess, sys, time
child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])
pathlib.Path(sys.argv[1]).write_text(str(child.pid))
time.sleep(30)
'''
    unrelated = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])
    try:
        with pytest.raises(subprocess.TimeoutExpired):
            _run_libreoffice([sys.executable, '-c', script, str(child_pid_path)], timeout=1)
        assert unrelated.poll() is None
        child_pid = int(child_pid_path.read_text())
        status = Path(f'/proc/{child_pid}/status')
        # A killed child may briefly remain a zombie awaiting the system reaper.
        deadline = time.monotonic() + 2
        while True:
            try:
                state = status.read_text()
            except FileNotFoundError:
                break
            if 'State:\tZ' in state:
                break
            assert time.monotonic() < deadline, 'Converter child is still running'
            time.sleep(0.01)
    finally:
        unrelated.kill()
        unrelated.wait()


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
