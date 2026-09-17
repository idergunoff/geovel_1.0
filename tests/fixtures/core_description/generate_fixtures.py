"""Generate de-identified DOCX fixtures without storing binary files in Git."""

from __future__ import annotations

import argparse
from pathlib import Path
from xml.sax.saxutils import escape
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

CONTENT_TYPE = (
    "application/vnd.openxmlformats-officedocument."
    "wordprocessingml.document.main+xml"
)
FIXTURES = {
    "standard_separate_columns.docx": {
        "paragraphs": [
            "Геологический журнал",
            "Площадь: Тестовая Северная",
            "СКВАЖИНА № TEST-101",
        ],
        "tables": [[
            ["От, м", "До, м", "Описание породы", "Проявление"],
            ["100,0", "102,5", "Песчаник серый, слабо битумонасыщенный.", "битум"],
            ["102,5", "104,0", "Глина с прослоями известняка, без признаков нефти.", ""],
        ]],
        "trailing_paragraphs": ["Описал геолог: Иванов И. И."],
    },
    "single_range_column.docx": {
        "paragraphs": ["Описание керна", "Скв. TEST-202", "Пл. Учебная"],
        "tables": [[
            ["Интервал, м", "Литологическое описание"],
            ["205.0–207.25", "Переслаивание песчаника и глины; местами запах нефти."],
            ["207,25-208,0 м", "Известняк, предположительно нефтенасыщенный."],
        ]],
        "trailing_paragraphs": ["Описал: Петров П.П."],
    },
    "merged_header.docx": {
        "paragraphs": ["Площадь Учебная Восточная", "№ скв.: TEST-303"],
        "tables": [[
            [("Глубина залегания слоя, м", 2), "Мощность, м", "Описание керна"],
            ["От", "До", "", ""],
            ["310,0", "311,2", "1,2", "Доломит с включениями сульфидных минералов."],
            ["311,2", "313,0", "1,8", "Песчаник интенсивно нефтенасыщенный."],
        ]],
        "trailing_paragraphs": ["Описал геолог Сидорова А. Б."],
    },
    "continuation_and_boundary.docx": {
        "paragraphs": ["СКВАЖИНА: TEST-404", "Площадь: Демонстрационная"],
        "tables": [[
            ["От", "До", "Описание"],
            ["400,0", "401,0", "Песчаник, средне нефтенасыщенный."],
            ["", "", "Продолжение описания: с прослоями глины."],
            ["401,0", "401,0", "Кровля условного горизонта"],
            ["401,0", "403,0", "Изолированный керн."],
        ]],
        "trailing_paragraphs": [],
    },
}


def _paragraph(text: str) -> str:
    return f'<w:p><w:r><w:t xml:space="preserve">{escape(text)}</w:t></w:r></w:p>'


def _cell(value: str | tuple[str, int]) -> str:
    text, span = value if isinstance(value, tuple) else (value, None)
    properties = f'<w:tcPr><w:gridSpan w:val="{span}"/></w:tcPr>' if span else "<w:tcPr/>"
    return f"<w:tc>{properties}{_paragraph(text)}</w:tc>"


def _table(rows: list[list[str | tuple[str, int]]]) -> str:
    body = "".join(f"<w:tr>{''.join(_cell(cell) for cell in row)}</w:tr>" for row in rows)
    return f"<w:tbl><w:tblPr/><w:tblGrid/>{body}</w:tbl>"


def _write_member(archive: ZipFile, name: str, content: str) -> None:
    # A fixed timestamp makes generated fixtures byte-for-byte reproducible.
    info = ZipInfo(name, date_time=(2020, 1, 1, 0, 0, 0))
    info.compress_type = ZIP_DEFLATED
    archive.writestr(info, content.encode("utf-8"))


def generate_fixtures(output_dir: Path) -> list[Path]:
    """Create all DOCX fixtures in *output_dir* and return their paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    generated = []
    for filename, fixture in FIXTURES.items():
        body = "".join(_paragraph(value) for value in fixture["paragraphs"])
        body += "".join(_table(table) for table in fixture["tables"])
        body += "".join(_paragraph(value) for value in fixture["trailing_paragraphs"])
        body += "<w:sectPr/>"
        document = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<w:document xmlns:w="http://schemas.openxmlformats.org/'
            f'wordprocessingml/2006/main"><w:body>{body}</w:body></w:document>'
        )
        path = output_dir / filename
        with ZipFile(path, "w") as archive:
            _write_member(
                archive,
                "[Content_Types].xml",
                '<?xml version="1.0"?><Types xmlns="http://schemas.openxmlformats.org/'
                'package/2006/content-types"><Default Extension="rels" ContentType="application/'
                'vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" '
                f'ContentType="application/xml"/><Override PartName="/word/document.xml" '
                f'ContentType="{CONTENT_TYPE}"/></Types>',
            )
            _write_member(
                archive,
                "_rels/.rels",
                '<?xml version="1.0"?><Relationships xmlns="http://schemas.openxmlformats.org/'
                'package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.'
                'openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
                'Target="word/document.xml"/></Relationships>',
            )
            _write_member(archive, "word/document.xml", document)
        generated.append(path)
    return generated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output_dir",
        nargs="?",
        type=Path,
        default=Path(__file__).with_name("generated"),
        help="output directory (default: ./generated)",
    )
    args = parser.parse_args()
    for path in generate_fixtures(args.output_dir):
        print(path)


if __name__ == "__main__":
    main()
