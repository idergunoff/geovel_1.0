"""Minimal, defensive Open XML extraction preserving body element order."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree
from zipfile import BadZipFile, ZipFile

WORD_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
W = f"{{{WORD_NS}}}"
MAX_ARCHIVE_MEMBERS = 2_000
MAX_DOCUMENT_XML = 32 * 1024 * 1024
MAX_TOTAL_UNCOMPRESSED = 128 * 1024 * 1024


class DocxExtractionError(ValueError):
    """Raised for an invalid or unsafe DOCX package."""


@dataclass(frozen=True)
class ExtractedTable:
    rows: list[list[str]]


@dataclass(frozen=True)
class ExtractedDocx:
    blocks: list[str | ExtractedTable]


def _text(element: ElementTree.Element) -> str:
    pieces: list[str] = []
    for node in element.iter():
        if node.tag == W + "t" and node.text:
            pieces.append(node.text)
        elif node.tag in {W + "br", W + "cr"}:
            pieces.append("\n")
        elif node.tag == W + "tab":
            pieces.append("\t")
    return "".join(pieces).strip()


def _table(element: ElementTree.Element) -> ExtractedTable:
    rows: list[list[str]] = []
    for row in element.findall(W + "tr"):
        values: list[str] = []
        for cell in row.findall(W + "tc"):
            value = _text(cell)
            span_node = cell.find(f"{W}tcPr/{W}gridSpan")
            span = 1
            if span_node is not None:
                try:
                    span = max(1, int(span_node.attrib.get(W + "val", "1")))
                except ValueError:
                    pass
            values.extend([value] + [""] * (span - 1))
        rows.append(values)
    return ExtractedTable(rows)


def extract_docx(path: Path) -> ExtractedDocx:
    """Extract paragraphs and tables without executing relationships or macros."""

    try:
        with ZipFile(path) as archive:
            members = archive.infolist()
            if len(members) > MAX_ARCHIVE_MEMBERS:
                raise DocxExtractionError("DOCX contains too many archive members")
            if sum(item.file_size for item in members) > MAX_TOTAL_UNCOMPRESSED:
                raise DocxExtractionError("DOCX uncompressed size exceeds the safety limit")
            try:
                document = archive.getinfo("word/document.xml")
            except KeyError as error:
                raise DocxExtractionError("DOCX has no word/document.xml") from error
            if document.file_size > MAX_DOCUMENT_XML:
                raise DocxExtractionError("DOCX document XML exceeds the safety limit")
            xml = archive.read(document)
    except (BadZipFile, OSError) as error:
        raise DocxExtractionError(f"Cannot open DOCX: {error}") from error

    try:
        root = ElementTree.fromstring(xml)
    except ElementTree.ParseError as error:
        raise DocxExtractionError(f"Invalid document XML: {error}") from error
    body = root.find(W + "body")
    if body is None:
        raise DocxExtractionError("DOCX document has no body")
    blocks: list[str | ExtractedTable] = []
    for child in body:
        if child.tag == W + "p":
            blocks.append(_text(child))
        elif child.tag == W + "tbl":
            blocks.append(_table(child))
    return ExtractedDocx(blocks)
