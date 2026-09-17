"""Pure parser for Word documents containing core descriptions."""

from .models import ParsedCoreDocument, ParsedCoreInterval
from .parser import CoreDescriptionParseError, parse_core_document

__all__ = [
    "CoreDescriptionParseError",
    "ParsedCoreDocument",
    "ParsedCoreInterval",
    "parse_core_document",
]
