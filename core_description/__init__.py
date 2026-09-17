"""Pure parser for Word documents containing core descriptions."""

from .models import ParsedCoreDocument, ParsedCoreInterval, RockMention, SemanticMatch
from .parser import CoreDescriptionParseError, parse_core_document
from .semantics import DictionaryValidationError, analyze_description, load_dictionary

__all__ = [
    "CoreDescriptionParseError",
    "ParsedCoreDocument",
    "ParsedCoreInterval",
    "RockMention",
    "SemanticMatch",
    "DictionaryValidationError",
    "analyze_description",
    "load_dictionary",
    "parse_core_document",
]
