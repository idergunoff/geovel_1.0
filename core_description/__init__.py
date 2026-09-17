"""Pure parser for Word documents containing core descriptions."""

from .models import ParsedCoreDocument, ParsedCoreInterval, RockMention, SemanticMatch
from .parser import CoreDescriptionParseError, parse_core_document
from .batch import BatchParseItem, BatchParseResult, discover_documents, parse_batch
from .semantics import DictionaryValidationError, analyze_description, load_dictionary

__all__ = [
    "CoreDescriptionParseError",
    "BatchParseItem",
    "BatchParseResult",
    "discover_documents",
    "parse_batch",
    "ParsedCoreDocument",
    "ParsedCoreInterval",
    "RockMention",
    "SemanticMatch",
    "DictionaryValidationError",
    "analyze_description",
    "load_dictionary",
    "parse_core_document",
]
