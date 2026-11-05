"""Build literal dictionary rules from selected source text, with conflict checks."""

from copy import deepcopy
import re
from uuid import uuid4

from .semantics import DictionaryValidationError, _matches, validate_dictionary
from .terminology import ROCK_LABELS, SATURATION_LABELS, display_term

LEARNING_CATEGORIES = ("rock", "oil_indicator", "negation", "oil_saturation")


def normalize_expression(text):
    return " ".join(text.split()).casefold()


def literal_rules(payload, expression):
    normalized = normalize_expression(expression)
    return [rule for rule in payload["rules"] if rule["pattern_type"] != "regex"
            and any(normalize_expression(pattern) == normalized for pattern in rule["patterns"])]


def build_dictionary_draft(payload, *, category, canonical_value, expressions, rule_id=None):
    """Return a validated draft; never modify bundled or user data in place."""
    canonical_value = canonical_value.strip()
    if category not in LEARNING_CATEGORIES:
        raise DictionaryValidationError("Выберите категорию словаря.")
    if not canonical_value.strip():
        raise DictionaryValidationError("Укажите, как распознавать выражение.")
    if category == "oil_indicator" and canonical_value != "present":
        raise DictionaryValidationError("Для признака нефти выберите наличие нефтенасыщенности.")
    if category == "negation" and canonical_value != "none":
        raise DictionaryValidationError("Для отрицания выберите отсутствие признаков нефти.")
    if category == "oil_saturation" and canonical_value not in set(SATURATION_LABELS) - {"none", "unknown", "present", "uncertain"}:
        raise DictionaryValidationError("Выберите степень нефтенасыщенности из списка.")
    patterns = list({normalize_expression(text): " ".join(text.split())
                     for text in expressions if text.strip()}.values())
    if not patterns:
        raise DictionaryValidationError("Добавьте хотя бы одно слово или фразу.")
    ambiguous = r"(?:слаб|сильн|средн|интенсивн)(?:о|е|ый|ая|ое|ые|ого|ой|ому|ым|ом|ую|ою|ых|ыми|ий|яя|ее|ие|его|ей|ему|им|ем|юю|ею|их|ими)"
    if category == "oil_saturation" and any(re.fullmatch(ambiguous, normalize_expression(p)) for p in patterns):
        raise DictionaryValidationError("Выделите степень вместе с признаком нефти, например «слабо нефтенасыщенный».")
    draft = deepcopy(payload)
    existing = next((r for r in draft["rules"] if r["id"] == rule_id), None)
    if rule_id is not None and (existing is None or existing["pattern_type"] == "regex"):
        raise DictionaryValidationError("Исходное правило изменилось. Откройте форму заново.")
    others = deepcopy(draft)
    others["rules"] = [r for r in others["rules"] if r["id"] != rule_id]
    dictionary = validate_dictionary(others)
    for pattern in patterns:
        for match in _matches(pattern, dictionary.rules):
            if match.start != 0 or match.end != len(pattern):
                continue
            same = (match.category, match.canonical_value) == (category, canonical_value)
            positive_pair = {match.category, category} == {"oil_indicator", "oil_saturation"}
            if not same and not positive_pair and match.category in LEARNING_CATEGORIES:
                existing_label = display_term(match.canonical_value,
                                              ROCK_LABELS if match.category == "rock" else SATURATION_LABELS)
                raise DictionaryValidationError(
                    f'Выражение «{pattern}» уже относится к другому значению: «{existing_label}». '
                    'Измените существующую запись в редакторе словаря.')
    if existing is None:
        existing = {"id": f"user.{uuid4().hex}", "priority": 200, "enabled": True}
        draft["rules"].append(existing)
    existing.update(category=category, canonical_value=canonical_value, patterns=patterns,
                    pattern_type="phrase" if any(len(p.split()) > 1 for p in patterns) else "word",
                    enabled=True)
    validate_dictionary(draft)
    return draft
