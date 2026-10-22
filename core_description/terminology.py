"""Russian display terms with stable codes for stored classifications."""

ROCK_LABELS = {
    "sandstone": "Песчаник", "limestone": "Известняк", "dolomite": "Доломит",
    "clay": "Глина", "claystone": "Аргиллит", "siltstone": "Алевролит",
    "marl": "Мергель", "conglomerate": "Конгломерат", "gravelite": "Гравелит",
    "silt": "Алеврит", "chalk": "Мел", "tripoli": "Трепел",
    "diatomite": "Диатомит", "tuff": "Туф", "gypsum": "Гипс",
    "anhydrite": "Ангидрит", "breccia": "Брекчия", "sand": "Песок",
    "opoka": "Опока", "coal": "Уголь", "shale": "Сланец",
    "rock_salt": "Каменная соль",
}

SATURATION_LABELS = {
    "unknown": "Не определена",
    "none": "Признаки нефти отсутствуют",
    "weak": "Слабая",
    "medium": "Средняя",
    "intense": "Интенсивная",
    "weak_to_medium": "От слабой до средней",
    "medium_to_intense": "От средней до интенсивной",
    "weak_to_intense": "От слабой до интенсивной",
    "residual": "Остаточная",
    "odor": "Запах",
    "patchy": "Пятнами",
    "present": "Нефтенасыщенность без указания степени",
    "uncertain": "Нефтенасыщенность под вопросом",
}

RELATION_LABELS = {
    "primary": "основная", "secondary": "дополнительная",
    "interlayer": "прослой", "interbedding": "переслаивание", "inclusion": "включение",
}


def display_term(value: str, labels: dict[str, str]) -> str:
    """Keep custom dictionary values visible as supplied."""
    return labels.get(value, value)


def canonical_term(value: str, labels: dict[str, str]) -> str:
    """Accept Russian labels and legacy codes, preserving custom values."""
    value = value.strip()
    normalized = value.casefold()
    for code, label in labels.items():
        if normalized in (code.casefold(), label.casefold()):
            return code
    return value
