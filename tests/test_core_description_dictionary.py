"""Regression examples for the extended lithology and saturation dictionary."""

from itertools import permutations

import pytest

from core_description import analyze_description, load_dictionary


@pytest.mark.parametrize("canonical, forms", [
    ("claystone", ("Аргиллит", "аргиллиты", "аргиллитов")),
    ("siltstone", ("Алевролит", "алевролиты", "алевролитов")),
    ("marl", ("Мергель", "мергели", "мергелями")),
    ("conglomerate", ("Конгломерат", "конгломераты", "конгломератов")),
    ("gravelite", ("Гравелит", "гравелиты", "гравелитов")),
    ("silt", ("Алеврит", "алевриты", "алевритов")),
    ("chalk", ("Мел", "мела", "мелом")),
    ("tripoli", ("Трепел", "трепелы", "трепела")),
    ("diatomite", ("Диатомит", "диатомиты", "диатомитов")),
    ("tuff", ("Туф", "туфы", "туфов")),
    ("gypsum", ("Гипс", "гипсы", "гипса")),
    ("anhydrite", ("Ангидрит", "ангидриты", "ангидритов")),
    ("breccia", ("Брекчия", "брекчии", "брекчий")),
    ("sand", ("Песок", "пески", "песков")),
    ("opoka", ("Опока", "опоки", "опок")),
    ("coal", ("Уголь", "угли", "углём")),
    ("shale", ("Сланец", "сланцы", "сланцев")),
    ("rock_salt", ("Каменная соль", "каменные соли", "каменной солью")),
])
def test_extended_rock_names_and_inflections(canonical, forms):
    for word in forms:
        rocks, oil, _, matches, _ = analyze_description(word)
        assert [(rock.canonical_value, rock.relation) for rock in rocks] == [(canonical, "primary")]
        assert oil == "unknown"
        assert matches[0].matched_text == word


def test_new_rocks_and_additional_rock_relations():
    rocks = analyze_description("Аргиллиты с тонкими прослойками алевролитов")[0]
    assert [(rock.canonical_value, rock.relation) for rock in rocks] == [
        ("claystone", "primary"), ("siltstone", "interlayer")
    ]


def test_rock_adjectives_and_substrings_do_not_create_additional_rocks():
    text = "Песчаник алевритистый, аргиллитоподобный, мергелистый, углистый, гипсовый"
    assert [rock.canonical_value for rock in analyze_description(text)[0]] == ["sandstone"]


@pytest.mark.parametrize("fluid", ["битумо", "нефте"])
@pytest.mark.parametrize("degrees, expected", [
    ("слабо и средне", "weak_to_medium"),
    ("слабо, средне", "weak_to_medium"),
    ("средне, слабо", "weak_to_medium"),
    ("средне и интенсивно", "medium_to_intense"),
    ("интенсивно или средне", "medium_to_intense"),
    ("слабо-сильно", "weak_to_intense"),
    ("сильно/слабо", "weak_to_intense"),
    ("от слабо до средне и интенсивно", "weak_to_intense"),
    ("слабо- и средне", "weak_to_medium"),
    ("от слабо- до интенсивно", "weak_to_intense"),
    ("средне, интенсивно и средне", "medium_to_intense"),
    ("средне, слабо, интенсивно, средне", "weak_to_intense"),
    ("слабо и слабо", "weak"),
])
def test_multiple_degrees_before_one_saturation_word(fluid, degrees, expected):
    text = f"Алевролиты {degrees} {fluid}насыщенные."
    _, oil, _, matches, _ = analyze_description(text)
    assert oil == expected
    if "_to_" in expected:
        assert any(
            match.canonical_value == expected
            and match.matched_text == f"{degrees} {fluid}насыщенные"
            for match in matches
        )
    assert all(text[match.start:match.end] == match.matched_text for match in matches)
    assert {match.rule_id for match in matches} <= {rule.id for rule in load_dictionary().rules}


@pytest.mark.parametrize("degrees", list(permutations(("слабо", "средне", "интенсивно"))))
def test_three_degrees_are_independent_of_order(degrees):
    text = ", ".join(degrees) + " битумонасыщенные"
    _, oil, _, matches, _ = analyze_description(text)
    assert oil == "weak_to_intense"
    assert any(match.canonical_value == oil and match.matched_text == text for match in matches)


@pytest.mark.parametrize("degrees", [
    "средне, местами интенсивно", "средне и участками интенсивно",
    "средне, а местами интенсивно", "интенсивно, участками средне",
])
def test_locality_with_multiple_degrees_is_preserved(degrees):
    _, oil, confidence, _, warnings = analyze_description(f"Мергели {degrees} битумонасыщенные")
    assert oil == "medium_to_intense"
    assert warnings == ["locality_marker"]
    assert confidence == 0.8


@pytest.mark.parametrize("text, expected", [
    ("Интенсивно трещиноватые, слабо битумонасыщенные", "weak"),
    ("Слабо сцементированные, интенсивно битумонасыщенные", "intense"),
    ("Слабо трещиноватые, средне и интенсивно битумонасыщенные", "medium_to_intense"),
    ("Интенсивно. Слабо битумонасыщенные", "weak"),
    ("Слабо сцементированные, битумонасыщенные", "present"),
    ("Слабо и интенсивно трещиноватые", "unknown"),
    ("Без признаков нефти; слабо и средне битумонасыщенные", "none"),
    ("Возможно слабо и интенсивно битумонасыщенные", "uncertain"),
])
def test_multiple_degrees_do_not_capture_unrelated_properties(text, expected):
    assert analyze_description(text)[1] == expected
