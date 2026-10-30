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


@pytest.mark.parametrize("text, expected", [
    ("Песчаник остаточно нефтенасыщенный", "residual"),
    ("Остаточная нефтенасыщенность", "residual"),
    ("Остаточное битумонасыщение", "residual"),
    ("Остаточная нефть в порах", "residual"),
    ("Песчаник с запахом нефти", "odor"),
    ("Слабый запах битума", "odor"),
    ("Запах углеводородов", "odor"),
    ("Нефтяной запах", "odor"),
    ("Песчаник, пятна нефти", "patchy"),
    ("Песчаник пятнами нефтенасыщенный", "patchy"),
    ("Пятнистая нефтенасыщенность", "patchy"),
    ("Битумонасыщение пятнами", "patchy"),
    ("Нефть отдельными пятнами", "patchy"),
    ("Битумные пятна", "patchy"),
    ("Остаточная влажность", "unknown"),
    ("Запах сероводорода", "unknown"),
    ("Песчаник с пятнами окислов железа", "unknown"),
    ("Пятнистая окраска", "unknown"),
    ("Без запаха нефти", "none"),
    ("Запах битума отсутствует", "none"),
    ("Запах нефти не ощущается", "none"),
    ("Без нефтяных пятен", "none"),
    ("Без признаков нефти, остаточная нефтенасыщенность", "none"),
    ("Возможно остаточная нефтенасыщенность", "uncertain"),
    ("Возможно запах нефти", "uncertain"),
    ("Возможно пятна нефти", "uncertain"),
])
def test_residual_odor_and_patchy_saturation(text, expected):
    _, oil, _, matches, _ = analyze_description(text)
    assert oil == expected
    assert all(text[m.start:m.end] == m.matched_text for m in matches)
    if expected in {"residual", "odor", "patchy"}:
        assert any(m.canonical_value == expected and m.category == "oil_saturation" for m in matches)


@pytest.mark.parametrize("double_n", [False, True])
@pytest.mark.parametrize("degree, expected", [
    ("", "present"), ("слабо", "weak"), ("средне", "medium"),
    ("интенсивно", "intense"), ("сильно", "intense"),
    ("слабо и средне", "weak_to_medium"),
    ("средне, интенсивно", "medium_to_intense"),
    ("от слабо до сильно", "weak_to_intense"),
    ("остаточно", "residual"), ("пятнами", "patchy"),
])
def test_bitumino_spelling_and_inflections_with_all_degrees(double_n, degree, expected):
    endings = ["ый", "ая", "ое", "ые", "ого", "ой", "ому", "ым", "ом", "ую", "ою", "ых", "ыми", "а", "о", "ы"]
    if not double_n:
        endings.append("")
    for ending in endings:
        word = "битуминонасыщен" + ("н" if double_n else "") + ending
        text = f"Песчаник {degree} {word}."
        _, oil, _, matches, _ = analyze_description(text)
        assert oil == expected, text
        assert any(word in match.matched_text for match in matches), text
        assert all(text[match.start:match.end] == match.matched_text for match in matches)


@pytest.mark.parametrize("fluid", ["битумо", "битумино", "нефте"])
@pytest.mark.parametrize("degree, expected", [("weak", "weak"), ("medium", "medium"), ("intense", "intense"), ("strong", "intense")])
@pytest.mark.parametrize("after_noun", [False, True])
def test_saturation_nouns_and_degree_adjectives_in_all_cases(fluid, degree, expected, after_noun):
    # Singular/plural forms of both насыщение and насыщенность, with agreeing adjectives.
    noun_endings = ["ие", "ия", "ию", "ием", "ии", "ия", "ий", "иям", "иями", "иях",
                    "ность", "ности", "ностью", "ностей", "ностям", "ностями", "ностях"]
    hard_endings = ["ое", "ого", "ому", "ым", "ом", "ые", "ых", "ым", "ыми", "ых",
                    "ая", "ой", "ой", "ых", "ым", "ыми", "ых"]
    soft_endings = ["ее", "его", "ему", "им", "ем", "ие", "их", "им", "ими", "их",
                    "яя", "ей", "ей", "их", "им", "ими", "их"]
    stems = {"weak": "слаб", "medium": "средн", "intense": "интенсивн", "strong": "сильн"}
    adjective_endings = soft_endings if degree == "medium" else hard_endings
    for noun_ending, adjective_ending in zip(noun_endings, adjective_endings):
        noun = fluid + "насыщен" + noun_ending
        adjective = stems[degree] + adjective_ending
        phrase = f"{noun}: {adjective}" if after_noun else f"{adjective} {noun}"
        text = "Песчаник " + phrase
        _, oil, _, matches, _ = analyze_description(text)
        assert oil == expected, text
        assert any(match.canonical_value == expected and match.matched_text == phrase for match in matches), text


@pytest.mark.parametrize("fluid", ["битумо", "битумино", "нефте"])
@pytest.mark.parametrize("phrase, expected", [
    ("со слабым {fluid}насыщением", "weak"),
    ("с {fluid}насыщением", "present"),
    ("с {fluid}насыщенностью", "present"),
    ("со слабым и средним {fluid}насыщением", "weak_to_medium"),
    ("от среднего до интенсивного {fluid}насыщения", "medium_to_intense"),
    ("от слабой до сильной {fluid}насыщенности", "weak_to_intense"),
    ("{fluid}насыщение от слабого до среднего", "weak_to_medium"),
    ("{fluid}насыщенность: средняя и интенсивная", "medium_to_intense"),
    ("{fluid}насыщение — от слабого до сильного", "weak_to_intense"),
    ("{fluid}насыщенность:слабая", "weak"),
    ("с остаточным {fluid}насыщением", "residual"),
    ("{fluid}насыщение пятнами", "patchy"),
])
def test_saturation_noun_phrases_and_ranges(fluid, phrase, expected):
    text = phrase.format(fluid=fluid)
    _, oil, _, matches, _ = analyze_description(text)
    assert oil == expected
    assert all(text[match.start:match.end] == match.matched_text for match in matches)


@pytest.mark.parametrize("text, expected", [
    ("БИТУМИНОНАСЫЩЕНЫЕ", "present"),
    ("слабобитуминонасыщеные", "weak"),
    ("средне-битуминонасыщеные", "medium"),
    ("без битумонасыщения", "none"),
    ("без слабого битуминонасыщения", "none"),
    ("без слабого и среднего битумонасыщения", "none"),
    ("без остаточной битумонасыщенности", "none"),
    ("битумонасыщение отсутствует", "none"),
    ("битумонасыщенность: отсутствует", "none"),
    ("слабое битумонасыщение не отмечается", "none"),
    ("битуминонасыщение не обнаружено", "none"),
    ("не битуминонасыщеные", "none"),
    ("не сильно битуминонасыщеные", "none"),
    ("без признаков нефти, со слабым битумонасыщением", "none"),
    ("возможно со слабым битумонасыщением", "uncertain"),
    ("интенсивно трещиноватый, со слабым битумонасыщением", "weak"),
    ("слабо сцементированный, с интенсивным битуминонасыщением", "intense"),
    ("со слабым запахом битума", "odor"),
    ("со слабым цементом", "unknown"),
    ("со средними прослоями", "unknown"),
    ("с интенсивной трещиноватостью", "unknown"),
    ("слабая окраска, битумонасыщение", "present"),
    ("битумонасыщение. Слабая окраска", "present"),
    ("со слабым битумонасыщениемподобным", "unknown"),
    ("слабо битуминонасыщеноподобный", "unknown"),
    ("антибитуминонасыщеные", "unknown"),
])
def test_new_word_forms_preserve_negation_context_and_boundaries(text, expected):
    assert analyze_description(text)[1] == expected
