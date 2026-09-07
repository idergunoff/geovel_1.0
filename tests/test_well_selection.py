from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from well_selection import find_well_row, well_id_from_text


class Item:
    def __init__(self, text):
        self._text = text

    def text(self):
        return self._text


def test_well_id_from_text_reads_only_final_id_suffix():
    assert well_id_from_text("скв.№ 10 id10") == 10
    assert well_id_from_text("скв.№ id42 [profile] id7") == 7
    assert well_id_from_text("скважина без идентификатора") is None


def test_find_well_row_does_not_confuse_id_with_longer_id_prefix():
    items = [Item("скв.№ первая id10"), Item("скв.№ вторая id1")]

    assert find_well_row(items, 1) == 1
    assert find_well_row(items, 10) == 0


def test_find_well_row_accepts_string_id_and_handles_invalid_value():
    items = [Item("скв.№ первая id3")]

    assert find_well_row(items, "3") == 0
    assert find_well_row(items, None) is None
