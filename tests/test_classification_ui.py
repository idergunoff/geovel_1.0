from pathlib import Path
from xml.etree import ElementTree


ROOT = Path(__file__).resolve().parents[1]


def test_classification_has_one_add_all_wells_button():
    """The wired button must not be covered by a duplicate in its grid cell."""
    ui_root = ElementTree.parse(ROOT / "qt" / "geovel_main_window.ui").getroot()
    buttons = ui_root.findall(
        ".//widget[@class='QPushButton'][@name='pushButton_add_all_well_cls']"
    )
    generated_ui = (ROOT / "qt" / "geovel_main_window.py").read_text(encoding="utf-8")

    assert len(buttons) == 1
    assert generated_ui.count(
        "self.pushButton_add_all_well_cls = QtWidgets.QPushButton"
    ) == 1
    assert "pushButton_add_all_well_cls1" not in generated_ui
