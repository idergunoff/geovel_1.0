"""Wire the optional, mutually exclusive L1 normalization controls."""


def configure_l1_controls(ui, *, allow_all=True):
    selected = ui.checkBox_norm_l1
    all_components = ui.checkBox_norm_l1_all
    all_components.setVisible(allow_all)
    if not allow_all:
        all_components.setChecked(False)
    elif all_components.isChecked():
        selected.setChecked(False)
    selected.toggled.connect(lambda checked: all_components.setChecked(False) if checked else None)
    all_components.toggled.connect(lambda checked: selected.setChecked(False) if checked else None)
