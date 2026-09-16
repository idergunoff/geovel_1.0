"""Reusable object picker for bulk profile operations."""

from __future__ import annotations

from collections.abc import Iterable

from PyQt5 import QtCore, QtWidgets


def choose_profile_object_ids(
    parent,
    objects: Iterable[object],
    *,
    title: str,
    description: str,
) -> set[int] | None:
    """Let the user choose objects for a bulk profile operation.

    All objects are enabled initially.  ``None`` means that the dialog was
    cancelled, while an empty set means that the user accepted without
    selecting an object.
    """
    object_list = list(objects)
    dialog = QtWidgets.QDialog(parent)
    dialog.setWindowTitle(title)
    dialog.setMinimumSize(460, 420)

    layout = QtWidgets.QVBoxLayout(dialog)
    label = QtWidgets.QLabel(description)
    label.setWordWrap(True)
    layout.addWidget(label)

    object_widget = QtWidgets.QListWidget()
    object_widget.setAlternatingRowColors(True)
    for georadar_object in object_list:
        item = QtWidgets.QListWidgetItem(
            f"{georadar_object.title or 'Без названия'} (id {georadar_object.id})"
        )
        item.setData(QtCore.Qt.UserRole, georadar_object.id)
        item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
        item.setCheckState(QtCore.Qt.Checked)
        object_widget.addItem(item)
    layout.addWidget(object_widget)

    selection_buttons = QtWidgets.QHBoxLayout()
    select_all_button = QtWidgets.QPushButton("Выбрать все")
    clear_button = QtWidgets.QPushButton("Убрать все")
    selection_buttons.addWidget(select_all_button)
    selection_buttons.addWidget(clear_button)
    selection_buttons.addStretch()
    layout.addLayout(selection_buttons)

    def set_all(check_state):
        for row in range(object_widget.count()):
            object_widget.item(row).setCheckState(check_state)

    select_all_button.clicked.connect(lambda: set_all(QtCore.Qt.Checked))
    clear_button.clicked.connect(lambda: set_all(QtCore.Qt.Unchecked))

    button_box = QtWidgets.QDialogButtonBox(
        QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
    )
    button_box.accepted.connect(dialog.accept)
    button_box.rejected.connect(dialog.reject)
    layout.addWidget(button_box)

    if dialog.exec_() != QtWidgets.QDialog.Accepted:
        return None
    return {
        int(object_widget.item(row).data(QtCore.Qt.UserRole))
        for row in range(object_widget.count())
        if object_widget.item(row).checkState() == QtCore.Qt.Checked
    }
