import os

import pytest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from PyQt5 import QtCore, QtTest, QtWidgets, sip

from qt.work_windows import ask_in_window, show_work_window


@pytest.fixture
def application():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_work_windows_can_be_used_and_closed_independently(application):
    first, second = QtWidgets.QDialog(), QtWidgets.QDialog()
    button = QtWidgets.QPushButton('Action', second)
    actions = []
    button.clicked.connect(lambda: actions.append('clicked'))

    show_work_window(first)
    show_work_window(second)
    assert application.activeModalWidget() is None
    assert first.isVisible() and second.isVisible()

    QtTest.QTest.mouseClick(button, QtCore.Qt.LeftButton)
    first.close()
    QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
    assert sip.isdeleted(first)
    assert second.isVisible()
    QtTest.QTest.mouseClick(button, QtCore.Qt.LeftButton)
    assert actions == ['clicked', 'clicked']

    second.close()
    QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)


def test_scoped_question_leaves_other_work_windows_available(application):
    owner = QtWidgets.QDialog()
    other = QtWidgets.QDialog()
    question = QtWidgets.QDialog()
    button = QtWidgets.QPushButton('Action', other)
    actions = []
    button.clicked.connect(lambda: actions.append('clicked'))
    show_work_window(owner)
    show_work_window(other)

    def answer():
        QtTest.QTest.mouseClick(button, QtCore.Qt.LeftButton)
        question.accept()

    QtCore.QTimer.singleShot(0, answer)
    assert ask_in_window(question, owner) == QtWidgets.QDialog.Accepted
    assert actions == ['clicked']
    assert question.windowModality() == QtCore.Qt.WindowModal
    owner.close()
    other.close()
    QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
