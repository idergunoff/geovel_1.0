"""Check the empty analysis selection without starting the Qt application."""

import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import unittest


def check_empty_analysis(analysis_id):
    source = Path(__file__).resolve().parents[1] / 'mlp.py'
    tree = ast.parse(source.read_text(encoding='utf-8'))
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name in {'update_list_param_mlp', 'update_line_edit_exception_mlp'}]
    session = Mock()
    session.query.return_value.filter_by.return_value.first.return_value = None
    ui = SimpleNamespace(
        listWidget_param_mlp=Mock(), label_count_param_mlp=Mock(),
        lineEdit_signal_except=Mock(), lineEdit_crl_except=Mock(),
    )
    build_table = Mock(side_effect=AssertionError('No analysis to build'))
    namespace = dict(session=session, ui=ui, get_MLP_id=lambda: analysis_id,
                     AnalysisMLP=object(), ExceptionMLP=object(),
                     build_table_train=build_table)
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), 'exec'), namespace)

    namespace['update_list_param_mlp'](db=True)

    ui.listWidget_param_mlp.clear.assert_called_once_with()
    ui.label_count_param_mlp.setText.assert_called_once_with('<i><u>0</u></i> параметров')
    ui.lineEdit_signal_except.clear.assert_called_once_with()
    ui.lineEdit_crl_except.clear.assert_called_once_with()
    build_table.assert_not_called()


class EmptyAnalysisTests(unittest.TestCase):
    def test_empty_database(self):
        check_empty_analysis('')

    def test_missing_selected_analysis(self):
        check_empty_analysis('42')


if __name__ == '__main__':
    unittest.main()
