import ast
import json
from pathlib import Path
from types import SimpleNamespace


def load_auto_result_actions():
    source = Path("cluster/auto_runner.py").read_text()
    tree = ast.parse(source)
    names = {
        "_auto_result_identity",
        "_remove_auto_results_from_saved_cache",
        "remove_selected_cluster_auto_tune_results",
        "clear_cluster_auto_tune_results_with_confirm",
    }
    selected_nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *selected_nodes],
        type_ignores=[],
    )
    namespace = {"json": json, "CandidateResult": dict, "ClusterRunContext": dict}
    exec(compile(ast.fix_missing_locations(module), "cluster/auto_runner.py", "exec"), namespace)
    return namespace


class IndexStub:
    def __init__(self, row):
        self._row = row

    def row(self):
        return self._row


class SelectionModelStub:
    def __init__(self, rows):
        self._rows = rows

    def selectedRows(self):
        return [IndexStub(row) for row in self._rows]


class TableStub:
    def __init__(self, selected_rows=(), current_row=-1, row_count=0):
        self._selection_model = SelectionModelStub(selected_rows)
        self._current_row = current_row
        self._row_count = row_count

    def selectionModel(self):
        return self._selection_model

    def currentRow(self):
        return self._current_row

    def rowCount(self):
        return self._row_count


def result(candidate_id):
    return {
        "candidate_id": candidate_id,
        "candidate_config": {"method": "kmeans", "method_params": {"kmeans_n_clusters": 2}},
        "stats": {"partition_hash": f"hash-{candidate_id}"},
    }


def test_remove_selected_auto_result_updates_runtime_table():
    actions = load_auto_result_actions()
    results = [result("C001"), result("C002"), result("C003")]
    rendered = []
    messages = []
    actions.update(
        {
            "ui": SimpleNamespace(tableWidget_cluster_auto_result=TableStub(selected_rows=[1])),
            "cluster_auto_results_cache": results,
            "build_cluster_run_context": lambda **kwargs: None,
            "render_auto_results_table": lambda rows: rendered.append(rows),
            "set_info": lambda message, color: messages.append((message, color)),
        }
    )

    actions["remove_selected_cluster_auto_tune_results"]()

    assert rendered == [[results[0], results[2]]]
    assert messages[-1] == ("AUTO: удалено строк результатов: 1.", "green")


def test_remove_auto_results_updates_matching_persistent_cache_row():
    actions = load_auto_result_actions()
    results = [result("C001"), result("C002")]
    cache_row = SimpleNamespace(top_results=json.dumps(results))

    class QueryStub:
        def filter_by(self, **kwargs):
            assert kwargs == {"object_set_id": 7}
            return self

        def all(self):
            return [cache_row]

    class SessionStub:
        def __init__(self):
            self.commits = 0

        def query(self, model):
            return QueryStub()

        def commit(self):
            self.commits += 1

        def delete(self, row):
            raise AssertionError("row should be updated, not deleted")

    session = SessionStub()
    gpr_cache_model = type("GprCache", (), {})
    well_cache_model = type("WellCache", (), {})
    actions.update(
        {
            "session": session,
            "WellLogClusterAutoTuningCache": well_cache_model,
            "_auto_tuning_cache_model": lambda source_type: gpr_cache_model,
            "_ensure_cluster_auto_tuning_cache_table": lambda source_type: None,
            "_cache_row_matches_source_type": lambda row, source_type: True,
        }
    )

    updated = actions["_remove_auto_results_from_saved_cache"](
        {"source_type": "gpr", "dataset_id": 7},
        results,
        [results[0]],
    )

    assert updated == 1
    assert json.loads(cache_row.top_results) == [results[1]]
    assert session.commits == 1


def test_clear_auto_results_keeps_empty_runtime_context_and_table_headers():
    actions = load_auto_result_actions()
    context = {"source_type": "gpr", "dataset_id": 0, "data_hash": "hash"}
    context_key = ("gpr", 0, "hash")
    rendered = []
    messages = []

    class MessageBoxStub:
        Yes = 1
        No = 2

        @staticmethod
        def warning(*args, **kwargs):
            return MessageBoxStub.Yes

    actions.update(
        {
            "ui": SimpleNamespace(tableWidget_cluster_auto_result=TableStub(row_count=2)),
            "cluster_auto_results_cache": [result("C001")],
            "cluster_auto_results_by_context": {context_key: [result("C001")]},
            "build_cluster_run_context": lambda **kwargs: context,
            "_auto_context_cache_key": lambda value: context_key,
            "render_auto_results_table": lambda rows: rendered.append(rows),
            "set_info": lambda message, color: messages.append((message, color)),
            "QMessageBox": MessageBoxStub,
            "MainWindow": object(),
        }
    )

    actions["clear_cluster_auto_tune_results_with_confirm"]()

    assert actions["cluster_auto_results_by_context"][context_key] == []
    assert rendered == [[]]
    assert messages[-1] == ("AUTO: результаты автоподбора очищены.", "green")
