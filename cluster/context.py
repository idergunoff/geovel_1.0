from __future__ import annotations

from .common import *

def _estimate_array_like_nbytes(value: Any) -> int:
    """
    Грубая оценка объема памяти array-like объекта в байтах.
    """
    if value is None:
        return 0
    if isinstance(value, np.ndarray):
        return int(value.nbytes)
    if hasattr(value, "nbytes"):
        try:
            return int(value.nbytes)  # type: ignore[attr-defined]
        except Exception:
            pass
    try:
        return int(np.array(value).nbytes)
    except Exception:
        return 0


def _estimate_transform_cache_item_nbytes(value: Any) -> int:
    """
    Оценивает объем одного элемента transform-cache.
    """
    if isinstance(value, tuple):
        return int(sum(_estimate_array_like_nbytes(v) for v in value))
    return _estimate_array_like_nbytes(value)


def _trim_transform_cache(
        transform_cache: "OrderedDict[tuple, Any]",
        cache_sizes: "OrderedDict[tuple, int]",
        cache_total_bytes: int,
        *,
        max_cache_bytes: int
) -> int:
    """
    Поддерживает LRU-ограничение transform-cache по памяти.
    """
    while cache_total_bytes > max_cache_bytes and transform_cache:
        evicted_key, _ = transform_cache.popitem(last=False)
        cache_total_bytes -= int(cache_sizes.pop(evicted_key, 0))
    return max(0, int(cache_total_bytes))


def _sample_rows_for_auto_tuning(data, max_rows: int) -> Any:
    """
    Ограничивает число строк для AUTO-подбора, чтобы снизить риск OOM.
    """
    if data is None:
        return data
    try:
        max_rows = max(2, int(max_rows))
    except (TypeError, ValueError):
        max_rows = AUTO_TUNING_MAX_ROWS
    data_len = len(data)
    if data_len <= max_rows:
        return data
    n_features = None
    if data_len > 0:
        try:
            first_row = data[0]
            n_features = len(first_row)
        except Exception:
            n_features = None
    if n_features and n_features > 0:
        approx_bytes_per_row = int(max(8, n_features * 8 * 6))
        max_rows_by_memory = int(AUTO_TUNING_MAX_WORKING_SET_BYTES // approx_bytes_per_row)
        max_rows = min(max_rows, max(2, max_rows_by_memory))
    max_rows = max(2, min(max_rows, data_len))
    if data_len <= max_rows:
        return data
    rng = np.random.default_rng(42)
    sampled_idx = np.sort(rng.choice(data_len, size=max_rows, replace=False))
    if isinstance(data, np.ndarray):
        return data[sampled_idx]
    return [data[int(i)] for i in sampled_idx]




def _build_sample_indices_for_auto_tuning(data_len: int, max_rows: int, *, seed: int) -> np.ndarray:
    """
    Строит детерминированные индексы подвыборки для AUTO по seed.
    """
    data_len = max(0, int(data_len))
    if data_len == 0:
        return np.array([], dtype=int)
    try:
        max_rows = max(2, int(max_rows))
    except (TypeError, ValueError):
        max_rows = AUTO_TUNING_MAX_ROWS
    if data_len <= max_rows:
        return np.arange(data_len, dtype=int)
    rng = np.random.default_rng(int(seed))
    sampled_idx = np.sort(rng.choice(data_len, size=max_rows, replace=False))
    return np.asarray(sampled_idx, dtype=int)


def _apply_sample_indices(data, sampled_idx: np.ndarray) -> Any:
    if data is None:
        return data
    if sampled_idx is None or len(sampled_idx) == 0:
        return data
    if isinstance(data, np.ndarray):
        return data[sampled_idx]
    return [data[int(i)] for i in sampled_idx]

def _reduce_feature_space_for_auto_tuning(data, max_features: int) -> Any:
    """
    Ограничивает число признаков для AUTO-подбора при очень высокой размерности.
    """
    if data is None:
        return data
    try:
        max_features = max(2, int(max_features))
    except (TypeError, ValueError):
        max_features = AUTO_TUNING_MAX_FEATURES

    data_np = np.asarray(data, dtype=np.float64)
    if data_np.ndim != 2:
        return data
    n_rows = int(data_np.shape[0])
    n_features = int(data_np.shape[1])
    if AUTO_TUNING_FEATURE_REDUCTION_MODE == "off":
        return data_np
    matrix_bytes = int(max(1, n_rows) * max(1, n_features) * 4)
    if AUTO_TUNING_FEATURE_REDUCTION_MODE == "auto":
        if (
                n_features <= int(AUTO_TUNING_REDUCE_FEATURES_THRESHOLD)
                and matrix_bytes <= int(AUTO_TUNING_MAX_WORKING_SET_BYTES)
        ):
            return data_np
        # Оценка безопасной целевой размерности с учетом временных копий.
        approx_copies_factor = 3
        max_features_by_memory = int(
            AUTO_TUNING_MAX_WORKING_SET_BYTES // max(1, n_rows * 4 * approx_copies_factor)
        )
        max_features = max(32, min(int(max_features), int(max_features_by_memory)))
    if n_features <= int(max_features):
        return data_np

    if AUTO_TUNING_FEATURE_REDUCTION_MODE in {"random_projection", "auto"}:
        projector = SparseRandomProjection(
            n_components=max_features,
            dense_output=True,
            random_state=42
        )
        reduced = projector.fit_transform(data_np)
        return np.asarray(reduced, dtype=np.float64)

    feature_idx = np.linspace(0, n_features - 1, num=max_features, dtype=int)
    return np.asarray(data_np[:, feature_idx], dtype=np.float64)


def _serialize_cluster_dataset(data: list[list[Any]]) -> str:
    """
    Сериализует набор данных кластера в компактный JSON,
    а для крупных наборов — в gzip+base64 строку.
    """
    json_payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    encoded_size = len(json_payload.encode("utf-8"))

    # Сжимаем только крупные payload, чтобы снизить размер записи в БД.
    if encoded_size < 10 * 1024 * 1024:
        return json_payload

    compressed = gzip.compress(json_payload.encode("utf-8"), compresslevel=6)
    b64_payload = base64.b64encode(compressed).decode("ascii")
    return f"{CLUSTER_DATA_GZIP_PREFIX}{b64_payload}"


def _deserialize_cluster_dataset(raw_data: str) -> list[list[Any]]:
    """
    Десериализует набор данных кластера из JSON или gzip+base64 формата.
    """
    if not raw_data:
        return []

    if raw_data.startswith(CLUSTER_DATA_GZIP_PREFIX):
        compressed_b64 = raw_data[len(CLUSTER_DATA_GZIP_PREFIX):]
        decompressed = gzip.decompress(base64.b64decode(compressed_b64.encode("ascii")))
        return json.loads(decompressed.decode("utf-8"))

    return json.loads(raw_data)


WELL_LOG_CLUSTER_REQUIRED_META_COLUMNS = ["well_id", "well_name", "depth_md", "row_index_in_well"]
WELL_LOG_CLUSTER_SAMPLE_INDEX_FEATURE_NAME = "sample_index_in_well"
WELL_LOG_CLUSTER_MIN_VALID_ROWS_PER_WELL = 2
WELL_LOG_CLUSTER_MIN_TOTAL_ROWS = 4


def _cluster_data_hash(raw_data: Any) -> str:
    """
    Возвращает стабильный hash runtime-данных для будущей инвалидации результатов.
    """
    payload = json.dumps(raw_data, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _current_tab_matches(tab_widget: Any, *, expected_widget: Any = None, expected_titles: set[str] | None = None) -> bool:
    """
    Проверяет активную вкладку без жесткой привязки только к сгенерированному имени.

    В старом UI вкладка Well Log известна как ``tab_14``, но для тестов/будущих форм
    надежнее дополнительно сверять текст вкладки.
    """
    if tab_widget is None:
        return False

    try:
        current_widget = tab_widget.currentWidget()
    except Exception:
        current_widget = None
    if expected_widget is not None and current_widget is expected_widget:
        return True

    try:
        current_text = str(tab_widget.tabText(tab_widget.currentIndex())).strip().casefold()
    except Exception:
        current_text = ""
    return bool(expected_titles and current_text in expected_titles)


def get_active_cluster_source_type() -> Literal["gpr", "well_log"]:
    """
    Определяет активный источник кластеризации по вложенной вкладке Cluster.

    Возвращает ``well_log`` только когда активна вкладка Well Log; во всех остальных
    случаях используется исторический источник ``gpr``.
    """
    tab_widget = getattr(ui, "tabWidget", None)
    well_log_tab = getattr(ui, "tab_14", None)
    if _current_tab_matches(
        tab_widget,
        expected_widget=well_log_tab,
        expected_titles={"well log", "well_log", "каротаж", "скважинный каротаж"},
    ):
        return "well_log"

    return "gpr"


def _show_cluster_context_error(source_type: str, error_text: str) -> None:
    title = "WELL LOG CLUSTER" if source_type == "well_log" else "CLUSTER"
    try:
        QMessageBox.warning(MainWindow, title, error_text)
    except Exception:
        pass
    set_info(error_text, "brown")


def build_cluster_run_context(*, show_errors: bool = True) -> ClusterRunContext | None:
    """
    Строит единый контекст расчета для активной вкладки Cluster.
    """
    source_type = get_active_cluster_source_type()
    try:
        if source_type == "well_log":
            return build_well_log_cluster_context()
        return build_gpr_cluster_context()
    except ClusterContextError as exc:
        if show_errors:
            _show_cluster_context_error(source_type, str(exc))
        return None


def _parse_gpr_profile_trace(value: Any) -> tuple[int | None, int | None]:
    """Парсит исторический ключ профиля/трассы вида ``profile_id_trace_index``."""
    text = str(value or "").strip()
    if "_" not in text:
        return None, None
    profile_part, trace_part = text.rsplit("_", 1)
    try:
        return int(profile_part), int(trace_part)
    except (TypeError, ValueError):
        return None, None


def _build_gpr_meta_rows(raw_rows: list[Any]) -> tuple[list[dict[str, Any]], list[list[float]], dict[str, int]]:
    """
    Формирует meta/X-представление GPR для общего ClusterRunContext.

    ``raw_rows`` сохраняются в историческом формате для существующего pipeline, а этот
    адаптер добавляет явную metadata-модель, аналогичную Well Log контексту.
    """
    meta_rows: list[dict[str, Any]] = []
    x_rows: list[list[float]] = []
    invalid_meta_rows = 0
    invalid_feature_rows = 0

    for source_idx, row in enumerate(raw_rows):
        if not isinstance(row, (list, tuple)) or len(row) < 4:
            invalid_meta_rows += 1
            continue

        profile_id, trace_index = _parse_gpr_profile_trace(row[0])
        if profile_id is None or trace_index is None:
            invalid_meta_rows += 1

        try:
            x_coord = float(row[1])
        except (TypeError, ValueError):
            x_coord = float("nan")
        try:
            y_coord = float(row[2])
        except (TypeError, ValueError):
            y_coord = float("nan")

        feature_values: list[float] = []
        for value in row[3:]:
            try:
                feature_values.append(float(value))
            except (TypeError, ValueError):
                feature_values.append(float("nan"))
        if not any(np.isfinite(value) for value in feature_values):
            invalid_feature_rows += 1

        meta_rows.append({
            "profile_trace_key": str(row[0]),
            "profile_id": profile_id,
            "trace_index": trace_index,
            "x": x_coord,
            "y": y_coord,
            "source_row_index": int(source_idx),
        })
        x_rows.append(feature_values)

    diagnostics = {
        "invalid_meta_rows": invalid_meta_rows,
        "invalid_feature_rows": invalid_feature_rows,
    }
    return meta_rows, x_rows, diagnostics


def build_gpr_cluster_context() -> ClusterRunContext:
    """
    Адаптер текущего ObjectSet георадара к единому ClusterRunContext.
    """
    clust_object_id = get_curr_clust_object_id()
    if not clust_object_id:
        raise ClusterContextError("Выберите/соберите ObjectSet для кластеризации георадара.")

    clust_object = session.query(ObjectSet).filter_by(id=int(clust_object_id)).first()
    if clust_object is None:
        raise ClusterContextError(f"ObjectSet id={clust_object_id} не найден. Обновите список наборов.")
    if not clust_object.data:
        raise ClusterContextError("В выбранном ObjectSet нет собранных данных. Выполните COLLECT.")

    try:
        raw_rows = _deserialize_cluster_dataset(clust_object.data)
    except Exception as exc:
        raise ClusterContextError(f"Не удалось прочитать данные ObjectSet: {exc}") from exc
    if not raw_rows:
        raise ClusterContextError("В выбранном ObjectSet пустой набор данных. Выполните COLLECT.")

    try:
        analysis = session.query(AnalysisCluster).filter_by(id=int(clust_object.analysis_id)).first()
        feature_names = json.loads(analysis.parameter) if analysis and analysis.parameter else []
    except Exception:
        feature_names = []
    if not isinstance(feature_names, list):
        feature_names = []
    feature_count = max(0, len(raw_rows[0]) - 3) if raw_rows and isinstance(raw_rows[0], (list, tuple)) else 0
    if not feature_names or len(feature_names) != feature_count:
        feature_names = [f"feature_{idx + 1}" for idx in range(feature_count)]

    dataset_title = str(getattr(ui, "comboBox_clust_obj", None).currentText()) if hasattr(ui, "comboBox_clust_obj") else f"ObjectSet id{clust_object_id}"
    meta_rows, x_rows, meta_diagnostics = _build_gpr_meta_rows(raw_rows)
    diagnostics = {
        "invalid_rows": 0,
        "excluded_wells": [],
        **meta_diagnostics,
    }
    return ClusterRunContext(
        source_type="gpr",
        dataset_id=int(clust_object_id),
        dataset_title=dataset_title,
        raw_rows=raw_rows,
        feature_names=[str(name) for name in feature_names],
        meta_columns=["profile_trace_key", "profile_id", "trace_index", "x", "y"],
        feature_columns=[str(name) for name in feature_names],
        row_count=len(raw_rows),
        ui_tab_key="cluster_georadar",
        data_hash=_cluster_data_hash(raw_rows),
        meta=meta_rows,
        X=x_rows,
        diagnostics=diagnostics,
    )


def _coerce_well_log_feature_value(value: Any) -> float:
    if value is None or value == "":
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _parse_well_log_row_key(value: Any) -> tuple[int, float]:
    text = str(value or "").strip()
    if "_" not in text:
        raise ValueError("ожидается формат well_id_depth")
    well_part, depth_part = text.split("_", 1)
    return int(well_part), float(depth_part)


def _normalize_well_log_source_curve_names(raw_value: Any, feature_names: list[str]) -> dict[str, str]:
    """
    Нормализует трассировку исходных curve-name для runtime-строки Well Log.

    Компактный формат старых COLLECT-записей не хранит отдельные source_curve_names,
    поэтому для него каноническое имя признака считается одновременно исходным именем.
    """
    if isinstance(raw_value, dict):
        return {str(name): str(raw_value.get(name, name)) for name in feature_names}
    return {str(name): str(name) for name in feature_names}


def _normalize_well_log_runtime_rows(stored_rows: Any) -> tuple[list[str], list[dict[str, Any]], dict[str, Any]]:
    """
    Приводит WellLogClusterDatasetData.data к единому runtime-формату этапа 4.

    Поддерживаются два формата хранения:
    - компактная таблица COLLECT: header ``[well_id_depth, GR, ...]`` + строки;
    - явные JSON-строки: ``well_id``, ``depth_md``, ``features``, ``source_curve_names``.

    Возвращает ``feature_names``, список нормализованных строк с meta/features и
    диагностический summary по отброшенным строкам.
    """
    invalid_rows: list[dict[str, Any]] = []
    normalized_rows: list[dict[str, Any]] = []

    if not isinstance(stored_rows, list) or not stored_rows:
        return [], [], {"invalid_row_count": 0, "invalid_rows_preview": []}

    first_row = stored_rows[0]
    if isinstance(first_row, dict):
        feature_order: OrderedDict[str, None] = OrderedDict()
        for source_idx, row in enumerate(stored_rows):
            if not isinstance(row, dict):
                invalid_rows.append({"row": source_idx, "reason": "row is not an object"})
                continue
            features = row.get("features")
            if not isinstance(features, dict):
                invalid_rows.append({"row": source_idx, "reason": "missing features object"})
                continue
            for feature_name in features:
                feature_order.setdefault(str(feature_name), None)

        feature_names = list(feature_order.keys())
        for source_idx, row in enumerate(stored_rows):
            if not isinstance(row, dict):
                continue
            features = row.get("features")
            if not isinstance(features, dict):
                continue
            try:
                well_id = int(row["well_id"])
                depth_md = float(row["depth_md"])
            except Exception as exc:
                invalid_rows.append({"row": source_idx, "reason": f"invalid well_id/depth_md: {exc}"})
                continue
            feature_values = {name: features.get(name) for name in feature_names}
            normalized_rows.append(
                {
                    "source_row_index": int(source_idx),
                    "well_id": well_id,
                    "well_name": str(row.get("well_name") or f"well_id={well_id}"),
                    "depth_md": depth_md,
                    "features": feature_values,
                    "source_curve_names": _normalize_well_log_source_curve_names(row.get("source_curve_names"), feature_names),
                }
            )
        return feature_names, normalized_rows, {
            "invalid_row_count": len(invalid_rows),
            "invalid_rows_preview": invalid_rows[:10],
        }

    header = first_row
    if not isinstance(header, list) or len(header) < 2:
        invalid_rows.append({"row": 0, "reason": "invalid compact header"})
        return [], [], {"invalid_row_count": len(invalid_rows), "invalid_rows_preview": invalid_rows[:10]}

    feature_names = [str(name) for name in header[1:]]
    for source_idx, row in enumerate(stored_rows[1:]):
        stored_row_number = source_idx + 1
        if not isinstance(row, list) or len(row) < len(feature_names) + 1:
            invalid_rows.append({"row": stored_row_number, "reason": "row length/header mismatch"})
            continue
        try:
            well_id, depth_md = _parse_well_log_row_key(row[0])
        except Exception as exc:
            invalid_rows.append({"row": stored_row_number, "reason": f"invalid well_id/depth_md: {exc}"})
            continue
        normalized_rows.append(
            {
                "source_row_index": int(source_idx),
                "well_id": int(well_id),
                "well_name": f"well_id={well_id}",
                "depth_md": float(depth_md),
                "features": dict(zip(feature_names, row[1:len(feature_names) + 1])),
                "source_curve_names": _normalize_well_log_source_curve_names(None, feature_names),
            }
        )

    return feature_names, normalized_rows, {
        "invalid_row_count": len(invalid_rows),
        "invalid_rows_preview": invalid_rows[:10],
    }


def build_well_log_cluster_context(dataset_id: Optional[int] = None) -> ClusterRunContext:
    """
    Адаптер WellLogClusterDatasetData.data к единому ClusterRunContext.

    Если ``dataset_id`` не передан, dataset берется из активного comboBox вкладки
    Well Log. Явный ``dataset_id`` используется batch-режимом, чтобы обрабатывать
    наборы каротажа независимо от текущего выбора в UI.
    """
    if dataset_id is None:
        combo = getattr(ui, "comboBox_cluster_well_set", None)
        if combo is None:
            raise ClusterContextError("Не найден comboBox_cluster_well_set для выбора Well Log dataset.")

        dataset_id = combo.currentData()
        if dataset_id is None:
            raise ClusterContextError("Выберите Well Log dataset и выполните COLLECT.")
    dataset_id = int(dataset_id)

    dataset = session.query(WellLogClusterDataset).filter_by(id=dataset_id).first()
    if dataset is None:
        raise ClusterContextError(f"Well Log dataset id={dataset_id} не найден. Обновите список наборов.")

    data_row = (
        session.query(WellLogClusterDatasetData)
        .filter(WellLogClusterDatasetData.dataset_id == dataset_id)
        .order_by(WellLogClusterDatasetData.id.desc())
        .first()
    )
    if data_row is None or not data_row.data:
        raise ClusterContextError(
            f'Для Well Log dataset "{dataset.name}" нет собранных data. Выполните COLLECT перед CALC/AUTO.'
        )

    try:
        stored_rows = _deserialize_cluster_dataset(data_row.data)
    except Exception as exc:
        raise ClusterContextError(f"Не удалось прочитать Well Log data. Повторите COLLECT: {exc}") from exc
    if not stored_rows or len(stored_rows) < 2:
        raise ClusterContextError(
            f'Well Log dataset "{dataset.name}" не содержит строк расчета. Выполните COLLECT.'
        )

    feature_names, runtime_rows, normalize_diagnostics = _normalize_well_log_runtime_rows(stored_rows)
    if not feature_names:
        raise ClusterContextError("В Well Log data нет признаков каротажа. Добавьте параметры и выполните COLLECT.")
    feature_names = [WELL_LOG_CLUSTER_SAMPLE_INDEX_FEATURE_NAME] + [
        name for name in feature_names if name != WELL_LOG_CLUSTER_SAMPLE_INDEX_FEATURE_NAME
    ]
    if not runtime_rows:
        raise ClusterContextError(
            f'Well Log dataset "{dataset.name}" не содержит валидных runtime-строк. Повторите COLLECT.'
        )

    well_names = {
        int(row.id): str(row.name or f"well_id={row.id}")
        for row in session.query(Well.id, Well.name).all()
    }

    invalid_rows: list[dict[str, Any]] = list(normalize_diagnostics.get("invalid_rows_preview", []))
    invalid_row_count = int(normalize_diagnostics.get("invalid_row_count", len(invalid_rows)))
    prepared_rows: list[list[Any]] = []
    meta_rows: list[dict[str, Any]] = []
    x_rows: list[list[float]] = []
    row_index_by_well: dict[int, int] = {}
    dropped_rows_by_reason: Counter[str] = Counter()

    runtime_rows.sort(key=lambda item: (int(item["well_id"]), float(item["depth_md"]), int(item["source_row_index"])))
    for row in runtime_rows:
        well_id = int(row["well_id"])
        depth_md = float(row["depth_md"])
        source_idx = int(row.get("source_row_index", len(meta_rows)))
        row_index = row_index_by_well.get(well_id, 0)
        row_index_by_well[well_id] = row_index + 1

        raw_features = dict(row.get("features", {}) or {})
        # Обязательный признак последовательности всегда пересчитываем после фильтрации
        # выбранного интервала и сортировки глубин внутри каждой скважины. Так CALC/AUTO
        # используют именно номер отсчета в текущем интервале, даже для старых COLLECT-данных.
        raw_features[WELL_LOG_CLUSTER_SAMPLE_INDEX_FEATURE_NAME] = int(row_index)
        feature_values = [_coerce_well_log_feature_value(raw_features.get(name)) for name in feature_names]
        log_feature_values = [
            value for name, value in zip(feature_names, feature_values)
            if name != WELL_LOG_CLUSTER_SAMPLE_INDEX_FEATURE_NAME
        ]
        if not any(np.isfinite(value) for value in log_feature_values):
            invalid_row_count += 1
            dropped_rows_by_reason["all log feature values are non-finite"] += 1
            if len(invalid_rows) < 10:
                invalid_rows.append({"row": source_idx, "reason": "all log feature values are non-finite"})
            continue

        well_name = well_names.get(well_id, str(row.get("well_name") or f"well_id={well_id}"))
        source_curve_names = _normalize_well_log_source_curve_names(row.get("source_curve_names"), feature_names)
        meta = {
            "well_id": well_id,
            "well_name": well_name,
            "depth_md": depth_md,
            "row_index_in_well": int(row_index),
            "source_row_index": source_idx,
            "source_curve_values": dict(zip(feature_names, feature_values)),
            "source_curve_names": source_curve_names,
        }
        meta_rows.append(meta)
        x_rows.append(feature_values)
        prepared_rows.append([source_idx, well_id, depth_md] + feature_values)

    rows_by_well: dict[int, int] = Counter(int(meta["well_id"]) for meta in meta_rows)
    excluded_wells = [
        {"well_id": well_id, "well_name": well_names.get(well_id, f"well_id={well_id}"), "valid_rows": count}
        for well_id, count in rows_by_well.items()
        if count < WELL_LOG_CLUSTER_MIN_VALID_ROWS_PER_WELL
    ]
    excluded_ids = {int(item["well_id"]) for item in excluded_wells}
    excluded_row_count = 0
    if excluded_ids:
        keep_mask = [int(meta["well_id"]) not in excluded_ids for meta in meta_rows]
        excluded_row_count = sum(1 for keep in keep_mask if not keep)
        prepared_rows = [row for row, keep in zip(prepared_rows, keep_mask) if keep]
        meta_rows = [meta for meta, keep in zip(meta_rows, keep_mask) if keep]
        x_rows = [row for row, keep in zip(x_rows, keep_mask) if keep]

    remaining_wells = {int(meta["well_id"]) for meta in meta_rows}
    if len(remaining_wells) < 2:
        raise ClusterContextError(
            "Well Log data недостаточно для кластеризации: после валидации осталось меньше двух скважин. "
            "Проверьте интервалы/параметры и выполните COLLECT."
        )
    if len(prepared_rows) < WELL_LOG_CLUSTER_MIN_TOTAL_ROWS:
        raise ClusterContextError(
            "Well Log data содержит слишком мало валидных строк для кластеризации. Выполните COLLECT с большим интервалом."
        )

    diagnostics = {
        "invalid_row_count": invalid_row_count,
        "invalid_rows_preview": invalid_rows[:10],
        "dropped_rows_by_reason": dict(dropped_rows_by_reason),
        "excluded_wells": excluded_wells,
        "excluded_well_count": len(excluded_wells),
        "excluded_row_count": excluded_row_count,
        "valid_well_count": len(remaining_wells),
        "valid_row_count": len(prepared_rows),
        "feature_count": len(feature_names),
        "runtime_format": "object_rows" if isinstance(stored_rows[0], dict) else "compact_table",
    }

    return ClusterRunContext(
        source_type="well_log",
        dataset_id=dataset_id,
        dataset_title=str(dataset.name),
        raw_rows=prepared_rows,
        feature_names=feature_names,
        meta_columns=list(WELL_LOG_CLUSTER_REQUIRED_META_COLUMNS),
        feature_columns=feature_names,
        row_count=len(prepared_rows),
        ui_tab_key="cluster_well_log",
        data_hash=_cluster_data_hash(stored_rows),
        meta=meta_rows,
        X=x_rows,
        diagnostics=diagnostics,
    )


CLUSTER_MAP_INTERP_RESOLUTION_MIN = 50
CLUSTER_MAP_INTERP_RESOLUTION_MAX = 500
CLUSTER_MAP_INTERP_RESOLUTION_DEFAULT = 200


def _normalize_interpolation_resolution(value: Any) -> int:
    """
    Нормализует разрешение сетки для интерполяции карты кластеров.
    """
    try:
        resolution = int(value)
    except (TypeError, ValueError):
        return CLUSTER_MAP_INTERP_RESOLUTION_DEFAULT
    if resolution < CLUSTER_MAP_INTERP_RESOLUTION_MIN:
        return CLUSTER_MAP_INTERP_RESOLUTION_MIN
    if resolution > CLUSTER_MAP_INTERP_RESOLUTION_MAX:
        return CLUSTER_MAP_INTERP_RESOLUTION_MAX
    return resolution


def _get_ui_control_by_names(*control_names: str) -> Any:
    """
    Возвращает первый найденный UI-контрол по списку имен.
    """
    for name in control_names:
        control = getattr(ui, name, None)
        if control is not None:
            return control
    return None


def _normalize_smoothing_window(window: int) -> int:
    """
    Приводит окно сглаживания к валидному нечетному значению.
    """
    try:
        normalized = int(window)
    except (TypeError, ValueError):
        return 0
    if normalized <= 0:
        return 0
    if normalized < 3:
        normalized = 3
    if normalized % 2 == 0:
        normalized += 1
    return normalized


def _smooth_label_sequence(
        labels: list[int],
        *,
        method: str = "maj",
        window: int = 5,
        preserve_noise: bool = True
) -> list[int]:
    """
    Сглаживает последовательность меток кластера в 1D окне.

    method:
      - maj: мажоритарное голосование (mode)
      - med: медиана по окну
    """
    norm_window = _normalize_smoothing_window(window)
    if norm_window <= 0 or len(labels) <= 1:
        return list(labels)

    radius = norm_window // 2
    method_norm = (method or "maj").strip().lower()
    smoothed = list(labels)

    for idx, center_label in enumerate(labels):
        if preserve_noise and int(center_label) == -1:
            continue

        left = max(0, idx - radius)
        right = min(len(labels), idx + radius + 1)
        neighborhood = [int(v) for v in labels[left:right]]
        if not neighborhood:
            continue

        if preserve_noise:
            neighborhood_wo_noise = [v for v in neighborhood if v != -1]
            if neighborhood_wo_noise:
                neighborhood = neighborhood_wo_noise

        if method_norm == "med":
            target_label = int(np.median(np.array(neighborhood, dtype=float)))
        else:
            target_label = int(Counter(neighborhood).most_common(1)[0][0])
        smoothed[idx] = target_label

    return smoothed


def _smooth_labels_by_profile_trace(
        labels: list[int],
        profile_trace_rows: dict[int, dict[int, int]],
        *,
        method: str = "maj",
        window: int = 5,
        preserve_noise: bool = True
) -> tuple[list[int], int]:
    """
    Сглаживает метки отдельно по каждому профилю и только внутри непрерывных
    сегментов trace_index (без «перетекания» через разрывы).
    """
    smoothed_labels = list(labels)
    changes = 0

    for trace_rows in profile_trace_rows.values():
        if not trace_rows:
            continue
        sorted_trace_row = sorted((int(trace), int(row_idx)) for trace, row_idx in trace_rows.items())
        segment: list[tuple[int, int]] = []

        def flush_segment(items: list[tuple[int, int]]) -> None:
            nonlocal changes
            if not items:
                return
            rows = [row_idx for _, row_idx in items]
            original = [int(labels[row_idx]) for row_idx in rows]
            smoothed_segment = _smooth_label_sequence(
                original,
                method=method,
                window=window,
                preserve_noise=preserve_noise
            )
            for row_idx, old_label, new_label in zip(rows, original, smoothed_segment):
                smoothed_labels[row_idx] = int(new_label)
                if int(old_label) != int(new_label):
                    changes += 1

        for trace_idx, row_idx in sorted_trace_row:
            if not segment:
                segment = [(trace_idx, row_idx)]
                continue
            prev_trace_idx = segment[-1][0]
            if trace_idx == prev_trace_idx + 1:
                segment.append((trace_idx, row_idx))
            else:
                flush_segment(segment)
                segment = [(trace_idx, row_idx)]
        flush_segment(segment)

    return smoothed_labels, changes


