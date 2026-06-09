from __future__ import annotations

from .common import *
from .models import AUTO_SILHOUETTE_MAX_SAMPLES

def cluster_data(
        data,
        method="kmeans",  # "kmeans" | "hdbscan" | "gmm"

        # KMeans
        kmeans_n_clusters=4,
        kmeans_n_init=10,
        kmeans_random_state=42,

        # HDBSCAN
        hdbscan_min_cluster_size=30,
        hdbscan_min_samples=5,
        hdbscan_metric="euclidean",
        hdbscan_cluster_selection_method="eom",

        # GMM
        gmm_n_components=4,
        gmm_covariance_type="full",
        gmm_random_state=42,
        gmm_reg_covar=1e-6,
        gmm_max_iter=200
):
    """
    Кластеризация данных.

    Parameters
    ----------
    data : list[list] | np.ndarray
        Таблица признаков

    method : str
        "kmeans", "hdbscan", "gmm"

    Returns
    -------
    labels_list : list[int]
        Метка кластера для каждой строки

    cluster_info : dict
        Сводка для GUI
    """

    X = np.array(data, dtype=float)

    if method == "kmeans":
        from sklearn.cluster import KMeans

        model = KMeans(
            n_clusters=int(kmeans_n_clusters),
            n_init=kmeans_n_init,
            random_state=kmeans_random_state
        )

        labels = model.fit_predict(X)

        n_clusters = len(set(labels))
        noise_points = 0
        noise_fraction = 0.0

        cluster_info = {
            "method": "KMeans",
            "n_clusters": int(n_clusters),
            "noise_points": int(noise_points),
            "noise_fraction": float(noise_fraction),
            "inertia": float(model.inertia_)
        }

    elif method == "hdbscan":

        model = hdbscan.HDBSCAN(
            min_cluster_size=int(hdbscan_min_cluster_size),
            min_samples=int(hdbscan_min_samples),
            metric=hdbscan_metric,
            cluster_selection_method=hdbscan_cluster_selection_method
        )

        labels = model.fit_predict(X)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        noise_points = int(np.sum(labels == -1))
        noise_fraction = float(noise_points / len(labels)) if len(labels) > 0 else 0.0

        cluster_info = {
            "method": "HDBSCAN",
            "n_clusters": int(n_clusters),
            "noise_points": int(noise_points),
            "noise_fraction": float(noise_fraction)
        }

    elif method == "gmm":
        from sklearn.mixture import GaussianMixture

        model = GaussianMixture(
            n_components=int(gmm_n_components),
            covariance_type=gmm_covariance_type,
            random_state=gmm_random_state,
            reg_covar=gmm_reg_covar,
            max_iter=gmm_max_iter
        )

        model.fit(X)
        labels = model.predict(X)

        n_clusters = len(set(labels))
        noise_points = 0
        noise_fraction = 0.0

        cluster_info = {
            "method": "GaussianMixture",
            "n_clusters": int(n_clusters),
            "noise_points": int(noise_points),
            "noise_fraction": float(noise_fraction),
            "bic": float(model.bic(X)),
            "aic": float(model.aic(X))
        }

    else:
        raise ValueError("method must be 'kmeans', 'hdbscan' or 'gmm'")

    return labels.tolist(), cluster_info


def plot_cluster_map(
        label_list,
        data,
        figsize=(12, 8),
        point_size=20,
        interpolation_resolution=200,
        title="Cluster map",
        noise_color="gray",
        noise_marker=".",
        noise_label="noise",
        legend=True,
        show_interpolation=True,
        settings_caption: str | None = None
):
    """
    Визуализация кластеров на карте (без подписей профилей).

    Parameters
    ----------
    label_list : list[int]
        Метки кластеров.

    data : list[list]
        Исходные данные:
        data[i][1] - x
        data[i][2] - y

    figsize : tuple
        Размер графика.

    point_size : int
        Размер точек.

    interpolation_resolution : int
        Разрешение регулярной сетки для фоновой интерполяции.
        Чем больше значение, тем более детальная (и более тяжелая по времени)
        отрисовка. Рабочий диапазон: 50..500.

    noise_color : str
        Цвет шума (label = -1).

    noise_marker : str
        Маркер шума.

    legend : bool
        Показывать легенду.

    show_interpolation : bool
        Если True — строится фоновая интерполяция по всей области участка.
        Если False — рисуются только исходные точки.

    settings_caption : str | None
        Дополнительная подпись с параметрами расчета (scaler/PCA/метод и т.д.).
    """

    labels = np.asarray(label_list)
    arr = np.asarray(data, dtype=float)

    if arr.ndim != 2 or arr.shape[1] < 3:
        raise ValueError("data must have at least 3 columns")

    if len(labels) != len(arr):
        raise ValueError("label_list length must match data")

    x = arr[:, 1]
    y = arr[:, 2]

    # Явно создаем новую фигуру/ось на каждый вызов, чтобы исключить
    # повторное использование текущей активной оси matplotlib.
    fig, ax = plt.subplots(figsize=figsize)

    unique_labels = np.unique(labels)
    unique_labels_sorted = sorted([int(v) for v in unique_labels])

    if show_interpolation and len(unique_labels_sorted) > 1 and len(x) >= 3:
        min_x, max_x = float(np.min(x)), float(np.max(x))
        min_y, max_y = float(np.min(y)), float(np.max(y))
        span_x = max(max_x - min_x, 1e-9)
        span_y = max(max_y - min_y, 1e-9)
        pad_x = span_x * 0.03
        pad_y = span_y * 0.03
        min_x -= pad_x
        max_x += pad_x
        min_y -= pad_y
        max_y += pad_y

        grid_n = _normalize_interpolation_resolution(interpolation_resolution)
        grid_x, grid_y = np.meshgrid(
            np.linspace(min_x, max_x, grid_n),
            np.linspace(min_y, max_y, grid_n)
        )

        points = np.column_stack((x, y))
        grid_z = griddata(points, labels.astype(float), (grid_x, grid_y), method="nearest")

        if grid_z is not None and np.any(~np.isnan(grid_z)):
            color_list = [get_cluster_color(label) for label in unique_labels_sorted]
            cmap = ListedColormap(color_list)
            boundaries = [unique_labels_sorted[0] - 0.5]
            boundaries.extend([label + 0.5 for label in unique_labels_sorted])
            norm = BoundaryNorm(boundaries, cmap.N, clip=True)

            ax.pcolormesh(
                grid_x,
                grid_y,
                grid_z,
                shading="auto",
                cmap=cmap,
                norm=norm,
                alpha=0.35,
                zorder=1
            )

    # кластеры
    for label in unique_labels:
        if label == -1:
            continue

        mask = labels == label
        cluster_color = get_cluster_color(label)

        ax.scatter(
            x[mask],
            y[mask],
            s=point_size,
            label=f"cluster {int(label)}",
            c=cluster_color,
            edgecolors=cluster_color,
            alpha=0.95,
            zorder=3
        )

    # шум
    if -1 in unique_labels:
        mask = labels == -1
        noise_cluster_color = get_cluster_color(-1)
        ax.scatter(
            x[mask],
            y[mask],
            s=point_size,
            c=noise_cluster_color if noise_color == "gray" else noise_color,
            marker=noise_marker,
            label=noise_label,
            edgecolors=noise_cluster_color if noise_color == "gray" else noise_color,
            alpha=0.9,
            zorder=4
        )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(title)
    ax.axis("equal")
    ax.grid(True, alpha=0.3)

    if legend:
        ax.legend()

    if settings_caption:
        fig.text(
            0.01,
            0.01,
            settings_caption,
            ha="left",
            va="bottom",
            fontsize=8,
            color="dimgray"
        )
        fig.tight_layout(rect=(0, 0.05, 1, 1))
    else:
        fig.tight_layout()
    plt.show()


def show_cluster_diagnostics(
        data_for_clustering,
        labels,
        method_name: str,
        model=None,
        cached_image_base64: str | None = None
) -> str | None:
    """
    Показывает набор диагностических графиков на одном листе:
    PCA 2D/3D, t-SNE 2D/3D, матрица расстояний, silhouette.
    """
    from sklearn.decomposition import PCA
    from sklearn.metrics import pairwise_distances, silhouette_samples
    from sklearn.manifold import TSNE
    import io

    if cached_image_base64:
        try:
            image_bytes = base64.b64decode(cached_image_base64.encode("ascii"))
            image = plt.imread(io.BytesIO(image_bytes), format="png")
            fig = plt.figure(figsize=(20, 12))
            ax = fig.add_subplot(1, 1, 1)
            ax.imshow(image)
            ax.axis("off")
            fig.tight_layout()
            plt.show()
            return cached_image_base64
        except Exception:
            pass

    X = np.asarray(data_for_clustering, dtype=float)
    y = np.asarray(labels, dtype=int)
    if len(X) == 0 or len(X) != len(y):
        return

    if min(X.shape[1], len(X)) < 2:
        return
    uniq_lbl = [v for v in sorted(np.unique(y)) if v != -1]
    centroids = np.array([X[y == lbl].mean(axis=0) for lbl in uniq_lbl]) if uniq_lbl else np.empty((0, X.shape[1]))
    has_centroids = len(centroids) > 0

    pca_2d_model = PCA(n_components=2).fit(X)
    pca_2d = pca_2d_model.transform(X)
    cent_pca_2d = pca_2d_model.transform(centroids) if has_centroids else np.empty((0, 2))

    pca_3d = None
    cent_pca_3d = None
    if min(3, X.shape[1], len(X)) >= 3:
        pca_3d_model = PCA(n_components=3).fit(X)
        pca_3d = pca_3d_model.transform(X)
        cent_pca_3d = pca_3d_model.transform(centroids) if has_centroids else np.empty((0, 3))

    perplexity = max(5, min(30, len(X) - 1))
    tsne_2d_model = TSNE(n_components=2, random_state=42, init="pca", learning_rate="auto", perplexity=perplexity)
    if has_centroids:
        comb2 = np.vstack([X, centroids])
        emb2 = tsne_2d_model.fit_transform(comb2)
        tsne_2d, cent_tsne_2d = emb2[:len(X)], emb2[len(X):]
    else:
        tsne_2d = tsne_2d_model.fit_transform(X)
        cent_tsne_2d = np.empty((0, 2))

    tsne_3d = None
    cent_tsne_3d = None
    if min(3, X.shape[1], len(X)) >= 3:
        tsne_3d_model = TSNE(n_components=3, random_state=42, init="pca", learning_rate="auto", perplexity=perplexity)
        if has_centroids:
            comb3 = np.vstack([X, centroids])
            emb3 = tsne_3d_model.fit_transform(comb3)
            tsne_3d, cent_tsne_3d = emb3[:len(X)], emb3[len(X):]
        else:
            tsne_3d = tsne_3d_model.fit_transform(X)
            cent_tsne_3d = np.empty((0, 3))

    fig = plt.figure(figsize=(20, 12))
    # Порядок графиков:
    # [1] PCA 2D, [2] t-SNE 2D, [3] Silhouette
    # [4] PCA 3D, [5] t-SNE 3D, [6] Distance matrix
    ax_pca2 = fig.add_subplot(2, 3, 1)
    ax_tsne2 = fig.add_subplot(2, 3, 2)
    ax_sil = fig.add_subplot(2, 3, 3)
    ax_pca3 = fig.add_subplot(2, 3, 4, projection="3d")
    ax_tsne3 = fig.add_subplot(2, 3, 5, projection="3d")
    ax_dm = fig.add_subplot(2, 3, 6)

    def _plot_2d(ax, pts, title, centroids_2d=None):
        for lbl in sorted(np.unique(y)):
            m = y == lbl
            color = "gray" if lbl == -1 else get_cluster_color(int(lbl))
            ax.scatter(pts[m, 0], pts[m, 1], s=14, c=color, alpha=0.8, label=f"cluster {lbl}")
        if centroids_2d is not None and len(centroids_2d) > 0:
            ax.scatter(centroids_2d[:, 0], centroids_2d[:, 1], marker="X", c="black", s=90, label="centroids")
        ax.set_title(title)
        ax.grid(alpha=0.2)

    _plot_2d(ax_pca2, pca_2d, f"{method_name.upper()} • PCA 2D", cent_pca_2d)
    ax_pca2.set_xlabel("PC1")
    ax_pca2.set_ylabel("PC2")
    ax_pca2.legend(loc="best", fontsize=7)

    if pca_3d is not None:
        for lbl in sorted(np.unique(y)):
            m = y == lbl
            color = "gray" if lbl == -1 else get_cluster_color(int(lbl))
            ax_pca3.scatter(pca_3d[m, 0], pca_3d[m, 1], pca_3d[m, 2], s=12, c=color, alpha=0.75)
        if cent_pca_3d is not None and len(cent_pca_3d) > 0:
            ax_pca3.scatter(cent_pca_3d[:, 0], cent_pca_3d[:, 1], cent_pca_3d[:, 2], marker="X", c="black", s=90)
        ax_pca3.set_title("PCA 3D")
    else:
        ax_pca3.set_title("PCA 3D unavailable")

    _plot_2d(ax_tsne2, tsne_2d, "t-SNE 2D", cent_tsne_2d)
    ax_tsne2.legend(loc="best", fontsize=7)

    if tsne_3d is not None:
        for lbl in sorted(np.unique(y)):
            m = y == lbl
            color = "gray" if lbl == -1 else get_cluster_color(int(lbl))
            ax_tsne3.scatter(tsne_3d[m, 0], tsne_3d[m, 1], tsne_3d[m, 2], s=12, c=color, alpha=0.75)
        if cent_tsne_3d is not None and len(cent_tsne_3d) > 0:
            ax_tsne3.scatter(cent_tsne_3d[:, 0], cent_tsne_3d[:, 1], cent_tsne_3d[:, 2], marker="X", c="black", s=90)
        ax_tsne3.set_title("t-SNE 3D")
    else:
        ax_tsne3.set_title("t-SNE 3D unavailable")

    order = np.argsort(y)
    dist_mx = pairwise_distances(X[order], metric="euclidean")
    im = ax_dm.imshow(dist_mx, cmap="turbo", aspect="auto")
    ax_dm.set_title("Distance matrix (ordered)")
    fig.colorbar(im, ax=ax_dm, shrink=0.8)

    mask_valid = y != -1
    y_eval = y[mask_valid]
    X_eval = X[mask_valid]
    if len(X_eval) > 2 and len(np.unique(y_eval)) > 1:
        sil_values = silhouette_samples(X_eval, y_eval)
        y_lower = 10
        for lbl in sorted(np.unique(y_eval)):
            vals = np.sort(sil_values[y_eval == lbl])
            y_upper = y_lower + len(vals)
            ax_sil.fill_betweenx(np.arange(y_lower, y_upper), 0, vals, alpha=0.7, color=get_cluster_color(int(lbl)))
            y_lower = y_upper + 10
        sil_avg = float(np.mean(sil_values))
        ax_sil.axvline(sil_avg, color="red", linestyle="--", linewidth=1.2, label=f"avg={sil_avg:.3f}")
        ax_sil.set_title("Silhouette plot")
        ax_sil.legend(loc="best", fontsize=8)
    else:
        ax_sil.set_title("Silhouette unavailable")

    fig.suptitle(f"Cluster diagnostics • {method_name.upper()}", fontsize=14)
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    image_buffer = io.BytesIO()
    fig.savefig(image_buffer, format="png", dpi=110, bbox_inches="tight")
    image_base64 = base64.b64encode(image_buffer.getvalue()).decode("ascii")
    plt.show()
    return image_base64

def evaluate_clustering(
        data,
        labels,
        use_silhouette=False,
        use_db=False,
        use_ch=False,
        max_silhouette_samples: int = AUTO_SILHOUETTE_MAX_SAMPLES
):
    """
    Оценка качества кластеризации без эталонной разметки.

    Parameters
    ----------
    data : list[list] | np.ndarray
        Матрица признаков, на которой выполнялась кластеризация.

    labels : list[int] | np.ndarray
        Метки кластеров.

    use_silhouette : bool
        Считать ли silhouette score.

    use_db : bool
        Считать ли Davies-Bouldin index.

    use_ch : bool
        Считать ли Calinski-Harabasz index.

    Returns
    -------
    dict
        {
            "metrics": {...},
            "interpretation": {...},
            "overall_score": int,
            "overall_label": str
        }
    """

    X = np.array(data, dtype=float)
    labels = np.array(labels)

    results = {
        "metrics": {},
        "interpretation": {},
        "overall_score": None,
        "overall_label": None
    }

    # Убираем шум HDBSCAN для внутренних метрик
    mask = labels != -1
    labels_eval = labels[mask]
    X_eval = X[mask]

    unique_clusters = np.unique(labels_eval)

    # Нельзя оценивать, если после удаления шума осталось < 2 кластеров
    if len(unique_clusters) < 2 or len(X_eval) < 2:
        results["overall_score"] = 0
        results["overall_label"] = "Недостаточно кластеров для оценки"
        return results

    scores_for_summary = []

    # --------------------------
    # Silhouette
    # --------------------------
    if use_silhouette:
        silhouette_sample_size = len(X_eval)
        if max_silhouette_samples is not None:
            try:
                max_silhouette_samples = max(2, int(max_silhouette_samples))
            except (TypeError, ValueError):
                max_silhouette_samples = AUTO_SILHOUETTE_MAX_SAMPLES
            silhouette_sample_size = min(silhouette_sample_size, int(max_silhouette_samples))
        val = float(
            silhouette_score(
                X_eval,
                labels_eval,
                sample_size=silhouette_sample_size if silhouette_sample_size < len(X_eval) else None,
                random_state=42
            )
        )
        results["metrics"]["silhouette"] = val
        results["metrics"]["silhouette_n_samples"] = int(silhouette_sample_size)

        if val > 0.5:
            label = "Хорошо"
            text = "Кластеры хорошо отделены друг от друга."
            score = 2
        elif val >= 0.2:
            label = "Нормально"
            text = "Структура кластеров присутствует, но разделение умеренное."
            score = 1
        else:
            label = "Слабо"
            text = "Кластеры плохо отделены или сильно пересекаются."
            score = 0

        results["interpretation"]["silhouette"] = {
            "label": label,
            "text": text
        }
        scores_for_summary.append(score)

    # --------------------------
    # Davies-Bouldin
    # --------------------------
    if use_db:
        val = float(davies_bouldin_score(X_eval, labels_eval))
        results["metrics"]["davies_bouldin"] = val

        if val < 1.0:
            label = "Хорошо"
            text = "Кластеры компактны и хорошо разделены."
            score = 2
        elif val <= 2.0:
            label = "Нормально"
            text = "Кластеры различимы, но разделение неидеально."
            score = 1
        else:
            label = "Слабо"
            text = "Кластеры плохо разделены или слишком растянуты."
            score = 0

        results["interpretation"]["davies_bouldin"] = {
            "label": label,
            "text": text
        }
        scores_for_summary.append(score)

    # --------------------------
    # Calinski-Harabasz
    # --------------------------
    if use_ch:
        val = float(calinski_harabasz_score(X_eval, labels_eval))
        results["metrics"]["calinski_harabasz"] = val

        label = "Справочно"
        text = "Чем больше значение, тем лучше разделение кластеров. Удобно для сравнения разных запусков."
        results["interpretation"]["calinski_harabasz"] = {
            "label": label,
            "text": text
        }

    # --------------------------
    # Общий вердикт
    # --------------------------
    if len(scores_for_summary) == 0:
        results["overall_score"] = None
        results["overall_label"] = "Метрики не выбраны"
        return results

    avg_score = sum(scores_for_summary) / len(scores_for_summary)

    if avg_score >= 1.5:
        overall = "Хорошее качество кластеризации"
    elif avg_score >= 0.75:
        overall = "Приемлемое качество кластеризации"
    else:
        overall = "Слабое качество кластеризации"

    results["overall_score"] = avg_score
    results["overall_label"] = overall

    return results


def build_clustering_report(
        preprocess_mode,
        pca_mode,
        pca_info,
        cluster_info,
        result_info,
        evaluation
):
    """
    Формирует HTML-строку для set_info.
    """

    lines = []

    # --------------------------
    # Настройки
    # --------------------------
    settings = []

    # preprocess
    settings.append(f"Preprocess: {preprocess_mode}")

    # PCA
    if pca_mode == "fixed_components":
        settings.append(f"PCA: {pca_info['components_after_pca']} components")
    elif pca_mode == "variance_ratio":
        settings.append(
            f"PCA: (var={pca_info['explained_variance']:.2f})"
        )
    else:
        settings.append(
            "PCA: off"
        )

    # clustering

    smoothing_desc = str(cluster_info.get("smoothing", "off"))
    settings.append(f"Smoothing: {smoothing_desc}")

    if cluster_info["method"] == "kmeans":
        settings.append(
            f"KMeans (k={cluster_info['kmeans_n']}, init={cluster_info['kmeans_n_init']})"
        )

    elif cluster_info["method"] == "hdbscan":
        settings.append(
            f"HDBSCAN (min size={cluster_info['min_size']}, min sample={cluster_info['min_sample']}, "
            f"type={cluster_info['hdbscan_type']})"
        )

    elif cluster_info["method"] == "gmm":
        settings.append(
            f"GMM (k={cluster_info['n']}, type={cluster_info['gmm_type']})"
        )

    lines.append(" | ".join(settings))
    lines.append("")  # пустая строка
    lines.append(f"Результат: {result_info}")
    lines.append("")  # пустая строка

    # --------------------------
    # Метрики
    # --------------------------
    metrics = evaluation["metrics"]
    interp = evaluation["interpretation"]

    if "silhouette" in metrics and metrics["silhouette"] is not None:
        lines.append(
            f"Silhouette: {metrics['silhouette']:.2f} — {interp['silhouette']['label']}"
        )

    if "davies_bouldin" in metrics and metrics["davies_bouldin"] is not None:
        lines.append(
            f"Davies-Bouldin: {metrics['davies_bouldin']:.2f} — {interp['davies_bouldin']['label']}"
        )

    if "calinski_harabasz" in metrics and metrics["calinski_harabasz"] is not None:
        lines.append(
            f"Calinski-Harabasz: {metrics['calinski_harabasz']:.1f} — {interp['calinski_harabasz']['label']}"
        )

    lines.append("")  # пустая строка

    # --------------------------
    # Итог
    # --------------------------
    lines.append(f"Итог: {evaluation['overall_label']}")

    # --------------------------
    # HTML
    # --------------------------
    return "<br>".join(lines)



def _read_manual_cluster_ui_config() -> dict[str, Any]:
    """
    Считывает текущие настройки ручного CALC из общей панели кластеризации.
    """
    if ui.radioButton_clust_scaler_none.isChecked():
        preprocess_mode = "none"
    elif ui.radioButton_clust_scaler_stnd.isChecked():
        preprocess_mode = "standard"
    elif ui.radioButton_clust_scaler_rob.isChecked():
        preprocess_mode = "robust"
    elif ui.radioButton_clust_scaler_l2.isChecked():
        preprocess_mode = "l2_norm"
    else:
        preprocess_mode = "row_center"

    if ui.radioButton_clust_kmean.isChecked():
        method = "kmeans"
    elif ui.radioButton_clust_hdbscan.isChecked():
        method = "hdbscan"
    elif ui.radioButton_clust_gaussmix.isChecked():
        method = "gmm"
    else:
        method = "kmeans"

    selected_button = ui.buttonGroup_3.checkedButton()
    text_method_nan = selected_button.text() if selected_button else "impute"
    pca_enabled = bool(ui.checkBox_cluster_pca.isChecked())
    pca_mode = "fixed_components" if ui.radioButton_clust_pca_fix.isChecked() else "variance_ratio"
    pca_value = ui.spinBox_clust_pca_fix.value() if pca_mode == "fixed_components" else ui.doubleSpinBox_clust_pca_disp.value()

    return {
        "clean": {
            "use_non_finite": bool(ui.checkBox_clust_clean_nan.isChecked()),
            "non_finite_mode": text_method_nan,
            "use_variance_threshold": bool(ui.checkBox_clust_clear_vartresh.isChecked()),
            "use_correlation_filter": bool(ui.checkBox_clust_clear_corr.isChecked()),
        },
        "preprocess_mode": preprocess_mode,
        "pca": {
            "enabled": pca_enabled,
            "mode": pca_mode if pca_enabled else None,
            "value": pca_value if pca_enabled else None,
            "fixed_components": ui.spinBox_clust_pca_fix.value(),
            "variance_ratio": ui.doubleSpinBox_clust_pca_disp.value(),
        },
        "method": method,
        "method_params": {
            "kmeans_n_clusters": ui.spinBox_clust_kmeans_n.value(),
            "kmeans_n_init": ui.spinBox_clust_kmean_ninint.value(),
            "hdbscan_min_cluster_size": ui.spinBox_clust_hdbsc_minsize.value(),
            "hdbscan_min_samples": ui.spinBox_clust_hdbsc_minsamp.value(),
            "hdbscan_metric": ui.comboBox_clust_hdbsc_type.currentText(),
            "gmm_n_components": ui.spinBox_clust_gmm_n.value(),
            "gmm_covariance_type": ui.comboBox_clust_gmm_type.currentText(),
        },
        "metrics": {
            "use_silhouette": bool(ui.checkBox_cluster_silhoutte.isChecked()),
            "use_db": bool(ui.checkBox_cluster_dav_boul.isChecked()),
            "use_ch": bool(ui.checkBox_cluster_calin_har.isChecked()),
        },
        "smoothing": {
            "enabled": bool(ui.checkBox_cluster_smooth.isChecked()),
            "method": "maj" if ui.radioButton_cluster_smooth_maj.isChecked() else "med",
            "window": ui.spinBox_cluster_smooth_window.value(),
        },
    }


