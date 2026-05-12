"""Runtime tooltip overrides for Cluster UI.

Keep this out of generated Qt files so changes survive re-generation from Qt Designer.
"""


def apply_cluster_tooltips(ui) -> None:
    """Apply Russian explanatory tooltips for Cluster tab controls."""
    tips = {
        "groupBox_4": "Выбор алгоритма и его базовых параметров для разбиения объектов на кластеры.",
        "radioButton_clust_kmean": "Алгоритм KMeans: разбивает данные на заранее заданное число кластеров.",
        "radioButton_clust_hdbscan": "Алгоритм HDBSCAN: находит кластеры разной плотности и может выделять шум.",
        "radioButton_clust_gaussmix": "Gaussian Mixture: вероятностная кластеризация через смесь нормальных распределений.",
        "groupBox_6": "Метрики, по которым оценивается качество полученной кластеризации.",
        "checkBox_cluster_silhoutte": "Silhouette: показывает, насколько хорошо объекты отделены от соседних кластеров (больше — лучше).",
        "checkBox_cluster_dav_boul": "Davies-Bouldin: отношение компактности к разделимости кластеров (меньше — лучше).",
        "checkBox_cluster_calin_har": "Calinski-Harabasz: чем выше значение, тем лучше разделены кластеры.",
        "checkBox_cluster_ari_nmi": "ARI/NMI: внешние метрики согласованности с эталонной разметкой (если она доступна).",
        "pushButton_clust_save_conf": "Сохранить текущую конфигурацию параметров кластеризации.",
        "pushButton_clust_prev_prof": "Показать предыдущий профиль результата кластеризации.",
        "pushButton_clust_next_prof": "Показать следующий профиль результата кластеризации.",
        "pushButton_clust_calc": "Запустить расчёт кластеризации с текущими настройками.",
        "checkBox_clust_interp": "Интерполировать карту кластеров для более плавного отображения.",
        "pushButton_clust_auto": "Автоматический подбор параметров кластеризации по выбранным ограничениям.",
        "groupBox": "Управление набором объектов и признаков, используемых для кластеризации.",
        "pushButton_clust_add_cls": "Добавить набор признаков/классов в текущий объект кластеризации.",
        "pushButton_clust_collect_obj": "Собрать данные из выбранных источников в единый набор для кластеризации.",
        "pushButton_clust_add_reg": "Добавить признаки из регрессионного модуля в набор кластеризации.",
        "pushButton_clust_rm_obj": "Удалить выбранный элемент из набора данных кластеризации.",
        "pushButton_clust_add_well": "Добавить скважинные данные в текущий набор кластеризации.",
        "pushButton_clust_del_set": "Удалить выбранный ObjectSet кластеризации.",
        "groupBox_7": "Ограничения и режимы для автоматического подбора параметров (AUTO).",
        "checkBox_cluster_auto_candidate_time": "Ограничить время расчёта одного кандидата (в секундах).",
        "checkBox_cluster_auto_total_time": "Ограничить суммарное время AUTO-поиска (в секундах).",
        "radioButton_cluster_fine_auto": "Подробный поиск: больше кандидатов, выше шанс найти лучшее решение, но дольше расчёт.",
        "checkBox_cluster_auto_scaler_only": "Перебирать только варианты предобработки со скейлерами.",
        "radioButton_cluster_coarse_auto": "Быстрый грубый поиск: меньше кандидатов и быстрее расчёт.",
        "checkBox_cluster_auto_pca_only": "Ограничить подбор конфигурациями, использующими PCA.",
        "label_46": "Минимально допустимое количество объектов в кластере для AUTO-валидации кандидатов.",
        "toolButton_cluster_auto_min_n_reset": "Сбросить минимальный размер кластера к рекомендуемым 5% от объёма выборки.",
    }

    for name, tip in tips.items():
        widget = getattr(ui, name, None)
        if widget is not None:
            widget.setToolTip(tip)

    if hasattr(ui, "label_46"):
        ui.label_46.setText("Min cluster size")
