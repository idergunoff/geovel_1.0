# RFC: Well Log Feature Constructor Stage 1

## Назначение

Конструктор расчётных признаков Well Log — отдельное окно, открываемое из вкладки **Cluster analysis → Well Log** по кнопке **CONSTR**. На этапе 1 окно является безопасным техническим каркасом: оно показывает контекст текущего `well_log_cluster_dataset` и глобальную библиотеку `feature_calculator`, но не выполняет расчёты, не сохраняет новые признаки и не меняет состав dataset.

## UI-точка входа

- Object name кнопки: `pushButton_cluster_well_log_constructor`.
- Нажатие без выбранного Well Log dataset показывает предупреждение.
- Нажатие с выбранным dataset открывает `WellLogFeatureConstructorDialog`.
- Открытие формы не запускает `COLLECT` и не меняет связи `cluster_well_log_parameter` / `cluster_well_log_parameter_from_calculator`.

## Контекст, передаваемый в форму

Форма получает снимок текущего состояния:

1. `dataset_id` и имя dataset;
2. список скважин dataset с интервалами `top_md` / `bottom_md`;
3. список выбранных canonical-параметров;
4. список уже привязанных calculator-параметров;
5. глобальный список записей `feature_calculator`.

Если в dataset нет скважин или параметров, форма всё равно может открыться, но кнопка `Add to dataset` на этапе 1 отключена.

## Версия `params_json`

Поддерживаемая версия конфигурации для MVP: `version = 1`.

### Operation mode

```json
{
  "version": 1,
  "mode": "operation",
  "operation": "zscore",
  "inputs": [
    {
      "canonical_id": 12,
      "canonical_name": "GR"
    }
  ],
  "normalization_scope": "whole_well",
  "outlier_policy": "none"
}
```

### Formula mode

```json
{
  "version": 1,
  "mode": "formula",
  "expression": "(GR - RHOB) / (GR + RHOB)",
  "inputs": [
    {
      "canonical_id": 12,
      "canonical_name": "GR"
    },
    {
      "canonical_id": 17,
      "canonical_name": "RHOB"
    }
  ],
  "depth_grid_policy": "strict_equal",
  "invalid_math_policy": "block",
  "outlier_policy": "none"
}
```

## Правила MVP, закреплённые в каркасе backend

- `depth_grid_policy`: только `strict_equal` — интерполяция не используется.
- `invalid_math_policy`: только `block` — некорректная математика должна блокировать признак на следующих этапах.
- `outlier_policy`: только `none` — коррекция выбросов в MVP не выполняется.
- `normalization_scope`: значение по умолчанию `whole_well`; поддержка `interval` добавляется на backend-этапах.
- Функции backend не должны падать неожиданно: ошибки возвращаются как `FeatureCalculatorError`.
- Расчёт на этапе 1 возвращает контролируемый `not_implemented`.

## Публичный backend-каркас

Модуль: `cluster/well_feature_calculator.py`.

Публичные функции:

- `parse_feature_calculator_config(calculator)`;
- `validate_feature_calculator_config(config)`;
- `validate_feature_calculator_for_dataset(dataset_id, calculator_id)`;
- `evaluate_feature_calculator_for_well(config, well_id, top_md, bottom_md)`.

Полноценные операции, parser формул, preview, сохранение новых признаков и интеграция в `COLLECT` не входят в этап 1.
