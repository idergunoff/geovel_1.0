# Валидация качества AUTO-подбора (этап 8)

В модуле `cluster.py` добавлены функции:

- `validate_auto_tuning_run(...)`
- `validate_auto_tuning_quality(...)`

Они закрывают проверки этапа 8 из `cluster_auto_tuning_plan.md`:

1. **COARSE**: проверка, что запуск возвращает непустой leaderboard.
2. **FINE**: проверка, что есть и coarse-часть, и fine-часть.
3. **Стабильность**: оценка близости top-N между повторными прогонами
   (через пересечение сигнатур конфигураций).
4. **Граничные кейсы**: должны подаваться во входные `coarse_result`/`fine_result`
   и/или `repeated_results` из отдельных запусков на соответствующих выборках.

## Минимальный шаблон использования

```python
coarse = run_auto_cluster_tuning(base_data, auto_mode="COARSE")
fine = run_auto_cluster_tuning(base_data, auto_mode="FINE")
repeats = [run_auto_cluster_tuning(base_data, auto_mode="COARSE") for _ in range(3)]

report = validate_auto_tuning_quality(
    coarse_result=coarse,
    fine_result=fine,
    repeated_results=repeats,
    stability_top_n=5,
)
print(report)
```

## Интерпретация stability

- `min_overlap >= 0.60` — проверка пройдена.
- `min_overlap < 0.60` — ранжирование нестабильно (нужно смотреть `random_state`
  и/или сужать search-space).
