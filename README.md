# GeoVel

## Установка зависимостей

Рекомендуется **64-разрядный Python 3.12** с `pip`. Откройте терминал
в папке GeoVel (там, где находится `geovel.py`). Для скачивания пакетов
нужен интернет; установка PyTorch и библиотек машинного обучения может
занять несколько минут и потребовать несколько гигабайт свободного места.

### Windows (PowerShell или cmd)

```text
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m pip check
.venv\Scripts\python.exe geovel.py
```

### Linux / macOS

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip check
.venv/bin/python geovel.py
```

Активация окружения не требуется: команды явно используют Python из `.venv`.
Если виртуальное окружение уже настроено в IDE, достаточно выполнить в нём:

```sh
python -m pip install -r requirements.txt
```

Версии прямых зависимостей взяты из настроенного окружения GeoVel с Python 3.12.
Это не полный lock-файл: версии транзитивных зависимостей выбирает `pip`.
PyTorch включён для функций нейросетей; для ускорения на NVIDIA установите
подходящую сборку по инструкции https://pytorch.org/get-started/locally/.

## Системные компоненты и дополнительные возможности

- **Linux:** Qt требует графического сеанса и системных библиотек. Для Ubuntu/Debian
  при ошибках загрузки Qt/XCB установите `libgl1 libegl1 libxkbcommon-x11-0
  libxcb-xinerama0 libxcb-cursor0`; LightGBM/XGBoost могут потребовать `libgomp1`.
  Если создание окружения недоступно, установите пакет `venv` для вашей версии Python.
- **macOS:** для LightGBM может потребоваться OpenMP: `brew install libomp`.
- **MapInfo TAB:** дополнительно нужен **GDAL** с модулем Python `osgeo` и драйвером
  `MapInfo File`. Устанавливайте GDAL способом, подходящим вашей ОС и версии Python
  (например, через conda-forge в отдельном окружении). При установке через `pip`
  версия Python-пакета GDAL должна соответствовать установленной системной библиотеке
  GDAL. Поэтому GDAL не включён в общий `requirements.txt`. Экспорт MIF/MID его не требует.
- **Старые документы `.doc`:** установите LibreOffice; в Windows также поддерживается
  Microsoft Word. Для `.docx` внешний конвертер не нужен. Подробнее:
  [импорт описаний керна](docs/core_description_import_guide.md).
- Установка пакетов не переносит существующую базу данных, проекты и обученные модели.

## Для разработчиков

`requirements-dev.txt` дополнительно устанавливает pytest и python-docx
для тестов и генерации тестовых документов:

```sh
python -m pip install -r requirements-dev.txt
```

Используйте Python из выбранного виртуального окружения.
