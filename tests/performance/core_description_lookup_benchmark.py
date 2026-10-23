"""Measure core-import preview lookup on a synthetic in-memory SQLite database."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from time import perf_counter

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from PyQt5 import QtWidgets  # noqa: E402
from sqlalchemy import create_engine, event  # noqa: E402
from sqlalchemy.orm import Session  # noqa: E402

from models_db.model import Well, WellOptionally  # noqa: E402
import models_db.model_cluster  # noqa: E402,F401
import models_db.model_profile_features  # noqa: E402,F401
from core_description.models import ParsedCoreDocument  # noqa: E402
from qt.core_description_import import CoreDescriptionImportDialog  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wells', type=int, default=10000)
    args = parser.parse_args()
    if args.wells < 1:
        parser.error('--wells must be positive')
    engine = create_engine('sqlite:///:memory:')
    Well.__table__.create(engine)
    WellOptionally.__table__.create(engine)
    attributes = [('Площадь', 'Северная'), ('Начало бурения', '01.02.1980'),
                  ('Окончание бурения', '01.03.1980'), ('Альтитуда устья', '120'),
                  ('Глубина', '500'), ('Регион', 'Татарстан'),
                  ('Категория', 'Поисковая'), ('Состояние', 'Ликвидирована')]
    with engine.begin() as connection:
        connection.execute(Well.__table__.insert(), [
            dict(id=i, name=str(i), alt=120.) for i in range(1, args.wells + 1)])
        connection.execute(WellOptionally.__table__.insert(), [
            dict(well_id=i, option=label, value=value)
            for i in range(1, args.wells + 1) for label, value in attributes])

    queries = []
    event.listen(engine, 'before_cursor_execute', lambda c, cur, sql, *rest: queries.append(sql))
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    with Session(engine) as session:
        started = perf_counter()
        dialog = CoreDescriptionImportDialog(session)
        opening_seconds = perf_counter() - started
        opening_queries = len(queries)
        queries.clear()
        document = ParsedCoreDocument('core.docx', 'docx', 'a' * 64,
                                      '12', '12', 'Северная', 'Северная', None, None)
        started = perf_counter()
        dialog.set_document(document)
        first_seconds = perf_counter() - started
        first_queries = len(queries)
        queries.clear()
        document.source_path = 'other.docx'
        document.well_name = '13'
        started = perf_counter()
        dialog.set_document(document)
        second_seconds = perf_counter() - started
        print(json.dumps(dict(
            wells=args.wells, attributes=args.wells * len(attributes),
            opening_seconds=round(opening_seconds, 4), opening_queries=opening_queries,
            first_seconds=round(first_seconds, 4), first_queries=first_queries,
            second_seconds=round(second_seconds, 4), second_queries=len(queries))))
        dialog.close()
    engine.dispose()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
