"""Shared read-only presentation of saved core descriptions."""

from html import escape

from sqlalchemy.orm import joinedload, selectinload

from models_db.model import CoreDescriptionDocument, CoreDescriptionInterval
from core_description.terminology import ROCK_LABELS, SATURATION_LABELS, display_term
from well_selection import CORE_MARKER


CORE_COLOR = "#C9F2F2"


def core_well_ids(session):
    """Read availability without loading documents or interval descriptions."""
    return {well_id for (well_id,) in session.query(CoreDescriptionDocument.well_id).distinct()}


def load_core_intervals(session, well_id):
    return (session.query(CoreDescriptionInterval)
            .options(selectinload(CoreDescriptionInterval.rocks),
                     joinedload(CoreDescriptionInterval.document))
            .filter_by(well_id=well_id)
            .order_by(CoreDescriptionInterval.top_depth, CoreDescriptionInterval.bottom_depth,
                      CoreDescriptionInterval.document_id, CoreDescriptionInterval.id).all())


def number_text(value):
    return "—" if value is None else format(value, ".10g")


def primary_rock_text(interval):
    names = dict.fromkeys(display_term(rock.rock_name, ROCK_LABELS)
                          for rock in sorted(interval.rocks, key=lambda rock: rock.id or 0)
                          if rock.role == "primary")
    return ", ".join(names) or "Не определена"


def saturation_text(value):
    return display_term(value or "unknown", SATURATION_LABELS)


def core_summary_html(intervals):
    parts = ['<h3>Описание керна</h3>',
             '<p><i>Индекс — глубина, м — мощность, м — основная порода — нефтенасыщенность</i></p>']
    for interval in intervals:
        values = (interval.geological_index_raw or "—",
                  f'{number_text(interval.top_depth)}–{number_text(interval.bottom_depth)}',
                  number_text(interval.thickness), primary_rock_text(interval),
                  saturation_text(interval.oil_saturation))
        parts.append('<p>' + ' — '.join(escape(value) for value in values) + '</p>')
    return ''.join(parts)
