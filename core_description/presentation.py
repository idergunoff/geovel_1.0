"""Shared read-only presentation of saved core descriptions."""

from html import escape

from sqlalchemy.orm import joinedload, selectinload

from models_db.model import CoreDescriptionDocument, CoreDescriptionInterval
from core_description.terminology import (
    ROCK_LABELS, SATURATION_LABELS, RELATION_LABELS, canonical_term, display_term,
)
from well_selection import CORE_MARKER


CORE_COLOR = "#C9F2F2"
OIL_LAYER_COLOR = "#F2DCC6"


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


def secondary_rock_text(interval):
    """Keep only non-primary mentions, with short meaningful roles."""
    names = []
    for rock in sorted(interval.rocks, key=lambda rock: rock.id or 0):
        if rock.role == "primary":
            continue
        name = display_term(rock.rock_name, ROCK_LABELS)
        if rock.role in {"interlayer", "interbedding", "inclusion"}:
            name += f' ({display_term(rock.role, RELATION_LABELS)})'
        if name not in names:
            names.append(name)
    return ", ".join(names)


def saturation_text(value, *, omit_unknown=False):
    code = canonical_term(value or "unknown", SATURATION_LABELS)
    if omit_unknown and code == "unknown":
        return ""
    return display_term(code, SATURATION_LABELS)


def has_oil(interval):
    """Any oil indication, including odor and uncertain or rock-only mentions."""
    values = [interval.oil_saturation] + [rock.oil_saturation for rock in interval.rocks]
    return any(canonical_term(value or "unknown", SATURATION_LABELS) not in {"unknown", "none"}
               for value in values)


def core_summary_html(intervals):
    parts = ['<h3>Описание керна</h3>',
             '<p><i>Индекс — кровля, м — мощность, м — основная порода — нефтенасыщенность</i></p>']
    for interval in intervals:
        values = [interval.geological_index_raw or "—", number_text(interval.top_depth),
                  number_text(interval.thickness), primary_rock_text(interval)]
        saturation = saturation_text(interval.oil_saturation, omit_unknown=True)
        if saturation:
            values.append(saturation)
        style = f' style="background-color:{OIL_LAYER_COLOR};color:#17212b"' if has_oil(interval) else ''
        parts.append(f'<p{style}>' + ' — '.join(escape(value) for value in values) + '</p>')
    return ''.join(parts)
