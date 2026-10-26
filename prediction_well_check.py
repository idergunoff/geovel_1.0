"""Sample saved predictions at wells using the application's proximity search."""
import json
import math
from dataclasses import dataclass, field

from models_db.model import Profile, ProfileModelPrediction, Research


@dataclass
class PredictionWellSample:
    well: object
    profile: object
    prediction_id: int
    trace_index: int
    distance: float
    value: float
    corrected: bool


@dataclass
class PredictionWellReport:
    samples: list = field(default_factory=list)
    issues: list[str] = field(default_factory=list)
    profiles_without_prediction: int = 0


def collect_prediction_wells(session, object_id, selected, nearest_wells, use_corrected=False):
    """Keep each well/profile/prediction pair; never run inference or change data.

    ``nearest_wells(profile_id)`` is the shared radius search, returning
    (well, closest trace index, distance). Arrays must match profile geometry:
    truncating or stretching them could silently compare the wrong locations.
    """
    report = PredictionWellReport()
    profiles = (session.query(Profile).join(Research)
                .filter(Research.object_id == object_id).order_by(Profile.id).all())
    for profile in profiles:
        predictions = (session.query(ProfileModelPrediction).filter_by(
            profile_id=profile.id, model_id=selected.model_id,
            type_model=selected.type_model).order_by(ProfileModelPrediction.id).all())
        if not predictions:
            report.profiles_without_prediction += 1
            continue
        try:
            x, y = json.loads(profile.x_pulc or '[]'), json.loads(profile.y_pulc or '[]')
            if not isinstance(x, list) or not isinstance(y, list) or not x or len(x) != len(y):
                raise ValueError()
            if not all(math.isfinite(float(v)) for v in x + y):
                raise ValueError()
        except (TypeError, ValueError, OverflowError):
            report.issues.append(f'{profile.title} (id{profile.id}): нет корректных координат профиля')
            continue
        wells = nearest_wells(profile.id) or []
        for prediction in predictions:
            corrected = bool(use_corrected and prediction.corrected)
            raw = prediction.corrected[0].correct if corrected else prediction.prediction
            try:
                values = json.loads(raw)
                if not isinstance(values, list) or len(values) != len(x):
                    raise ValueError()
            except (TypeError, ValueError):
                report.issues.append(f'{profile.title}, прогноз id{prediction.id}: '
                                     'массив прогноза повреждён или не совпадает с длиной профиля')
                continue
            for well, index, distance in wells:
                try:
                    if index < 0 or index >= len(values) or isinstance(values[index], bool):
                        raise ValueError()
                    value = float(values[index])
                    if not math.isfinite(value):
                        raise ValueError()
                except (TypeError, ValueError, OverflowError):
                    report.issues.append(f'{profile.title}, прогноз id{prediction.id}, '
                                         f'{well.name}: нет числового значения прогноза')
                    continue
                report.samples.append(PredictionWellSample(
                    well, profile, prediction.id, index, float(distance), value, corrected))
    return report
