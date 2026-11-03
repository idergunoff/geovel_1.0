"""Terrain sources and derived values. Never changes saved training datasets.

Grid payloads are immutable XYZ nodes in the profile coordinate system (Pulkovo).
Local grids are opt-in; existing common grids retain their automatic selection.
"""
import hashlib
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree
from sqlalchemy import inspect

from models_db.model import CommonGrid, GeoradarObject, Research, Profile, Formation


class TerrainError(ValueError):
    pass


def ensure_terrain_schema(engine):
    """Additive startup compatibility for databases not yet upgraded by Alembic."""
    additions = {
        'georadar_object': {'relief_grid_id': 'INTEGER REFERENCES common_grid(id)'},
        'common_grid': {'comment': 'TEXT', 'source_name': 'VARCHAR', 'created_at': 'DATETIME',
                        'is_local': 'BOOLEAN NOT NULL DEFAULT 0'},
        'profile': {'relief_source_key': 'VARCHAR'},
        'formation': {'land_source_key': 'VARCHAR'},
        'profile_model_prediction': {'is_stale': 'BOOLEAN NOT NULL DEFAULT 0'},
    }
    with engine.begin() as connection:
        inspector = inspect(connection)
        for table, columns in additions.items():
            existing = {column['name'] for column in inspector.get_columns(table)}
            for name, sql_type in columns.items():
                if name not in existing:
                    connection.exec_driver_sql(f'ALTER TABLE {table} ADD COLUMN {name} {sql_type}')
        # Repair flags written by the initial terrain manager. Predictions and
        # their corrections remain saved results, independent of the terrain.
        connection.exec_driver_sql(
            'UPDATE profile_model_prediction SET is_stale = 0 WHERE is_stale <> 0')


class TerrainSurface:
    def __init__(self, nodes):
        try:
            data = np.asarray(nodes, dtype=float)
        except (TypeError, ValueError) as exc:
            raise TerrainError('Сетка должна содержать числовые столбцы X Y Z.') from exc
        if data.ndim != 2 or data.shape[1] != 3 or len(data) < 4:
            raise TerrainError('Нужны как минимум четыре узла регулярной сетки X Y Z.')
        if not np.isfinite(data[:, :2]).all():
            raise TerrainError('В координатах сетки есть пропуски или бесконечные значения.')
        self.x, self.y = np.unique(data[:, 0]), np.unique(data[:, 1])
        if min(len(self.x), len(self.y)) < 2:
            raise TerrainError('Сетка должна иметь минимум два узла по каждой оси.')
        for axis in (self.x, self.y):
            steps = np.diff(axis)
            if not np.allclose(steps / steps.min(), np.round(steps / steps.min()), rtol=1e-5, atol=1e-5):
                raise TerrainError('Узлы должны лежать на регулярной сетке.')
        self.steps = (float(np.diff(self.x).min()), float(np.diff(self.y).min()))
        # Keep missing nodes in axes, but exclude them from interpolation/coverage.
        valid = np.isfinite(data[:, 2]) & (np.abs(data[:, 2]) < 1.7e38) & (data[:, 2] != -9999)
        self.data = data[valid]
        if len(self.data) < 4:
            raise TerrainError('В сетке недостаточно узлов с корректной высотой.')
        ix = np.searchsorted(self.x, self.data[:, 0])
        iy = np.searchsorted(self.y, self.data[:, 1])
        self.keys = np.sort(ix * len(self.y) + iy)
        if len(np.unique(self.keys)) != len(self.keys):
            raise TerrainError('В сетке повторяются координаты узлов.')
        self.complete = (len(self.keys) == len(self.x) * len(self.y)
                         and all(np.allclose(np.diff(axis), step)
                                 for axis, step in zip((self.x, self.y), self.steps)))
        self.tree = cKDTree(self.data[:, :2])

    def covers(self, points):
        """Check the polyline, including cells between consecutive traces."""
        if not self._covers_points(points):
            return False
        if self.complete:
            return True
        points = np.asarray(points, dtype=float)
        for start, stop in zip(points[:-1], points[1:]):
            delta = stop - start
            crossings = [0., 1.]
            for axis, a, b, d in zip((self.x, self.y), start, stop, delta):
                if d == 0:
                    continue
                lo, hi = np.searchsorted(axis, [min(a, b), max(a, b)])
                crossings.extend(((axis[lo:hi] - a) / d).tolist())
            times = np.unique(crossings)
            midpoints = (times[1:] + times[:-1]) / 2
            if not self._covers_points(start + midpoints[:, None] * delta):
                return False
        return True

    def _covers_points(self, points):
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 2 or not len(points):
            return False
        if not np.isfinite(points).all():
            return False
        bounds = ((points[:, 0] >= self.x[0]) & (points[:, 0] <= self.x[-1]) &
                  (points[:, 1] >= self.y[0]) & (points[:, 1] <= self.y[-1]))
        if not bounds.all():
            return False
        indices = []
        for axis, coords, step in zip((self.x, self.y), points.T, self.steps):
            hi = np.minimum(np.searchsorted(axis, coords), len(axis) - 1)
            exact = np.isclose(axis[hi], coords, rtol=0, atol=max(1e-8, step * 1e-7))
            lo = np.where(exact, hi, np.maximum(hi - 1, 0))
            # A wholly missing row/column must not be bridged.
            if np.any((axis[hi] - axis[lo]) > step * 1.00001):
                return False
            indices.append((lo, hi))
        for ix in indices[0]:
            for iy in indices[1]:
                keys = ix * len(self.y) + iy
                found = np.searchsorted(self.keys, keys)
                if np.any(found >= len(self.keys)) or not np.all(self.keys[np.minimum(found, len(self.keys)-1)] == keys):
                    return False
        return True

    def sample(self, points):
        if not self.covers(points):
            raise TerrainError('Сетка не полностью покрывает профиль или содержит пропуски на его пути.')
        distances, indices = self.tree.query(points, k=4)
        exact = distances[:, 0] == 0
        result = np.empty(len(points), dtype=float)
        result[exact] = self.data[indices[exact, 0], 2]
        weights = 1 / np.maximum(distances[~exact], np.finfo(float).eps) ** 2
        result[~exact] = np.sum(weights * self.data[indices[~exact], 2], axis=1) / weights.sum(axis=1)
        return result.tolist()

    def metadata(self):
        return {'nodes': len(self.data), 'bounds': [float(self.x[0]), float(self.y[0]),
                float(self.x[-1]), float(self.y[-1])], 'step': list(self.steps)}


@lru_cache(maxsize=3)
def _surface(payload):
    try:
        return TerrainSurface(json.loads(payload))
    except (TypeError, json.JSONDecodeError) as exc:
        raise TerrainError('Не удалось прочитать данные сетки.') from exc


def surface(grid):
    return _surface(grid.grid_table)


def read_grid_file(filename):
    """XYZ .dat/.txt, optional header, whitespace or semicolon separated."""
    rows = []
    with open(filename, encoding='utf-8-sig') as source:
        for number, line in enumerate(source, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            fields = line.replace(';', ' ').split()
            try:
                if len(fields) != 3:
                    raise ValueError()
                row = [float(value.replace(',', '.')) for value in fields]
            except ValueError as exc:
                if not rows and all(value.lower() in ('x', 'y', 'z', 'value', 'height') for value in fields) and len(fields) == 3:
                    continue
                raise TerrainError(f'Строка {number}: ожидаются три числа X Y Z.') from exc
            # JSON must remain portable: encode NoData as null rather than NaN.
            rows.append([value if np.isfinite(value) else None for value in row])
    TerrainSurface(rows)
    return rows


def add_grid(db, filename, title=None, comment='', is_local=True):
    rows = read_grid_file(filename)
    grid = CommonGrid(title=(title or Path(filename).stem).strip(), type='r', comment=comment,
                      source_name=Path(filename).name, is_local=is_local,
                      grid_table=json.dumps(rows, allow_nan=False))
    if not grid.title:
        raise TerrainError('Введите название сетки.')
    db.add(grid)
    db.flush()
    return grid


def profile_points(profile):
    try:
        xs, ys = json.loads(profile.x_pulc), json.loads(profile.y_pulc)
        if not xs or len(xs) != len(ys):
            raise ValueError()
        points = np.column_stack((xs, ys)).astype(float)
        if not np.isfinite(points).all():
            raise ValueError()
        return points
    except (TypeError, ValueError) as exc:
        raise TerrainError(f'Профиль «{profile.title}»: отсутствуют или некорректны координаты.') from exc


def object_profiles(db, object_id):
    return db.query(Profile).join(Research).filter(Research.object_id == object_id).all()


def validate_object_grid(db, object_id, grid):
    if grid is None or grid.type != 'r':
        raise TerrainError('Выберите сетку рельефа.')
    terrain = surface(grid)
    errors = []
    for profile in object_profiles(db, object_id):
        try:
            if not terrain.covers(profile_points(profile)):
                errors.append(f'{profile.title} (исследование id{profile.research_id}): неполное покрытие')
        except TerrainError as exc:
            errors.append(str(exc))
    if errors:
        raise TerrainError('Сетка не может быть привязана:\n' + '\n'.join(errors))


def resolve_grid(db, profile):
    points = profile_points(profile)
    obj = profile.research.object if profile.research else None
    try:
        return _resolve_grid(db, obj, points)
    except TerrainError as exc:
        raise TerrainError(f'Профиль «{profile.title}»: {exc}') from exc


def _resolve_grid(db, obj, points):
    if obj and obj.relief_grid_id is not None:
        grid = db.get(CommonGrid, obj.relief_grid_id)
        if grid is None or grid.type != 'r' or not surface(grid).covers(points):
            raise TerrainError('Привязанная сетка не полностью покрывает координаты.')
        return grid
    for grid in db.query(CommonGrid).filter_by(type='r', is_local=False).order_by(CommonGrid.id):
        try:
            if surface(grid).covers(points):
                return grid
        except TerrainError:
            continue
    raise TerrainError('Нет общей сетки с полным покрытием координат.')


def sample_object_terrain(db, object_id, points):
    obj = db.get(GeoradarObject, object_id)
    if obj is None:
        raise TerrainError('Объект не найден.')
    return surface(_resolve_grid(db, obj, points)).sample(points)


def invalidate_profile(db, profile):
    profile.abs_relief = profile.depth_relief = profile.relief_source_key = None
    for formation in profile.formations:
        formation.land = formation.speed_cover = formation.land_source_key = None


def bind_grid(db, object_id, grid_id, progress=None):
    """Change the source and rebuild current values in the caller's transaction.

    The caller commits the complete operation or rolls it back on error. Returns
    descriptions of formations lacking the inputs needed for speed_cover.
    """
    obj = db.get(GeoradarObject, object_id)
    if obj is None:
        raise TerrainError('Объект не найден.')
    if grid_id is not None:
        validate_object_grid(db, object_id, db.get(CommonGrid, grid_id))
    if obj.relief_grid_id == grid_id:
        return []
    obj.relief_grid_id = grid_id
    profiles = object_profiles(db, object_id)
    missing_speed_cover = []
    for index, profile in enumerate(profiles):
        if progress:
            progress(index, len(profiles), profile)
        invalidate_profile(db, profile)
        try:
            ensure_profile_terrain(db, profile)
            for formation in profile.formations:
                ensure_formation_terrain(db, formation)
                if formation.speed_cover is None:
                    missing_speed_cover.append(
                        f'{obj.title} / {profile.title} / {formation.title or formation.id}')
        except TerrainError:
            raise
        except (TypeError, ValueError, IndexError) as exc:
            raise TerrainError(
                f'Не удалось пересчитать параметры профиля «{profile.title}». '
                'Проверьте высоты кровли и границы пластов.') from exc
    if progress:
        progress(len(profiles), len(profiles), None)
    db.flush()
    return missing_speed_cover


def move_profile_to_research(db, profile, research_id):
    research = db.get(Research, research_id)
    if research is None:
        raise TerrainError('Исследование не найдено.')
    previous = profile.research
    with db.no_autoflush:
        profile.research = research
        try:
            if research.object.relief_grid_id is not None:
                resolve_grid(db, profile)
        except TerrainError:
            profile.research = previous
            raise
    invalidate_profile(db, profile)
    db.flush()


def delete_grid(db, grid_id):
    grid = db.get(CommonGrid, grid_id)
    if grid is None or grid.type != 'r':
        raise TerrainError('Сетка не найдена.')
    objects = db.query(GeoradarObject).filter_by(relief_grid_id=grid_id).all()
    if objects:
        raise TerrainError('Сетка привязана к объектам: ' + ', '.join(obj.title for obj in objects))
    # Legacy results have no source key. Invalidate unbound objects as well when
    # deleting a common source, so their old predictions cannot bypass refresh.
    profiles = db.query(Profile).filter(Profile.relief_source_key.like(f'v2:{grid_id}:%')).all()
    if not grid.is_local:
        profiles.extend(db.query(Profile).join(Research).join(GeoradarObject).filter(
            GeoradarObject.relief_grid_id.is_(None)).all())
    for profile in set(profiles):
        invalidate_profile(db, profile)
    db.delete(grid)
    db.flush()
    _surface.cache_clear()


def ensure_profile_terrain(db, profile):
    grid = resolve_grid(db, profile)  # Validate even when a cached result exists.
    digest = hashlib.sha256((profile.x_pulc + ':' + profile.y_pulc).encode()).hexdigest()
    key = f'v2:{grid.id}:{digest}'
    if profile.relief_source_key != key or not profile.abs_relief or not profile.depth_relief:
        heights = surface(grid).sample(profile_points(profile))
        invalidate_profile(db, profile)
        profile.abs_relief = json.dumps(heights)
        profile.depth_relief = json.dumps((max(heights) - np.asarray(heights)).tolist())
        profile.relief_source_key = key
        db.flush()
    return json.loads(profile.abs_relief)


def ensure_formation_terrain(db, formation):
    heights = ensure_profile_terrain(db, formation.profile)
    if formation.land_source_key != formation.profile.relief_source_key or not formation.land:
        formation.land = json.dumps(heights)
        formation.speed_cover = None
        if formation.top and formation.layer_up:
            top = np.asarray(json.loads(formation.top), dtype=float)
            times = np.asarray(json.loads(formation.layer_up.layer_line), dtype=float) * 8
            if len(top) == len(times) == len(heights):
                values = np.divide((np.asarray(heights) - top) * 100, times,
                                   out=np.zeros(len(times)), where=times != 0)
                formation.speed_cover = json.dumps(values.tolist())
        formation.land_source_key = formation.profile.relief_source_key
        db.flush()
    return heights


def display_heights(profile, scale_m=0):
    heights = np.asarray(json.loads(profile.abs_relief), dtype=float)
    if scale_m > 0 and len(heights) > 2:
        points = profile_points(profile)
        distance = np.r_[0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
        if distance[-1] > 0:
            # Resample by distance so smoothing has a physical, not trace-count scale.
            uniform = np.linspace(0, distance[-1], len(heights))
            values = np.interp(uniform, distance, heights)
            values = gaussian_filter1d(values, scale_m / (uniform[1] - uniform[0]), mode='nearest')
            heights = np.interp(distance, uniform, values)
    return heights.tolist()
