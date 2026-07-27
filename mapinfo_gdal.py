"""GDAL-backed native MapInfo writer, isolated from the dependency-free validation code."""

import math
from pathlib import Path
from typing import Sequence

from osgeo import ogr, osr

from mapinfo_export import (
    ExportProfile,
    ProfileExportError,
    WGS_MATCH_TOLERANCE_METERS,
    profile_length,
    tab_sidecar_paths,
)


def _spatial_reference(epsg: int):
    reference = osr.SpatialReference()
    if reference.ImportFromEPSG(epsg) != 0:
        raise ProfileExportError(f"GDAL не удалось загрузить EPSG:{epsg}.")
    if hasattr(reference, "SetAxisMappingStrategy"):
        reference.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    return reference


def _validate_against_wgs(profiles: Sequence[ExportProfile], projected_srs) -> None:
    wgs_srs = _spatial_reference(4326)
    transformation = osr.CoordinateTransformation(projected_srs, wgs_srs)
    for profile in profiles:
        # Checking every point is deliberate: the WGS arrays are our safety net
        # against a wrong zone, swapped axes, or an incorrectly labelled dataset.
        for index, ((easting, northing), (expected_lon, expected_lat)) in enumerate(
            zip(profile.projected_points, profile.wgs_points), start=1
        ):
            actual_lon, actual_lat, *_ = transformation.TransformPoint(easting, northing)
            mean_latitude = math.radians((expected_lat + actual_lat) / 2.0)
            latitude_delta = math.radians(actual_lat - expected_lat)
            longitude_delta = math.radians(actual_lon - expected_lon)
            distance = 6_371_008.8 * math.hypot(
                latitude_delta,
                longitude_delta * math.cos(mean_latitude),
            )
            if not distance <= WGS_MATCH_TOLERANCE_METERS:
                raise ProfileExportError(
                    f'Профиль "{profile.title}", точка {index}: координаты СК-42 расходятся '
                    f"с WGS 84 на {distance:.1f} м (допуск {WGS_MATCH_TOLERANCE_METERS:.0f} м)."
                )


def write_native_tab(tab_path: str | Path, profiles: Sequence[ExportProfile], zone: int) -> Path:
    """Write profiles to a MapInfo Native TAB dataset using GDAL/OGR."""
    output_path = Path(tab_path).with_suffix(".tab")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    projected_srs = _spatial_reference(28400 + zone)
    _validate_against_wgs(profiles, projected_srs)

    driver = ogr.GetDriverByName("MapInfo File")
    if driver is None:
        raise ProfileExportError("Установленный GDAL не содержит драйвер MapInfo File.")
    for sidecar in tab_sidecar_paths(output_path):
        if sidecar.exists():
            sidecar.unlink()
    data_source = driver.CreateDataSource(str(output_path))
    if data_source is None:
        raise ProfileExportError(f"Не удалось создать MapInfo TAB: {output_path}")
    layer = data_source.CreateLayer(
        output_path.stem,
        srs=projected_srs,
        geom_type=ogr.wkbLineString,
        options=["FORMAT=NATIVE", "ENCODING=UTF-8"],
    )
    if layer is None:
        raise ProfileExportError("GDAL не удалось создать линейный слой MapInfo.")
    for name, field_type, width in (
        ("PROFILE_ID", ogr.OFTInteger, 0),
        ("RESEARCH_ID", ogr.OFTInteger, 0),
        ("TITLE", ogr.OFTString, 254),
        ("POINTS", ogr.OFTInteger, 0),
        ("LENGTH_M", ogr.OFTReal, 0),
        ("ZONE", ogr.OFTInteger, 0),
        ("EPSG", ogr.OFTInteger, 0),
    ):
        field = ogr.FieldDefn(name, field_type)
        if width:
            field.SetWidth(width)
        if layer.CreateField(field) != 0:
            raise ProfileExportError(f"Не удалось создать поле {name} в MapInfo TAB.")

    for profile in profiles:
        feature = ogr.Feature(layer.GetLayerDefn())
        feature.SetField("PROFILE_ID", profile.profile_id)
        feature.SetField("RESEARCH_ID", profile.research_id)
        feature.SetField("TITLE", profile.title)
        feature.SetField("POINTS", len(profile.projected_points))
        feature.SetField("LENGTH_M", profile_length(profile.projected_points))
        feature.SetField("ZONE", zone)
        feature.SetField("EPSG", 28400 + zone)
        line = ogr.Geometry(ogr.wkbLineString)
        for easting, northing in profile.projected_points:
            line.AddPoint_2D(easting, northing)
        feature.SetGeometry(line)
        if layer.CreateFeature(feature) != 0:
            raise ProfileExportError(f'Не удалось записать профиль "{profile.title}".')
        feature = None
    layer.SyncToDisk()
    data_source = None
    return output_path
