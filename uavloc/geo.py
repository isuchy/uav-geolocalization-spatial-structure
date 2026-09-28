"""Geo utilities: coordinate transforms used by eval scripts."""
from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=4)
def _make_transformer(src_epsg: int, dst_epsg: int):
    from pyproj import Transformer
    return Transformer.from_crs(f"EPSG:{src_epsg}", f"EPSG:{dst_epsg}", always_xy=True)


def latlon_to_epsg5514(lat: float, lon: float) -> tuple[float, float]:
    """WGS84 (EPSG:4326) -> S-JTSK / Krovak East North (EPSG:5514).
    Note: pyproj expects (lon, lat) when always_xy=True.
    """
    t = _make_transformer(4326, 5514)
    x, y = t.transform(lon, lat)
    return float(x), float(y)


def epsg5514_to_latlon(x: float, y: float) -> tuple[float, float]:
    t = _make_transformer(5514, 4326)
    lon, lat = t.transform(x, y)
    return float(lat), float(lon)
