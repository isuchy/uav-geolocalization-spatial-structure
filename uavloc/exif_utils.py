"""Minimal EXIF GPS extraction for real drone images."""
from __future__ import annotations

from pathlib import Path
from typing import Optional


def read_gps_from_image(path: Path) -> Optional[tuple[float, float, Optional[float]]]:
    """
    Return (lat, lon, alt_m) from EXIF, or None if absent.
    Returns degrees (WGS84). Altitude is metres above sea level if present.
    """
    try:
        from PIL import Image
        from PIL.ExifTags import GPSTAGS, TAGS
    except ImportError:
        return None

    img = Image.open(path)
    exif = img._getexif() if hasattr(img, "_getexif") else None
    if not exif:
        return None
    gps_info_raw = None
    for tag, val in exif.items():
        if TAGS.get(tag) == "GPSInfo":
            gps_info_raw = val
            break
    if not gps_info_raw:
        return None
    gps = {GPSTAGS.get(t, t): v for t, v in gps_info_raw.items()}

    def _to_deg(values, ref):
        d = float(values[0]); m = float(values[1]); s = float(values[2])
        sign = -1.0 if ref in ("S", "W") else 1.0
        return sign * (d + m / 60.0 + s / 3600.0)

    if "GPSLatitude" not in gps or "GPSLongitude" not in gps:
        return None
    lat = _to_deg(gps["GPSLatitude"], gps.get("GPSLatitudeRef", "N"))
    lon = _to_deg(gps["GPSLongitude"], gps.get("GPSLongitudeRef", "E"))
    alt = None
    if "GPSAltitude" in gps:
        try:
            alt_raw = float(gps["GPSAltitude"])
            if gps.get("GPSAltitudeRef") in (1, b"\x01"):
                alt_raw = -alt_raw
            alt = alt_raw
        except Exception:
            alt = None
    return (lat, lon, alt)
