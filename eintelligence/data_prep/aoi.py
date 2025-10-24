from __future__ import annotations
from shapely.geometry import box, mapping, Point
from shapely.ops import transform as shp_transform
from pyproj import CRS, Transformer

def square_aoi(lat: float, lon: float):
    """
    Create a square AOI GeoJSON around a lat/lon point (10 * 10 km).
    """
    wsg84 = CRS.from_epsg(4326)

    # UTM zone for the given lat/lon
    utm = CRS.from_proj4(f"+proj=utm +zone={int((lon + 180) / 6) + 1} +datum=WGS84 +units=m +no_defs")
    to_utm = Transformer.from_crs(wsg84, utm, always_xy=True).transform
    to_wsg84 = Transformer.from_crs(utm, wsg84, always_xy=True).transform

    px, py = to_utm(lon, lat)
    half = 5_000.0 # 5 km half-size
    aoi_utm = box(px - half, py - half, px + half, py + half)

    aoi_wsg84 = shp_transform(to_wsg84, aoi_utm)

    return {"type": "Feature", "geometry": mapping(aoi_wsg84), "properties": {}}