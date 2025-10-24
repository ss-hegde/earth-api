from __future__ import annotations
from datetime import date
from typing import Any, Dict, List, Optional, Tuple, Sequence
from shapely.geometry import Point, mapping, shape, box
from pystac_client import Client
import planetary_computer

stac_api_url = "https://planetarycomputer.microsoft.com/api/stac/v1"

def search_s2_items(
        aoi_geojson_or_geom,
        start_date: str | date,
        end_date: str | date,
        max_cloud: int = 20,
        limit: Optional[int] = None,
        same_mgrs_tile: bool = True
) -> List:
    
    """Search for Sentinel-2 items in a given area and date range.

    Args:
        stac_api_url (str): URL of the STAC API.
        aoi_geojson: GeoJSON geometry of the area of interest.
        start_date (str | date): Start date for the search (YYYY-MM-DD).
        end_date (str | date): End date for the search (YYYY-MM-DD).
        max_cloud (int, optional): Maximum cloud cover percentage. Defaults to 20.
        limit (Optional[int], optional): Maximum number of items to return. Defaults to None.
        same_mgrs_tile (bool, optional): If True, only return items from the same MGRS tile. Defaults to True.

    Returns:
        List: List of STAC items matching the search criteria.
    """
    catalog = Client.open(stac_api_url, modifier=planetary_computer.sign_inplace)
    
    geom = (aoi_geojson_or_geom.get("geometry") if isinstance(aoi_geojson_or_geom, dict)
            and "type" in aoi_geojson_or_geom and aoi_geojson_or_geom["type"] != "Polygon"
            else aoi_geojson_or_geom)

    search = catalog.search(
        collections=["sentinel-2-l2a"],
        intersects=geom,
        datetime=f"{start_date}/{end_date}",
        query={"eo:cloud_cover": {"lt": max_cloud}},
    )

    items = list(search.items())
    if not items:
        return []
    
    if same_mgrs_tile:
        counts = {}
        for item in items:
            t = item.properties.get("s2:mgrs_tile")
            counts[t] = counts.get(t, 0) + 1
        winner = max(counts, key=counts.get)
        items = [item for item in items if item.properties.get("s2:mgrs_tile") == winner]

    # Sort by time
    items.sort(key=lambda i: i.properties.get("datetime"))

    if limit: items = items[:limit]
    return items