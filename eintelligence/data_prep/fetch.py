import rioxarray as rxr
from rioxarray.merge import merge_arrays
import pystac_client 
from shapely.geometry import mapping, Point
from datetime import datetime
import warnings
import planetary_computer
import numpy as np
import xarray as xr
import rioxarray as rxr

STAC_API_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"

def get_sentinel2_data(lat, lon, start_date, end_date, max_cloud_cover=5):
    """
    Fetch Sentinel-2 data from the Microsoft Planetary Computer STAC API for a given location and date range.

    Parameters:
    lat (float): Latitude of the location.
    lon (float): Longitude of the location.
    max_cloud_cover (int): Maximum cloud cover percentage.

    Returns:
    list: List of URLs to the Sentinel-2 data assets.
    """
    # Create a STAC client
    catalog = pystac_client.Client.open(STAC_API_URL, modifier=planetary_computer.sign_inplace)
    
    # Define the search area as a point geometry
    point = Point(lon, lat)
    aoi_polygon = point.buffer(0.01)  # Buffer to create a small area around the point

    # Perform the search
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        intersects=mapping(aoi_polygon),  # Buffer to create a small area around the point
        datetime=f"{start_date.isoformat()}/{end_date.isoformat()}",
        query={"eo:cloud_cover": {"lt": max_cloud_cover}}
    )

    # Get the items from the search results
    items = list(search.item_collection())

    if not items:
        warnings.warn("No suitable Sentinel-2 data found for the given parameters.")
        return None
    
    items.sort(key=lambda item: item.properties['eo:cloud_cover'])
    return items[0]

def load_sentinel2_bands(stac_item, bands=["B04", "B08", "B03", "B02"]):
    """
    Load specified bands from a Sentinel-2 STAC item into a multi-band xarray DataArray.

    Parameters:
    stac_item (pystac.Item): The STAC item representing the Sentinel-2 data.
    bands (list): List of band names to load.

    Returns:
    xarray.DataArray: Multi-band DataArray containing the specified bands.
    """

    rasters = []
    band_names: list[str] = []
    # band_data = {}
    for b in bands:
        asset = stac_item.assets[b]
        href = asset.href
        da = rxr.open_rasterio(href, chunks={"x": 1024, "y": 1024}, masked=True)
        
        
        scale = 1.0
        try:
            scale = float(asset.extra_fields.get("raster:bands", [{}])[0].get("scale", 0.0001))
        except Exception:
            scale = 0.0001

        da = (da.astype("float32") * scale).squeeze("band", drop=True)

        da.rio.write_crs(da.rio.crs or "EPSG:4326", inplace=True)

        rasters.append(da)
        band_names.append(b)
    
    ds = xr.concat(rasters, dim="band")
    ds = ds.assign_coords({"band": band_names})
    ds.name = "sentinel2_reflectance"

    return ds

# Caclulate NDVI

def calculate_ndvi(ds: xr.DataArray) -> xr.DataArray:
    """
    NDVI = (NIR - Red) / (NIR + Red), using B08 (NIR) and B04 (Red).
    Returns a float32 (y, x) DataArray with metadata preserved.
    """
    if "B08" not in ds.coords["band"].values or "B04" not in ds.coords["band"].values:
        raise ValueError("Bands B08 (NIR) and B04 (Red) are required.")

    nir = ds.sel(band="B08")
    red = ds.sel(band="B04")
    denom = (nir + red)
    ndvi = (nir - red) / denom.where(denom != 0)
    ndvi = ndvi.astype("float32")
    ndvi.name = "NDVI"
    ndvi.attrs.update(
        long_name="Normalized Difference Vegetation Index",
        units="dimensionless",
    )
    
    return ndvi