# eintelligence/data_prep/tiler_streaming.py
from __future__ import annotations
from pathlib import Path
import json
from typing import Optional, Sequence, Tuple, Iterable

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window
from rio_cogeo.cogeo import cog_translate
from rio_cogeo.profiles import cog_profiles
from shapely.geometry import box, mapping, shape
from shapely.ops import transform as shp_transform
from pyproj import CRS, Transformer
import rasterio.windows as rwin
import planetary_computer

def _iter_windows(H: int, W: int, tile: int, stride: Optional[int]) -> Iterable[Tuple[int,int,Window]]:
    step = tile if stride is None else stride
    ri = 0; r = 0
    while r < H:
        ci = 0; c = 0
        while c < W:
            win_h = min(tile, H - r); win_w = min(tile, W - c)
            if win_h == tile and win_w == tile:
                yield ri, ci, Window(c, r, win_w, win_h)
            c += step; ci += 1
        r += step; ri += 1

def _iter_windows_subset(H, W, tile, stride, row0, col0, h_sub, w_sub):
    step = tile if stride is None else stride
    r=row0
    while r < row0 + h_sub:
        c=col0
        win_h = min(tile, (row0 + h_sub) - r)
        while c < col0 + w_sub:
            win_w = min(tile, (col0 + w_sub) - c)
            yield rwin.Window(c, r, win_w, win_h)
            c += step
        r += step

import time
import rasterio
from rasterio.enums import Resampling

def _read_with_retry(href, band=1, window=None, out_shape=None, resampling=Resampling.nearest, retries=3, delay=3):
    """
    Read a raster window from a remote COG with retry logic.
    Helps avoid transient HTTP 503 or range-request errors from Planetary Computer.

    Parameters
    ----------
    href : str
        Remote URL or local path to the COG.
    band : int
        Band index to read (default: 1).
    window : rasterio.windows.Window, optional
        Pixel window to read.
    out_shape : tuple, optional
        Desired (height, width) of the output array.
    resampling : rasterio.enums.Resampling
        Resampling method (nearest by default).
    retries : int
        Number of times to retry if read fails.
    delay : int
        Delay (seconds) between retries.

    Returns
    -------
    np.ndarray
        The read array (single band).
    """
    last_err = None
    for i in range(retries):
        try:
            with rasterio.open(href) as src:
                return src.read(
                    band,
                    window=window,
                    out_shape=out_shape,
                    resampling=resampling
                )
        except Exception as e:
            last_err = e
            print(f"⚠️ Read failed ({i+1}/{retries}) for {href.split('/')[-1]}: {e}")
            time.sleep(delay)
    raise last_err


def tile_stac_item_to_cogs(
    stac_item,
    bands: Sequence[str] = ("B02","B03","B04","B08"),
    out_dir: str | Path = "data/tiles_s2",
    tile_size: int = 512,
    stride: Optional[int] = 256,
    min_valid_fraction: float = 0.3,
    web_optimized: bool = True,
    reflectance_uint16: bool = True,  # store original 0..10000 as uint16 for tiny RAM/IO
    aoi_geojson: None = None,
    ) -> Path:
    """
    Stream tiles directly from remote Sentinel-2 COG assets referenced by STAC.
    No full-scene GeoTIFF is written. Produces multi-band COG tiles + manifest.json.
    """
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    profile = cog_profiles.get("deflate")
    manifest = {"type": "FeatureCollection", "features": []}

    # Open the first 10 m band to get grid/CRS. (Ensure all chosen bands are same resolution here.)
    signed_item = planetary_computer.sign(stac_item)
    ref_href = signed_item.assets[bands[0]].href
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="YES"):  # speed up HTTP range reads
        with rasterio.open(ref_href) as ref:
            H, W = ref.height, ref.width
            crs = ref.crs
            transform = ref.transform
            nodata = ref.nodata
            # read dtype from ref; S2 L2A 10m bands are usually uint16 values 0..10000
            base_dtype = ref.dtypes[0]

            if aoi_geojson is not None:
                aoi = shape(aoi_geojson["geometry"])

                # reproject AOI to raster CRS
                to_raster = Transformer.from_crs(CRS.from_epsg(4326), crs, always_xy=True).transform
                aoi_raster = shp_transform(to_raster, aoi)  
                # intersect with raster bounds
                raster_bounds_poly = box(*ref.bounds)
                aoi_raster = aoi_raster.intersection(raster_bounds_poly)
                if aoi_raster.is_empty:
                    raise RuntimeError("AOI does not intersect raster bounds")
                # get pixel bounds of intersected AOI
                minx, miny, maxx, maxy = aoi_raster.bounds
                sub_window = rwin.from_bounds(minx, miny, maxx, maxy, transform)
                sub_window = sub_window.round_offsets().round_lengths()
                row0, col0 = int(sub_window.row_off), int(sub_window.col_off)
                h_sub, w_sub = int(sub_window.height), int(sub_window.width)
            else:
                row0, col0 = 0, 0
                h_sub, w_sub = H, W

        for win in _iter_windows_subset(H, W, tile_size, stride, row0, col0, h_sub, w_sub):
            tile_bands = []
            valid_fraction = 1.0

            for b in bands:
                href = signed_item.assets[b].href
                arr = _read_with_retry(href, 1, window=win, out_shape=(int(win.height), int(win.width)), resampling=Resampling.nearest)
                tile_bands.append(arr)
                if nodata is not None:
                    valid_fraction = min(valid_fraction, (arr != nodata).mean())
            
            if valid_fraction < min_valid_fraction:
                continue

            arr_stack = np.stack(tile_bands, axis=0)  # (B, tile, tile)

            # Keep as uint16 to minimize memory/IO; scale to float later when needed
            if reflectance_uint16:
                dst_dtype = "uint16"
                dst_nodata = 0  # S2 reflectance rarely uses 0; acceptable as nodata
            else:
                arr_stack = (arr_stack.astype("float32") * 1.0/10000.0)
                dst_dtype = "float32"
                dst_nodata = None
            
            r,c = int(win.row_off), int(win.col_off)
            tile_name = f"r{r:04d}_c{c:04d}.tif"
            tmp_path = out_dir / f"_{tile_name}"
            tile_path = out_dir / tile_name

            meta = {
                "driver": "GTiff",
                "height": int(win.height),
                "width": int(win.width),
                "count": len(bands),
                "crs": crs,
                "transform": rwin.transform(win, transform),
                "dtype": dst_dtype,
                "tiled": True,
                "compress": None,   # handled by cog profile
                "nodata": dst_nodata,
            }

            with rasterio.open(tmp_path, "w", **meta) as dst:
                dst.write(arr_stack)
            
            cog_translate(
                tmp_path,
                tile_path,
                profile,
                indexes=list(range(1, len(bands) + 1)),
                overview_level=4,
                overview_resampling="nearest",
                web_optimized=False,
                quiet=True,
            )
            tmp_path.unlink(missing_ok=True)
            left, bottom, right, top = rwin.bounds(win, transform)
            geom = mapping(box(left, bottom, right, top))
            manifest["features"].append(
                {
                    "type": "Feature",
                    "geometry": geom,
                    "properties": {
                        "path": tile_name,
                        "row_off": r, "col_off": c,
                        "height": int(win.height), "width": int(win.width),
                        "size": tile_size, "stride": tile_size if stride is None else stride,
                        "crs": crs.to_string(), "bands": list(bands),
                        "valid_fraction": float(valid_fraction),
                    }
                }
            )


        # for ri, ci, win in _iter_windows(H, W, tile_size, stride):
        #     tile_bands = []
        #     valid_fraction = 1.0

        #     for b in bands:
        #         href = signed_item.assets[b].href
        #         with rasterio.open(href) as src:
        #             arr = src.read(1, window=win, out_shape=(tile_size, tile_size), resampling=Resampling.nearest)
        #             tile_bands.append(arr)

        #             if nodata is not None:
        #                 valid_fraction = min(valid_fraction, (arr != nodata).mean())

        #     if valid_fraction < min_valid_fraction:
        #         continue

        #     arr_stack = np.stack(tile_bands, axis=0)  # (B, tile, tile)

        #     # Keep as uint16 to minimize memory/IO; scale to float later when needed
        #     if reflectance_uint16:
        #         dst_dtype = "uint16"
        #         dst_nodata = 0  # S2 reflectance rarely uses 0; acceptable as nodata
        #     else:
        #         arr_stack = (arr_stack.astype("float32") * 1.0/10000.0)
        #         dst_dtype = "float32"
        #         dst_nodata = None

        #     tile_name = f"r{ri:04d}_c{ci:04d}.tif"
        #     tmp_path = out_dir / f"_{tile_name}"
        #     tile_path = out_dir / tile_name

        #     meta = {
        #         "driver": "GTiff",
        #         "height": tile_size,
        #         "width": tile_size,
        #         "count": len(bands),
        #         "crs": crs,
        #         "transform": rasterio.windows.transform(win, transform),
        #         "dtype": dst_dtype,
        #         "tiled": True,
        #         "compress": None,   # handled by cog profile
        #         "nodata": dst_nodata,
        #     }

        #     # Write small window GTiff then translate to COG
        #     with rasterio.open(tmp_path, "w", **meta) as dst:
        #         dst.write(arr_stack)

        #     cog_translate(
        #         tmp_path,
        #         tile_path,
        #         profile,
        #         indexes=list(range(1, len(bands) + 1)),
        #         overview_level=4,
        #         quiet=True,
        #     )
        #     tmp_path.unlink(missing_ok=True)

        #     left, bottom, right, top = rasterio.windows.bounds(win, transform)
        #     manifest["features"].append(
        #         {
        #             "type": "Feature",
        #             "geometry": mapping(box(left, bottom, right, top)),
        #             "properties": {
        #                 "path": tile_name,
        #                 "row": ri,
        #                 "col": ci,
        #                 "size": tile_size,
        #                 "stride": tile_size if stride is None else stride,
        #                 "crs": crs.to_string(),
        #                 "bands": list(bands),
        #                 "valid_fraction": float(valid_fraction),
        #                 "dtype": dst_dtype,
        #             },
        #         }
        #     )

    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest_path
