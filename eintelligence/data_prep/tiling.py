from __future__ import annotations
from pathlib import Path
import json
from typing import Tuple, Iterable, Optional, Sequence

import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.transform import Affine
from rasterio.enums import Resampling
from rio_cogeo.cogeo import cog_translate
from rio_cogeo.profiles import cog_profiles
from shapely.geometry import box, mapping

def _window_transform(base_transform: Affine, window:Window) -> Affine:
    return Affine.translation(window.col_off, window.row_off) * base_transform

def _iter_windows(height:int, width:int, tile:int, stride: Optional[int]) -> Iterable[Tuple[int, int, Window]]:
    """ Yield (row_idx, col_idx, Window). If stride is none, stride == tile (there is no overlap) """

    step = tile if stride is None else stride
    r = 0
    row_idx = 0
    while r < height:
        c = 0
        col_idx = 0
        win_h = min(tile, height - r)
        while c<width:
            win_w = min(tile, width - c)
            yield row_idx, col_idx, Window(c, r, win_w, win_h)
            c += step
            col_idx += 1
        r += step
        row_idx += 1

def tile_to_cog(
        src_tif: str | Path,
        out_dir: str | Path,
        tile_size: int = 512,
        stride: Optional[int] = None, 
        min_valid_fraction: float = 0.3, # skip tiles with mostly nodata
        overview_levels: tuple[int,...] = (2, 4, 8, 16),
        web_optimized: bool = True,
        ) -> Path:
    """
    Slice a GeoTIFF into COG tiles and build a manifest.json.
    Returns the manifest path.
    """

    src_tif = Path(src_tif)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"type": "FeatureCollection", "features":[]}

    # COG Profile
    profile = cog_profiles.get("webp" if web_optimized else "deflate")

    with rasterio.open(src_tif) as src:
        H, W = src.height, src.width
        base_transform = src.transform
        crs = src.crs
        nodata = src.nodata
        count = src.count

        band_names = None
        if "band" in src.tags():
            band_names = src.tags()["band"].split(",")

        if band_names is None:
            band_names = [f"band{b}" for b in range(1, count+1)]

        for ri, ci, win in _iter_windows(H, W, tile_size, stride):
            if win.height < tile_size or win.width < tile_size:
                continue

            # Read the window
            arr = src.read(window=win, out_shape=(count, tile_size, tile_size), resampling=Resampling.nearest) # Shape (bands, tile, tile)


            # skip if nodata
            if nodata is not None:
                valid = (arr[0] != nodata).mean()
            else:
                # if nodata is unknown, consider all valid
                valid = 1.0
            
            if valid < min_valid_fraction:
                continue

            tile_name = f"r{ri:04d}_c{ci:04d}.tif"
            tmp_path = out_dir / f"_{tile_name}"
            tile_path = out_dir / tile_name

            # Build a temporary in-memory dataset profile for cog_translate

            meta = src.profile.copy()
            meta.update(
                driver="GTiff",
                height=tile_size,
                width=tile_size,
                transform=rasterio.windows.transform(win, base_transform),
                count=count,
                tiled=True,
                compress=None, # no compression for temp file
                nodata=nodata,
                dtype=str(arr.dtype),
            )

            # rio-cogeo expects an input path; easiest path: write a small GTiff then translate.
            # To avoid double-write in future: use MemoryFile -> cog_translate. For clarity now, write once.
            with rasterio.open(tmp_path, "w", **meta) as dst:
                dst.write(arr, 1)


            # Build COG
            cog_translate(
                tmp_path,
                tile_path,
                profile,
                indexes=list(range(1, count + 1)),
                overview_level=overview_levels,
                quiet=True,
            )

            tmp_path.unlink(missing_ok=True)

            # Manifest entry
            left, bottom, right, top = rasterio.windows.bounds(win, base_transform)
            geom = mapping(box(left, bottom, right, top))
            manifest["features"].append(
                {
                    "type": "Feature",
                    "geometry": geom,
                    "properties": {
                        "path": tile_name,
                        "row": ri,
                        "col": ci,
                        "size": tile_size,
                        "stride": tile_size if stride is None else stride,
                        "crs": crs.to_string(),
                        "bands": band_names,
                        "valid_fraction": float(valid),
                    },
                }
            )
    
    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    return manifest_path
