from __future__ import annotations
from pathlib import Path



def save_multiband_to_geotiff(s2_da, out_path:str | Path, nodata: float | None = None):
    """
    Persist a multi-band Sentinel-2 DataArray to GeoTIFF.
    s2_da dims must be ('band','y','x') with band coord = list of band names.
    """

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    da = s2_da.astype("float32")
    if nodata is not None:
        da = da.rio.write_nodata(nodata, inplace=False)

    da.rio.to_raster(
        out_path,
        BIGTIFF="IF_SAFER",
        tiled=True,
        compress="deflate",
        predictor=2,
    )

    return str(out_path)