from pathlib import Path
import json
from typing import Sequence, Dict
from shapely.geometry import shape, mapping, box
import planetary_computer
# from eintelligence.data_prep.tiler_streaming import tile_stac_item_to_cogs

from .tiler_streaming import tile_stac_item_to_cogs

def build_s2_data_collection(
    items: Sequence,
    out_dir: Path, 
    bands=("B02","B03","B04","B08"),
    tile_size: int = 512,
    stride: int = 256,
    aoi_geojson = None) -> Path:
    """
    For each STAC item: tile -> tiles_dir + scene manifest.
    Create a collection_manifest.json listing all scenes (scene_id, date, mgrs, manifest_path).
    """

    out_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for item in items:
        scene_id = item.id.replace(":", "_")
        scene_dir = out_dir / scene_id
        tiles_dir = scene_dir / "tiles_s2"
        tiles_dir.mkdir(parents=True, exist_ok=True)

        signed_item = planetary_computer.sign(item)
        manifest_path = tile_stac_item_to_cogs(
            signed_item, bands=bands, out_dir=tiles_dir, tile_size=tile_size, stride=stride, min_valid_fraction=0.3, web_optimized=False, aoi_geojson=aoi_geojson
        )

        entries.append({
            "scene_id": scene_id,
            "datetime": item.properties.get("datetime"),
            "mgrs_tile": item.properties.get("s2:mgrs_tile"),
            "manifest_path": str(manifest_path)
        })
    
    collection_path = out_dir / "collection_manifest.json"
    with open(collection_path, "w") as f:
        json.dump({"scenes": entries}, f, indent=2)

    return collection_path