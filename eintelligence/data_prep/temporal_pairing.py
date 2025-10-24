import json
from pathlib import Path
from typing import List, Dict

def _tile_key(props: dict) -> tuple:
    """
    Produce a stable key for matching tiles between scenes.
    Supports both old schema (row/col) and new AOI schema (row_off/col_off).

    Preference order:
    1) explicit grid indices: (row, col)
    2) derive grid indices from pixel offsets: (row_off // stride, col_off // stride)
       falling back to // size if stride is missing
    3) last resort: exact pixel offsets (row_off, col_off)
    """
    if "row" in props and "col" in props:
        return (int(props["row"]), int(props["col"]))

    if "row_off" in props and "col_off" in props:
        row_off = int(props["row_off"])
        col_off = int(props["col_off"])
        # derive grid step
        step = int(props.get("stride", props.get("size", 1)))
        if step > 0:
            return (row_off // step, col_off // step)
        # fallback if step missing/bad
        return (row_off, col_off)

    # ultimate fallback: use bounds (rounded) if nothing else available
    if "geometry" in props:  # not typical; geometry is on the feature, not props
        pass
    # Unusual manifest; return something deterministic to avoid crash
    return (hash(frozenset(props.items())), 0)

# def build_rowcol_index(manifest_path: Path) -> Dict[tuple, str]:
#     """
#     Given a scene manifest.json, build an index mapping (row, col) to tile file path.
#     """
#     m = json.loads(Path(manifest_path).read_text())
#     index = {}
#     for feature in m["features"]:
#         props = Path(manifest_path).parent / feature["properties"]["path"]
#         index[(feature["properties"]["row"], feature["properties"]["col"])] = str(props)
#     return index

def build_rowcol_index(manifest_path: Path) -> dict:
    """
    Build an index mapping tile-key -> absolute tile path for a single scene manifest.
    """
    m = json.loads(Path(manifest_path).read_text())
    idx = {}
    tiles_base = Path(manifest_path).parent
    for ft in m["features"]:
        props = ft["properties"]
        path = tiles_base / props["path"]
        key = _tile_key(props)
        idx[key] = str(path)
    return idx

def build_temporal_pairs(
    collection_manifest_path: Path, 
) -> Path:
    
    """
    For consecutive scenes s_i, s_(i+1), pair tiles with same (row, col).
    Returns a JSON file, pairs_manifest.json, with entries:
    { t1_path, t2_path, row, col, s1_id, s2_id}
    """

    collection = json.loads(Path(collection_manifest_path).read_text())["scenes"]
    collection = sorted(collection, key=lambda e: e["datetime"])
    pairs = []
    for i in range(len(collection) - 1):
        A, B = collection[i], collection[i+1]
        index_A = build_rowcol_index(Path(A["manifest_path"]))
        index_B = build_rowcol_index(Path(B["manifest_path"]))
        common_keys = set(index_A.keys()).intersection(index_B.keys())
        for (r, c) in common_keys:
            pairs.append({
                "row": r, "col": c,
                "t1_path": index_A[(r, c)], "t2_path": index_B[(r, c)],
                "s1_id": A["scene_id"], "s2_id": B["scene_id"]
            })
    
    out_path = Path(collection_manifest_path).parent / "pairs_manifest.json"
    out_path.write_text(json.dumps({"pairs": pairs}, indent=2))
    return out_path
    
    