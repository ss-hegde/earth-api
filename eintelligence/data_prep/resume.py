# eintelligence/data_prep/resume.py
from __future__ import annotations
from pathlib import Path
import json
import re
from typing import List, Dict

_TS_RE = re.compile(r"(\d{8}T\d{6})")  # e.g., 20230720T183929

def _guess_datetime(scene_id: str, fallback: str = "1970-01-01T00:00:00Z") -> str:
    """
    Try to parse a datetime string from a scene_id like 'S2A_MSIL2A_20230720T183929_...'
    Falls back to 1970-01-01 if no timestamp is found (still sortable).
    """
    m = _TS_RE.search(scene_id)
    if not m:
        return fallback
    ts = m.group(1)
    # naive ISO; you can improve to full ISO if needed
    return f"{ts[:4]}-{ts[4:6]}-{ts[6:8]}T{ts[9:11]}:{ts[11:13]}:{ts[13:15]}Z"

def discover_s2_collection(region_dir: Path) -> Path:
    """
    Scan an existing region directory for scene manifests and build/overwrite
    collection_manifest.json *without* re-tiling or re-fetching.
    Expected layout:
        region_dir/
          <scene_id>/
            tiles_s2/
              manifest.json
    Returns the path to the (re)created collection_manifest.json.
    """
    region_dir = Path(region_dir)
    entries: List[Dict] = []

    for scene_dir in sorted(region_dir.iterdir()):
        if not scene_dir.is_dir():
            continue
        tiles_dir = scene_dir / "tiles_s2"
        mani = tiles_dir / "manifest.json"
        if mani.exists():
            # print(f"Found scene manifest: {mani}")
            scene_id = scene_dir.name
            # try to read mgrs from a hint file if you saved one earlier; otherwise None
            scene_meta = scene_dir / "scene_meta.json"
            mgrs = None
            dt = None
            if scene_meta.exists():
                try:
                    meta = json.loads(scene_meta.read_text())
                    dt = meta.get("datetime")
                    mgrs = meta.get("mgrs_tile")
                except Exception:
                    pass
            if dt is None:
                dt = _guess_datetime(scene_id)
            entries.append({
                "scene_id": scene_id,
                "datetime": dt,
                "mgrs_tile": mgrs,
                "manifest_path": str(mani),
            })

    if not entries:
        raise RuntimeError(f"No scene manifests found under {region_dir}. "
                           f"Expected subfolders like <scene_id>/tiles_s2/manifest.json")

    # sort by datetime string
    entries.sort(key=lambda e: e["datetime"])

    coll_path = region_dir / "collection_manifest.json"
    coll_path.write_text(json.dumps({"scenes": entries}, indent=2))
    return coll_path
