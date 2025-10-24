from __future__ import annotations
from pathlib import Path
from typing import List, Sequence, Tuple, Optional, Dict, Any, Literal
import os, json, glob
import uuid, time

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from fastapi import Request
from pydantic import BaseModel, Field, validator

from orchestrator.workflow_manager import (
    DeforestationWorkflow, TrainingConfig, TilingConfig
)

from eintelligence.data_prep.aoi import square_aoi
from eintelligence.data_prep.resume import discover_s2_collection
from eintelligence.data_prep.temporal_pairing import build_temporal_pairs

# ------- Config & Model Registry -------   

env_root = os.environ.get("EARTH_PROJECT_ROOT")
if env_root:
    PROJECT_ROOT = Path(env_root)
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
print(f"Using PROJECT_ROOT: {PROJECT_ROOT}")
DATA_ROOT =  PROJECT_ROOT / "data"
MODELS_ROOT = PROJECT_ROOT / "models"

# Map Task -> (workflow_ctor, checkpoint_subdir, band_names)

MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "deforestation": {
        "workflow": DeforestationWorkflow,
        "ckpt": "deforestation_resnet18.pt",
        "bands": ("B02", "B03", "B04", "B08"),  # Blue, Green, Red, NIR
    },
    # Add more tasks and their configurations here
}

# ------- request / response schemas -------

class AOIInput(BaseModel):
    kind: Literal["point10km", "geojson"] = Field(
        default="point10km",
        description="How to define AOI: 10 km square around point, or full GeoJSON geometry."
    )
    lat: Optional[float] = None
    lon: Optional[float] = None
    geometry: Optional[Dict[str, Any]] = None  # GeoJSON geometry

    @validator("geometry", always=True)
    def _check_geojson(cls, v, values):
        if values.get("kind") == "geojson" and v is None:
            raise ValueError("geometry must be provided when kind is 'geojson'")
        return v
    
class QueryRequest(BaseModel):
    task: Literal["deforestation"] ="deforestation" # expandable
    # build vs resume
    mode: Literal["build", "resume"] = "build"
    region: str = Field(..., description="Folder name under data/ for outputs, e.g. 'central_ca_10km'")
    start: Optional[str] = None
    end: Optional[str] = None
    aoi: Optional[AOIInput] = None

    # inference options
    max_tiles: int = 32
    make_quicklooks: bool = True

class QueryResponse(BaseModel):
    task: str
    mode: str
    region: str
    outputs_dir: str
    quicklooks: List[str] # URLs
    details: Dict[str, Any] 


# ------- app -------

app = FastAPI(title="Earth Intelligence API", version="0.1.0")

# serve data/ as static so clients can open PNG quicklooks
app.mount("/data", StaticFiles(directory=str(DATA_ROOT), html=False), name="data")

# ------- utility functions -------

def _resolve_registry(task: str):
    if task not in MODEL_REGISTRY:
        raise HTTPException(status_code=400, detail=f"Unknown task '{task}'")
    entry = MODEL_REGISTRY[task]
    ckpt_path = MODELS_ROOT / entry["ckpt"]
    if not ckpt_path.exists():  
        raise HTTPException(status_code=500, detail=f"Model checkpoint not found: {ckpt_path}")
    return entry, ckpt_path

def _build_aoi(aoi: Optional[AOIInput]) -> Dict[str, Any]:
    if aoi is None or aoi.kind == "point10km":
        if aoi is None:
            raise HTTPException(status_code=400, detail="Area of Interest (AOI) not provided; need lat/lon to build 10km square AOI.")
        if aoi.lat is None or aoi.lon is None:
            raise HTTPException(status_code=400, detail="Latitude and Longitude must be provided for point10km AOI.")
        return square_aoi(aoi.lat, aoi.lon)
    else:
        return {"type": "Feature", "geometry": aoi.geometry, "properties": {}}
    
def _collect_quicklook_urls(region: str, subdir: str) -> List[str]:
    quicklook_dir = DATA_ROOT / region / subdir / "quicklooks"
    if not quicklook_dir.exists():
        return []
    # return list of URLs relative to /data/
    base = f"/data/{region}/{subdir}/quicklooks/"
    files = sorted(quicklook_dir.glob("*.png"))[:24]
    return [f"{base}{f.name}" for f in files]

# ------- API endpoints -------
@app.post("/earth/query", response_model=QueryResponse)
def earth_query(req: QueryRequest):
    # resolve task -> workflow & checkpoint
    registry, ckpt_path = _resolve_registry(req.task)
    bands = registry["bands"]
    Workflow = registry["workflow"]

    tiling_cfg = TilingConfig(bands_s2 =bands, tile_size=512, stride=256, max_cloud = 20)
    train_cfg = TrainingConfig(num_epochs=1) # Training is not the part of query; kept for the sake of completeness

    wf = Workflow(PROJECT_ROOT, tiling_cfg, train_cfg)

    region_dir = DATA_ROOT / req.region
    region_dir.mkdir(parents=True, exist_ok=True)

    # build or resume -> pairs_manifest
    if req.mode == "build":
        if req.start is None or req.end is None or req.aoi is None:
            raise HTTPException(status_code=400, detail="start, end, and aoi must be provided in build mode.")
        aoi_feature = _build_aoi(req.aoi)
        pairs_manifest = wf.build_data(aoi_feature, req.start, req.end, req.region)
        pairing_source = "built_from_STAC"

    else:  # resume
        coll_manifest = discover_s2_collection(region_dir)
        pairs_manifest = build_temporal_pairs(coll_manifest)
        pairing_source = "resumed_from_tiles"

    # run inference
    out_dir = region_dir / f"pred_{req.task}_{req.mode}"
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs_path = Path(pairs_manifest)

    # deforestation workflow exposes infer_latest(pairs, ckpt,out_dir)
    # wf.infer_latest(
    #     pairs_manifest=pairs_path,
    #     ckpt_path=ckpt_path,
    #     out_dir=out_dir,
    #     max_tiles=req.max_tiles,
    #     make_quicklooks=req.make_quicklooks,
    #     band_names=bands,
    # )

    wf.run_deforestation_workflow(
        pairs_manifest=pairs_path,
        ckpt_path=ckpt_path,
        out_dir=out_dir,
        retrain=False)

    quicklook_urls = _collect_quicklook_urls(req.region, f"pred_{req.task}_{req.mode}")

    return QueryResponse(
        task=req.task,
        mode=req.mode,
        region=req.region,
        outputs_dir=str(out_dir),
        quicklooks=quicklook_urls,
        details={
            "pairs_manifest": str(pairs_path),
            "checkpoint": str(ckpt_path),
            "pairing_source": pairing_source,
            "bands": bands,
        },
    )
