from __future__ import annotations
from pathlib import Path
from typing import Sequence, Optional, Dict, List, Tuple

import numpy as np
import json
import rasterio
from rasterio.env import Env
import torch
from torch.utils.data import Dataset


CURL_ENV = dict(
    GDAL_DISABLE_READDIR_ON_OPEN="YES",
    CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif",
    CPL_VSIL_CURL_NON_CACHED=".tif",
    VSI_CACHE="TRUE",
    VSI_CACHE_SIZE="1000000",
)

def _read_tile(path: Path) -> np.ndarray:
    """Read a multi-band COG tile -> numpy array (C, H, W)"""
    with Env(**CURL_ENV):
        with rasterio.open(path) as src:
            arr = src.read()  # (C, H, W)
    return arr

def _reflectance_uint16_to_float(arr: np.ndarray, scale: float = 1/10000.0) -> np.ndarray:
    """ # Sentinel-2 L2A reflectance stored as 0..10000 uint16"""
    return (arr.astype(np.float32) * scale)

def _compute_ndvi_float(red: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """NDVI = (NIR - Red) / (NIR + Red)"""
    denom = (nir + red)
    ndvi = (nir - red) / np.where(denom != 0.0, denom, np.nan)
    return ndvi.astype(np.float32)

class Sentinel2TileDataset(Dataset):
    """
    Streams tiles referenced in manifest.json.
    - Returns (image_tensor, target_tensor, meta) where:
        image_tensor: float32 [C,H,W] in reflectance 0..1 (S2) or 0..255/1.0 depending on input
        target_tensor: float32 [1,H,W] (NDVI>threshold mask) -- for the demo task
        meta: dict with path/row/col, can be ignored by the model but useful for saving outputs
    - You can easily replace the pseudo-label generator with real labels later.
    """

    def __init__(
        self,
        tiles_dir: str | Path,
        manifest_path: str | Path,
        band_names: Sequence[str] = ("B02","B03","B04","B08"),
        ndvi_threshold: float = 0.3, # pseudo-label threshold
        normalize: bool = True,      # simple per channel normalization
        augment: Optional[object] = None, 
        use_uint16_reflectance: bool = True, 
        ):
        self.tiles_dir = Path(tiles_dir)
        self.items: List[Dict] = []

        with open(manifest_path) as f:
            manifest = json.load(f)
        feats = manifest["features"]

        self.band_names = list(band_names)
        manifest_bands = feats[0]["properties"].get("bands", self.band_names)

        # Build index of band -> index
        self.band_idx = {name: i for i, name in enumerate(manifest_bands)}

        # Keep only tiles that actually exist
        for feat in feats:
            p = self.tiles_dir / feat["properties"]["path"]
            if p.exists():
                self.items.append({"path": p, "row": feat["properties"]["row"], "col": feat["properties"]["col"]})

        self.ndvi_threshold = ndvi_threshold
        self.normalize = normalize
        self.augment = augment
        self.use_uint16_reflectance = use_uint16_reflectance

        # Basic normalizations (mean/std) for S2 0..1 reflectance (rough defaults; refine later)
        self._mean = np.array([0.12, 0.14, 0.16, 0.28], dtype=np.float32)  # B02,B03,B04,B08
        self._std  = np.array([0.08, 0.07, 0.08, 0.10], dtype=np.float32)

    def __len__(self) -> int:
        return len(self.items)
    
    def __getitem__(self, idx: int):
        item = self.items[idx]
        arr = _read_tile(item["path"])  # (C, H, W)

        # reorder channels by band names
        take = [self.band_idx[b] for b in self.band_names]
        arr = arr[take]  # (C, H, W)

        # Convert to float reflectance if uint16
        if self.use_uint16_reflectance and arr.dtype == np.uint16:
            arr = _reflectance_uint16_to_float(arr)  # (C, H, W) float32 0..1

        # Psuedo-label: NDVI>threshold mask
        red = arr[self.band_names.index("B04")]
        nir = arr[self.band_names.index("B08")]
        ndvi = _compute_ndvi_float(red, nir)  
        target = (ndvi > self.ndvi_threshold).astype(np.float32)  
        target = np.expand_dims(target, 0)  # (1, H, W)

        #  Albumentations branch (if provided) expects HWC; convert then back
        if self.augment is not None:
            # to HWC for image, HW for mask
            img_hwc = np.transpose(arr, (1,2,0))  # (H, W, C)
            m_hw = target[0]          # (H, W)
            res = self.augment(image=img_hwc, mask=m_hw)
            img_hwc = res["image"].astype(np.float32)
            m_hw = res["mask"].astype(np.float32)
            arr = np.transpose(img_hwc, (2,0,1))  # (C, H, W)
            target = np.expand_dims(m_hw, 0)      # (1, H, W)

        # Normalize
        if self.normalize:
            arr = (arr - self._mean[:,None,None]) / (self._std[:,None,None] + 1e-6)

        # to torch tensors
        x = torch.from_numpy(arr)      # (C, H, W)
        y = torch.from_numpy(target)   # (1, H, W)
        meta = {"path": str(item["path"]), "row": item["row"], "col": item["col"]}
        return x, y, meta