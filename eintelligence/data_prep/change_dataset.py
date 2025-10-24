from __future__ import annotations
import numpy as np
import torch
from torch.utils.data import Dataset
import rasterio
from rasterio.env import Env
import json
from pathlib import Path

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

class DeforestationChangeDataset(Dataset):
    """
    Returns (X, y): X = concat([t1_bands, t2_bands]) along channel dim (8xHxW if 4 bands),
    y = weak label (change mask) from NDVI/NBR drop (later swap for Hansen labels).
    """

    def __init__(
        self,
        pairs_manifest_path,
        band_names = ("B02","B03","B04","B08"),
        ndvi_drop_threshold: float = 0.2, # pseudo-label threshold
        normalize: bool = True,           # simple per channel normalization
        tile_size: int = 512,
    ):
        pairs = json.loads(open(pairs_manifest_path).read())["pairs"]
        self.pairs = pairs
        self.band_names = band_names
        self.ndvi_drop_threshold = ndvi_drop_threshold
        self.normalize = normalize
        self.tile_size = tile_size

        # rough per-channel mean/std from a few tiles, for simple normalization
        self.mean = np.array([0.12, 0.14, 0.16, 0.28], np.float32)
        self.std = np.array([0.08, 0.07, 0.08, 0.10], np.float32)

    def __len__(self):
        return len(self.pairs)

    @staticmethod
    def _pad_hw(arr, Ht, Wt):
        # arr: (C,H,W) -> pad to (C,Ht,Wt) with zeros
        C,H,W = arr.shape
        if H==Ht and W==Wt:
            return arr
        pad_h = max(0, Ht - H)
        pad_w = max(0, Wt - W)
        # (before_H, after_H), (before_W, after_W)
        return np.pad(arr, ((0,0),(0,pad_h),(0,pad_w)), mode="constant", constant_values=0)
    
    def __getitem__(self, idx):
        pair = self.pairs[idx]
        t1 = _reflectance_uint16_to_float(_read_tile(pair["t1_path"])) # (C,H,W) float32 0..1
        t2 = _reflectance_uint16_to_float(_read_tile(pair["t2_path"])) # (C,H,W) float32 0..1

        red1, nir1 = t1[2], t1[3]  # B04, B08
        red2, nir2 = t2[2], t2[3]  # B04, B08
        ndvi1 = _compute_ndvi_float(red1, nir1)
        ndvi2 = _compute_ndvi_float(red2, nir2)
        ndvi_drop = ((ndvi1 - ndvi2) > self.ndvi_drop_threshold).astype(np.float32)  # (H,W) 0/1 float32
        y = ndvi_drop[None, ...]  # (1,H,W) float32

        # input - concat t1, t2 along channel dim  
        x = np.concatenate([t1, t2], axis=0)  # (2C,H,W) float32 0..1

        # pad both to fixed size
        x = self._pad_hw(x, self.tile_size, self.tile_size)
        y = self._pad_hw(y, self.tile_size, self.tile_size)

        if self.normalize:
            x[:4] = (x[:4] - self.mean[:, None, None]) / (self.std[:, None, None]+1e-6)
            x[4:] = (x[4:] - self.mean[:, None, None]) / (self.std[:, None, None]+1e-6)

        meta = {
            "t1_path": pair["t1_path"],
            "t2_path": pair["t2_path"],
            "row": pair.get("row"),
            "col": pair.get("col"),
            "s1_id": pair.get("s1_id"),
            "s2_id": pair.get("s2_id"),
        }

        return torch.from_numpy(x), torch.from_numpy(y), meta
    