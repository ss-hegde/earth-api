from __future__ import annotations
import numpy as np
import json
import torch
from torch.utils.data import Dataset
import rasterio
from rasterio.env import Env
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

class FloodS1Dataset(Dataset):
    """
    Returns (X, y): X = concat([t1_VV,VH, t2_VV,VH]) -> [4,H,W]
    y = weak label threshold on backscatter drop 
    """

    def __init__(
        self,
        pairs_manifest_path,
        vv_index = 1,
        vh_index = 2,
        drop_db = 2.5, # pseudo-label threshold
        normalize: bool = True,           # simple per channel normalization
    ):
        pairs = json.loads(open(pairs_manifest_path).read())["pairs"]
        self.pairs = pairs
        self.vv_index = vv_index-1
        self.vh_index = vh_index-1
        self.drop_db = drop_db
        self.normalize = normalize

    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        pair = self.pairs[idx]
        t1_path = _read_tile(pair["t1_path"]).astype(np.float32)  # assume bands = [VV, VH] reflectivity or dB
        t2_path = _read_tile(pair["t2_path"]).astype(np.float32)

        X = np.stack((t1_path[self.vv_index], t1_path[self.vh_index], t2_path[self.vv_index], t2_path[self.vh_index]), axis=0)  # [4,H,W]

        y = ((t1_path[self.vv_index] - t2_path[self.vv_index]) > self.drop_db).astype(np.float32)[None, ...]

        if self.normalize:
            mu = X.mean(axis=(1,2), keepdims=True)
            sigma = X.std(axis=(1,2), keepdims=True) + 1e-6
            X = (X - mu) / sigma

        return torch.from_numpy(X), torch.from_numpy(y)
    
      