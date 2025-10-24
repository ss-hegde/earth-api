from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict, Any, Sequence, Tuple
import json, os, time, sys

import torch    
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import GradScaler, autocast
import torch.nn.functional as F
import numpy as np
import rasterio
import matplotlib
matplotlib.use('Agg')  # for headless servers
import matplotlib.pyplot as plt

# Get the path of the project root directory 
# Assumes you are running jupyter lab from inside the 'earth-intelligence-platform' directory
project_root = os.path.abspath(os.path.join(os.getcwd(), '..'))

# # Add the project root to the system path so Python can find 'eintelligence'
# if project_root not in sys.path:
#     sys.path.append(project_root)
    
# print(f"Project root added to path: {project_root}")

# Data discovery and preparation
from eintelligence.data_prep.fetch_multi_data import search_s2_items
from eintelligence.data_prep.build_data_collection import build_s2_data_collection
from eintelligence.data_prep.temporal_pairing import build_temporal_pairs
from eintelligence.data_prep.aoi import square_aoi
from eintelligence.data_prep.resume import discover_s2_collection

# Datasets and dataloaders
from eintelligence.data_prep.change_dataset import DeforestationChangeDataset
from eintelligence.data_prep.flood_s1_dataset import FloodS1Dataset

# Models and training
from eintelligence.backbone.resnet_encoder import ResNetBackbone
from eintelligence.adapters.change_head import ChangeUNetHead

# utility - save a single band mask aligned to a source COG
def _save_mask_like(src_tile_path:Path, mask_uint8:np.ndarray, out_path:Path) -> None:
    """Save a single band uint8 mask aligned to a source COG"""
    with rasterio.open(src_tile_path) as src:
        profile = src.profile
        profile.update(count=1, dtype="uint8", compress="deflate")
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(mask_uint8, 1)

def _pick_rgb_indices(band_names: Sequence[str]) -> Tuple[int, int, int]:
    """Given a sequence of band names, return the indices for R,G,B bands."""
    band_map = {name: idx for idx, name in enumerate(band_names)}
    for need in ("B04", "B03", "B02"):
        if need not in band_map:
            raise ValueError(f"Band {need} not found in band names: {band_names}")
    return band_map["B04"], band_map["B03"], band_map["B02"]

def _load_rgb_reflectance(tile_path: Path, band_names: Sequence[str]) -> np.ndarray:
    """Load RGB float image in [0,1] from a multiband reflectance COG tile.
    returns: [H,W,3] float32
    """
    with rasterio.open(tile_path) as src:
        r_idx, g_idx, b_idx = _pick_rgb_indices(band_names)
        R = src.read(r_idx + 1).astype(np.float32) / 10000.0
        G = src.read(g_idx + 1).astype(np.float32) / 10000.0
        B = src.read(b_idx + 1).astype(np.float32) / 10000.0
    rgb = np.stack([R, G, B], axis=-1)
    return np.clip(rgb, 0.0, 1.0)

def _save_quicklook_png(rgb: np.ndarray, mask_uint8: np.ndarray, out_png: Path, alpha: float = 0.3):
    """
    Save an RGB + semi-transparent mask overlay (green) as a PNG.
    rgb: HxWx3 in [0,1], mask_uint8: HxW (0 or 255)
    """
    h, w, _ = rgb.shape
    plt.figure(figsize=(max(4, w/512), max(4, h/512)), dpi=150)
    plt.imshow(rgb)
    # overlay mask - green
    m = (mask_uint8 > 0).astype(np.float32)
    overlay = np.zeros((h, w, 4), dtype=np.float32)
    overlay[..., 1] = 1.0  # green channel
    overlay[..., 3] = m * alpha  # alpha channel
    plt.imshow(overlay)
    plt.axis('off')
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(pad=0)
    plt.savefig(out_png, bbox_inches='tight', pad_inches=0)
    plt.close()

@dataclass
class TrainingConfig:
    batch_size: int = 8
    num_workers: int = 4
    lr: float = 1e-3
    weight_decay: float = 1e-4
    num_epochs: int = 10
    val_fraction: float = 0.2
    amp: bool = True  # automatic mixed precision

@dataclass
class TilingConfig:
    bands_s2: Tuple[str, ...] = ("B02","B03","B04","B08")  # S2 bands to use
    tile_size: int = 512
    stride: Optional[int] = 256  # None = non-overlapping
    max_cloud: int = 20  # max cloud cover % for S2 scenes
    same_mgrs_tile: bool = True  # pair scenes from same MGRS tile  

class _Trainer:
    """ reusable training loop for both adapters
    """

    def __init__(self, device: torch.device, cfg: TrainingConfig):
        self.device = device
        self.cfg = cfg
        self.scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and cfg.amp))
        torch.backends.cudnn.benchmark = True  # may help speed

    @staticmethod
    def _loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """Binary cross entropy with logits loss, ignoring nan targets"""
        bce = F.binary_cross_entropy_with_logits(logits, targets)
        probs = torch.sigmoid(logits)
        inter = (probs * targets).sum(dim=(2,3))
        dice = 1 - (2*inter + eps) / (probs.sum(dim=(2,3)) + targets.sum(dim=(2,3)) + eps)
        return bce + dice.mean()
    
    def _run_epoch(self, loader: DataLoader, enc: torch.nn.Module, head: torch.nn.Module, optimizer=None, train=True) -> float:
        if train:
            enc.train(); head.train()
        else:
            enc.eval(); head.eval()
        
        total = 0.0
        with torch.set_grad_enabled(train):
            for x, y, _meta in loader:
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)

                with autocast(enabled=(self.device.type == "cuda" and self.cfg.amp)):
                    features = enc(x)
                    logits = head(features)
                    loss = self._loss(logits, y)

                if train:
                    optimizer.zero_grad(set_to_none=True)
                    self.scaler.scale(loss).backward()
                    self.scaler.step(optimizer)
                    self.scaler.update()
                total += loss.item() * x.size(0)
        return total / len(loader.dataset)
    
    def fit(self, dataset, in_ch: int, ckpt_path: Path) -> None:
        n_val = int(len(dataset) * self.cfg.val_fraction)
        n_train = len(dataset) - n_val
        train_set, val_set = random_split(dataset, [n_train, n_val])

        train_loader = DataLoader(train_set, batch_size=self.cfg.batch_size, shuffle=True, num_workers=self.cfg.num_workers, pin_memory=True, persistent_workers=True)
        val_loader = DataLoader(val_set, batch_size=self.cfg.batch_size, shuffle=False, num_workers=self.cfg.num_workers, pin_memory=True, persistent_workers=True)

        # model
        enc = ResNetBackbone(in_channels=in_ch).to(self.device)
        head = ChangeUNetHead().to(self.device)
        optimizer = torch.optim.AdamW(list(enc.parameters()) + list(head.parameters()), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)

        best = float('inf')
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)

        for epoch in range(self.cfg.num_epochs):
            tr = self._run_epoch(train_loader, enc, head, optimizer, train=True)
            vl = self._run_epoch(val_loader, enc, head, optimizer=None, train=False)
            print(f"[epoch {epoch:02d}] train={tr:.4f}  val={vl:.4f}")
            if vl < best:
                best = vl
                torch.save({"enc": enc.state_dict(), "head": head.state_dict()}, ckpt_path)
                print(f"  ↳ saved: {ckpt_path}")


class DeforestationWorkflow:
    """
    End-to-end deforestation workflow (Sentinel-2 temporal change):
      1) search multi-scene S2 (same MGRS) over AOI/time
      2) tile scenes to COGs + scene manifests
      3) build temporal tile pairs (t1,t2)
      4) train change-detector (adapter)
      5) run inference on the last pair and write masks
    """

    def __init__(
        self, 
        project_root, 
        tiling_cfg: TilingConfig = TilingConfig(), 
        train_cfg: TrainingConfig = TrainingConfig(), 
        skip_to_pairing: bool = False
        ):
        
        self.root = Path(project_root)
        self.tcfg = tiling_cfg
        self.tn = _Trainer(torch.device("cuda" if torch.cuda.is_available() else "cpu"), train_cfg)
        self.skip_to_pairing = skip_to_pairing

    def build_data(
        self, 
        aoi_geojson: Dict[str, Any], 
        start: str, end: str, 
        region_name: str
        ) -> Path:
        
        region_dir = self.root / "data" / region_name

        if self.skip_to_pairing:
            print("Skipping to temporal pairing using existing tiled data...")
            coll_manifest = discover_s2_collection(region_dir)
        else:
            # 1) search multi-scene
            items = search_s2_items(
                aoi_geojson, start, end,
                max_cloud=self.tcfg.max_cloud,
                same_mgrs_tile=self.tcfg.same_mgrs_tile
            )

            if not items:
                raise RuntimeError("No Sentinel-2 items found for the given AOI/time.")


            # 2) tile each scene -> per-scene manifest; 3) write collection_manifest.json
            
            coll_manifest = build_s2_data_collection(
                items, out_dir=region_dir,
                bands=self.tcfg.bands_s2,
                tile_size=self.tcfg.tile_size,
                stride=self.tcfg.stride,
                aoi_geojson=aoi_geojson
            )

        # 3) build temporal pairs across consecutive scenes
        pairs_manifest = build_temporal_pairs(coll_manifest)
        print(f"pairs manifest: {pairs_manifest}")
        return pairs_manifest

    def train(self, pairs_manifest: Path, ckpt_path: Path) -> None:
        print("Training deforestation change detector...")
        # S2 change: input channels = 2 * len(bands) (t1||t2)
        in_ch = 2 * len(self.tcfg.bands_s2)
        ds = DeforestationChangeDataset(str(pairs_manifest), band_names=self.tcfg.bands_s2, ndvi_drop_threshold=0.2)
        self.tn.fit(ds, in_ch=in_ch, ckpt_path=ckpt_path)

    def infer_latest(self, 
        pairs_manifest: Path, 
        ckpt_path: Path, 
        out_dir: Path, 
        max_tiles: int = 32,
        make_quicklooks: bool = True,
        band_names: Sequence[str] = ("B02", "B03", "B04", "B08")
        ) -> Path:

        device = self.tn.device
        # load model
        in_ch = 2 * len(self.tcfg.bands_s2)
        enc = ResNetBackbone(in_channels=in_ch).to(device)
        head = ChangeUNetHead().to(device)
        ckpt = torch.load(ckpt_path, map_location=device)
        enc.load_state_dict(ckpt["enc"]); head.load_state_dict(ckpt["head"])
        enc.eval(); head.eval()

        pairs = json.loads(Path(pairs_manifest).read_text())["pairs"]
        # only use last scene pair (most recent)
        if not pairs:
            raise RuntimeError("No pairs in the manifest.")
        # group by s2_id and pick the last pair set
        last_s2 = pairs[-1]["s2_id"]
        last_pairs = [p for p in pairs if p["s2_id"] == last_s2]
        out_dir.mkdir(parents=True, exist_ok=True)

        saved = 0
        for p in last_pairs:
            # read t1,t2 stacks (uint16) and convert to reflectance
            with rasterio.open(p["t1_path"]) as s1, rasterio.open(p["t2_path"]) as s2:
                A = s1.read().astype(np.float32) / 10000.0  # [C,H,W]
                B = s2.read().astype(np.float32) / 10000.0
            x = np.concatenate([A, B], axis=0)[None, ...]     # [1,2C,H,W]
            x = torch.from_numpy(x).to(device, non_blocking=True)

            with torch.no_grad(), autocast(enabled=(device.type=="cuda")):
                feats = enc(x)
                logits = head(feats)
                probs = torch.sigmoid(logits)[0, 0].float().cpu().numpy()
            mask = (probs > 0.5).astype(np.uint8) * 255

            src_tile_path = Path(p["t2_path"])  # write aligned to t2
            out_path = out_dir / (src_tile_path.stem + "_deforest.tif")
            _save_mask_like(src_tile_path, mask, out_path)
            saved += 1
            if saved >= max_tiles: break

            if make_quicklooks:
                try:
                    rgb = _load_rgb_reflectance(src_tile_path, band_names)
                    png_dir = out_dir / "quicklooks"
                    png_path = png_dir / (src_tile_path.stem + "_flood.png")
                    _save_quicklook_png(rgb, mask, png_path)
                except Exception as e:
                    print(f"Error creating quicklook for {src_tile_path}: {e}")

        print(f"wrote {saved} deforestation mask tiles -> {out_dir}")
        return out_dir
    
    def run_deforestation_workflow(
        self,
        pairs_manifest: Path,
        ckpt_path: Path,
        out_dir: Path,
        retrain: bool = False,
        ):
        if retrain or not ckpt_path.exists():
            print("Training deforestation change detector...")
            self.train(pairs_manifest, ckpt_path)
        else:
            print(f"Using existing checkpoint: {ckpt_path}, skipping training.")

        self.infer_latest(pairs_manifest, ckpt_path, out_dir)

    

class FloodWorkflow:
    """
    Flood change workflow (Sentinel-1 RTC before/after).
    NOTE: This assumes you have already built a pairs_manifest.json for S1-RTC tiles
          (VV/VH, same grid). You can mirror the S2 build steps for S1 RTC.
    """
    def __init__(self, project_root: Path, train_cfg: TrainingConfig = TrainingConfig()):
        self.root = project_root
        self.tn = _Trainer(torch.device("cuda" if torch.cuda.is_available() else "cpu"), train_cfg)

    def train(self, s1_pairs_manifest: Path, ckpt_path: Path) -> None:
        ds = FloodS1Dataset(str(s1_pairs_manifest))  # X=[VV,VH]_t1|t2 -> [4,H,W]
        self.tn.fit(ds, in_ch=4, ckpt_path=ckpt_path)

    def infer_latest(self, 
        s1_pairs_manifest: Path, 
        ckpt_path: Path, 
        out_dir: Path, 
        max_tiles: int = 32,
        make_quicklooks: bool = True,
        band_names: Sequence[str] = ("B02", "B03", "B04", "B08")
        ) -> Path:

        device = self.tn.device
        enc = ResNetBackbone(in_channels=4).to(device)
        head = ChangeUNetHead().to(device)
        ckpt = torch.load(ckpt_path, map_location=device)
        enc.load_state_dict(ckpt["enc"]); head.load_state_dict(ckpt["head"])
        enc.eval(); head.eval()

        pairs = json.loads(Path(s1_pairs_manifest).read_text())["pairs"]
        if not pairs:
            raise RuntimeError("No pairs in the S1 pairs manifest.")
        last_s2 = pairs[-1]["s2_id"] if "s2_id" in pairs[-1] else None  # field name may differ for S1
        last_pairs = [p for p in pairs if p.get("s2_id", last_s2) == last_s2] if last_s2 else pairs

        out_dir.mkdir(parents=True, exist_ok=True)
        saved = 0
        for p in last_pairs:
            with rasterio.open(p["t1_path"]) as a, rasterio.open(p["t2_path"]) as b:
                A = a.read().astype(np.float32)
                B = b.read().astype(np.float32)
            x = np.stack([A[0], A[1], B[0], B[1]], axis=0)[None, ...]  # [1,4,H,W]
            x = torch.from_numpy(x).to(device, non_blocking=True)
            with torch.no_grad(), autocast(enabled=(device.type=="cuda")):
                probs = torch.sigmoid(head(enc(x)))[0, 0].float().cpu().numpy()
            mask = (probs > 0.5).astype(np.uint8) * 255
            src_tile_path = Path(p["t2_path"])
            out_path = out_dir / (src_tile_path.stem + "_flood.tif")
            _save_mask_like(src_tile_path, mask, out_path)
            saved += 1
            if saved >= max_tiles: break
            if make_quicklooks:
                try:
                    rgb = _load_rgb_reflectance(src_tile_path, band_names)
                    png_dir = out_dir / "quicklooks"
                    png_path = png_dir / (src_tile_path.stem + "_flood.png")
                    _save_quicklook_png(rgb, mask, png_path)
                except Exception as e:
                    print(f"Error creating quicklook for {src_tile_path}: {e}")

        print(f"wrote {saved} flood mask tiles -> {out_dir}")
        return out_dir