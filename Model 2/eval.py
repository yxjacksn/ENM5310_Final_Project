# eval.py
"""
External Evaluation Script (Hugging Face).
"""

from __future__ import annotations

import json
import os
import random
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm
import model

# ----------------------------
# CONFIG
# ----------------------------
HF_DATASET = "yxjacksn/Collected_Separate"
HF_SPLIT = "train"
IMAGE_COL = "image"
LAT_COL = "Latitude"
LON_COL = "Longitude"

MODEL_FILE = "model.pt"
SAMPLE_SIZE = 500       # -1 = evaluate ALL rows (5144)
BATCH_SIZE = 32
SEED = 42
HF_CACHE_DIR = None     

SAVE_JSON = True
JSON_OUT = "external_eval_metrics.json"

ACC_THRESHOLDS_M = [1, 2, 5, 10, 25, 50, 100, 200]

# ----------------------------
# Utilities
# ----------------------------
def haversine_m_np(true_lat, true_lon, pred_lat, pred_lon) -> np.ndarray:
    R = 6_371_000.0
    phi1 = np.radians(true_lat)
    phi2 = np.radians(pred_lat)
    dphi = np.radians(pred_lat - true_lat)
    dlambda = np.radians(pred_lon - true_lon)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * (np.sin(dlambda / 2.0) ** 2)
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(np.maximum(1.0 - a, 1e-12)))
    return R * c

def meters_per_degree(lat0_deg: float) -> Tuple[float, float]:
    m_per_deg_lat = 111_320.0
    m_per_deg_lon = 111_320.0 * np.cos(np.radians(lat0_deg))
    return float(m_per_deg_lat), float(m_per_deg_lon)

def _normalize_state_dict_keys(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in sd.items():
        kk = k
        if kk.startswith("module."): kk = kk[len("module.") :]
        if kk.startswith("model."): kk = kk[len("model.") :]
        out[kk] = v
    return out

def load_model_safely(model_file: str, device: str):
    print("🧠 Loading model...")
    m = model.get_model()

    if not os.path.exists(model_file):
        raise FileNotFoundError(f"❌ Could not find '{model_file}'.")

    ckpt = torch.load(model_file, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    elif isinstance(ckpt, dict):
        sd = ckpt
    else:
        raise RuntimeError(f"Unsupported checkpoint format: {type(ckpt)}")

    sd = _normalize_state_dict_keys(sd)

    for key in ("gallery_feats", "gallery_coords", "retrieval_emb", "retrieval_coords"):
        if key in sd and hasattr(m, key):
            current_shape = m.state_dict()[key].shape
            saved_shape = sd[key].shape
            if current_shape != saved_shape:
                print(f"   ⟳ Resizing buffer '{key}' from {tuple(current_shape)} to {tuple(saved_shape)}")
                m.register_buffer(key, torch.zeros_like(sd[key]))

    m.load_state_dict(sd, strict=False)
    m.to(device)
    m.eval()
    print(f"   ✅ Model loaded. Engine: {device}")
    return m

def run_predict(m, batch_items: List[Any]) -> np.ndarray:
    if hasattr(m, "predict"):
        out = m.predict(batch_items)
    else:
        out = m(batch_items)
    
    if torch.is_tensor(out):
        out = out.detach().cpu().numpy()
    out = np.asarray(out, dtype=np.float32)
    if out.ndim == 1: out = out.reshape(1, -1)
    return out

@dataclass
class Metrics:
    dataset_name: str
    model_file: str
    n_evaluated: int
    mean_m: float
    median_m: float
    rmse_m: float
    std_m: float
    p75_m: float
    p90_m: float
    p95_m: float
    p99_m: float
    acc: Dict[str, float]
    mean_dlat_m: float
    mean_dlon_m: float
    avg_infer_ms: float
    throughput_img_s: float

def summarize_and_print(
    *,
    name: str,
    model_file: str,
    device: str,
    true_coords: np.ndarray,
    pred_coords: np.ndarray,
    infer_times_s: List[float],
) -> Metrics:
    errors = haversine_m_np(true_coords[:, 0], true_coords[:, 1], pred_coords[:, 0], pred_coords[:, 1])

    mean_m = float(np.mean(errors))
    median_m = float(np.median(errors))
    rmse_m = float(np.sqrt(np.mean(errors ** 2)))
    std_m = float(np.std(errors))
    p75_m, p90_m, p95_m, p99_m = [float(np.percentile(errors, q)) for q in (75, 90, 95, 99)]
    acc = {f"@{t}m": float(np.mean(errors < t) * 100.0) for t in ACC_THRESHOLDS_M}

    lat0 = float(np.mean(true_coords[:, 0]))
    mlat, mlon = meters_per_degree(lat0)
    dlat_m = (pred_coords[:, 0] - true_coords[:, 0]) * mlat
    dlon_m = (pred_coords[:, 1] - true_coords[:, 1]) * mlon
    mean_dlat_m = float(np.mean(dlat_m))
    mean_dlon_m = float(np.mean(dlon_m))

    total_infer_s = float(np.sum(infer_times_s))
    avg_infer_ms = float((total_infer_s / max(len(errors), 1)) * 1000.0)
    throughput = float(len(errors) / max(total_infer_s, 1e-9))

    print("\n📊 FINAL REPORT CARD (EXTERNAL)")
    print("=" * 60)
    print(f"Dataset:      {name}")
    print(f"Model:        {model_file}")
    print(f"Evaluated:    {len(errors)} images")
    print("-" * 60)
    print("🎯 Error metrics (meters)")
    print(f"  Mean:       {mean_m:8.2f}")
    print(f"  Median:     {median_m:8.2f}")
    print(f"  RMSE:       {rmse_m:8.2f}")
    print(f"  P95:        {p95_m:8.2f}")
    print("-" * 60)
    print("🏆 Accuracy @ thresholds")
    for t in ACC_THRESHOLDS_M:
        print(f"  @ {t:>3d}m:      {acc[f'@{t}m']:6.2f}%")
    print("-" * 60)
    print("🧭 Bias (+N / +E)")
    print(f"  dLat:       {mean_dlat_m:+7.2f} m")
    print(f"  dLon:       {mean_dlon_m:+7.2f} m")
    print("-" * 60)
    print(f"⏱️  Speed:       {avg_infer_ms:7.2f} ms/img ({throughput:.1f} img/s)")
    print("=" * 60)

    worst_k = min(5, len(errors))
    worst_idx = np.argsort(-errors)[:worst_k]
    print("\n🔍 Worst samples")
    for rank, j in enumerate(worst_idx, start=1):
        print(f"  #{rank}: {errors[j]:8.2f} m | True:({true_coords[j,0]:.5f},{true_coords[j,1]:.5f}) Pred:({pred_coords[j,0]:.5f},{pred_coords[j,1]:.5f})")

    return Metrics(
        dataset_name=name, model_file=model_file, n_evaluated=len(errors),
        mean_m=mean_m, median_m=median_m, rmse_m=rmse_m, std_m=std_m,
        p75_m=p75_m, p90_m=p90_m, p95_m=p95_m, p99_m=p99_m, acc=acc,
        mean_dlat_m=mean_dlat_m, mean_dlon_m=mean_dlon_m,
        avg_infer_ms=avg_infer_ms, throughput_img_s=throughput
    )

def hf_image_to_item(image_field: Any) -> Any:
    if isinstance(image_field, dict) and "bytes" in image_field:
        from PIL import Image
        import io
        img = Image.open(io.BytesIO(image_field["bytes"])).convert("RGB")
        return np.asarray(img)
    try:
        from PIL import Image as PILImage
        if isinstance(image_field, PILImage.Image):
            return np.asarray(image_field.convert("RGB"))
    except Exception:
        pass
    return np.asarray(image_field)

def main():
    try:
        from datasets import load_dataset
    except ImportError:
        print("❌ Missing dependency 'datasets'. Run: pip install datasets")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not os.path.exists(MODEL_FILE):
        print(f"❌ Could not find '{MODEL_FILE}'.")
        return

    m = load_model_safely(MODEL_FILE, device)

    print(f"📡 Downloading/Loading HF Dataset: {HF_DATASET}...")
    ds = load_dataset(HF_DATASET, split=HF_SPLIT, cache_dir=HF_CACHE_DIR)
    
    n_total = len(ds)
    rng = random.Random(SEED)
    
    if SAMPLE_SIZE == -1 or SAMPLE_SIZE >= n_total:
        indices = list(range(n_total))
        print(f"   Testing on ALL {n_total} external images.")
    else:
        indices = rng.sample(range(n_total), SAMPLE_SIZE)
        print(f"   Testing on random {SAMPLE_SIZE} external images.")

    print("\n🚀 Running External Evaluation...")
    pred_list = []
    true_list = []
    infer_times = []

    pbar = tqdm(range(0, len(indices), BATCH_SIZE))
    for start in pbar:
        batch_idx = indices[start : start + BATCH_SIZE]
        
        batch_items = []
        batch_true = []
        
        for i in batch_idx:
            ex = ds[i]
            img_np = hf_image_to_item(ex[IMAGE_COL])
            batch_items.append(img_np)
            batch_true.append([float(ex[LAT_COL]), float(ex[LON_COL])])

        t0 = time.perf_counter()
        if device == "cuda": torch.cuda.synchronize()
        
        try:
            batch_pred = run_predict(m, batch_items)
        except Exception:
            batch_pred = []
            valid_true = []
            for item, truth in zip(batch_items, batch_true):
                try:
                    p = run_predict(m, [item])
                    batch_pred.append(p[0])
                    valid_true.append(truth)
                except:
                    continue
            if not batch_pred: continue
            batch_pred = np.array(batch_pred)
            batch_true = np.array(valid_true)

        if device == "cuda": torch.cuda.synchronize()
        infer_times.append(time.perf_counter() - t0)

        pred_list.append(batch_pred)
        true_list.append(batch_true)

    if not pred_list:
        print("❌ All images failed.")
        return

    pred = np.concatenate(pred_list, axis=0)
    true = np.concatenate(true_list, axis=0)

    metrics = summarize_and_print(
        name=f"EXTERNAL ({HF_DATASET})",
        model_file=MODEL_FILE,
        device=device,
        true_coords=true,
        pred_coords=pred,
        infer_times_s=infer_times,
    )

    if SAVE_JSON:
        with open(JSON_OUT, "w", encoding="utf-8") as f:
            json.dump(asdict(metrics), f, indent=2)
        print(f"\n💾 Saved metrics JSON -> {JSON_OUT}")

if __name__ == "__main__":
    main()