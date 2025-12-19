# eval.py
"""
External evaluation script (Hugging Face dataset: yxjacksn/Collected_Separate).

Goal:
- Mirror test_model.py style, but evaluate on a *completely separate* dataset (5.14k images).
- Produce comparable metrics so you can write about internal-vs-external generalization.

Dataset page: https://huggingface.co/datasets/yxjacksn/Collected_Separate
Columns visible in the viewer: image, Latitude, Longitude.  (single 'train' split)

Requirements:
  pip install -U datasets pillow

Run:
  python eval.py
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
# CONFIG (edit these if needed)
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

# If True: request file paths from HF cache (more like your internal path-based pipeline).
# If False: use decoded PIL images directly.
USE_IMAGE_PATHS = True

# HuggingFace cache directory (optional)
HF_CACHE_DIR = None     # e.g. "./hf_cache"

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
        if kk.startswith("module."):
            kk = kk[len("module.") :]
        if kk.startswith("model."):
            kk = kk[len("model.") :]
        out[kk] = v
    return out


def load_model_safely(model_file: str, device: str):
    print("🧠 Loading model...")
    m = model.get_model()

    ckpt = torch.load(model_file, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        sd = ckpt["state_dict"]
    elif isinstance(ckpt, dict):
        sd = ckpt
    else:
        raise RuntimeError(f"Unsupported checkpoint format: {type(ckpt)}")

    sd = _normalize_state_dict_keys(sd)

    # Dynamic buffer fix for common memory-bank keys
    for key in ("gallery_feats", "gallery_coords", "retrieval_emb", "retrieval_coords"):
        if key in sd and (not hasattr(m, key) or (key in m.state_dict() and m.state_dict()[key].shape != sd[key].shape)):
            print(f"   ⟳ Resizing/adding buffer '{key}' to {tuple(sd[key].shape)}")
            m.register_buffer(key, torch.zeros_like(sd[key]))

    incompatible = m.load_state_dict(sd, strict=False)
    missing = list(getattr(incompatible, "missing_keys", []))
    unexpected = list(getattr(incompatible, "unexpected_keys", []))
    if missing:
        print(f"   ⚠️  Missing keys (ok if intentional): {missing[:8]}{' ...' if len(missing)>8 else ''}")
    if unexpected:
        print(f"   ⚠️  Unexpected keys (ok if buffers/extra): {unexpected[:8]}{' ...' if len(unexpected)>8 else ''}")

    m.to(device)
    m.eval()
    print(f"   ✅ Model loaded. Engine: {device}")
    return m


def run_predict(m, batch_items: List[Any]) -> np.ndarray:
    if hasattr(m, "predict") and callable(getattr(m, "predict")):
        out = m.predict(batch_items)
        out = np.asarray(out, dtype=np.float32)
        if out.ndim == 1:
            out = out.reshape(1, -1)
        return out

    out = m(batch_items)
    if torch.is_tensor(out):
        out = out.detach().cpu().numpy()
    out = np.asarray(out, dtype=np.float32)
    if out.ndim == 1:
        out = out.reshape(1, -1)
    return out


@dataclass
class Metrics:
    dataset_name: str
    model_file: str
    device: str
    n_requested: int
    n_processed: int
    n_skipped: int
    batch_size: int
    sample_size: int

    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    approx_ns_m: float
    approx_ew_m: float

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
    median_dlat_m: float
    median_dlon_m: float

    clamp_rate_lat_pct: Optional[float]
    clamp_rate_lon_pct: Optional[float]
    gt_outside_model_bbox_pct: Optional[float]

    avg_infer_ms: float
    total_infer_s: float
    throughput_img_s: float

    tail_ratio_p95_over_median: float
    mean_over_median: float


def summarize_and_print(
    *,
    name: str,
    model_file: str,
    device: str,
    indices: List[int],
    true_coords: np.ndarray,
    pred_coords: np.ndarray,
    infer_times_s: List[float],
    item_ids: Optional[List[str]] = None,
    model_bbox: Optional[Tuple[float, float, float, float]] = None,
) -> Metrics:
    errors = haversine_m_np(true_coords[:, 0], true_coords[:, 1], pred_coords[:, 0], pred_coords[:, 1])

    mean_m = float(np.mean(errors))
    median_m = float(np.median(errors))
    rmse_m = float(np.sqrt(np.mean(errors ** 2)))
    std_m = float(np.std(errors))
    p75_m, p90_m, p95_m, p99_m = [float(np.percentile(errors, q)) for q in (75, 90, 95, 99)]
    acc = {f"@{t}m": float(np.mean(errors < t) * 100.0) for t in ACC_THRESHOLDS_M}

    lat_min, lat_max = float(np.min(true_coords[:, 0])), float(np.max(true_coords[:, 0]))
    lon_min, lon_max = float(np.min(true_coords[:, 1])), float(np.max(true_coords[:, 1]))
    lat0 = float(np.mean(true_coords[:, 0]))
    mlat, mlon = meters_per_degree(lat0)
    approx_ns_m = (lat_max - lat_min) * mlat
    approx_ew_m = (lon_max - lon_min) * mlon

    dlat_m = (pred_coords[:, 0] - true_coords[:, 0]) * mlat
    dlon_m = (pred_coords[:, 1] - true_coords[:, 1]) * mlon
    mean_dlat_m = float(np.mean(dlat_m))
    mean_dlon_m = float(np.mean(dlon_m))
    median_dlat_m = float(np.median(dlat_m))
    median_dlon_m = float(np.median(dlon_m))

    clamp_lat = clamp_lon = outside_gt = None
    if model_bbox is not None:
        m_lat_min, m_lat_max, m_lon_min, m_lon_max = model_bbox
        eps = 1e-8
        clamp_lat = float(np.mean((np.abs(pred_coords[:, 0] - m_lat_min) < eps) | (np.abs(pred_coords[:, 0] - m_lat_max) < eps)) * 100.0)
        clamp_lon = float(np.mean((np.abs(pred_coords[:, 1] - m_lon_min) < eps) | (np.abs(pred_coords[:, 1] - m_lon_max) < eps)) * 100.0)
        outside_gt = float(
            np.mean(
                (true_coords[:, 0] < m_lat_min)
                | (true_coords[:, 0] > m_lat_max)
                | (true_coords[:, 1] < m_lon_min)
                | (true_coords[:, 1] > m_lon_max)
            ) * 100.0
        )

    total_infer_s = float(np.sum(infer_times_s))
    avg_infer_ms = float((total_infer_s / max(len(errors), 1)) * 1000.0)
    throughput_img_s = float(len(errors) / max(total_infer_s, 1e-9))

    tail_ratio = float(p95_m / max(median_m, 1e-9))
    mean_over_median = float(mean_m / max(median_m, 1e-9))

    print("\n📊 FINAL REPORT CARD")
    print("=" * 72)
    print(f"Dataset:          {name}")
    print(f"Model:            {model_file}")
    print(f"Device:           {device}")
    print(f"Evaluated:        {len(errors)} images  (requested {len(indices)})")
    print("-" * 72)
    print("📌 Ground-truth coordinate coverage (true labels)")
    print(f"  Lat range:      [{lat_min:.6f}, {lat_max:.6f}]")
    print(f"  Lon range:      [{lon_min:.6f}, {lon_max:.6f}]")
    print(f"  Approx area:    {approx_ns_m:,.1f} m (N–S) × {approx_ew_m:,.1f} m (E–W)")
    if model_bbox is not None:
        m_lat_min, m_lat_max, m_lon_min, m_lon_max = model_bbox
        print("📦 Model bounding box (if your model clamps output)")
        print(f"  Model lat:      [{m_lat_min:.6f}, {m_lat_max:.6f}]")
        print(f"  Model lon:      [{m_lon_min:.6f}, {m_lon_max:.6f}]")
        print(f"  GT outside bbox:{outside_gt:.2f}%")
        print(f"  Clamp rate:     lat={clamp_lat:.2f}% | lon={clamp_lon:.2f}%")

    print("-" * 72)
    print("🎯 Error metrics (meters)")
    print(f"  Mean:           {mean_m:8.2f}")
    print(f"  Median:         {median_m:8.2f}")
    print(f"  RMSE:           {rmse_m:8.2f}")
    print(f"  Std:            {std_m:8.2f}")
    print(f"  P75/P90/P95/P99:{p75_m:6.2f} / {p90_m:6.2f} / {p95_m:6.2f} / {p99_m:6.2f}")
    print("-" * 72)
    print("🏆 Accuracy @ thresholds")
    for t in ACC_THRESHOLDS_M:
        print(f"  @ {t:>3d}m:         {acc[f'@{t}m']:6.2f}%")
    print("-" * 72)
    print("🧭 Directional bias (approx meters; +North / +East)")
    print(f"  Mean dLat/dLon: {mean_dlat_m:+7.2f} m / {mean_dlon_m:+7.2f} m")
    print(f"  Med  dLat/dLon: {median_dlat_m:+7.2f} m / {median_dlon_m:+7.2f} m")
    print("-" * 72)
    print("🦾 Robustness / tail diagnostics")
    print(f"  Mean/Median:    {mean_over_median:6.2f}×   (≫1 means long-tail outliers)")
    print(f"  P95/Median:     {tail_ratio:6.2f}×   (tail heaviness)")
    print("-" * 72)
    print("⏱️  Speed (timed around model.predict / model(batch))")
    print(f"  Avg infer:      {avg_infer_ms:7.2f} ms/img")
    print(f"  Total infer:    {total_infer_s:7.3f} s")
    print(f"  Throughput:     {throughput_img_s:7.2f} img/s")
    print("=" * 72)

    worst_k = min(5, len(errors))
    worst_idx = np.argsort(-errors)[:worst_k]
    print("\n🔍 Worst samples (for manual inspection)")
    for rank, j in enumerate(worst_idx, start=1):
        src = item_ids[j] if item_ids is not None else f"idx={indices[j]}"
        print(
            f"  #{rank}: {errors[j]:8.2f} m | true=({true_coords[j,0]:.6f},{true_coords[j,1]:.6f}) "
            f"pred=({pred_coords[j,0]:.6f},{pred_coords[j,1]:.6f}) | {src}"
        )

    return Metrics(
        dataset_name=name,
        model_file=model_file,
        device=device,
        n_requested=len(indices),
        n_processed=len(errors),
        n_skipped=len(indices) - len(errors),
        batch_size=BATCH_SIZE,
        sample_size=SAMPLE_SIZE,
        lat_min=lat_min,
        lat_max=lat_max,
        lon_min=lon_min,
        lon_max=lon_max,
        approx_ns_m=float(approx_ns_m),
        approx_ew_m=float(approx_ew_m),
        mean_m=mean_m,
        median_m=median_m,
        rmse_m=rmse_m,
        std_m=std_m,
        p75_m=p75_m,
        p90_m=p90_m,
        p95_m=p95_m,
        p99_m=p99_m,
        acc=acc,
        mean_dlat_m=mean_dlat_m,
        mean_dlon_m=mean_dlon_m,
        median_dlat_m=median_dlat_m,
        median_dlon_m=median_dlon_m,
        clamp_rate_lat_pct=clamp_lat,
        clamp_rate_lon_pct=clamp_lon,
        gt_outside_model_bbox_pct=outside_gt,
        avg_infer_ms=avg_infer_ms,
        total_infer_s=total_infer_s,
        throughput_img_s=throughput_img_s,
        tail_ratio_p95_over_median=tail_ratio,
        mean_over_median=mean_over_median,
    )


def hf_image_to_item(image_field: Any) -> Any:
    """
    Convert HF 'image' field into something your model can consume.
    Preference order:
      1) cached file path (string)
      2) numpy RGB array
      3) PIL Image (as-is)
    """
    # decode=False -> {'path': ..., 'bytes': ...}
    if isinstance(image_field, dict):
        p = image_field.get("path")
        if isinstance(p, str) and p and os.path.exists(p):
            return p
        b = image_field.get("bytes")
        if b is not None:
            try:
                from PIL import Image
                import io

                img = Image.open(io.BytesIO(b)).convert("RGB")
                return np.asarray(img)
            except Exception:
                return image_field

    # PIL image
    try:
        from PIL import Image as PILImage
        if isinstance(image_field, PILImage.Image):
            return np.asarray(image_field.convert("RGB"))
    except Exception:
        pass

    return image_field


def main():
    # HF datasets import (lazy)
    try:
        from datasets import load_dataset, Image
    except Exception as e:
        print("❌ Missing dependency 'datasets' (and/or pillow).")
        print("   Install with: pip install -U datasets pillow")
        print(f"   Import error: {e}")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cudnn.benchmark = True

    if not os.path.exists(MODEL_FILE):
        print(f"❌ Could not find '{MODEL_FILE}'. Put your checkpoint next to eval.py.")
        return

    m = load_model_safely(MODEL_FILE, device=device)

    model_bbox = None
    if all(hasattr(m, a) for a in ("lat_min", "lat_max", "lon_min", "lon_max")):
        model_bbox = (float(m.lat_min), float(m.lat_max), float(m.lon_min), float(m.lon_max))

    print("📡 Downloading / loading HF dataset...")
    ds = load_dataset(HF_DATASET, split=HF_SPLIT, cache_dir=HF_CACHE_DIR)

    if USE_IMAGE_PATHS:
        # Make dataset return cached file paths (more like your local path-based pipeline)
        try:
            ds = ds.cast_column(IMAGE_COL, Image(decode=False))
        except Exception as e:
            print(f"   ⚠️ Could not cast image column to decode=False ({e}). Falling back to decoded images.")
            pass

    n_total = len(ds)
    rng = random.Random(SEED)
    if SAMPLE_SIZE == -1 or SAMPLE_SIZE >= n_total:
        indices = list(range(n_total))
        print(f"   Testing on ALL {n_total} images.")
    else:
        indices = rng.sample(range(n_total), SAMPLE_SIZE)
        print(f"   Testing on random {SAMPLE_SIZE} images (seed={SEED}).")

    print("\n🚀 Running Evaluation...")
    pred_list = []
    true_list = []
    infer_times = []
    item_ids = []

    # warmup
    if len(indices) > 0:
        ex0 = ds[indices[0]]
        warm_item = hf_image_to_item(ex0[IMAGE_COL])
        try:
            _ = run_predict(m, [warm_item])
        except Exception:
            pass

    pbar = tqdm(range(0, len(indices), BATCH_SIZE))
    for start in pbar:
        batch_idx = indices[start : start + BATCH_SIZE]

        batch_items = []
        batch_true = []
        batch_ids = []
        for i in batch_idx:
            ex = ds[i]
            batch_items.append(hf_image_to_item(ex[IMAGE_COL]))
            batch_true.append([float(ex[LAT_COL]), float(ex[LON_COL])])
            batch_ids.append(f"hf_idx={i}")

        batch_true = np.asarray(batch_true, dtype=np.float32)

        t0 = time.perf_counter()
        if device == "cuda":
            torch.cuda.synchronize()

        try:
            batch_pred = run_predict(m, batch_items)
        except Exception:
            batch_pred = []
            batch_true_ok = []
            batch_ids_ok = []
            for it, gt, sid in zip(batch_items, batch_true, batch_ids):
                try:
                    one = run_predict(m, [it])[0]
                    batch_pred.append(one)
                    batch_true_ok.append(gt)
                    batch_ids_ok.append(sid)
                except Exception:
                    continue
            if len(batch_pred) == 0:
                continue
            batch_pred = np.stack(batch_pred, axis=0).astype(np.float32)
            batch_true = np.stack(batch_true_ok, axis=0).astype(np.float32)
            batch_ids = batch_ids_ok

        if device == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        infer_times.append(float(t1 - t0))

        pred_list.append(batch_pred)
        true_list.append(batch_true)
        item_ids.extend(batch_ids[: len(batch_pred)])

    if not pred_list:
        print("❌ No images successfully processed.")
        return

    pred = np.concatenate(pred_list, axis=0)
    true = np.concatenate(true_list, axis=0)

    metrics = summarize_and_print(
        name=f"EXTERNAL (HF: {HF_DATASET}/{HF_SPLIT})",
        model_file=MODEL_FILE,
        device=device,
        indices=indices,
        true_coords=true,
        pred_coords=pred,
        infer_times_s=infer_times,
        item_ids=item_ids,
        model_bbox=model_bbox,
    )

    if SAVE_JSON:
        with open(JSON_OUT, "w", encoding="utf-8") as f:
            json.dump(asdict(metrics), f, indent=2)
        print(f"\n💾 Saved metrics JSON -> {JSON_OUT}")


if __name__ == "__main__":
    main()
