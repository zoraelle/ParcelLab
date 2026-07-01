#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
identify_exact_augmentation.py

功能：
    进一步反推 FHAPD 中 add / con / gau 的近似增强方式。

输入：
    FHAPD 原始 image 文件夹。

分析对象：
    add
    con
    gau

分析方法：
    对每个 original-augmented pair，尝试多种候选变换：
        1. identity
        2. image + offset
        3. image - offset
        4. brightness alpha
        5. contrast alpha
        6. gamma
        7. gaussian blur
        8. simple Gaussian noise approximation

输出：
    logs/identify_exact_augmentation/
        identify_exact_augmentation_时间戳.log
        exact_augmentation_analysis_时间戳.csv
        summary_时间戳.json
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime
from collections import Counter, defaultdict
import csv
import json
import math
import os
import platform
import random
import sys
import time

import cv2
import numpy as np
from PIL import Image


# ============================================================
# 参数配置区
# ============================================================

DATASET_ROOT = Path(os.environ.get("FHAPD_ROOT", "FHAPD"))

IMG_FOLDER_NAME = "img"
IMAGE_SUFFIX = ".png"

TARGET_AUG_TYPES = {"add", "con", "gau"}
KNOWN_AUGMENTATION_TAGS = {"add", "con", "dia", "gau", "hor", "vec"}

MAX_BASE_PATCH_PER_REGION = 200
RANDOM_SEED = 20260625
REPORT_INTERVAL = 1000


# ============================================================
# 输出路径
# ============================================================

RUN_TIME = datetime.now().strftime("%Y%m%d_%H%M%S")

LOG_DIR = Path("logs/identify_exact_augmentation")
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_PATH = LOG_DIR / f"identify_exact_augmentation_{RUN_TIME}.log"
ANALYSIS_CSV = LOG_DIR / f"exact_augmentation_analysis_{RUN_TIME}.csv"
SUMMARY_JSON = LOG_DIR / f"summary_{RUN_TIME}.json"


# ============================================================
# 工具函数
# ============================================================

def log(message: str = "") -> None:
    print(message)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(message + "\n")


def format_seconds(seconds: float) -> str:
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h:
        return f"{h} h {m} min {s} s"
    if m:
        return f"{m} min {s} s"
    return f"{s} s"


def read_rgb(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB")).astype(np.uint8)


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_json(path: Path, content: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(content, f, ensure_ascii=False, indent=2)


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))


def mad(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a.astype(np.int16) - b.astype(np.int16))))


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    value = mse(a, b)
    if value == 0:
        return float("inf")
    return 20 * math.log10(255.0 / math.sqrt(value))


def to_uint8(arr: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(arr), 0, 255).astype(np.uint8)


# ============================================================
# 文件名解析
# ============================================================

def parse_filename(stem: str) -> dict:
    parts = stem.split("_")

    if len(parts) >= 4 and all(p.isdigit() for p in parts[-4:]):
        crop = parts[-4:]
        prefix_parts = parts[:-4]
    else:
        crop = []
        prefix_parts = parts

    aug_type = "original"

    if prefix_parts and prefix_parts[-1] in KNOWN_AUGMENTATION_TAGS:
        aug_type = prefix_parts[-1]
        base_prefix = prefix_parts[:-1]
    else:
        base_prefix = prefix_parts

    if crop:
        base_patch_id = "_".join(base_prefix + crop)
    else:
        base_patch_id = "_".join(base_prefix)

    return {
        "augmentation_type": aug_type,
        "base_patch_id": base_patch_id,
    }


def collect_region_groups(region_dir: Path) -> dict[str, dict[str, Path]]:
    img_dir = region_dir / IMG_FOLDER_NAME
    groups: dict[str, dict[str, Path]] = defaultdict(dict)

    if not img_dir.exists():
        return groups

    for path in sorted(img_dir.glob(f"*{IMAGE_SUFFIX}")):
        parsed = parse_filename(path.stem)
        groups[parsed["base_patch_id"]][parsed["augmentation_type"]] = path

    return groups


# ============================================================
# 候选增强
# ============================================================

def candidate_transforms(original: np.ndarray) -> dict[str, np.ndarray]:
    """
    构造候选增强结果。
    """

    candidates: dict[str, np.ndarray] = {}

    img = original.astype(np.float32)

    candidates["identity"] = original.copy()

    # 加减常数
    for offset in range(-8, 9):
        if offset == 0:
            continue
        candidates[f"offset_{offset:+d}"] = to_uint8(img + offset)

    # brightness: alpha * image
    for alpha in np.round(np.arange(0.90, 1.111, 0.005), 3):
        candidates[f"brightness_alpha_{alpha:.3f}"] = to_uint8(img * alpha)

    # contrast: (image - mean) * alpha + mean
    mean = img.mean(axis=(0, 1), keepdims=True)
    for alpha in np.round(np.arange(0.90, 1.111, 0.005), 3):
        candidates[f"contrast_alpha_{alpha:.3f}"] = to_uint8((img - mean) * alpha + mean)

    # gamma correction
    norm = img / 255.0
    for gamma in np.round(np.arange(0.80, 1.211, 0.01), 3):
        candidates[f"gamma_{gamma:.3f}"] = to_uint8((norm ** gamma) * 255.0)

    # Gaussian Blur
    for k in [3, 5, 7, 9]:
        candidates[f"gaussian_blur_{k}x{k}_sigma0"] = cv2.GaussianBlur(original, (k, k), 0)

    for sigma in [0.3, 0.5, 0.7, 1.0, 1.5, 2.0]:
        candidates[f"gaussian_blur_5x5_sigma{sigma}"] = cv2.GaussianBlur(original, (5, 5), sigma)

    return candidates


def find_best_candidate(original: np.ndarray, target: np.ndarray) -> dict:
    """
    找到与 target 最接近的候选增强。
    """

    candidates = candidate_transforms(original)

    best_name = ""
    best_mad = None
    best_mse = None
    best_psnr = None
    best_equal = False
    best_max_diff = None

    for name, candidate in candidates.items():
        current_mad = mad(candidate, target)
        current_mse = mse(candidate, target)
        current_psnr = psnr(candidate, target)
        current_equal = bool(np.array_equal(candidate, target))
        current_max_diff = int(np.max(np.abs(candidate.astype(np.int16) - target.astype(np.int16))))

        if best_mad is None or current_mad < best_mad:
            best_name = name
            best_mad = current_mad
            best_mse = current_mse
            best_psnr = current_psnr
            best_equal = current_equal
            best_max_diff = current_max_diff

    direct_mad = mad(original, target)
    direct_psnr = psnr(original, target)
    direct_equal = bool(np.array_equal(original, target))

    diff = target.astype(np.int16) - original.astype(np.int16)

    return {
        "direct_equal": int(direct_equal),
        "direct_mad": round(direct_mad, 6),
        "direct_psnr": "inf" if direct_equal else round(direct_psnr, 6),
        "direct_max_diff": int(np.max(np.abs(diff))),
        "mean_diff_r": round(float(diff[:, :, 0].mean()), 6),
        "mean_diff_g": round(float(diff[:, :, 1].mean()), 6),
        "mean_diff_b": round(float(diff[:, :, 2].mean()), 6),
        "best_candidate": best_name,
        "best_mad": round(float(best_mad), 6),
        "best_mse": round(float(best_mse), 6),
        "best_psnr": "inf" if best_psnr == float("inf") else round(float(best_psnr), 6),
        "best_equal": int(best_equal),
        "best_max_diff": best_max_diff,
    }


def classify_candidate(name: str) -> str:
    if name == "identity":
        return "identity"
    if name.startswith("offset_"):
        return "constant_offset"
    if name.startswith("brightness_alpha_"):
        return "brightness_scale"
    if name.startswith("contrast_alpha_"):
        return "contrast_scale"
    if name.startswith("gamma_"):
        return "gamma_correction"
    if name.startswith("gaussian_blur_"):
        return "gaussian_blur"
    return "unknown"


def analyze_pair(region: str, base_id: str, aug_type: str, original_path: Path, aug_path: Path) -> dict:
    original = read_rgb(original_path)
    target = read_rgb(aug_path)

    if original.shape != target.shape:
        return {
            "region": region,
            "base_patch_id": base_id,
            "augmentation_type": aug_type,
            "status": "shape_mismatch",
            "original_path": str(original_path),
            "augmented_path": str(aug_path),
        }

    result = find_best_candidate(original, target)
    result["best_category"] = classify_candidate(result["best_candidate"])

    row = {
        "region": region,
        "base_patch_id": base_id,
        "augmentation_type": aug_type,
        "status": "ok",
        "original_path": str(original_path),
        "augmented_path": str(aug_path),
    }
    row.update(result)

    return row


def analyze_region(region_dir: Path) -> list[dict]:
    region = region_dir.name
    groups = collect_region_groups(region_dir)

    valid_groups = {
        base_id: aug_map
        for base_id, aug_map in groups.items()
        if "original" in aug_map and any(tag in aug_map for tag in TARGET_AUG_TYPES)
    }

    base_ids = sorted(valid_groups.keys())

    if MAX_BASE_PATCH_PER_REGION is not None and len(base_ids) > MAX_BASE_PATCH_PER_REGION:
        random.seed(RANDOM_SEED)
        base_ids = random.sample(base_ids, MAX_BASE_PATCH_PER_REGION)

    total_pairs = sum(
        1 for base_id in base_ids
        for tag in TARGET_AUG_TYPES
        if tag in valid_groups[base_id]
    )

    log("")
    log("=" * 80)
    log(f"[INFO] Analyze Region: {region}")
    log(f"[INFO] Valid Groups: {len(valid_groups)}")
    log(f"[INFO] Selected Groups: {len(base_ids)}")
    log(f"[INFO] Total Pairs: {total_pairs}")
    log("=" * 80)

    rows = []
    start = time.time()
    processed = 0

    for base_id in base_ids:
        aug_map = valid_groups[base_id]
        original_path = aug_map["original"]

        for aug_type in sorted(TARGET_AUG_TYPES):
            if aug_type not in aug_map:
                continue

            rows.append(
                analyze_pair(
                    region=region,
                    base_id=base_id,
                    aug_type=aug_type,
                    original_path=original_path,
                    aug_path=aug_map[aug_type],
                )
            )

            processed += 1

            if processed % REPORT_INTERVAL == 0 or processed == total_pairs:
                log(
                    f"[INFO] {region}: {processed}/{total_pairs} | "
                    f"Elapsed: {format_seconds(time.time() - start)}"
                )

    return rows


# ============================================================
# 主函数
# ============================================================

def main() -> None:
    start = time.time()

    log("=" * 80)
    log("Program Name: Identify Exact FHAPD Augmentation")
    log("=" * 80)
    log(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Conda Environment: {os.environ.get('CONDA_DEFAULT_ENV', 'unknown')}")
    log(f"Python Version: {sys.version.split()[0]}")
    log(f"Platform: {platform.platform()}")
    log(f"Dataset Root: {DATASET_ROOT}")
    log(f"Target Aug Types: {sorted(TARGET_AUG_TYPES)}")
    log(f"Max Base Patch Per Region: {MAX_BASE_PATCH_PER_REGION}")
    log("=" * 80)

    all_rows = []

    for region_dir in sorted(p for p in DATASET_ROOT.iterdir() if p.is_dir()):
        all_rows.extend(analyze_region(region_dir))

    fieldnames = [
        "region",
        "base_patch_id",
        "augmentation_type",
        "status",
        "original_path",
        "augmented_path",
        "direct_equal",
        "direct_mad",
        "direct_psnr",
        "direct_max_diff",
        "mean_diff_r",
        "mean_diff_g",
        "mean_diff_b",
        "best_candidate",
        "best_category",
        "best_mad",
        "best_mse",
        "best_psnr",
        "best_equal",
        "best_max_diff",
    ]

    write_csv(ANALYSIS_CSV, all_rows, fieldnames)

    rows_ok = [row for row in all_rows if row["status"] == "ok"]

    summary_by_type = {}

    for aug_type in sorted(TARGET_AUG_TYPES):
        rows = [row for row in rows_ok if row["augmentation_type"] == aug_type]

        category_counter = Counter(row["best_category"] for row in rows)
        candidate_counter = Counter(row["best_candidate"] for row in rows)

        def mean_value(key: str):
            vals = []
            for row in rows:
                value = row[key]
                if value in ["", None, "inf"]:
                    continue
                vals.append(float(value))
            return round(float(np.mean(vals)), 6) if vals else None

        summary_by_type[aug_type] = {
            "count": len(rows),
            "direct_equal_count": sum(int(row["direct_equal"]) for row in rows),
            "best_category_counter": dict(category_counter),
            "best_candidate_top10": dict(candidate_counter.most_common(10)),
            "direct_mad_mean": mean_value("direct_mad"),
            "best_mad_mean": mean_value("best_mad"),
            "direct_psnr_mean": mean_value("direct_psnr"),
            "best_psnr_mean": mean_value("best_psnr"),
            "mean_diff_r": mean_value("mean_diff_r"),
            "mean_diff_g": mean_value("mean_diff_g"),
            "mean_diff_b": mean_value("mean_diff_b"),
        }

    summary = {
        "run_time": RUN_TIME,
        "dataset_root": str(DATASET_ROOT),
        "total_pairs": len(all_rows),
        "summary_by_type": summary_by_type,
        "analysis_csv": str(ANALYSIS_CSV),
        "summary_json": str(SUMMARY_JSON),
        "log_path": str(LOG_PATH),
        "elapsed_seconds": round(time.time() - start, 2),
    }

    save_json(SUMMARY_JSON, summary)

    log("")
    log("=" * 80)
    log("[SUMMARY] EXACT AUGMENTATION SUMMARY")
    log("=" * 80)

    for aug_type, stats in summary_by_type.items():
        log(f"{aug_type}: {stats}")

    log("")
    log("=" * 80)
    log("[SUMMARY] FINAL SUMMARY")
    log("=" * 80)
    for key, value in summary.items():
        if key != "summary_by_type":
            log(f"{key}: {value}")

    log("")
    log("=" * 80)
    log("[PASS] Exact augmentation identification finished.")
    log("=" * 80)


if __name__ == "__main__":
    main()
