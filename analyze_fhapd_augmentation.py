#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
analyze_fhapd_augmentation.py

功能：
    分析 FHAPD 中 original / add / con / dia / gau / hor / vec 等增强文件
    与原始 patch 的差异。

当前版本 v1：
    1. 自动识别增强组；
    2. 逐组比较 original 与增强图；
    3. 计算是否像素完全一致；
    4. 计算 Mean Abs Diff / Max Diff / Std Diff；
    5. 计算 PSNR；
    6. 计算 SSIM；
    7. 保存 difference heatmap；
    8. 输出 CSV / JSON / log。

输出：
    logs/analyze_fhapd_augmentation/
        analyze_fhapd_augmentation_时间戳.log
        augmentation_analysis_时间戳.csv
        summary_时间戳.json
        diff_heatmaps_时间戳/
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter
import csv
import json
import math
import os
import platform
import random
import re
import sys
import time

import cv2
import numpy as np
from PIL import Image


# ============================================================
# 一、参数配置区
# ============================================================

DATASET_ROOT = Path(os.environ.get("FHAPD_ROOT", "FHAPD"))

IMG_FOLDER_NAME = "img"
IMAGE_SUFFIX = ".png"

KNOWN_AUGMENTATION_TAGS = {"add", "con", "dia", "gau", "hor", "vec"}

# 每个区域最多分析多少个原始 patch。
# None 表示全量分析。建议先用 100，确认无误后再改 None。
MAX_BASE_PATCH_PER_REGION = 100

RANDOM_SEED = 20260625

SAVE_DIFF_HEATMAP = True
REPORT_INTERVAL = 1000


# ============================================================
# 二、输出路径
# ============================================================

RUN_TIME = datetime.now().strftime("%Y%m%d_%H%M%S")

LOG_DIR = Path("logs/analyze_fhapd_augmentation")
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_PATH = LOG_DIR / f"analyze_fhapd_augmentation_{RUN_TIME}.log"
ANALYSIS_CSV = LOG_DIR / f"augmentation_analysis_{RUN_TIME}.csv"
SUMMARY_JSON = LOG_DIR / f"summary_{RUN_TIME}.json"
DIFF_ROOT = LOG_DIR / f"diff_heatmaps_{RUN_TIME}"


# ============================================================
# 三、工具函数
# ============================================================

def log(message: str = "") -> None:
    print(message)
    with open(LOG_PATH, "a", encoding="utf-8") as file:
        file.write(message + "\n")


def format_seconds(seconds: float) -> str:
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h > 0:
        return f"{h} h {m} min {s} s"
    if m > 0:
        return f"{m} min {s} s"
    return f"{s} s"


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with open(path, "w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_json(path: Path, content: dict) -> None:
    with open(path, "w", encoding="utf-8") as file:
        json.dump(content, file, ensure_ascii=False, indent=2)


def read_rgb_array(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB")).astype(np.uint8)


def psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    mse = np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)
    if mse == 0:
        return float("inf")
    return 20 * math.log10(255.0 / math.sqrt(mse))


def ssim_gray(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    计算灰度 SSIM。
    优先使用 skimage；如果没有，则返回 -1。
    """
    try:
        from skimage.metrics import structural_similarity as ssim

        gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
        return float(ssim(gray1, gray2, data_range=255))
    except Exception:
        return -1.0


def save_diff_heatmap(original: np.ndarray, augmented: np.ndarray, save_path: Path) -> None:
    """
    保存 original 与 augmented 的绝对差异热力图。
    """
    diff = np.abs(original.astype(np.int16) - augmented.astype(np.int16)).astype(np.uint8)
    diff_gray = diff.max(axis=2)

    if diff_gray.max() > diff_gray.min():
        diff_vis = ((diff_gray - diff_gray.min()) / (diff_gray.max() - diff_gray.min()) * 255).astype(np.uint8)
    else:
        diff_vis = np.zeros_like(diff_gray, dtype=np.uint8)

    heat_bgr = cv2.applyColorMap(diff_vis, cv2.COLORMAP_JET)
    heat_rgb = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(heat_rgb).save(save_path)


# ============================================================
# 四、文件名解析
# ============================================================

def parse_filename(stem: str) -> dict:
    """
    解析增强类型和 base patch id。

    例：
        0000000009_4_4_0_0_256_256
        0000000009_4_4_add_0_0_256_256

    base_patch_id 都归为：
        0000000009_4_4_0_0_256_256
    """
    parts = stem.split("_")

    has_crop_window = False
    if len(parts) >= 4 and all(p.isdigit() for p in parts[-4:]):
        crop = parts[-4:]
        prefix_parts = parts[:-4]
        has_crop_window = True
    else:
        crop = []
        prefix_parts = parts

    aug_type = "original"

    if prefix_parts and prefix_parts[-1] in KNOWN_AUGMENTATION_TAGS:
        aug_type = prefix_parts[-1]
        base_prefix = prefix_parts[:-1]
    else:
        base_prefix = prefix_parts

    if has_crop_window:
        base_patch_id = "_".join(base_prefix + crop)
    else:
        base_patch_id = "_".join(base_prefix)

    return {
        "augmentation_type": aug_type,
        "base_patch_id": base_patch_id,
    }


def collect_region_groups(region_dir: Path) -> dict[str, dict[str, Path]]:
    """
    收集一个区域中每个 base patch 对应的 original / aug 文件。
    """
    img_dir = region_dir / IMG_FOLDER_NAME
    groups: dict[str, dict[str, Path]] = defaultdict(dict)

    if not img_dir.exists():
        return groups

    for path in sorted(img_dir.glob(f"*{IMAGE_SUFFIX}")):
        parsed = parse_filename(path.stem)
        groups[parsed["base_patch_id"]][parsed["augmentation_type"]] = path

    return groups


# ============================================================
# 五、增强分析
# ============================================================

def analyze_pair(
    region: str,
    base_patch_id: str,
    aug_type: str,
    original_path: Path,
    augmented_path: Path,
) -> dict:
    """
    分析 original 与某个增强版本之间的差异。
    """
    original = read_rgb_array(original_path)
    augmented = read_rgb_array(augmented_path)

    if original.shape != augmented.shape:
        return {
            "region": region,
            "base_patch_id": base_patch_id,
            "augmentation_type": aug_type,
            "original_path": str(original_path),
            "augmented_path": str(augmented_path),
            "status": "shape_mismatch",
            "shape_original": str(original.shape),
            "shape_augmented": str(augmented.shape),
            "pixel_identical": "",
            "mean_abs_diff": "",
            "max_abs_diff": "",
            "std_abs_diff": "",
            "psnr": "",
            "ssim": "",
            "mean_diff_r": "",
            "mean_diff_g": "",
            "mean_diff_b": "",
            "diff_heatmap_path": "",
        }

    diff = augmented.astype(np.int16) - original.astype(np.int16)
    abs_diff = np.abs(diff)

    pixel_identical = bool(np.array_equal(original, augmented))

    diff_heatmap_path = ""

    if SAVE_DIFF_HEATMAP:
        diff_heatmap_path = (
            DIFF_ROOT
            / region
            / base_patch_id
            / f"{base_patch_id}_{aug_type}_diff.png"
        )
        save_diff_heatmap(original, augmented, diff_heatmap_path)

    return {
        "region": region,
        "base_patch_id": base_patch_id,
        "augmentation_type": aug_type,
        "original_path": str(original_path),
        "augmented_path": str(augmented_path),
        "status": "ok",
        "shape_original": str(original.shape),
        "shape_augmented": str(augmented.shape),
        "pixel_identical": int(pixel_identical),
        "mean_abs_diff": round(float(abs_diff.mean()), 6),
        "max_abs_diff": int(abs_diff.max()),
        "std_abs_diff": round(float(abs_diff.std()), 6),
        "psnr": "inf" if pixel_identical else round(psnr(original, augmented), 6),
        "ssim": round(ssim_gray(original, augmented), 6),
        "mean_diff_r": round(float(diff[:, :, 0].mean()), 6),
        "mean_diff_g": round(float(diff[:, :, 1].mean()), 6),
        "mean_diff_b": round(float(diff[:, :, 2].mean()), 6),
        "diff_heatmap_path": str(diff_heatmap_path),
    }


def analyze_region(region_dir: Path) -> list[dict]:
    """
    分析一个区域的增强文件。
    """
    region = region_dir.name
    groups = collect_region_groups(region_dir)

    valid_groups = {
        base_id: aug_map
        for base_id, aug_map in groups.items()
        if "original" in aug_map and len(aug_map) > 1
    }

    base_ids = sorted(valid_groups.keys())

    if MAX_BASE_PATCH_PER_REGION is not None and len(base_ids) > MAX_BASE_PATCH_PER_REGION:
        random.seed(RANDOM_SEED)
        base_ids = random.sample(base_ids, MAX_BASE_PATCH_PER_REGION)

    rows = []

    log("")
    log("=" * 80)
    log(f"[INFO] Analyze Region: {region}")
    log(f"[INFO] Groups with augmentation: {len(valid_groups)}")
    log(f"[INFO] Groups selected: {len(base_ids)}")
    log("=" * 80)

    start = time.time()

    total_pairs = sum(len(valid_groups[base_id]) - 1 for base_id in base_ids)
    processed = 0

    for base_id in base_ids:
        aug_map = valid_groups[base_id]
        original_path = aug_map["original"]

        for aug_type, augmented_path in sorted(aug_map.items()):
            if aug_type == "original":
                continue

            row = analyze_pair(
                region=region,
                base_patch_id=base_id,
                aug_type=aug_type,
                original_path=original_path,
                augmented_path=augmented_path,
            )
            rows.append(row)
            processed += 1

            if processed % REPORT_INTERVAL == 0 or processed == total_pairs:
                elapsed = time.time() - start
                log(
                    f"[INFO] {region}: {processed}/{total_pairs} pairs | "
                    f"Elapsed: {format_seconds(elapsed)}"
                )

    return rows


# ============================================================
# 六、主函数
# ============================================================

def main() -> None:
    start = time.time()

    log("=" * 80)
    log("Program Name: FHAPD Augmentation Reverse Analysis v1")
    log("=" * 80)
    log(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Conda Environment: {os.environ.get('CONDA_DEFAULT_ENV', 'unknown')}")
    log(f"Python Version: {sys.version.split()[0]}")
    log(f"Platform: {platform.platform()}")
    log(f"Dataset Root: {DATASET_ROOT}")
    log(f"Max Base Patch Per Region: {MAX_BASE_PATCH_PER_REGION}")
    log(f"Save Diff Heatmap: {SAVE_DIFF_HEATMAP}")
    log("=" * 80)

    if not DATASET_ROOT.exists():
        log(f"[FAIL] Dataset root not found: {DATASET_ROOT}")
        return

    region_dirs = sorted(p for p in DATASET_ROOT.iterdir() if p.is_dir())

    all_rows = []

    for region_dir in region_dirs:
        rows = analyze_region(region_dir)
        all_rows.extend(rows)

    fieldnames = [
        "region",
        "base_patch_id",
        "augmentation_type",
        "original_path",
        "augmented_path",
        "status",
        "shape_original",
        "shape_augmented",
        "pixel_identical",
        "mean_abs_diff",
        "max_abs_diff",
        "std_abs_diff",
        "psnr",
        "ssim",
        "mean_diff_r",
        "mean_diff_g",
        "mean_diff_b",
        "diff_heatmap_path",
    ]

    write_csv(ANALYSIS_CSV, all_rows, fieldnames)

    type_counter = Counter(row["augmentation_type"] for row in all_rows)
    identical_counter = Counter(
        row["augmentation_type"]
        for row in all_rows
        if row["pixel_identical"] == 1
    )

    summary_by_type = {}

    for aug_type in sorted(type_counter.keys()):
        rows = [row for row in all_rows if row["augmentation_type"] == aug_type and row["status"] == "ok"]

        def mean_numeric(key: str):
            vals = []
            for row in rows:
                value = row[key]
                if value == "inf" or value == "":
                    continue
                vals.append(float(value))
            return round(float(np.mean(vals)), 6) if vals else None

        summary_by_type[aug_type] = {
            "count": type_counter[aug_type],
            "pixel_identical_count": identical_counter[aug_type],
            "mean_abs_diff": mean_numeric("mean_abs_diff"),
            "max_abs_diff_mean": mean_numeric("max_abs_diff"),
            "std_abs_diff": mean_numeric("std_abs_diff"),
            "psnr_mean": mean_numeric("psnr"),
            "ssim_mean": mean_numeric("ssim"),
            "mean_diff_r": mean_numeric("mean_diff_r"),
            "mean_diff_g": mean_numeric("mean_diff_g"),
            "mean_diff_b": mean_numeric("mean_diff_b"),
        }

    total_elapsed = time.time() - start

    summary = {
        "run_time": RUN_TIME,
        "dataset_root": str(DATASET_ROOT),
        "total_pairs_analyzed": len(all_rows),
        "augmentation_type_counter": dict(type_counter),
        "pixel_identical_counter": dict(identical_counter),
        "summary_by_type": summary_by_type,
        "analysis_csv": str(ANALYSIS_CSV),
        "diff_root": str(DIFF_ROOT),
        "summary_json": str(SUMMARY_JSON),
        "log_path": str(LOG_PATH),
        "total_elapsed_seconds": round(total_elapsed, 2),
    }

    save_json(SUMMARY_JSON, summary)

    log("")
    log("=" * 80)
    log("[SUMMARY] AUGMENTATION TYPE SUMMARY")
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
    log("[PASS] FHAPD augmentation reverse analysis v1 finished.")
    log("=" * 80)

    log("")
    log("[INFO] Saved Reports:")
    log(f"CSV: {ANALYSIS_CSV}")
    log(f"Summary JSON: {SUMMARY_JSON}")
    log(f"Diff Heatmaps: {DIFF_ROOT}")
    log(f"Log: {LOG_PATH}")


if __name__ == "__main__":
    main()
