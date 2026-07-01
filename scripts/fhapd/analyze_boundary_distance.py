#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
analyze_boundary_distance.py

功能：
    分析已经生成好的 boundary / dist 结果，定位异常样本。

重点分析对象：
    1. mask 非空，但 boundary 为空；
    2. mask 非空，但 dist 为空；
    3. boundary / dist 生成异常的样本。

输出：
    logs/analyze_boundary_distance/
        analyze_boundary_distance_时间戳.log
        abnormal_samples_时间戳.csv
        normal_samples_时间戳.csv
        summary_时间戳.json
        debug_samples_时间戳/
            abnormal/
                Region/
                    sample_name/
                        image.png
                        mask.png
                        boundary.png
                        dist.png
                        montage.png
            normal_reference/
                Region/
                    sample_name/
                        image.png
                        mask.png
                        boundary.png
                        dist.png
                        montage.png
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime
import csv
import json
import os
import platform
import random
import shutil
import sys
import time
from collections import defaultdict

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


# ============================================================
# 一、参数配置区
# ============================================================

DATASET_ROOT = Path(os.environ.get("FHAPD_ROOT", "FHAPD"))

IMG_FOLDER_NAME = "img"
MASK_FOLDER_NAME = "mask"
BOUNDARY_FOLDER_NAME = "boundary"
DIST_FOLDER_NAME = "dist"

IMAGE_SUFFIX = ".png"

# 正常样本对照组随机抽样数量
NORMAL_REFERENCE_SAMPLE_NUM = 100

# 是否复制异常样本到 debug 文件夹
COPY_DEBUG_SAMPLES = True

# 是否生成四宫格可视化
GENERATE_MONTAGE = True

RANDOM_SEED = 1234


# ============================================================
# 二、日志与输出路径
# ============================================================

RUN_TIME = datetime.now().strftime("%Y%m%d_%H%M%S")

LOG_DIR = Path("logs/analyze_boundary_distance")
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_PATH = LOG_DIR / f"analyze_boundary_distance_{RUN_TIME}.log"
ABNORMAL_CSV = LOG_DIR / f"abnormal_samples_{RUN_TIME}.csv"
NORMAL_CSV = LOG_DIR / f"normal_samples_{RUN_TIME}.csv"
SUMMARY_JSON = LOG_DIR / f"summary_{RUN_TIME}.json"

DEBUG_ROOT = LOG_DIR / f"debug_samples_{RUN_TIME}"
ABNORMAL_DEBUG_DIR = DEBUG_ROOT / "abnormal"
NORMAL_DEBUG_DIR = DEBUG_ROOT / "normal_reference"


# ============================================================
# 三、基础工具函数
# ============================================================

def log(message: str = "") -> None:
    """同时输出到终端和日志文件。"""
    print(message)
    with open(LOG_PATH, "a", encoding="utf-8") as file:
        file.write(message + "\n")


def format_seconds(seconds: float) -> str:
    """将秒数格式化为易读字符串。"""
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60

    if h > 0:
        return f"{h} h {m} min {s} s"
    if m > 0:
        return f"{m} min {s} s"
    return f"{s} s"


def read_gray(path: Path) -> np.ndarray:
    """读取单通道图像。"""
    arr = np.array(Image.open(path))
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    return arr.astype(np.uint8)


def read_rgb(path: Path) -> Image.Image:
    """读取 RGB 图像。"""
    return Image.open(path).convert("RGB")


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    """保存 CSV。"""
    with open(path, "w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_json(path: Path, content: dict) -> None:
    """保存 JSON。"""
    with open(path, "w", encoding="utf-8") as file:
        json.dump(content, file, ensure_ascii=False, indent=2)


def safe_copy(src: Path, dst: Path) -> None:
    """安全复制文件。"""
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    """将数组归一化到 0~255 uint8，用于可视化。"""
    arr = arr.astype(np.float32)
    min_value = float(arr.min())
    max_value = float(arr.max())

    if max_value > min_value:
        arr = (arr - min_value) / (max_value - min_value) * 255.0
    else:
        arr = np.zeros_like(arr, dtype=np.float32)

    return arr.astype(np.uint8)


def compute_connected_components(mask_arr: np.ndarray) -> dict:
    """统计 mask 连通域信息。"""
    binary = (mask_arr > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    # 去掉背景 label=0
    component_count = max(num_labels - 1, 0)

    if component_count == 0:
        return {
            "component_count": 0,
            "largest_component_area": 0,
            "smallest_component_area": 0,
            "mean_component_area": 0.0,
        }

    areas = stats[1:, cv2.CC_STAT_AREA]

    return {
        "component_count": int(component_count),
        "largest_component_area": int(areas.max()),
        "smallest_component_area": int(areas.min()),
        "mean_component_area": float(areas.mean()),
    }


def create_montage(
    image_path: Path,
    mask_path: Path,
    boundary_path: Path,
    dist_path: Path,
    save_path: Path,
) -> None:
    """生成 image / mask / boundary / dist 四宫格可视化。"""

    image = read_rgb(image_path).resize((256, 256))

    mask = Image.fromarray(read_gray(mask_path)).convert("L").resize((256, 256))
    boundary = Image.fromarray(read_gray(boundary_path)).convert("L").resize((256, 256))
    dist = Image.fromarray(normalize_to_uint8(read_gray(dist_path))).convert("L").resize((256, 256))

    mask_rgb = Image.merge("RGB", (mask, mask, mask))
    boundary_rgb = Image.merge("RGB", (boundary, boundary, boundary))
    dist_rgb = Image.merge("RGB", (dist, dist, dist))

    canvas = Image.new("RGB", (512, 560), "white")
    canvas.paste(image, (0, 30))
    canvas.paste(mask_rgb, (256, 30))
    canvas.paste(boundary_rgb, (0, 304))
    canvas.paste(dist_rgb, (256, 304))

    draw = ImageDraw.Draw(canvas)
    draw.text((10, 8), "Image", fill=(255, 0, 0))
    draw.text((266, 8), "Mask", fill=(255, 0, 0))
    draw.text((10, 282), "Boundary", fill=(255, 0, 0))
    draw.text((266, 282), "Distance", fill=(255, 0, 0))

    save_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(save_path)


# ============================================================
# 四、样本分析
# ============================================================

def analyze_sample(region: str, name: str, image_path: Path, mask_path: Path, boundary_path: Path, dist_path: Path) -> dict:
    """分析单个样本的 mask / boundary / dist 统计信息。"""

    mask_arr = read_gray(mask_path)
    boundary_arr = read_gray(boundary_path)
    dist_arr = read_gray(dist_path)

    mask_nonzero = int(np.count_nonzero(mask_arr))
    boundary_nonzero = int(np.count_nonzero(boundary_arr))
    dist_nonzero = int(np.count_nonzero(dist_arr))

    cc = compute_connected_components(mask_arr)

    return {
        "region": region,
        "name": name,
        "image_path": str(image_path),
        "mask_path": str(mask_path),
        "boundary_path": str(boundary_path),
        "dist_path": str(dist_path),
        "mask_unique": str(np.unique(mask_arr).tolist()[:20]),
        "boundary_unique": str(np.unique(boundary_arr).tolist()[:20]),
        "dist_unique": str(np.unique(dist_arr).tolist()[:20]),
        "mask_nonzero_pixels": mask_nonzero,
        "boundary_nonzero_pixels": boundary_nonzero,
        "dist_nonzero_pixels": dist_nonzero,
        "mask_empty": int(mask_nonzero == 0),
        "boundary_empty": int(boundary_nonzero == 0),
        "dist_empty": int(dist_nonzero == 0),
        "dist_min": int(dist_arr.min()),
        "dist_max": int(dist_arr.max()),
        "dist_mean": float(dist_arr.mean()),
        "component_count": cc["component_count"],
        "largest_component_area": cc["largest_component_area"],
        "smallest_component_area": cc["smallest_component_area"],
        "mean_component_area": cc["mean_component_area"],
    }


def copy_debug_sample(row: dict, target_root: Path) -> None:
    """复制 image/mask/boundary/dist 并生成 montage。"""

    region = row["region"]
    name = row["name"]

    sample_dir = target_root / region / name
    sample_dir.mkdir(parents=True, exist_ok=True)

    image_path = Path(row["image_path"])
    mask_path = Path(row["mask_path"])
    boundary_path = Path(row["boundary_path"])
    dist_path = Path(row["dist_path"])

    safe_copy(image_path, sample_dir / "image.png")
    safe_copy(mask_path, sample_dir / "mask.png")
    safe_copy(boundary_path, sample_dir / "boundary.png")
    safe_copy(dist_path, sample_dir / "dist.png")

    if GENERATE_MONTAGE:
        create_montage(
            image_path=image_path,
            mask_path=mask_path,
            boundary_path=boundary_path,
            dist_path=dist_path,
            save_path=sample_dir / "montage.png",
        )


# ============================================================
# 五、主流程
# ============================================================

def main() -> None:
    """主函数。"""

    random.seed(RANDOM_SEED)
    start_time = time.time()

    log("=" * 80)
    log("Program Name: Analyze Boundary and Distance Maps")
    log("=" * 80)
    log(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Conda Environment: {os.environ.get('CONDA_DEFAULT_ENV', 'unknown')}")
    log(f"Python Version: {sys.version.split()[0]}")
    log(f"Platform: {platform.platform()}")
    log(f"Dataset Root: {DATASET_ROOT}")
    log(f"Debug Root: {DEBUG_ROOT}")
    log(f"Normal Reference Sample Num: {NORMAL_REFERENCE_SAMPLE_NUM}")
    log("=" * 80)

    if not DATASET_ROOT.exists():
        log(f"[FAIL] Dataset root not found: {DATASET_ROOT}")
        return

    all_rows = []
    abnormal_rows = []
    normal_candidate_rows = []
    missing_rows = []

    region_dirs = sorted(p for p in DATASET_ROOT.iterdir() if p.is_dir())

    log("[INFO] Found Regions:")
    for region_dir in region_dirs:
        log(f"  - {region_dir.name}")

    for region_dir in region_dirs:
        region = region_dir.name

        img_dir = region_dir / IMG_FOLDER_NAME
        mask_dir = region_dir / MASK_FOLDER_NAME
        boundary_dir = region_dir / BOUNDARY_FOLDER_NAME
        dist_dir = region_dir / DIST_FOLDER_NAME

        log("")
        log("=" * 80)
        log(f"[INFO] Analyze Region: {region}")
        log("=" * 80)

        mask_files = sorted(mask_dir.glob(f"*{IMAGE_SUFFIX}"))

        for idx, mask_path in enumerate(mask_files, 1):
            name = mask_path.stem

            image_path = img_dir / mask_path.name
            boundary_path = boundary_dir / mask_path.name
            dist_path = dist_dir / mask_path.name

            if not image_path.exists() or not boundary_path.exists() or not dist_path.exists():
                missing_rows.append({
                    "region": region,
                    "name": name,
                    "image_exists": image_path.exists(),
                    "mask_exists": mask_path.exists(),
                    "boundary_exists": boundary_path.exists(),
                    "dist_exists": dist_path.exists(),
                })
                continue

            try:
                row = analyze_sample(region, name, image_path, mask_path, boundary_path, dist_path)
                all_rows.append(row)

                # 异常定义：
                # mask 非空，但是 boundary 或 dist 为空
                if row["mask_empty"] == 0 and (row["boundary_empty"] == 1 or row["dist_empty"] == 1):
                    abnormal_rows.append(row)
                else:
                    normal_candidate_rows.append(row)

            except Exception as error:
                missing_rows.append({
                    "region": region,
                    "name": name,
                    "image_exists": image_path.exists(),
                    "mask_exists": mask_path.exists(),
                    "boundary_exists": boundary_path.exists(),
                    "dist_exists": dist_path.exists(),
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                })

            if idx % 5000 == 0 or idx == len(mask_files):
                log(f"[INFO] {region}: {idx}/{len(mask_files)}")

    normal_reference_rows = random.sample(
        normal_candidate_rows,
        min(NORMAL_REFERENCE_SAMPLE_NUM, len(normal_candidate_rows)),
    )

    log("")
    log("[INFO] Copy debug samples...")

    if COPY_DEBUG_SAMPLES:
        for row in abnormal_rows:
            copy_debug_sample(row, ABNORMAL_DEBUG_DIR)

        for row in normal_reference_rows:
            copy_debug_sample(row, NORMAL_DEBUG_DIR)

    fieldnames = [
        "region",
        "name",
        "image_path",
        "mask_path",
        "boundary_path",
        "dist_path",
        "mask_unique",
        "boundary_unique",
        "dist_unique",
        "mask_nonzero_pixels",
        "boundary_nonzero_pixels",
        "dist_nonzero_pixels",
        "mask_empty",
        "boundary_empty",
        "dist_empty",
        "dist_min",
        "dist_max",
        "dist_mean",
        "component_count",
        "largest_component_area",
        "smallest_component_area",
        "mean_component_area",
    ]

    write_csv(ABNORMAL_CSV, abnormal_rows, fieldnames)
    write_csv(NORMAL_CSV, normal_reference_rows, fieldnames)

    total_elapsed = time.time() - start_time

    abnormal_by_region = defaultdict(int)
    for row in abnormal_rows:
        abnormal_by_region[row["region"]] += 1

    summary = {
        "run_time": RUN_TIME,
        "dataset_root": str(DATASET_ROOT),
        "total_samples_analyzed": len(all_rows),
        "abnormal_samples": len(abnormal_rows),
        "normal_reference_samples": len(normal_reference_rows),
        "missing_or_error_samples": len(missing_rows),
        "abnormal_by_region": dict(sorted(abnormal_by_region.items())),
        "debug_root": str(DEBUG_ROOT),
        "abnormal_csv": str(ABNORMAL_CSV),
        "normal_csv": str(NORMAL_CSV),
        "summary_json": str(SUMMARY_JSON),
        "total_elapsed_seconds": round(total_elapsed, 2),
    }

    save_json(SUMMARY_JSON, summary)

    log("")
    log("=" * 80)
    log("[SUMMARY] FINAL SUMMARY")
    log("=" * 80)
    for k, v in summary.items():
        log(f"{k}: {v}")

    log("")
    log("=" * 80)
    if len(abnormal_rows) == 0:
        log("[PASS] No abnormal samples found.")
    else:
        log("[WARNING] Abnormal samples found. Please inspect debug montage images.")
    log("=" * 80)

    log("")
    log("[INFO] Saved Files:")
    log(f"Abnormal CSV: {ABNORMAL_CSV}")
    log(f"Normal Reference CSV: {NORMAL_CSV}")
    log(f"Summary JSON: {SUMMARY_JSON}")
    log(f"Debug Root: {DEBUG_ROOT}")
    log(f"Log: {LOG_PATH}")


if __name__ == "__main__":
    main()
