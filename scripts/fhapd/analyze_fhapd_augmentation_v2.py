#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
analyze_fhapd_augmentation_v2.py

功能：
    在 v1 基础上，进一步反向分析 FHAPD 的增强类型。

重点：
    1. 自动判断 hor / vec / dia 是否对应几何翻转或旋转；
    2. 自动判断 gau 是否更接近 Gaussian Blur；
    3. 输出每种增强最可能的解释；
    4. 保存 CSV / JSON / log。

输出：
    logs/analyze_fhapd_augmentation_v2/
        analyze_fhapd_augmentation_v2_时间戳.log
        augmentation_reverse_v2_时间戳.csv
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
# 一、参数配置区
# ============================================================

DATASET_ROOT = Path(os.environ.get("FHAPD_ROOT", "FHAPD"))

IMG_FOLDER_NAME = "img"
IMAGE_SUFFIX = ".png"

KNOWN_AUGMENTATION_TAGS = {"add", "con", "dia", "gau", "hor", "vec"}

# 每个区域最多分析多少个原始 patch。
# 建议先 200；如果要全量，改为 None。
MAX_BASE_PATCH_PER_REGION = 200

RANDOM_SEED = 20260625
REPORT_INTERVAL = 1000


# ============================================================
# 二、输出路径
# ============================================================

RUN_TIME = datetime.now().strftime("%Y%m%d_%H%M%S")

LOG_DIR = Path("logs/analyze_fhapd_augmentation_v2")
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_PATH = LOG_DIR / f"analyze_fhapd_augmentation_v2_{RUN_TIME}.log"
ANALYSIS_CSV = LOG_DIR / f"augmentation_reverse_v2_{RUN_TIME}.csv"
SUMMARY_JSON = LOG_DIR / f"summary_{RUN_TIME}.json"


# ============================================================
# 三、基础工具函数
# ============================================================

def log(message: str = "") -> None:
    """同时输出到终端和日志文件。"""
    print(message)
    with open(LOG_PATH, "a", encoding="utf-8") as file:
        file.write(message + "\n")


def format_seconds(seconds: float) -> str:
    """格式化秒数。"""
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
    """保存 CSV 文件。"""
    with open(path, "w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_json(path: Path, content: dict) -> None:
    """保存 JSON 文件。"""
    with open(path, "w", encoding="utf-8") as file:
        json.dump(content, file, ensure_ascii=False, indent=2)


def read_rgb_array(path: Path) -> np.ndarray:
    """读取 RGB 图像为 uint8 numpy 数组。"""
    return np.array(Image.open(path).convert("RGB")).astype(np.uint8)


def mse(img1: np.ndarray, img2: np.ndarray) -> float:
    """计算 MSE。"""
    return float(np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2))


def psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """计算 PSNR。"""
    value = mse(img1, img2)
    if value == 0:
        return float("inf")
    return 20 * math.log10(255.0 / math.sqrt(value))


def mean_abs_diff(img1: np.ndarray, img2: np.ndarray) -> float:
    """计算平均绝对差。"""
    return float(np.mean(np.abs(img1.astype(np.int16) - img2.astype(np.int16))))


def max_abs_diff(img1: np.ndarray, img2: np.ndarray) -> int:
    """计算最大绝对差。"""
    return int(np.max(np.abs(img1.astype(np.int16) - img2.astype(np.int16))))


def pixel_equal(img1: np.ndarray, img2: np.ndarray) -> bool:
    """判断两张图是否像素完全一致。"""
    return bool(np.array_equal(img1, img2))


# ============================================================
# 四、文件名解析
# ============================================================

def parse_filename(stem: str) -> dict:
    """
    解析增强类型和 base patch id。
    """

    parts = stem.split("_")

    if len(parts) >= 4 and all(p.isdigit() for p in parts[-4:]):
        crop = parts[-4:]
        prefix_parts = parts[:-4]
        has_crop_window = True
    else:
        crop = []
        prefix_parts = parts
        has_crop_window = False

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
# 五、增强候选变换
# ============================================================

def candidate_geometric_transforms(original: np.ndarray) -> dict[str, np.ndarray]:
    """
    生成常见几何增强候选。

    返回：
        变换名称 -> 变换后的图像。
    """

    return {
        "identity": original,
        "horizontal_flip": np.ascontiguousarray(np.fliplr(original)),
        "vertical_flip": np.ascontiguousarray(np.flipud(original)),
        "rot90_ccw": np.ascontiguousarray(np.rot90(original, 1)),
        "rot180": np.ascontiguousarray(np.rot90(original, 2)),
        "rot270_ccw": np.ascontiguousarray(np.rot90(original, 3)),
        "transpose_main_diagonal": np.ascontiguousarray(np.transpose(original, (1, 0, 2))),
        "anti_diagonal_flip": np.ascontiguousarray(np.fliplr(np.flipud(np.transpose(original, (1, 0, 2))))),
    }


def candidate_gaussian_blurs(original: np.ndarray) -> dict[str, np.ndarray]:
    """
    生成 Gaussian Blur 候选。

    注意：
        如果实际 gau 是高斯噪声，这些候选不会很好匹配；
        如果实际 gau 是高斯模糊，则某个 kernel 的误差应明显最小。
    """

    candidates = {}

    for k in [3, 5, 7, 9]:
        candidates[f"gaussian_blur_{k}x{k}_sigma0"] = cv2.GaussianBlur(original, (k, k), 0)

    for sigma in [0.5, 1.0, 1.5, 2.0]:
        candidates[f"gaussian_blur_5x5_sigma{sigma}"] = cv2.GaussianBlur(original, (5, 5), sigma)

    return candidates


def best_match(original: np.ndarray, target: np.ndarray, candidates: dict[str, np.ndarray]) -> dict:
    """
    从候选变换中找出与 target 最接近的一项。
    """

    best_name = None
    best_mad = None
    best_mse = None
    best_psnr = None
    best_equal = False
    best_max_diff = None

    for name, candidate in candidates.items():
        if candidate.shape != target.shape:
            continue

        current_mad = mean_abs_diff(candidate, target)
        current_mse = mse(candidate, target)
        current_psnr = psnr(candidate, target)
        current_equal = pixel_equal(candidate, target)
        current_max_diff = max_abs_diff(candidate, target)

        if best_mad is None or current_mad < best_mad:
            best_name = name
            best_mad = current_mad
            best_mse = current_mse
            best_psnr = current_psnr
            best_equal = current_equal
            best_max_diff = current_max_diff

    return {
        "best_name": best_name,
        "best_mean_abs_diff": round(best_mad, 6) if best_mad is not None else None,
        "best_mse": round(best_mse, 6) if best_mse is not None else None,
        "best_psnr": "inf" if best_psnr == float("inf") else round(best_psnr, 6) if best_psnr is not None else None,
        "best_pixel_equal": int(best_equal),
        "best_max_abs_diff": best_max_diff,
    }


def estimate_brightness_contrast(original: np.ndarray, target: np.ndarray) -> dict:
    """
    粗略估计 target ≈ alpha * original + beta。

    使用所有 RGB 像素做一元线性拟合。
    """

    x = original.astype(np.float32).reshape(-1)
    y = target.astype(np.float32).reshape(-1)

    x_mean = float(x.mean())
    y_mean = float(y.mean())

    var_x = float(np.var(x))

    if var_x == 0:
        alpha = 0.0
    else:
        alpha = float(np.mean((x - x_mean) * (y - y_mean)) / var_x)

    beta = y_mean - alpha * x_mean

    pred = np.clip(alpha * original.astype(np.float32) + beta, 0, 255).astype(np.uint8)

    return {
        "linear_alpha": round(alpha, 6),
        "linear_beta": round(beta, 6),
        "linear_fit_mad": round(mean_abs_diff(pred, target), 6),
        "linear_fit_psnr": "inf" if pixel_equal(pred, target) else round(psnr(pred, target), 6),
    }


# ============================================================
# 六、单对增强分析
# ============================================================

def infer_possible_operation(aug_type: str, geom_match: dict, gau_match: dict, linear_fit: dict, direct_mad: float) -> str:
    """
    根据统计结果给出可能增强类型。
    """

    if geom_match["best_pixel_equal"] == 1:
        return geom_match["best_name"]

    if aug_type in {"hor", "vec", "dia"}:
        if geom_match["best_mean_abs_diff"] is not None and geom_match["best_mean_abs_diff"] < 1e-6:
            return geom_match["best_name"]
        return f"likely_geometric_{geom_match['best_name']}"

    if aug_type == "gau":
        if gau_match["best_mean_abs_diff"] is not None and gau_match["best_mean_abs_diff"] < direct_mad:
            return f"likely_{gau_match['best_name']}"
        return "likely_gaussian_noise_or_minus_constant"

    if aug_type in {"add", "con"}:
        alpha = linear_fit["linear_alpha"]
        beta = linear_fit["linear_beta"]
        if abs(alpha - 1.0) < 0.02 and abs(beta) < 2.0:
            return "very_weak_intensity_change"
        if abs(alpha - 1.0) >= 0.02:
            return "likely_contrast_change"
        if abs(beta) >= 2.0:
            return "likely_brightness_shift"
        return "weak_photometric_change"

    return "unknown"


def analyze_pair(region: str, base_patch_id: str, aug_type: str, original_path: Path, augmented_path: Path) -> dict:
    """
    分析 original 与增强版本之间的关系。
    """

    original = read_rgb_array(original_path)
    augmented = read_rgb_array(augmented_path)

    if original.shape != augmented.shape:
        return {
            "region": region,
            "base_patch_id": base_patch_id,
            "augmentation_type": aug_type,
            "status": "shape_mismatch",
            "original_path": str(original_path),
            "augmented_path": str(augmented_path),
        }

    direct_equal = pixel_equal(original, augmented)
    direct_mad = mean_abs_diff(original, augmented)
    direct_psnr = psnr(original, augmented)
    direct_max_diff = max_abs_diff(original, augmented)

    geom_match = best_match(
        original,
        augmented,
        candidate_geometric_transforms(original),
    )

    gau_match = best_match(
        original,
        augmented,
        candidate_gaussian_blurs(original),
    )

    linear_fit = estimate_brightness_contrast(original, augmented)

    possible_operation = infer_possible_operation(
        aug_type=aug_type,
        geom_match=geom_match,
        gau_match=gau_match,
        linear_fit=linear_fit,
        direct_mad=direct_mad,
    )

    return {
        "region": region,
        "base_patch_id": base_patch_id,
        "augmentation_type": aug_type,
        "status": "ok",
        "original_path": str(original_path),
        "augmented_path": str(augmented_path),

        "direct_pixel_equal": int(direct_equal),
        "direct_mean_abs_diff": round(direct_mad, 6),
        "direct_max_abs_diff": direct_max_diff,
        "direct_psnr": "inf" if direct_equal else round(direct_psnr, 6),

        "best_geometric_match": geom_match["best_name"],
        "best_geometric_mad": geom_match["best_mean_abs_diff"],
        "best_geometric_max_diff": geom_match["best_max_abs_diff"],
        "best_geometric_psnr": geom_match["best_psnr"],
        "best_geometric_pixel_equal": geom_match["best_pixel_equal"],

        "best_gaussian_match": gau_match["best_name"],
        "best_gaussian_mad": gau_match["best_mean_abs_diff"],
        "best_gaussian_max_diff": gau_match["best_max_abs_diff"],
        "best_gaussian_psnr": gau_match["best_psnr"],
        "best_gaussian_pixel_equal": gau_match["best_pixel_equal"],

        "linear_alpha": linear_fit["linear_alpha"],
        "linear_beta": linear_fit["linear_beta"],
        "linear_fit_mad": linear_fit["linear_fit_mad"],
        "linear_fit_psnr": linear_fit["linear_fit_psnr"],

        "possible_operation": possible_operation,
    }


def analyze_region(region_dir: Path) -> list[dict]:
    """
    分析单个区域。
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

    total_pairs = sum(len(valid_groups[base_id]) - 1 for base_id in base_ids)

    log("")
    log("=" * 80)
    log(f"[INFO] Analyze Region: {region}")
    log(f"[INFO] Groups with augmentation: {len(valid_groups)}")
    log(f"[INFO] Groups selected: {len(base_ids)}")
    log(f"[INFO] Total pairs: {total_pairs}")
    log("=" * 80)

    rows = []
    start = time.time()
    processed = 0

    for base_id in base_ids:
        aug_map = valid_groups[base_id]
        original_path = aug_map["original"]

        for aug_type, augmented_path in sorted(aug_map.items()):
            if aug_type == "original":
                continue

            rows.append(
                analyze_pair(
                    region=region,
                    base_patch_id=base_id,
                    aug_type=aug_type,
                    original_path=original_path,
                    augmented_path=augmented_path,
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
# 七、主函数
# ============================================================

def main() -> None:
    start = time.time()

    log("=" * 80)
    log("Program Name: FHAPD Augmentation Reverse Analysis v2")
    log("=" * 80)
    log(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Conda Environment: {os.environ.get('CONDA_DEFAULT_ENV', 'unknown')}")
    log(f"Python Version: {sys.version.split()[0]}")
    log(f"Platform: {platform.platform()}")
    log(f"Dataset Root: {DATASET_ROOT}")
    log(f"Max Base Patch Per Region: {MAX_BASE_PATCH_PER_REGION}")
    log("=" * 80)

    if not DATASET_ROOT.exists():
        log(f"[FAIL] Dataset root not found: {DATASET_ROOT}")
        return

    all_rows = []

    for region_dir in sorted(path for path in DATASET_ROOT.iterdir() if path.is_dir()):
        all_rows.extend(analyze_region(region_dir))

    fieldnames = [
        "region",
        "base_patch_id",
        "augmentation_type",
        "status",
        "original_path",
        "augmented_path",

        "direct_pixel_equal",
        "direct_mean_abs_diff",
        "direct_max_abs_diff",
        "direct_psnr",

        "best_geometric_match",
        "best_geometric_mad",
        "best_geometric_max_diff",
        "best_geometric_psnr",
        "best_geometric_pixel_equal",

        "best_gaussian_match",
        "best_gaussian_mad",
        "best_gaussian_max_diff",
        "best_gaussian_psnr",
        "best_gaussian_pixel_equal",

        "linear_alpha",
        "linear_beta",
        "linear_fit_mad",
        "linear_fit_psnr",

        "possible_operation",
    ]

    write_csv(ANALYSIS_CSV, all_rows, fieldnames)

    type_counter = Counter(row["augmentation_type"] for row in all_rows)
    operation_counter = Counter(row["possible_operation"] for row in all_rows)

    summary_by_aug_type = {}

    for aug_type in sorted(type_counter.keys()):
        rows = [
            row for row in all_rows
            if row["augmentation_type"] == aug_type and row["status"] == "ok"
        ]

        geom_counter = Counter(row["best_geometric_match"] for row in rows)
        operation_counter_by_type = Counter(row["possible_operation"] for row in rows)

        def mean_value(key: str):
            values = []
            for row in rows:
                value = row[key]
                if value in ["", None, "inf"]:
                    continue
                values.append(float(value))
            return round(float(np.mean(values)), 6) if values else None

        summary_by_aug_type[aug_type] = {
            "count": len(rows),
            "direct_pixel_equal_count": sum(int(row["direct_pixel_equal"]) for row in rows),
            "direct_mean_abs_diff_mean": mean_value("direct_mean_abs_diff"),
            "direct_psnr_mean": mean_value("direct_psnr"),
            "best_geometric_match_counter": dict(geom_counter),
            "possible_operation_counter": dict(operation_counter_by_type),
            "linear_alpha_mean": mean_value("linear_alpha"),
            "linear_beta_mean": mean_value("linear_beta"),
            "linear_fit_mad_mean": mean_value("linear_fit_mad"),
            "best_gaussian_match_counter": dict(Counter(row["best_gaussian_match"] for row in rows)),
            "best_gaussian_mad_mean": mean_value("best_gaussian_mad"),
        }

    total_elapsed = time.time() - start

    summary = {
        "run_time": RUN_TIME,
        "dataset_root": str(DATASET_ROOT),
        "total_pairs_analyzed": len(all_rows),
        "augmentation_type_counter": dict(type_counter),
        "possible_operation_counter": dict(operation_counter),
        "summary_by_aug_type": summary_by_aug_type,
        "analysis_csv": str(ANALYSIS_CSV),
        "summary_json": str(SUMMARY_JSON),
        "log_path": str(LOG_PATH),
        "total_elapsed_seconds": round(total_elapsed, 2),
    }

    save_json(SUMMARY_JSON, summary)

    log("")
    log("=" * 80)
    log("[SUMMARY] AUGMENTATION REVERSE SUMMARY")
    log("=" * 80)

    for aug_type, stats in summary_by_aug_type.items():
        log(f"{aug_type}: {stats}")

    log("")
    log("=" * 80)
    log("[SUMMARY] FINAL SUMMARY")
    log("=" * 80)
    for key, value in summary.items():
        if key != "summary_by_aug_type":
            log(f"{key}: {value}")

    log("")
    log("=" * 80)
    log("[PASS] FHAPD augmentation reverse analysis v2 finished.")
    log("=" * 80)

    log("")
    log("[INFO] Saved Reports:")
    log(f"CSV: {ANALYSIS_CSV}")
    log(f"Summary JSON: {SUMMARY_JSON}")
    log(f"Log: {LOG_PATH}")


if __name__ == "__main__":
    main()
