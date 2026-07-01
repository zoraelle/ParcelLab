#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
analyze_fhapd_filename_structure.py

功能：
    分析 FHAPD 数据集文件名结构、离线增强类型、真实独立 patch 数量、
    增强倍率、命名模式和区域差异。

输入数据结构：
    FHAPD/
        CQ/
            img/*.png
            mask/*.png
        GanSu/
            img/*.png
            mask/*.png
        ...

输出：
    logs/analyze_fhapd_filename/
        analyze_fhapd_filename_时间戳.log
        region_summary_时间戳.csv
        augmentation_summary_时间戳.csv
        unique_patch_summary_时间戳.csv
        filename_pattern_summary_时间戳.csv
        base_patch_detail_时间戳.csv
        summary_时间戳.json

用途：
    1. 判断哪些区域存在离线增强。
    2. 统计每个区域的增强类型和数量。
    3. 估算真正独立的原始 patch 数量。
    4. 判断 FHAPD 是否由多种命名体系构成。
    5. 为后续实验划分和论文数据集描述提供依据。
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime
from collections import Counter, defaultdict
import csv
import json
import os
import platform
import re
import sys
import time


# ============================================================
# 一、参数配置区
# ============================================================

DATASET_ROOT = Path(os.environ.get("FHAPD_ROOT", "FHAPD"))

IMAGE_FOLDER_NAME = "img"
MASK_FOLDER_NAME = "mask"
IMAGE_SUFFIX = ".png"

REPORT_INTERVAL = 10000

# 已知增强后缀。
# original 表示无增强后缀。
KNOWN_AUGMENTATION_TAGS = {
    "add",
    "con",
    "dia",
    "gau",
    "hor",
    "vec",
}


# ============================================================
# 二、日志与输出路径
# ============================================================

RUN_TIME = datetime.now().strftime("%Y%m%d_%H%M%S")

LOG_DIR = Path("logs/analyze_fhapd_filename")
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_PATH = LOG_DIR / f"analyze_fhapd_filename_{RUN_TIME}.log"
REGION_SUMMARY_CSV = LOG_DIR / f"region_summary_{RUN_TIME}.csv"
AUGMENTATION_SUMMARY_CSV = LOG_DIR / f"augmentation_summary_{RUN_TIME}.csv"
UNIQUE_PATCH_SUMMARY_CSV = LOG_DIR / f"unique_patch_summary_{RUN_TIME}.csv"
FILENAME_PATTERN_SUMMARY_CSV = LOG_DIR / f"filename_pattern_summary_{RUN_TIME}.csv"
BASE_PATCH_DETAIL_CSV = LOG_DIR / f"base_patch_detail_{RUN_TIME}.csv"
SUMMARY_JSON = LOG_DIR / f"summary_{RUN_TIME}.json"
RUN_CONFIG_JSON = LOG_DIR / f"run_config_{RUN_TIME}.json"


# ============================================================
# 三、基础工具函数
# ============================================================

def log(message: str = "") -> None:
    """同时输出到终端和日志文件。"""
    print(message)
    with open(LOG_PATH, "a", encoding="utf-8") as file:
        file.write(message + "\n")


def format_seconds(seconds: float) -> str:
    """把秒数转换为易读格式。"""
    seconds = int(seconds)
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60

    if hours > 0:
        return f"{hours} h {minutes} min {secs} s"
    if minutes > 0:
        return f"{minutes} min {secs} s"
    return f"{secs} s"


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


# ============================================================
# 四、文件名解析函数
# ============================================================

def remove_suffix(name: str) -> str:
    """
    去掉文件扩展名。

    参数：
        name: 文件名，例如 xxx.png。

    返回：
        stem: 不含扩展名的文件名。
    """
    return Path(name).stem


def parse_filename(stem: str) -> dict:
    """
    解析 FHAPD 文件名，识别增强类型、基础 patch 名和命名模式。

    支持例子：
        0000000009_4_4_0_0_256_256
        0000000009_4_4_add_0_0_256_256
        0000000009_4_4_con_0_0_256_256
        id_4_8_10_gau_0_0_256_256
        189_1_0_add_0_0_256_256
        DB1_000255_000255_0_0_256_256

    逻辑：
        1. 最后四段如果是数字，认为是 crop window：
           x, y, w, h。
        2. crop window 前一段如果是 add/con/dia/gau/hor/vec，
           则认为是增强标签。
        3. 基础 patch 名 = 去掉增强标签后的主体 + crop window。
        4. 命名模式根据主体部分判断。
    """

    parts = stem.split("_")

    crop_x = crop_y = crop_w = crop_h = ""
    has_crop_window = False

    if len(parts) >= 4 and all(part.isdigit() for part in parts[-4:]):
        crop_x, crop_y, crop_w, crop_h = parts[-4:]
        prefix_parts = parts[:-4]
        has_crop_window = True
    else:
        prefix_parts = parts

    augmentation_type = "original"

    if prefix_parts and prefix_parts[-1] in KNOWN_AUGMENTATION_TAGS:
        augmentation_type = prefix_parts[-1]
        base_prefix_parts = prefix_parts[:-1]
    else:
        base_prefix_parts = prefix_parts

    if has_crop_window:
        base_patch_id = "_".join(base_prefix_parts + [crop_x, crop_y, crop_w, crop_h])
    else:
        base_patch_id = "_".join(base_prefix_parts)

    prefix_without_aug = "_".join(base_prefix_parts)

    pattern_type = detect_pattern_type(prefix_without_aug)

    return {
        "stem": stem,
        "augmentation_type": augmentation_type,
        "base_patch_id": base_patch_id,
        "prefix_without_aug": prefix_without_aug,
        "has_crop_window": has_crop_window,
        "crop_x": crop_x,
        "crop_y": crop_y,
        "crop_w": crop_w,
        "crop_h": crop_h,
        "pattern_type": pattern_type,
    }


def detect_pattern_type(prefix_without_aug: str) -> str:
    """
    自动判断文件命名模式。

    返回：
        pattern_type: 命名模式字符串。
    """

    parts = prefix_without_aug.split("_")

    # 示例：0000000009_4_4
    if len(parts) == 3 and parts[0].isdigit() and len(parts[0]) >= 8:
        return "Pattern_A_numeric_scene_row_col"

    # 示例：189_1_0
    if len(parts) == 3 and all(part.isdigit() for part in parts):
        return "Pattern_B_short_numeric_triplet"

    # 示例：id_4_8_10
    if len(parts) == 4 and parts[0] == "id":
        return "Pattern_C_id_triplet"

    # 示例：DB1_000255_000255
    if len(parts) == 3 and re.match(r"^[A-Za-z]+[0-9]+$", parts[0]):
        return "Pattern_D_region_prefix_coords"

    # 示例：SC1_000255_000255
    if len(parts) == 3 and re.match(r"^[A-Za-z]+[0-9]+$", parts[0]):
        return "Pattern_D_region_prefix_coords"

    return "Pattern_Unknown"


# ============================================================
# 五、核心统计函数
# ============================================================

def analyze_region(region_dir: Path) -> dict:
    """
    分析单个区域的文件名结构。

    参数：
        region_dir: 区域目录。

    返回：
        region_result: 单区域统计结果。
    """

    region_start = time.time()
    region = region_dir.name

    img_dir = region_dir / IMAGE_FOLDER_NAME
    mask_dir = region_dir / MASK_FOLDER_NAME

    log("")
    log("=" * 80)
    log(f"[INFO] Analyze Region: {region}")
    log("=" * 80)

    if not img_dir.exists():
        log(f"[WARNING] Missing image folder: {img_dir}")
        image_files = []
    else:
        image_files = sorted(img_dir.glob(f"*{IMAGE_SUFFIX}"))

    if not mask_dir.exists():
        log(f"[WARNING] Missing mask folder: {mask_dir}")
        mask_files = []
    else:
        mask_files = sorted(mask_dir.glob(f"*{IMAGE_SUFFIX}"))

    img_names = {path.name for path in image_files}
    mask_names = {path.name for path in mask_files}

    paired_names = sorted(img_names & mask_names)
    img_without_mask = sorted(img_names - mask_names)
    mask_without_img = sorted(mask_names - img_names)

    augmentation_counter = Counter()
    pattern_counter = Counter()
    base_patch_to_augments: dict[str, set[str]] = defaultdict(set)
    base_patch_to_files: dict[str, list[str]] = defaultdict(list)

    base_patch_detail_rows = []

    total = len(paired_names)

    for index, filename in enumerate(paired_names, 1):
        stem = remove_suffix(filename)
        parsed = parse_filename(stem)

        augmentation_type = parsed["augmentation_type"]
        base_patch_id = parsed["base_patch_id"]
        pattern_type = parsed["pattern_type"]

        augmentation_counter[augmentation_type] += 1
        pattern_counter[pattern_type] += 1
        base_patch_to_augments[base_patch_id].add(augmentation_type)
        base_patch_to_files[base_patch_id].append(filename)

        base_patch_detail_rows.append(
            {
                "region": region,
                "filename": filename,
                "stem": stem,
                "base_patch_id": base_patch_id,
                "augmentation_type": augmentation_type,
                "pattern_type": pattern_type,
                "has_crop_window": parsed["has_crop_window"],
                "crop_x": parsed["crop_x"],
                "crop_y": parsed["crop_y"],
                "crop_w": parsed["crop_w"],
                "crop_h": parsed["crop_h"],
            }
        )

        if index % REPORT_INTERVAL == 0 or index == total:
            elapsed = time.time() - region_start
            percent = index / total * 100 if total else 100
            log(
                f"[INFO] {region}: {index}/{total} "
                f"({percent:.2f}%) | "
                f"Elapsed: {format_seconds(elapsed)}"
            )

    unique_patch_count = len(base_patch_to_augments)
    total_files = len(paired_names)
    augmented_files = total_files - augmentation_counter.get("original", 0)

    if unique_patch_count > 0:
        augmentation_ratio = total_files / unique_patch_count
    else:
        augmentation_ratio = 0.0

    complete_aug_set_counter = Counter(
        tuple(sorted(aug_set)) for aug_set in base_patch_to_augments.values()
    )

    elapsed = time.time() - region_start

    region_result = {
        "region": region,
        "img_files": len(image_files),
        "mask_files": len(mask_files),
        "paired_files": len(paired_names),
        "img_without_mask": len(img_without_mask),
        "mask_without_img": len(mask_without_img),
        "unique_patch_count": unique_patch_count,
        "original_files": augmentation_counter.get("original", 0),
        "augmented_files": augmented_files,
        "augmentation_ratio": round(augmentation_ratio, 4),
        "augmentation_types": "|".join(sorted(augmentation_counter.keys())),
        "pattern_types": "|".join(sorted(pattern_counter.keys())),
        "elapsed_seconds": round(elapsed, 2),
        "augmentation_counter": augmentation_counter,
        "pattern_counter": pattern_counter,
        "complete_aug_set_counter": complete_aug_set_counter,
        "base_patch_detail_rows": base_patch_detail_rows,
    }

    log("")
    log("-" * 80)
    log(f"[SUMMARY] Finished Region: {region}")
    log(f"Image Files: {len(image_files)}")
    log(f"Mask Files: {len(mask_files)}")
    log(f"Paired Files: {len(paired_names)}")
    log(f"Unique Patch Count: {unique_patch_count}")
    log(f"Original Files: {augmentation_counter.get('original', 0)}")
    log(f"Augmented Files: {augmented_files}")
    log(f"Augmentation Ratio: {augmentation_ratio:.4f}")
    log(f"Augmentation Types: {', '.join(sorted(augmentation_counter.keys()))}")
    log(f"Pattern Types: {', '.join(sorted(pattern_counter.keys()))}")
    log(f"Elapsed: {format_seconds(elapsed)}")
    log("-" * 80)

    return region_result


def save_reports(region_results: list[dict], total_elapsed: float) -> dict:
    """
    保存 CSV / JSON 报告。

    参数：
        region_results: 所有区域统计结果。
        total_elapsed: 总耗时。

    返回：
        summary: 总体统计。
    """

    region_summary_rows = []
    augmentation_summary_rows = []
    unique_patch_summary_rows = []
    filename_pattern_rows = []
    base_patch_detail_all = []

    global_aug_counter = Counter()
    global_pattern_counter = Counter()

    for result in region_results:
        region = result["region"]

        region_summary_rows.append(
            {
                "region": region,
                "img_files": result["img_files"],
                "mask_files": result["mask_files"],
                "paired_files": result["paired_files"],
                "img_without_mask": result["img_without_mask"],
                "mask_without_img": result["mask_without_img"],
                "unique_patch_count": result["unique_patch_count"],
                "original_files": result["original_files"],
                "augmented_files": result["augmented_files"],
                "augmentation_ratio": result["augmentation_ratio"],
                "augmentation_types": result["augmentation_types"],
                "pattern_types": result["pattern_types"],
                "elapsed_seconds": result["elapsed_seconds"],
            }
        )

        for aug_type, count in sorted(result["augmentation_counter"].items()):
            augmentation_summary_rows.append(
                {
                    "region": region,
                    "augmentation_type": aug_type,
                    "count": count,
                    "ratio_in_region": round(count / result["paired_files"], 6)
                    if result["paired_files"] else 0,
                }
            )
            global_aug_counter[aug_type] += count

        for pattern_type, count in sorted(result["pattern_counter"].items()):
            filename_pattern_rows.append(
                {
                    "region": region,
                    "pattern_type": pattern_type,
                    "count": count,
                    "ratio_in_region": round(count / result["paired_files"], 6)
                    if result["paired_files"] else 0,
                }
            )
            global_pattern_counter[pattern_type] += count

        for aug_set, count in sorted(result["complete_aug_set_counter"].items()):
            unique_patch_summary_rows.append(
                {
                    "region": region,
                    "augmentation_set": "|".join(aug_set),
                    "unique_patch_count": count,
                }
            )

        base_patch_detail_all.extend(result["base_patch_detail_rows"])

    write_csv(
        REGION_SUMMARY_CSV,
        region_summary_rows,
        [
            "region",
            "img_files",
            "mask_files",
            "paired_files",
            "img_without_mask",
            "mask_without_img",
            "unique_patch_count",
            "original_files",
            "augmented_files",
            "augmentation_ratio",
            "augmentation_types",
            "pattern_types",
            "elapsed_seconds",
        ],
    )

    write_csv(
        AUGMENTATION_SUMMARY_CSV,
        augmentation_summary_rows,
        [
            "region",
            "augmentation_type",
            "count",
            "ratio_in_region",
        ],
    )

    write_csv(
        UNIQUE_PATCH_SUMMARY_CSV,
        unique_patch_summary_rows,
        [
            "region",
            "augmentation_set",
            "unique_patch_count",
        ],
    )

    write_csv(
        FILENAME_PATTERN_SUMMARY_CSV,
        filename_pattern_rows,
        [
            "region",
            "pattern_type",
            "count",
            "ratio_in_region",
        ],
    )

    write_csv(
        BASE_PATCH_DETAIL_CSV,
        base_patch_detail_all,
        [
            "region",
            "filename",
            "stem",
            "base_patch_id",
            "augmentation_type",
            "pattern_type",
            "has_crop_window",
            "crop_x",
            "crop_y",
            "crop_w",
            "crop_h",
        ],
    )

    total_img_files = sum(r["img_files"] for r in region_results)
    total_mask_files = sum(r["mask_files"] for r in region_results)
    total_paired_files = sum(r["paired_files"] for r in region_results)
    total_unique_patches = sum(r["unique_patch_count"] for r in region_results)
    total_original_files = sum(r["original_files"] for r in region_results)
    total_augmented_files = sum(r["augmented_files"] for r in region_results)

    if total_unique_patches > 0:
        overall_augmentation_ratio = total_paired_files / total_unique_patches
    else:
        overall_augmentation_ratio = 0.0

    summary = {
        "run_time": RUN_TIME,
        "dataset_root": str(DATASET_ROOT),
        "total_regions": len(region_results),
        "total_img_files": total_img_files,
        "total_mask_files": total_mask_files,
        "total_paired_files": total_paired_files,
        "total_unique_patches": total_unique_patches,
        "total_original_files": total_original_files,
        "total_augmented_files": total_augmented_files,
        "overall_augmentation_ratio": round(overall_augmentation_ratio, 4),
        "global_augmentation_counter": dict(sorted(global_aug_counter.items())),
        "global_pattern_counter": dict(sorted(global_pattern_counter.items())),
        "region_summary_csv": str(REGION_SUMMARY_CSV),
        "augmentation_summary_csv": str(AUGMENTATION_SUMMARY_CSV),
        "unique_patch_summary_csv": str(UNIQUE_PATCH_SUMMARY_CSV),
        "filename_pattern_summary_csv": str(FILENAME_PATTERN_SUMMARY_CSV),
        "base_patch_detail_csv": str(BASE_PATCH_DETAIL_CSV),
        "summary_json": str(SUMMARY_JSON),
        "run_config_json": str(RUN_CONFIG_JSON),
        "log_path": str(LOG_PATH),
        "total_elapsed_seconds": round(total_elapsed, 2),
    }

    save_json(SUMMARY_JSON, summary)

    return summary


def save_run_config() -> None:
    """保存运行配置。"""
    config = {
        "dataset_root": str(DATASET_ROOT),
        "image_folder_name": IMAGE_FOLDER_NAME,
        "mask_folder_name": MASK_FOLDER_NAME,
        "image_suffix": IMAGE_SUFFIX,
        "known_augmentation_tags": sorted(KNOWN_AUGMENTATION_TAGS),
        "report_interval": REPORT_INTERVAL,
        "run_time": RUN_TIME,
        "conda_environment": os.environ.get("CONDA_DEFAULT_ENV", "unknown"),
        "python_version": sys.version,
        "platform": platform.platform(),
    }
    save_json(RUN_CONFIG_JSON, config)


# ============================================================
# 六、主函数
# ============================================================

def main() -> None:
    """主入口。"""

    start = time.time()
    save_run_config()

    log("=" * 80)
    log("Program Name: FHAPD Filename Structure Analysis")
    log("=" * 80)
    log(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Conda Environment: {os.environ.get('CONDA_DEFAULT_ENV', 'unknown')}")
    log(f"Python Version: {sys.version.split()[0]}")
    log(f"Platform: {platform.platform()}")
    log(f"Dataset Root: {DATASET_ROOT}")
    log(f"Image Folder Name: {IMAGE_FOLDER_NAME}")
    log(f"Mask Folder Name: {MASK_FOLDER_NAME}")
    log(f"Image Suffix: {IMAGE_SUFFIX}")
    log(f"Known Augmentation Tags: {sorted(KNOWN_AUGMENTATION_TAGS)}")
    log("=" * 80)

    if not DATASET_ROOT.exists():
        log(f"[FAIL] Dataset root not found: {DATASET_ROOT}")
        return

    region_dirs = sorted(path for path in DATASET_ROOT.iterdir() if path.is_dir())

    if not region_dirs:
        log(f"[FAIL] No region folders found under: {DATASET_ROOT}")
        return

    log("[INFO] Found Regions:")
    for region_dir in region_dirs:
        log(f"  - {region_dir.name}")

    region_results = []

    for region_dir in region_dirs:
        result = analyze_region(region_dir)
        region_results.append(result)

    total_elapsed = time.time() - start
    summary = save_reports(region_results, total_elapsed)

    log("")
    log("=" * 80)
    log("[SUMMARY] REGION SUMMARY")
    log("=" * 80)

    for result in region_results:
        log(
            f"{result['region']}: "
            f"paired={result['paired_files']}, "
            f"unique_patch={result['unique_patch_count']}, "
            f"original={result['original_files']}, "
            f"augmented={result['augmented_files']}, "
            f"ratio={result['augmentation_ratio']}, "
            f"aug_types={result['augmentation_types']}, "
            f"patterns={result['pattern_types']}"
        )

    log("")
    log("=" * 80)
    log("[SUMMARY] FINAL SUMMARY")
    log("=" * 80)

    for key, value in summary.items():
        log(f"{key}: {value}")

    log("")
    log("=" * 80)
    log("[PASS] FHAPD filename structure analysis finished.")
    log("=" * 80)

    log("")
    log("[INFO] Saved Reports:")
    log(f"Log: {LOG_PATH}")
    log(f"Region Summary CSV: {REGION_SUMMARY_CSV}")
    log(f"Augmentation Summary CSV: {AUGMENTATION_SUMMARY_CSV}")
    log(f"Unique Patch Summary CSV: {UNIQUE_PATCH_SUMMARY_CSV}")
    log(f"Filename Pattern Summary CSV: {FILENAME_PATTERN_SUMMARY_CSV}")
    log(f"Base Patch Detail CSV: {BASE_PATCH_DETAIL_CSV}")
    log(f"Summary JSON: {SUMMARY_JSON}")
    log(f"Run Config JSON: {RUN_CONFIG_JSON}")


if __name__ == "__main__":
    main()
