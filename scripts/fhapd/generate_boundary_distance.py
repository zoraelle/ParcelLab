#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
generate_boundary_distance.py

功能：
    为遥感地块分割数据集离线生成 boundary map 和 distance map。

适用数据结构：
    FHAPD/
        SC/
            img/
            mask/
            boundary/   # 本脚本自动生成
            dist/       # 本脚本自动生成
        JS/
            img/
            mask/
            boundary/
            dist/
        ...

生成规则：
    严格参考 HBGNet 官方推荐的 BsiNet-torch preprocess.py。

BsiNet 核心算法：
    distance map:
        result = cv2.distanceTransform(
            src=im_data,
            distanceType=cv2.DIST_L2,
            maskSize=3
        )
        scaled_image = ((result - min_value) / (max_value - min_value)) * 255
        result = scaled_image.astype(np.uint8)

    boundary map:
        boundary = cv2.Canny(im_data, 100, 200)

重要说明：
    1. FHAPD 的 mask 当前是 0/1 编码，而 BsiNet 的 Canny 阈值是 100/200。
    2. 如果直接对 0/1 mask 执行 Canny，会导致 boundary 全黑。
    3. 因此本脚本会先把 0/1 mask 转换为 0/255，再执行 BsiNet 原始算法。
    4. 这一步不是修改算法，而是统一标签编码，使 FHAPD mask 符合 BsiNet 预期输入范围。

输出：
    logs/generate_boundary_distance/
        generate_boundary_distance_时间戳.log
        region_stats_时间戳.csv
        sample_stats_时间戳.csv
        error_records_时间戳.csv
        summary_时间戳.json
        run_config_时间戳.json
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime
import csv
import json
import os
import platform
import sys
import time
from collections import defaultdict

import cv2
import numpy as np
from PIL import Image


# ============================================================
# 一、参数配置区
# ============================================================

DATASET_ROOT = Path(os.environ.get("FHAPD_ROOT", "FHAPD"))

MASK_FOLDER_NAME = "mask"
BOUNDARY_FOLDER_NAME = "boundary"
DIST_FOLDER_NAME = "dist"

IMAGE_SUFFIX = ".png"
REPORT_INTERVAL = 5000

# 重要：
# 你前面已经生成过一批错误的全黑 boundary。
# 所以这里必须设置为 True，重新覆盖生成。
OVERWRITE = True

# 是否保存每个样本的详细统计。
SAVE_SAMPLE_STATS = True


# ============================================================
# 二、日志与输出路径
# ============================================================

RUN_TIME = datetime.now().strftime("%Y%m%d_%H%M%S")

LOG_DIR = Path("logs/generate_boundary_distance")
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_PATH = LOG_DIR / f"generate_boundary_distance_{RUN_TIME}.log"
REGION_STATS_CSV = LOG_DIR / f"region_stats_{RUN_TIME}.csv"
SAMPLE_STATS_CSV = LOG_DIR / f"sample_stats_{RUN_TIME}.csv"
ERROR_RECORDS_CSV = LOG_DIR / f"error_records_{RUN_TIME}.csv"
SUMMARY_JSON = LOG_DIR / f"summary_{RUN_TIME}.json"
RUN_CONFIG_JSON = LOG_DIR / f"run_config_{RUN_TIME}.json"


# ============================================================
# 三、全局统计容器
# ============================================================

error_records: list[dict] = []
sample_stats: list[dict] = []

region_stats = defaultdict(
    lambda: {
        "mask_files": 0,
        "generated_boundary": 0,
        "generated_dist": 0,
        "skipped_both_exist": 0,
        "skipped_boundary_exist": 0,
        "skipped_dist_exist": 0,
        "failed": 0,
        "empty_mask": 0,
        "boundary_empty": 0,
        "dist_empty": 0,
        "mask_encoding_01": 0,
        "mask_encoding_0255": 0,
        "mask_encoding_other": 0,
        "elapsed_seconds": 0.0,
    }
)


# ============================================================
# 四、基础工具函数
# ============================================================

def log(message: str = "") -> None:
    """
    同时输出到终端和日志文件。

    参数：
        message: 需要输出的字符串。
    """

    print(message)
    with open(LOG_PATH, "a", encoding="utf-8") as file:
        file.write(message + "\n")


def format_seconds(seconds: float) -> str:
    """
    将秒数转换为更易读的 h/min/s 格式。

    参数：
        seconds: 秒数。

    返回：
        格式化后的字符串。
    """

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
    """
    保存 CSV 文件。

    参数：
        path: CSV 输出路径。
        rows: 字典列表。
        fieldnames: CSV 表头字段。
    """

    with open(path, "w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_json(path: Path, content: dict) -> None:
    """
    保存 JSON 文件。

    参数：
        path: JSON 输出路径。
        content: 需要保存的字典内容。
    """

    with open(path, "w", encoding="utf-8") as file:
        json.dump(content, file, ensure_ascii=False, indent=2)


# ============================================================
# 五、BsiNet / HBGNet 生成逻辑
# ============================================================

def read_mask_as_im_data(mask_path: Path) -> tuple[np.ndarray, str]:
    """
    读取 mask，并统一转换到 BsiNet preprocess.py 预期的 uint8 强度范围。

    BsiNet 原始 preprocess.py 对 im_data 直接执行：
        cv2.Canny(im_data, 100, 200)
        cv2.distanceTransform(im_data, cv2.DIST_L2, 3)

    因此 im_data 必须是 0/255 二值图。
    如果 FHAPD mask 是 0/1，需要转换为 0/255。

    参数：
        mask_path: mask 文件路径。

    返回：
        im_data: uint8 单通道 mask，取值通常为 0/255。
        encoding_type: 原始 mask 编码类型，可能为：
            - "0_1"
            - "0_255"
            - "other"
    """

    mask = np.array(Image.open(mask_path))

    if mask.ndim == 3:
        mask = mask[:, :, 0]

    mask = mask.astype(np.uint8)
    unique_values = set(np.unique(mask).tolist())

    if unique_values.issubset({0, 1}):
        im_data = mask * 255
        encoding_type = "0_1"
    elif unique_values.issubset({0, 255}):
        im_data = mask
        encoding_type = "0_255"
    else:
        # 对异常编码仍然采用非0即前景的方式转成0/255。
        # 这样可以保证 Canny 阈值100/200能够正常工作。
        im_data = (mask > 0).astype(np.uint8) * 255
        encoding_type = "other"

    return im_data, encoding_type


def generate_distance_map_bsinet(im_data: np.ndarray) -> np.ndarray:
    """
    按 BsiNet preprocess.py 生成 distance map。

    原始公式：
        result = cv2.distanceTransform(src=im_data, distanceType=cv2.DIST_L2, maskSize=3)
        min_value = np.min(result)
        max_value = np.max(result)
        scaled_image = ((result - min_value) / (max_value - min_value)) * 255
        result = scaled_image.astype(np.uint8)

    为兼容 empty mask：
        当 max_value == min_value 时，返回全0图。

    参数：
        im_data: 单通道 uint8 mask，取值通常为 0/255。

    返回：
        dist_uint8: uint8 distance map，范围 0~255。
    """

    result = cv2.distanceTransform(
        src=im_data,
        distanceType=cv2.DIST_L2,
        maskSize=3,
    )

    min_value = float(np.min(result))
    max_value = float(np.max(result))

    if max_value > min_value:
        scaled_image = ((result - min_value) / (max_value - min_value)) * 255.0
    else:
        scaled_image = np.zeros_like(result, dtype=np.float32)

    dist_uint8 = scaled_image.astype(np.uint8)

    return dist_uint8


def generate_boundary_map_bsinet(im_data: np.ndarray) -> np.ndarray:
    """
    按 BsiNet preprocess.py 生成 boundary map。

    原始公式：
        boundary = cv2.Canny(im_data, 100, 200)

    参数：
        im_data: 单通道 uint8 mask，取值通常为 0/255。

    返回：
        boundary: uint8 boundary map，通常为 0/255。
    """

    boundary = cv2.Canny(im_data, 100, 200)

    return boundary.astype(np.uint8)


def save_png(array: np.ndarray, path: Path) -> None:
    """
    保存 uint8 PNG 文件。

    参数：
        array: 待保存数组。
        path: 输出路径。
    """

    Image.fromarray(array).save(path)


# ============================================================
# 六、单区域处理逻辑
# ============================================================

def process_region(region_dir: Path) -> None:
    """
    处理单个区域，读取 mask 并生成 boundary/dist。

    参数：
        region_dir: 区域目录，例如 FHAPD/SC。
    """

    region_name = region_dir.name
    region_start = time.time()

    mask_dir = region_dir / MASK_FOLDER_NAME
    boundary_dir = region_dir / BOUNDARY_FOLDER_NAME
    dist_dir = region_dir / DIST_FOLDER_NAME

    log("")
    log("=" * 80)
    log(f"[INFO] Processing Region: {region_name}")
    log("=" * 80)

    if not mask_dir.exists():
        log(f"[WARNING] Skip region {region_name}: mask folder not found: {mask_dir}")
        return

    boundary_dir.mkdir(parents=True, exist_ok=True)
    dist_dir.mkdir(parents=True, exist_ok=True)

    mask_files = sorted(mask_dir.glob(f"*{IMAGE_SUFFIX}"))
    total = len(mask_files)

    region_stats[region_name]["mask_files"] = total

    log(f"[INFO] Mask Dir: {mask_dir}")
    log(f"[INFO] Boundary Dir: {boundary_dir}")
    log(f"[INFO] Dist Dir: {dist_dir}")
    log(f"[INFO] Mask Files: {total}")

    for index, mask_path in enumerate(mask_files, 1):
        boundary_path = boundary_dir / mask_path.name
        dist_path = dist_dir / mask_path.name

        boundary_exists = boundary_path.exists()
        dist_exists = dist_path.exists()

        if not OVERWRITE and boundary_exists and dist_exists:
            region_stats[region_name]["skipped_both_exist"] += 1

            if SAVE_SAMPLE_STATS:
                sample_stats.append(
                    {
                        "region": region_name,
                        "mask_name": mask_path.name,
                        "mask_path": str(mask_path),
                        "boundary_path": str(boundary_path),
                        "dist_path": str(dist_path),
                        "status": "skipped_both_exist",
                        "mask_encoding": "",
                        "is_empty_mask": "",
                        "boundary_empty": "",
                        "dist_empty": "",
                        "boundary_nonzero_pixels": "",
                        "dist_nonzero_pixels": "",
                    }
                )

        else:
            try:
                im_data, encoding_type = read_mask_as_im_data(mask_path)

                if encoding_type == "0_1":
                    region_stats[region_name]["mask_encoding_01"] += 1
                elif encoding_type == "0_255":
                    region_stats[region_name]["mask_encoding_0255"] += 1
                else:
                    region_stats[region_name]["mask_encoding_other"] += 1

                is_empty_mask = int(np.count_nonzero(im_data) == 0)

                if is_empty_mask:
                    region_stats[region_name]["empty_mask"] += 1

                boundary_empty = ""
                dist_empty = ""
                boundary_nonzero_pixels = ""
                dist_nonzero_pixels = ""

                if OVERWRITE or not boundary_exists:
                    boundary = generate_boundary_map_bsinet(im_data)
                    save_png(boundary, boundary_path)
                    region_stats[region_name]["generated_boundary"] += 1

                    boundary_nonzero_pixels = int(np.count_nonzero(boundary))
                    boundary_empty = int(boundary_nonzero_pixels == 0)

                    if boundary_empty:
                        region_stats[region_name]["boundary_empty"] += 1

                else:
                    region_stats[region_name]["skipped_boundary_exist"] += 1

                if OVERWRITE or not dist_exists:
                    dist = generate_distance_map_bsinet(im_data)
                    save_png(dist, dist_path)
                    region_stats[region_name]["generated_dist"] += 1

                    dist_nonzero_pixels = int(np.count_nonzero(dist))
                    dist_empty = int(dist_nonzero_pixels == 0)

                    if dist_empty:
                        region_stats[region_name]["dist_empty"] += 1

                else:
                    region_stats[region_name]["skipped_dist_exist"] += 1

                if SAVE_SAMPLE_STATS:
                    sample_stats.append(
                        {
                            "region": region_name,
                            "mask_name": mask_path.name,
                            "mask_path": str(mask_path),
                            "boundary_path": str(boundary_path),
                            "dist_path": str(dist_path),
                            "status": "processed",
                            "mask_encoding": encoding_type,
                            "is_empty_mask": is_empty_mask,
                            "boundary_empty": boundary_empty,
                            "dist_empty": dist_empty,
                            "boundary_nonzero_pixels": boundary_nonzero_pixels,
                            "dist_nonzero_pixels": dist_nonzero_pixels,
                        }
                    )

            except Exception as error:
                region_stats[region_name]["failed"] += 1

                error_records.append(
                    {
                        "region": region_name,
                        "mask_name": mask_path.name,
                        "mask_path": str(mask_path),
                        "error_type": type(error).__name__,
                        "error_message": str(error),
                    }
                )

        if index % REPORT_INTERVAL == 0 or index == total:
            elapsed = time.time() - region_start
            avg_time = elapsed / index if index > 0 else 0
            eta = avg_time * (total - index)
            percent = (index / total * 100.0) if total > 0 else 100.0

            log(
                f"[INFO] {region_name}: {index}/{total} "
                f"({percent:.2f}%) | "
                f"Elapsed: {format_seconds(elapsed)} | "
                f"ETA: {format_seconds(eta)}"
            )

    elapsed_region = time.time() - region_start
    region_stats[region_name]["elapsed_seconds"] = round(elapsed_region, 2)

    log("")
    log("-" * 80)
    log(f"[SUMMARY] Finished Region: {region_name}")
    log(f"Mask Files: {region_stats[region_name]['mask_files']}")
    log(f"Generated Boundary: {region_stats[region_name]['generated_boundary']}")
    log(f"Generated Dist: {region_stats[region_name]['generated_dist']}")
    log(f"Skipped Both Exist: {region_stats[region_name]['skipped_both_exist']}")
    log(f"Skipped Boundary Exist: {region_stats[region_name]['skipped_boundary_exist']}")
    log(f"Skipped Dist Exist: {region_stats[region_name]['skipped_dist_exist']}")
    log(f"Mask Encoding 0/1: {region_stats[region_name]['mask_encoding_01']}")
    log(f"Mask Encoding 0/255: {region_stats[region_name]['mask_encoding_0255']}")
    log(f"Mask Encoding Other: {region_stats[region_name]['mask_encoding_other']}")
    log(f"Empty Mask: {region_stats[region_name]['empty_mask']}")
    log(f"Boundary Empty: {region_stats[region_name]['boundary_empty']}")
    log(f"Dist Empty: {region_stats[region_name]['dist_empty']}")
    log(f"Failed: {region_stats[region_name]['failed']}")
    log(f"Elapsed: {format_seconds(elapsed_region)}")
    log("-" * 80)


# ============================================================
# 七、结果保存
# ============================================================

def save_reports(total_elapsed: float) -> dict:
    """
    保存 region_stats.csv、sample_stats.csv、error_records.csv、summary.json。

    参数：
        total_elapsed: 总耗时。

    返回：
        summary: 总体统计字典。
    """

    region_rows = []

    for region_name, stats in sorted(region_stats.items()):
        row = {"region": region_name}
        row.update(stats)
        region_rows.append(row)

    write_csv(
        REGION_STATS_CSV,
        region_rows,
        [
            "region",
            "mask_files",
            "generated_boundary",
            "generated_dist",
            "skipped_both_exist",
            "skipped_boundary_exist",
            "skipped_dist_exist",
            "mask_encoding_01",
            "mask_encoding_0255",
            "mask_encoding_other",
            "empty_mask",
            "boundary_empty",
            "dist_empty",
            "failed",
            "elapsed_seconds",
        ],
    )

    if SAVE_SAMPLE_STATS:
        write_csv(
            SAMPLE_STATS_CSV,
            sample_stats,
            [
                "region",
                "mask_name",
                "mask_path",
                "boundary_path",
                "dist_path",
                "status",
                "mask_encoding",
                "is_empty_mask",
                "boundary_empty",
                "dist_empty",
                "boundary_nonzero_pixels",
                "dist_nonzero_pixels",
            ],
        )

    write_csv(
        ERROR_RECORDS_CSV,
        error_records,
        [
            "region",
            "mask_name",
            "mask_path",
            "error_type",
            "error_message",
        ],
    )

    summary = {
        "run_time": RUN_TIME,
        "dataset_root": str(DATASET_ROOT),
        "mask_folder_name": MASK_FOLDER_NAME,
        "boundary_folder_name": BOUNDARY_FOLDER_NAME,
        "dist_folder_name": DIST_FOLDER_NAME,
        "image_suffix": IMAGE_SUFFIX,
        "overwrite": OVERWRITE,
        "save_sample_stats": SAVE_SAMPLE_STATS,
        "total_regions": len(region_stats),
        "total_mask_files": sum(stats["mask_files"] for stats in region_stats.values()),
        "total_generated_boundary": sum(stats["generated_boundary"] for stats in region_stats.values()),
        "total_generated_dist": sum(stats["generated_dist"] for stats in region_stats.values()),
        "total_skipped_both_exist": sum(stats["skipped_both_exist"] for stats in region_stats.values()),
        "total_skipped_boundary_exist": sum(stats["skipped_boundary_exist"] for stats in region_stats.values()),
        "total_skipped_dist_exist": sum(stats["skipped_dist_exist"] for stats in region_stats.values()),
        "total_mask_encoding_01": sum(stats["mask_encoding_01"] for stats in region_stats.values()),
        "total_mask_encoding_0255": sum(stats["mask_encoding_0255"] for stats in region_stats.values()),
        "total_mask_encoding_other": sum(stats["mask_encoding_other"] for stats in region_stats.values()),
        "total_empty_mask": sum(stats["empty_mask"] for stats in region_stats.values()),
        "total_boundary_empty": sum(stats["boundary_empty"] for stats in region_stats.values()),
        "total_dist_empty": sum(stats["dist_empty"] for stats in region_stats.values()),
        "total_failed": sum(stats["failed"] for stats in region_stats.values()),
        "total_elapsed_seconds": round(total_elapsed, 2),
        "log_path": str(LOG_PATH),
        "region_stats_csv": str(REGION_STATS_CSV),
        "sample_stats_csv": str(SAMPLE_STATS_CSV) if SAVE_SAMPLE_STATS else None,
        "error_records_csv": str(ERROR_RECORDS_CSV),
        "summary_json": str(SUMMARY_JSON),
        "run_config_json": str(RUN_CONFIG_JSON),
    }

    save_json(SUMMARY_JSON, summary)

    return summary


def save_run_config() -> None:
    """
    保存本次运行配置，便于复现实验。
    """

    config = {
        "dataset_root": str(DATASET_ROOT),
        "mask_folder_name": MASK_FOLDER_NAME,
        "boundary_folder_name": BOUNDARY_FOLDER_NAME,
        "dist_folder_name": DIST_FOLDER_NAME,
        "image_suffix": IMAGE_SUFFIX,
        "report_interval": REPORT_INTERVAL,
        "overwrite": OVERWRITE,
        "save_sample_stats": SAVE_SAMPLE_STATS,
        "run_time": RUN_TIME,
        "conda_environment": os.environ.get("CONDA_DEFAULT_ENV", "unknown"),
        "python_version": sys.version,
        "platform": platform.platform(),
    }

    save_json(RUN_CONFIG_JSON, config)


# ============================================================
# 八、主函数
# ============================================================

def main() -> None:
    """
    主函数：
        1. 输出运行信息；
        2. 扫描数据集区域；
        3. 逐区域生成 boundary/dist；
        4. 保存日志与统计报告；
        5. 输出 PASS / FAIL。
    """

    start_time = time.time()
    save_run_config()

    log("=" * 80)
    log("Program Name: Generate Boundary and Distance Maps")
    log("=" * 80)
    log("Task: Offline generation of boundary and distance maps for FHAPD")
    log(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Conda Environment: {os.environ.get('CONDA_DEFAULT_ENV', 'unknown')}")
    log(f"Python Version: {sys.version.split()[0]}")
    log(f"Platform: {platform.platform()}")
    log(f"Dataset Root: {DATASET_ROOT}")
    log(f"Mask Folder Name: {MASK_FOLDER_NAME}")
    log(f"Boundary Folder Name: {BOUNDARY_FOLDER_NAME}")
    log(f"Dist Folder Name: {DIST_FOLDER_NAME}")
    log(f"Image Suffix: {IMAGE_SUFFIX}")
    log(f"Overwrite Existing Files: {OVERWRITE}")
    log(f"Save Sample Stats: {SAVE_SAMPLE_STATS}")
    log(f"Report Interval: {REPORT_INTERVAL}")
    log(f"Log Path: {LOG_PATH}")
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

    for region_dir in region_dirs:
        process_region(region_dir)

    total_elapsed = time.time() - start_time
    summary = save_reports(total_elapsed)

    log("")
    log("=" * 80)
    log("[SUMMARY] REGION STATS")
    log("=" * 80)

    for region_name, stats in sorted(region_stats.items()):
        log(
            f"{region_name}: "
            f"mask_files={stats['mask_files']}, "
            f"generated_boundary={stats['generated_boundary']}, "
            f"generated_dist={stats['generated_dist']}, "
            f"skipped_both_exist={stats['skipped_both_exist']}, "
            f"mask_encoding_01={stats['mask_encoding_01']}, "
            f"mask_encoding_0255={stats['mask_encoding_0255']}, "
            f"mask_encoding_other={stats['mask_encoding_other']}, "
            f"empty_mask={stats['empty_mask']}, "
            f"boundary_empty={stats['boundary_empty']}, "
            f"dist_empty={stats['dist_empty']}, "
            f"failed={stats['failed']}"
        )

    log("")
    log("=" * 80)
    log("[SUMMARY] FINAL SUMMARY")
    log("=" * 80)

    for key, value in summary.items():
        log(f"{key}: {value}")

    log("")
    log("=" * 80)

    if summary["total_failed"] == 0:
        log("[PASS] Boundary and distance maps generated successfully.")
    else:
        log("[FAIL] Some samples failed. Please check error_records.csv and log file.")

    log("=" * 80)

    log("")
    log("[INFO] Saved Reports:")
    log(f"Log: {LOG_PATH}")
    log(f"Region Stats CSV: {REGION_STATS_CSV}")
    log(f"Sample Stats CSV: {SAMPLE_STATS_CSV}")
    log(f"Error Records CSV: {ERROR_RECORDS_CSV}")
    log(f"Summary JSON: {SUMMARY_JSON}")
    log(f"Run Config JSON: {RUN_CONFIG_JSON}")


if __name__ == "__main__":
    main()
