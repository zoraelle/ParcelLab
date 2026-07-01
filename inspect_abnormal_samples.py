#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
inspect_abnormal_samples.py

功能：
    对 analyze_boundary_distance.py 找出的异常样本进行进一步诊断。

输入：
    abnormal_samples_*.csv

当前默认读取：
    logs/analyze_boundary_distance/abnormal_samples_20260625_134120.csv

输出：
    logs/inspect_abnormal_samples/
        inspect_abnormal_samples_时间戳.log
        abnormal_diagnosis_时间戳.csv
        abnormal_summary_时间戳.json
        diagnosis_report_时间戳.md
        hist_foreground_pixels_时间戳.png
        hist_largest_component_area_时间戳.png
        hist_distance_max_时间戳.png

异常定义：
    来自上一阶段：
        mask 非空，但 boundary 或 dist 为空。

诊断目标：
    判断这些异常是否由极小地块、细线状区域、单像素碎片等造成。
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime
from collections import Counter, defaultdict
import csv
import json
import os
import platform
import sys
import time

import cv2
import numpy as np
from PIL import Image


# ============================================================
# 一、参数配置区
# ============================================================

ABNORMAL_CSV = Path(
    "logs/analyze_boundary_distance/abnormal_samples_20260625_134120.csv"
)

RUN_TIME = datetime.now().strftime("%Y%m%d_%H%M%S")

OUTPUT_DIR = Path("logs/inspect_abnormal_samples")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LOG_PATH = OUTPUT_DIR / f"inspect_abnormal_samples_{RUN_TIME}.log"
DIAGNOSIS_CSV = OUTPUT_DIR / f"abnormal_diagnosis_{RUN_TIME}.csv"
SUMMARY_JSON = OUTPUT_DIR / f"abnormal_summary_{RUN_TIME}.json"
REPORT_MD = OUTPUT_DIR / f"diagnosis_report_{RUN_TIME}.md"

HIST_FOREGROUND = OUTPUT_DIR / f"hist_foreground_pixels_{RUN_TIME}.png"
HIST_LARGEST_AREA = OUTPUT_DIR / f"hist_largest_component_area_{RUN_TIME}.png"
HIST_DISTANCE_MAX = OUTPUT_DIR / f"hist_distance_max_{RUN_TIME}.png"


# ============================================================
# 二、基础工具函数
# ============================================================

def log(message: str = "") -> None:
    """同时输出到终端和日志文件。"""

    print(message)
    with open(LOG_PATH, "a", encoding="utf-8") as file:
        file.write(message + "\n")


def format_seconds(seconds: float) -> str:
    """把秒数格式化为易读字符串。"""

    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60

    if h > 0:
        return f"{h} h {m} min {s} s"
    if m > 0:
        return f"{m} min {s} s"
    return f"{s} s"


def read_csv(path: Path) -> list[dict]:
    """读取 CSV 文件。"""

    with open(path, "r", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


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


def read_gray(path: Path) -> np.ndarray:
    """读取单通道图像。"""

    arr = np.array(Image.open(path))
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    return arr.astype(np.uint8)


def safe_int(value, default: int = 0) -> int:
    """安全转换整数。"""

    try:
        return int(float(value))
    except Exception:
        return default


def safe_float(value, default: float = 0.0) -> float:
    """安全转换浮点数。"""

    try:
        return float(value)
    except Exception:
        return default


# ============================================================
# 三、统计与诊断函数
# ============================================================

def connected_component_stats(mask: np.ndarray) -> dict:
    """计算 mask 连通域统计。"""

    binary = (mask > 0).astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8,
    )

    component_count = max(num_labels - 1, 0)

    if component_count == 0:
        return {
            "component_count": 0,
            "largest_component_area": 0,
            "smallest_component_area": 0,
            "mean_component_area": 0.0,
            "bbox_x": -1,
            "bbox_y": -1,
            "bbox_w": 0,
            "bbox_h": 0,
        }

    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_index = int(np.argmax(areas)) + 1

    x = int(stats[largest_index, cv2.CC_STAT_LEFT])
    y = int(stats[largest_index, cv2.CC_STAT_TOP])
    w = int(stats[largest_index, cv2.CC_STAT_WIDTH])
    h = int(stats[largest_index, cv2.CC_STAT_HEIGHT])

    return {
        "component_count": int(component_count),
        "largest_component_area": int(areas.max()),
        "smallest_component_area": int(areas.min()),
        "mean_component_area": float(areas.mean()),
        "bbox_x": x,
        "bbox_y": y,
        "bbox_w": w,
        "bbox_h": h,
    }


def diagnose_type(
    mask_nonzero_pixels: int,
    foreground_ratio: float,
    component_count: int,
    largest_component_area: int,
    bbox_w: int,
    bbox_h: int,
    boundary_nonzero_pixels: int,
    dist_nonzero_pixels: int,
    dist_max: int,
) -> str:
    """
    根据几何特征自动判断异常类型。

    类型说明：
        Mask Empty:
            mask 实际为空。

        Full Foreground Mask:
            整张 patch 几乎全部是前景，例如 256×256 全为地块。
            这种情况下图像内部没有背景-前景交界，Canny 可能得到全黑 boundary；
            distanceTransform 在全前景输入下也可能退化为全0。

        Tiny Component:
            前景非常小，最大连通域面积小于 10 像素。

        Very Small Region:
            前景像素总数很少，通常难以形成有效 distance map。

        One-pixel Thin Region:
            最大连通域外接框宽或高为 1，说明区域可能是单像素线。

        Thin / Fragmented Region:
            非空但极细碎，distance map 为空或接近为空。

        Boundary Generation Unexpected:
            从统计上看 mask 正常，但 boundary 为空，需要人工检查。

        Unexpected:
            其它未归类情况。
    """

    if mask_nonzero_pixels == 0:
        return "Mask Empty"

    # 新增：整幅图几乎全是前景。
    # 这里用 foreground_ratio >= 0.999，而不是写死 65536，
    # 是为了兼容不同 patch size。
    if foreground_ratio >= 0.999:
        return "Full Foreground Mask"

    if largest_component_area < 10:
        return "Tiny Component"

    if mask_nonzero_pixels < 30:
        return "Very Small Region"

    if bbox_w <= 1 or bbox_h <= 1:
        return "One-pixel Thin Region"

    if dist_max == 0 or dist_nonzero_pixels == 0:
        return "Thin / Fragmented Region"

    if boundary_nonzero_pixels == 0:
        return "Boundary Generation Unexpected"

    return "Unexpected"
    

def analyze_one(row: dict) -> dict:
    """重新读取异常样本并生成详细诊断结果。"""

    region = row["region"]
    name = row["name"]

    image_path = Path(row["image_path"])
    mask_path = Path(row["mask_path"])
    boundary_path = Path(row["boundary_path"])
    dist_path = Path(row["dist_path"])

    mask = read_gray(mask_path)
    boundary = read_gray(boundary_path)
    dist = read_gray(dist_path)

    mask_nonzero = int(np.count_nonzero(mask))
    boundary_nonzero = int(np.count_nonzero(boundary))
    dist_nonzero = int(np.count_nonzero(dist))

    total_pixels = int(mask.size)
    foreground_ratio = mask_nonzero / total_pixels if total_pixels > 0 else 0.0

    cc = connected_component_stats(mask)

    dist_min = int(dist.min())
    dist_max = int(dist.max())
    dist_mean = float(dist.mean())
    dist_std = float(dist.std())

    abnormal_type = diagnose_type(
        mask_nonzero_pixels=mask_nonzero,
        foreground_ratio=foreground_ratio,
        component_count=cc["component_count"],
        largest_component_area=cc["largest_component_area"],
        bbox_w=cc["bbox_w"],
        bbox_h=cc["bbox_h"],
        boundary_nonzero_pixels=boundary_nonzero,
        dist_nonzero_pixels=dist_nonzero,
        dist_max=dist_max,
    )

    return {
        "region": region,
        "name": name,
        "abnormal_type": abnormal_type,
        "image_path": str(image_path),
        "mask_path": str(mask_path),
        "boundary_path": str(boundary_path),
        "dist_path": str(dist_path),
        "mask_unique": str(np.unique(mask).tolist()[:20]),
        "boundary_unique": str(np.unique(boundary).tolist()[:20]),
        "dist_unique": str(np.unique(dist).tolist()[:20]),
        "mask_nonzero_pixels": mask_nonzero,
        "foreground_ratio": round(foreground_ratio, 8),
        "boundary_nonzero_pixels": boundary_nonzero,
        "dist_nonzero_pixels": dist_nonzero,
        "dist_min": dist_min,
        "dist_max": dist_max,
        "dist_mean": round(dist_mean, 6),
        "dist_std": round(dist_std, 6),
        "component_count": cc["component_count"],
        "largest_component_area": cc["largest_component_area"],
        "smallest_component_area": cc["smallest_component_area"],
        "mean_component_area": round(cc["mean_component_area"], 6),
        "bbox_x": cc["bbox_x"],
        "bbox_y": cc["bbox_y"],
        "bbox_w": cc["bbox_w"],
        "bbox_h": cc["bbox_h"],
    }


def save_histogram(values: list[int | float], title: str, xlabel: str, save_path: Path) -> bool:
    """保存直方图。若 matplotlib 不可用，则跳过。"""

    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 5))
        plt.hist(values, bins=30)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(save_path, dpi=200)
        plt.close()
        return True
    except Exception as error:
        log(f"[WARNING] Failed to save histogram {save_path}: {type(error).__name__}: {error}")
        return False


def write_markdown_report(summary: dict, diagnosis_rows: list[dict]) -> None:
    """生成 Markdown 诊断报告。"""

    type_counter = Counter(row["abnormal_type"] for row in diagnosis_rows)

    with open(REPORT_MD, "w", encoding="utf-8") as file:
        file.write("# Abnormal Sample Diagnosis Report\n\n")

        file.write("## 1. Basic Information\n\n")
        file.write(f"- Run Time: `{summary['run_time']}`\n")
        file.write(f"- Source CSV: `{summary['source_csv']}`\n")
        file.write(f"- Total Abnormal Samples: `{summary['total_abnormal_samples']}`\n")
        file.write(f"- Total Elapsed Seconds: `{summary['total_elapsed_seconds']}`\n\n")

        file.write("## 2. Abnormal Type Statistics\n\n")
        file.write("| Type | Count |\n")
        file.write("|---|---:|\n")
        for abnormal_type, count in type_counter.most_common():
            file.write(f"| {abnormal_type} | {count} |\n")

        file.write("\n## 3. Region Statistics\n\n")
        file.write("| Region | Count |\n")
        file.write("|---|---:|\n")
        for region, count in sorted(summary["abnormal_by_region"].items()):
            file.write(f"| {region} | {count} |\n")

        file.write("\n## 4. Diagnosis Conclusion\n\n")

        if summary["total_abnormal_samples"] == 0:
            file.write("No abnormal samples were detected.\n")
        else:
            dominant_type, dominant_count = type_counter.most_common(1)[0]
            file.write(
                f"A total of `{summary['total_abnormal_samples']}` abnormal samples were detected. "
                f"The dominant abnormal type is `{dominant_type}` with `{dominant_count}` samples.\n\n"
            )
            file.write(
                "These samples should be manually inspected before deciding whether to remove them, "
                "keep them, or treat them as special cases during training.\n"
            )

        file.write("\n## 5. Output Files\n\n")
        file.write(f"- Diagnosis CSV: `{DIAGNOSIS_CSV}`\n")
        file.write(f"- Summary JSON: `{SUMMARY_JSON}`\n")
        file.write(f"- Log File: `{LOG_PATH}`\n")
        file.write(f"- Foreground Histogram: `{HIST_FOREGROUND}`\n")
        file.write(f"- Largest Component Histogram: `{HIST_LARGEST_AREA}`\n")
        file.write(f"- Distance Max Histogram: `{HIST_DISTANCE_MAX}`\n")


# ============================================================
# 四、主函数
# ============================================================

def main() -> None:
    """主入口。"""

    start = time.time()

    log("=" * 80)
    log("Program Name: Inspect Abnormal Samples")
    log("=" * 80)
    log(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Conda Environment: {os.environ.get('CONDA_DEFAULT_ENV', 'unknown')}")
    log(f"Python Version: {sys.version.split()[0]}")
    log(f"Platform: {platform.platform()}")
    log(f"Input Abnormal CSV: {ABNORMAL_CSV}")
    log(f"Output Dir: {OUTPUT_DIR}")
    log("=" * 80)

    if not ABNORMAL_CSV.exists():
        log(f"[FAIL] Abnormal CSV not found: {ABNORMAL_CSV}")
        return

    rows = read_csv(ABNORMAL_CSV)
    total = len(rows)

    log(f"[INFO] Loaded abnormal samples: {total}")

    diagnosis_rows = []
    error_rows = []

    for index, row in enumerate(rows, 1):
        try:
            diagnosis = analyze_one(row)
            diagnosis_rows.append(diagnosis)
        except Exception as error:
            error_rows.append({
                "region": row.get("region", ""),
                "name": row.get("name", ""),
                "error_type": type(error).__name__,
                "error_message": str(error),
            })

        if index % 20 == 0 or index == total:
            elapsed = time.time() - start
            log(f"[INFO] Processed {index}/{total} | Elapsed: {format_seconds(elapsed)}")

    fieldnames = [
        "region",
        "name",
        "abnormal_type",
        "image_path",
        "mask_path",
        "boundary_path",
        "dist_path",
        "mask_unique",
        "boundary_unique",
        "dist_unique",
        "mask_nonzero_pixels",
        "foreground_ratio",
        "boundary_nonzero_pixels",
        "dist_nonzero_pixels",
        "dist_min",
        "dist_max",
        "dist_mean",
        "dist_std",
        "component_count",
        "largest_component_area",
        "smallest_component_area",
        "mean_component_area",
        "bbox_x",
        "bbox_y",
        "bbox_w",
        "bbox_h",
    ]

    write_csv(DIAGNOSIS_CSV, diagnosis_rows, fieldnames)

    foreground_values = [row["mask_nonzero_pixels"] for row in diagnosis_rows]
    largest_area_values = [row["largest_component_area"] for row in diagnosis_rows]
    dist_max_values = [row["dist_max"] for row in diagnosis_rows]

    save_histogram(
        foreground_values,
        "Foreground Pixels of Abnormal Samples",
        "Foreground Pixels",
        HIST_FOREGROUND,
    )

    save_histogram(
        largest_area_values,
        "Largest Component Area of Abnormal Samples",
        "Largest Component Area",
        HIST_LARGEST_AREA,
    )

    save_histogram(
        dist_max_values,
        "Distance Max of Abnormal Samples",
        "Distance Max",
        HIST_DISTANCE_MAX,
    )

    total_elapsed = time.time() - start

    abnormal_by_type = Counter(row["abnormal_type"] for row in diagnosis_rows)
    abnormal_by_region = Counter(row["region"] for row in diagnosis_rows)

    summary = {
        "run_time": RUN_TIME,
        "source_csv": str(ABNORMAL_CSV),
        "total_abnormal_samples": len(diagnosis_rows),
        "error_samples": len(error_rows),
        "abnormal_by_type": dict(abnormal_by_type),
        "abnormal_by_region": dict(abnormal_by_region),
        "min_foreground_pixels": min(foreground_values) if foreground_values else None,
        "max_foreground_pixels": max(foreground_values) if foreground_values else None,
        "min_largest_component_area": min(largest_area_values) if largest_area_values else None,
        "max_largest_component_area": max(largest_area_values) if largest_area_values else None,
        "min_dist_max": min(dist_max_values) if dist_max_values else None,
        "max_dist_max": max(dist_max_values) if dist_max_values else None,
        "total_elapsed_seconds": round(total_elapsed, 2),
        "diagnosis_csv": str(DIAGNOSIS_CSV),
        "summary_json": str(SUMMARY_JSON),
        "diagnosis_report_md": str(REPORT_MD),
        "hist_foreground": str(HIST_FOREGROUND),
        "hist_largest_area": str(HIST_LARGEST_AREA),
        "hist_distance_max": str(HIST_DISTANCE_MAX),
        "log_path": str(LOG_PATH),
    }

    save_json(SUMMARY_JSON, summary)
    write_markdown_report(summary, diagnosis_rows)

    log("")
    log("=" * 80)
    log("[SUMMARY] ABNORMAL TYPE STATS")
    log("=" * 80)
    for abnormal_type, count in abnormal_by_type.most_common():
        log(f"{abnormal_type}: {count}")

    log("")
    log("=" * 80)
    log("[SUMMARY] FINAL SUMMARY")
    log("=" * 80)
    for key, value in summary.items():
        log(f"{key}: {value}")

    log("")
    log("=" * 80)
    if len(error_rows) == 0:
        log("[PASS] Abnormal sample inspection finished successfully.")
    else:
        log("[WARNING] Some abnormal samples failed during inspection.")
    log("=" * 80)

    log("")
    log("[INFO] Saved Files:")
    log(f"Diagnosis CSV: {DIAGNOSIS_CSV}")
    log(f"Summary JSON: {SUMMARY_JSON}")
    log(f"Diagnosis Report: {REPORT_MD}")
    log(f"Histogram Foreground: {HIST_FOREGROUND}")
    log(f"Histogram Largest Area: {HIST_LARGEST_AREA}")
    log(f"Histogram Distance Max: {HIST_DISTANCE_MAX}")
    log(f"Log: {LOG_PATH}")


if __name__ == "__main__":
    main()
