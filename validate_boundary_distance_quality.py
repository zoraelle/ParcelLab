#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
validate_boundary_distance_quality.py

功能：
    随机抽样检查 FHAPD 中 image / mask / boundary / dist 的生成质量。

升级版输出 6 宫格：

    Image              Mask
    Boundary           Distance Gray
    Distance Color     Boundary Overlay

目的：
    通过更直观的可视化确认：
        1. boundary 是否沿着 mask 边界生成；
        2. dist 是否符合地块内部距离场；
        3. dist 灰度图是否只是显示不明显；
        4. boundary 是否能正确叠加到原始影像上；
        5. image / mask / boundary / distance 是否空间对应。

输出：
    logs/validate_boundary_distance_quality/
        validate_boundary_distance_quality_时间戳.log
        sampled_samples_时间戳.csv
        summary_时间戳.json
        montage_samples_时间戳/
            Region/
                sample_name_montage.png
                sample_name/
                    image.png
                    mask.png
                    boundary.png
                    dist.png
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

import cv2
import numpy as np
from PIL import Image, ImageDraw


# ============================================================
# 一、参数配置区
# ============================================================

DATASET_ROOT = Path(os.environ.get("FHAPD_ROOT", "FHAPD"))

IMG_FOLDER_NAME = "img"
MASK_FOLDER_NAME = "mask"
BOUNDARY_FOLDER_NAME = "boundary"
DIST_FOLDER_NAME = "dist"

IMAGE_SUFFIX = ".png"

RANDOM_SEED = 20260625
SAMPLE_NUM = 100

# True：每个区域尽量均衡抽样。
# False：从全体样本随机抽样。
BALANCED_BY_REGION = True

# 质量检查 boundary/dist 时，建议跳过空 mask。
SKIP_EMPTY_MASK = True

# 是否复制原始 image/mask/boundary/dist 到样本子文件夹。
COPY_RAW_FILES = True

# 可视化尺寸。FHAPD 是 256×256，这里保持一致。
VIS_SIZE = 256


# ============================================================
# 二、日志与输出路径
# ============================================================

RUN_TIME = datetime.now().strftime("%Y%m%d_%H%M%S")

LOG_DIR = Path("logs/validate_boundary_distance_quality")
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_PATH = LOG_DIR / f"validate_boundary_distance_quality_{RUN_TIME}.log"
SAMPLED_CSV = LOG_DIR / f"sampled_samples_{RUN_TIME}.csv"
SUMMARY_JSON = LOG_DIR / f"summary_{RUN_TIME}.json"
MONTAGE_ROOT = LOG_DIR / f"montage_samples_{RUN_TIME}"


# ============================================================
# 三、基础工具函数
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
    格式化秒数。

    参数：
        seconds: 秒数。

    返回：
        易读时间字符串。
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


def read_gray(path: Path) -> np.ndarray:
    """
    读取灰度图。如果是多通道，则取第一通道。

    参数：
        path: 图像路径。

    返回：
        uint8 单通道数组。
    """

    arr = np.array(Image.open(path))
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    return arr.astype(np.uint8)


def read_rgb(path: Path) -> Image.Image:
    """
    读取 RGB 图像。

    参数：
        path: 图像路径。

    返回：
        PIL RGB 图像。
    """

    return Image.open(path).convert("RGB")


def normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    """
    将数组归一化到 0~255，用于可视化。

    参数：
        arr: 任意数值数组。

    返回：
        uint8 数组。
    """

    arr = arr.astype(np.float32)
    min_value = float(arr.min())
    max_value = float(arr.max())

    if max_value > min_value:
        arr = (arr - min_value) / (max_value - min_value) * 255.0
    else:
        arr = np.zeros_like(arr, dtype=np.float32)

    return arr.astype(np.uint8)


def distance_to_color(dist_arr: np.ndarray) -> Image.Image:
    """
    将 distance map 转换为伪彩色图，便于观察距离梯度。

    参数：
        dist_arr: distance 灰度数组。

    返回：
        PIL RGB 伪彩图。
    """

    dist_uint8 = normalize_to_uint8(dist_arr)

    # OpenCV COLORMAP_JET 输出 BGR，需要转换为 RGB。
    color_bgr = cv2.applyColorMap(dist_uint8, cv2.COLORMAP_JET)
    color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)

    return Image.fromarray(color_rgb).convert("RGB")


def boundary_overlay(image: Image.Image, boundary_arr: np.ndarray) -> Image.Image:
    """
    将 boundary 叠加到原始图像上。

    说明：
        boundary 非零像素用红色显示，用于检查边界是否贴合影像地块边缘。

    参数：
        image: 原始 RGB 图像。
        boundary_arr: boundary 数组。

    返回：
        叠加后的 RGB 图像。
    """

    image_arr = np.array(image.convert("RGB")).copy()

    if boundary_arr.shape[:2] != image_arr.shape[:2]:
        boundary_arr = cv2.resize(
            boundary_arr,
            (image_arr.shape[1], image_arr.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    boundary_mask = boundary_arr > 0

    # 红色边界：[255, 0, 0]
    image_arr[boundary_mask] = [255, 0, 0]

    return Image.fromarray(image_arr).convert("RGB")


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    """
    保存 CSV 文件。

    参数：
        path: 输出 CSV 路径。
        rows: 字典列表。
        fieldnames: 表头字段。
    """

    with open(path, "w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_json(path: Path, content: dict) -> None:
    """
    保存 JSON 文件。

    参数：
        path: 输出 JSON 路径。
        content: 字典内容。
    """

    with open(path, "w", encoding="utf-8") as file:
        json.dump(content, file, ensure_ascii=False, indent=2)


def safe_copy(src: Path, dst: Path) -> None:
    """
    安全复制文件。

    参数：
        src: 源文件路径。
        dst: 目标文件路径。
    """

    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


# ============================================================
# 四、样本发现与抽样
# ============================================================

def discover_samples() -> list[dict]:
    """
    扫描所有 image / mask / boundary / dist 配对样本。

    返回：
        samples: 样本字典列表。
    """

    samples: list[dict] = []

    region_dirs = sorted(path for path in DATASET_ROOT.iterdir() if path.is_dir())

    for region_dir in region_dirs:
        region = region_dir.name

        img_dir = region_dir / IMG_FOLDER_NAME
        mask_dir = region_dir / MASK_FOLDER_NAME
        boundary_dir = region_dir / BOUNDARY_FOLDER_NAME
        dist_dir = region_dir / DIST_FOLDER_NAME

        if (
            not img_dir.exists()
            or not mask_dir.exists()
            or not boundary_dir.exists()
            or not dist_dir.exists()
        ):
            log(f"[WARNING] Skip region {region}: missing img/mask/boundary/dist folder.")
            continue

        mask_files = sorted(mask_dir.glob(f"*{IMAGE_SUFFIX}"))

        for mask_path in mask_files:
            name = mask_path.stem

            image_path = img_dir / mask_path.name
            boundary_path = boundary_dir / mask_path.name
            dist_path = dist_dir / mask_path.name

            if not image_path.exists() or not boundary_path.exists() or not dist_path.exists():
                continue

            if SKIP_EMPTY_MASK:
                mask_arr = read_gray(mask_path)
                if np.count_nonzero(mask_arr) == 0:
                    continue

            samples.append(
                {
                    "region": region,
                    "name": name,
                    "image_path": image_path,
                    "mask_path": mask_path,
                    "boundary_path": boundary_path,
                    "dist_path": dist_path,
                }
            )

    return samples


def sample_records(samples: list[dict]) -> list[dict]:
    """
    根据配置抽取样本。

    参数：
        samples: 全部候选样本。

    返回：
        sampled: 抽样样本列表。
    """

    random.seed(RANDOM_SEED)

    if not BALANCED_BY_REGION:
        return random.sample(samples, min(SAMPLE_NUM, len(samples)))

    region_to_samples: dict[str, list[dict]] = {}

    for sample in samples:
        region_to_samples.setdefault(sample["region"], []).append(sample)

    regions = sorted(region_to_samples.keys())
    per_region = max(SAMPLE_NUM // len(regions), 1)

    sampled: list[dict] = []

    for region in regions:
        candidates = region_to_samples[region]
        take_num = min(per_region, len(candidates))
        sampled.extend(random.sample(candidates, take_num))

    remaining = SAMPLE_NUM - len(sampled)

    if remaining > 0:
        already = {(sample["region"], sample["name"]) for sample in sampled}
        remaining_candidates = [
            sample for sample in samples
            if (sample["region"], sample["name"]) not in already
        ]
        sampled.extend(
            random.sample(
                remaining_candidates,
                min(remaining, len(remaining_candidates)),
            )
        )

    return sampled[:SAMPLE_NUM]


# ============================================================
# 五、可视化生成
# ============================================================

def create_montage(sample: dict, save_path: Path) -> dict:
    """
    生成 6 宫格：
        Image              Mask
        Boundary           Distance Gray
        Distance Color     Boundary Overlay

    参数：
        sample: 样本路径字典。
        save_path: montage 保存路径。

    返回：
        当前样本统计信息。
    """

    image = read_rgb(sample["image_path"]).resize((VIS_SIZE, VIS_SIZE))

    mask_arr = read_gray(sample["mask_path"])
    boundary_arr = read_gray(sample["boundary_path"])
    dist_arr = read_gray(sample["dist_path"])

    mask_vis = normalize_to_uint8(mask_arr)
    boundary_vis = normalize_to_uint8(boundary_arr)
    dist_gray_vis = normalize_to_uint8(dist_arr)

    mask_img = Image.fromarray(mask_vis).convert("RGB").resize((VIS_SIZE, VIS_SIZE))
    boundary_img = Image.fromarray(boundary_vis).convert("RGB").resize((VIS_SIZE, VIS_SIZE))
    dist_gray_img = Image.fromarray(dist_gray_vis).convert("RGB").resize((VIS_SIZE, VIS_SIZE))
    dist_color_img = distance_to_color(dist_arr).resize((VIS_SIZE, VIS_SIZE))
    overlay_img = boundary_overlay(image, boundary_arr).resize((VIS_SIZE, VIS_SIZE))

    # 3 行 × 2 列，每张 256×256，标题高度 30
    title_height = 30
    cell_width = VIS_SIZE
    cell_height = VIS_SIZE + title_height
    canvas_width = cell_width * 2
    canvas_height = cell_height * 3

    canvas = Image.new("RGB", (canvas_width, canvas_height), "white")
    draw = ImageDraw.Draw(canvas)

    panels = [
        ("Image", image, 0, 0),
        ("Mask", mask_img, 1, 0),
        ("Boundary", boundary_img, 0, 1),
        ("Distance Gray", dist_gray_img, 1, 1),
        ("Distance Color", dist_color_img, 0, 2),
        ("Boundary Overlay", overlay_img, 1, 2),
    ]

    for title, panel_img, col, row in panels:
        x = col * cell_width
        y = row * cell_height

        draw.text((x + 10, y + 8), title, fill=(255, 0, 0))
        canvas.paste(panel_img, (x, y + title_height))

    save_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(save_path)

    return {
        "region": sample["region"],
        "name": sample["name"],
        "image_path": str(sample["image_path"]),
        "mask_path": str(sample["mask_path"]),
        "boundary_path": str(sample["boundary_path"]),
        "dist_path": str(sample["dist_path"]),
        "montage_path": str(save_path),
        "mask_nonzero_pixels": int(np.count_nonzero(mask_arr)),
        "boundary_nonzero_pixels": int(np.count_nonzero(boundary_arr)),
        "dist_nonzero_pixels": int(np.count_nonzero(dist_arr)),
        "mask_unique": str(np.unique(mask_arr).tolist()[:20]),
        "boundary_unique": str(np.unique(boundary_arr).tolist()[:20]),
        "dist_min": int(dist_arr.min()),
        "dist_max": int(dist_arr.max()),
        "dist_mean": float(dist_arr.mean()),
        "dist_std": float(dist_arr.std()),
    }


# ============================================================
# 六、主函数
# ============================================================

def main() -> None:
    """
    主入口。
    """

    start_time = time.time()

    log("=" * 80)
    log("Program Name: Validate Boundary and Distance Quality V2")
    log("=" * 80)
    log(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Conda Environment: {os.environ.get('CONDA_DEFAULT_ENV', 'unknown')}")
    log(f"Python Version: {sys.version.split()[0]}")
    log(f"Platform: {platform.platform()}")
    log(f"Dataset Root: {DATASET_ROOT}")
    log(f"Sample Num: {SAMPLE_NUM}")
    log(f"Balanced By Region: {BALANCED_BY_REGION}")
    log(f"Skip Empty Mask: {SKIP_EMPTY_MASK}")
    log(f"Montage Root: {MONTAGE_ROOT}")
    log("=" * 80)

    if not DATASET_ROOT.exists():
        log(f"[FAIL] Dataset root not found: {DATASET_ROOT}")
        return

    log("[INFO] Discovering samples...")
    samples = discover_samples()
    log(f"[INFO] Available non-empty samples: {len(samples)}")

    if not samples:
        log("[FAIL] No valid samples found.")
        return

    sampled = sample_records(samples)
    log(f"[INFO] Sampled samples: {len(sampled)}")

    rows: list[dict] = []

    for index, sample in enumerate(sampled, 1):
        montage_path = MONTAGE_ROOT / sample["region"] / f"{sample['name']}_montage.png"
        row = create_montage(sample, montage_path)
        rows.append(row)

        if COPY_RAW_FILES:
            raw_dir = MONTAGE_ROOT / sample["region"] / sample["name"]
            safe_copy(sample["image_path"], raw_dir / "image.png")
            safe_copy(sample["mask_path"], raw_dir / "mask.png")
            safe_copy(sample["boundary_path"], raw_dir / "boundary.png")
            safe_copy(sample["dist_path"], raw_dir / "dist.png")

        if index % 20 == 0 or index == len(sampled):
            elapsed = time.time() - start_time
            log(
                f"[INFO] Generated montage {index}/{len(sampled)} | "
                f"Elapsed: {format_seconds(elapsed)}"
            )

    write_csv(
        SAMPLED_CSV,
        rows,
        [
            "region",
            "name",
            "image_path",
            "mask_path",
            "boundary_path",
            "dist_path",
            "montage_path",
            "mask_nonzero_pixels",
            "boundary_nonzero_pixels",
            "dist_nonzero_pixels",
            "mask_unique",
            "boundary_unique",
            "dist_min",
            "dist_max",
            "dist_mean",
            "dist_std",
        ],
    )

    total_elapsed = time.time() - start_time

    region_counts: dict[str, int] = {}

    for row in rows:
        region_counts[row["region"]] = region_counts.get(row["region"], 0) + 1

    summary = {
        "run_time": RUN_TIME,
        "dataset_root": str(DATASET_ROOT),
        "available_non_empty_samples": len(samples),
        "sample_num": SAMPLE_NUM,
        "actual_sampled": len(rows),
        "balanced_by_region": BALANCED_BY_REGION,
        "skip_empty_mask": SKIP_EMPTY_MASK,
        "region_counts": region_counts,
        "montage_root": str(MONTAGE_ROOT),
        "sampled_csv": str(SAMPLED_CSV),
        "summary_json": str(SUMMARY_JSON),
        "log_path": str(LOG_PATH),
        "total_elapsed_seconds": round(total_elapsed, 2),
    }

    save_json(SUMMARY_JSON, summary)

    log("")
    log("=" * 80)
    log("[SUMMARY] FINAL SUMMARY")
    log("=" * 80)

    for key, value in summary.items():
        log(f"{key}: {value}")

    log("")
    log("=" * 80)
    log("[PASS] Boundary / distance quality validation V2 samples generated.")
    log("=" * 80)

    log("")
    log("[INFO] Next step:")
    log(f"Open montage folder: {MONTAGE_ROOT}")


if __name__ == "__main__":
    main()
