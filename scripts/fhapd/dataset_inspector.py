from pathlib import Path
from PIL import Image
import numpy as np
from collections import defaultdict
from datetime import datetime
import csv
import json
import time
import platform
import os
import sys


# ============================================================
# 参数配置区
# ============================================================

ROOT = Path(os.environ.get("FHAPD_ROOT", "FHAPD"))
EXPECTED_SIZE = (256, 256)
REPORT_INTERVAL = 5000

RUN_TIME = datetime.now().strftime("%Y%m%d_%H%M%S")

LOG_DIR = Path("logs/dataset_inspector")
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_PATH = LOG_DIR / f"dataset_inspector_{RUN_TIME}.log"
REGION_CSV_PATH = LOG_DIR / f"region_stats_{RUN_TIME}.csv"
SUMMARY_JSON_PATH = LOG_DIR / f"summary_{RUN_TIME}.json"
EMPTY_MASKS_CSV_PATH = LOG_DIR / f"empty_masks_{RUN_TIME}.csv"
ERROR_CSV_PATH = LOG_DIR / f"error_records_{RUN_TIME}.csv"
SAMPLE_STATS_CSV_PATH = LOG_DIR / f"sample_stats_{RUN_TIME}.csv"


# ============================================================
# 全局统计容器
# ============================================================

bad_open = []
bad_pair = []
bad_image_mode = []
bad_image_size = []
bad_mask_size = []
bad_mask_channel = []
bad_mask_values = []
empty_masks = []

sample_stats = []

region_stats = defaultdict(lambda: {
    "img": 0,
    "mask": 0,
    "paired": 0,
    "success": 0,
    "bad_open": 0,
    "bad_pair": 0,
    "bad_image_mode": 0,
    "bad_image_size": 0,
    "bad_mask_size": 0,
    "bad_mask_channel": 0,
    "bad_mask_values": 0,
    "empty_mask": 0,
    "foreground_pixels": 0,
    "total_pixels": 0,
    "foreground_ratio": 0.0,
    "elapsed_seconds": 0.0,
})


# ============================================================
# 基础工具函数
# ============================================================

def log(message: str = ""):
    """同时输出到终端和日志文件。"""
    print(message)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(message + "\n")


def format_seconds(seconds: float) -> str:
    """将秒数格式化为更易读的时间字符串。"""
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60

    if h > 0:
        return f"{h} h {m} min {s} s"
    if m > 0:
        return f"{m} min {s} s"
    return f"{s} s"


def open_image(path: Path):
    """打开图像并真正读取像素，避免只检查文件头。"""
    with Image.open(path) as im:
        im.load()
        return im.copy()


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]):
    """保存 CSV 文件。"""
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# 核心检查函数
# ============================================================

def check_region(region_dir: Path):
    """检查单个区域下 img 与 mask 数据。

    检查内容：
    1. img/mask 是否一一对应；
    2. 图像是否能打开；
    3. 图像是否为 RGB；
    4. image/mask 尺寸是否为 EXPECTED_SIZE；
    5. mask 是否单通道；
    6. mask 像素值是否合法；
    7. mask 是否为空；
    8. 统计 foreground ratio。
    """

    region_start = time.time()
    region = region_dir.name

    img_dir = region_dir / "img"
    mask_dir = region_dir / "mask"

    log("")
    log("=" * 70)
    log(f"[INFO] Checking Region: {region}")
    log("=" * 70)

    if not img_dir.exists() or not mask_dir.exists():
        bad_pair.append((region, "missing_folder", str(region_dir)))
        region_stats[region]["bad_pair"] += 1
        log(f"[ERROR] Missing img or mask folder: {region_dir}")
        return

    img_files = {p.name: p for p in img_dir.glob("*.png")}
    mask_files = {p.name: p for p in mask_dir.glob("*.png")}

    region_stats[region]["img"] = len(img_files)
    region_stats[region]["mask"] = len(mask_files)

    img_without_mask = sorted(set(img_files) - set(mask_files))
    mask_without_img = sorted(set(mask_files) - set(img_files))

    for name in img_without_mask:
        bad_pair.append((region, "img_without_mask", str(img_files[name])))
        region_stats[region]["bad_pair"] += 1

    for name in mask_without_img:
        bad_pair.append((region, "mask_without_img", str(mask_files[name])))
        region_stats[region]["bad_pair"] += 1

    paired_names = sorted(set(img_files) & set(mask_files))
    total = len(paired_names)
    region_stats[region]["paired"] = total

    log(f"[INFO] img files: {len(img_files)}")
    log(f"[INFO] mask files: {len(mask_files)}")
    log(f"[INFO] paired files: {total}")

    for idx, name in enumerate(paired_names, 1):
        img_path = img_files[name]
        mask_path = mask_files[name]

        try:
            img = open_image(img_path)
        except Exception as e:
            bad_open.append((region, "image", str(img_path), type(e).__name__, str(e)))
            region_stats[region]["bad_open"] += 1
            continue

        try:
            mask_img = open_image(mask_path)
        except Exception as e:
            bad_open.append((region, "mask", str(mask_path), type(e).__name__, str(e)))
            region_stats[region]["bad_open"] += 1
            continue

        if img.size != EXPECTED_SIZE:
            bad_image_size.append((region, str(img_path), str(img.size)))
            region_stats[region]["bad_image_size"] += 1

        if mask_img.size != EXPECTED_SIZE:
            bad_mask_size.append((region, str(mask_path), str(mask_img.size)))
            region_stats[region]["bad_mask_size"] += 1

        if img.mode != "RGB":
            bad_image_mode.append((region, str(img_path), img.mode))
            region_stats[region]["bad_image_mode"] += 1

        mask_arr = np.array(mask_img)

        if mask_arr.ndim == 3:
            bad_mask_channel.append((region, str(mask_path), str(mask_arr.shape)))
            region_stats[region]["bad_mask_channel"] += 1
            mask_arr = mask_arr[:, :, 0]

        unique_values = np.unique(mask_arr)
        values_set = set(unique_values.tolist())

        allowed_01 = values_set.issubset({0, 1})
        allowed_0255 = values_set.issubset({0, 255})

        if not (allowed_01 or allowed_0255):
            bad_mask_values.append((region, str(mask_path), str(unique_values[:20].tolist())))
            region_stats[region]["bad_mask_values"] += 1

        foreground_pixels = int(np.count_nonzero(mask_arr))
        total_pixels = int(mask_arr.size)
        foreground_ratio = foreground_pixels / total_pixels if total_pixels > 0 else 0.0

        region_stats[region]["foreground_pixels"] += foreground_pixels
        region_stats[region]["total_pixels"] += total_pixels

        if foreground_pixels == 0:
            empty_masks.append((region, str(mask_path)))
            region_stats[region]["empty_mask"] += 1

        sample_stats.append({
            "region": region,
            "name": name,
            "image_path": str(img_path),
            "mask_path": str(mask_path),
            "image_mode": img.mode,
            "image_size": str(img.size),
            "mask_size": str(mask_img.size),
            "mask_unique_values": str(unique_values[:20].tolist()),
            "foreground_pixels": foreground_pixels,
            "total_pixels": total_pixels,
            "foreground_ratio": round(foreground_ratio, 8),
            "is_empty_mask": foreground_pixels == 0,
        })

        region_stats[region]["success"] += 1

        if idx % REPORT_INTERVAL == 0 or idx == total:
            elapsed = time.time() - region_start
            avg = elapsed / idx
            eta = avg * (total - idx)
            percent = idx / total * 100 if total > 0 else 100

            log(
                f"[INFO] {region}: {idx}/{total} "
                f"({percent:.2f}%) | "
                f"Elapsed: {format_seconds(elapsed)} | "
                f"ETA: {format_seconds(eta)}"
            )

    region_elapsed = time.time() - region_start
    region_stats[region]["elapsed_seconds"] = round(region_elapsed, 2)

    if region_stats[region]["total_pixels"] > 0:
        region_stats[region]["foreground_ratio"] = (
            region_stats[region]["foreground_pixels"] /
            region_stats[region]["total_pixels"]
        )

    log("")
    log("-" * 70)
    log(f"[SUMMARY] Finished Region: {region}")
    log(f"Images: {region_stats[region]['img']}")
    log(f"Masks: {region_stats[region]['mask']}")
    log(f"Paired: {region_stats[region]['paired']}")
    log(f"Success: {region_stats[region]['success']}")
    log(f"Empty Masks: {region_stats[region]['empty_mask']}")
    log(f"Foreground Ratio: {region_stats[region]['foreground_ratio']:.6f}")
    log(f"Bad Open: {region_stats[region]['bad_open']}")
    log(f"Bad Pair: {region_stats[region]['bad_pair']}")
    log(f"Elapsed: {format_seconds(region_elapsed)}")
    log("-" * 70)


# ============================================================
# 报告保存函数
# ============================================================

def save_reports(total_elapsed: float):
    """保存 CSV、JSON 等本地报告。"""

    region_rows = []

    for region, stat in sorted(region_stats.items()):
        row = {"region": region}
        row.update(stat)

        row["empty_mask_ratio"] = (
            round(stat["empty_mask"] / stat["paired"], 6)
            if stat["paired"] > 0 else 0
        )

        row["foreground_ratio"] = round(stat["foreground_ratio"], 8)
        region_rows.append(row)

    write_csv(
        REGION_CSV_PATH,
        region_rows,
        [
            "region",
            "img",
            "mask",
            "paired",
            "success",
            "bad_open",
            "bad_pair",
            "bad_image_mode",
            "bad_image_size",
            "bad_mask_size",
            "bad_mask_channel",
            "bad_mask_values",
            "empty_mask",
            "empty_mask_ratio",
            "foreground_pixels",
            "total_pixels",
            "foreground_ratio",
            "elapsed_seconds",
        ],
    )

    write_csv(
        EMPTY_MASKS_CSV_PATH,
        [{"region": r, "mask_path": p} for r, p in empty_masks],
        ["region", "mask_path"],
    )

    write_csv(
        SAMPLE_STATS_CSV_PATH,
        sample_stats,
        [
            "region",
            "name",
            "image_path",
            "mask_path",
            "image_mode",
            "image_size",
            "mask_size",
            "mask_unique_values",
            "foreground_pixels",
            "total_pixels",
            "foreground_ratio",
            "is_empty_mask",
        ],
    )

    error_rows = []

    for item in bad_open:
        error_rows.append({
            "category": "bad_open",
            "region": item[0],
            "file_type": item[1],
            "path": item[2],
            "error_type": item[3],
            "message": item[4],
        })

    for item in bad_pair:
        error_rows.append({
            "category": "bad_pair",
            "region": item[0],
            "file_type": item[1],
            "path": item[2],
            "error_type": "",
            "message": "",
        })

    for item in bad_image_mode:
        error_rows.append({
            "category": "bad_image_mode",
            "region": item[0],
            "file_type": "image",
            "path": item[1],
            "error_type": "WrongImageMode",
            "message": item[2],
        })

    for item in bad_image_size:
        error_rows.append({
            "category": "bad_image_size",
            "region": item[0],
            "file_type": "image",
            "path": item[1],
            "error_type": "WrongImageSize",
            "message": item[2],
        })

    for item in bad_mask_size:
        error_rows.append({
            "category": "bad_mask_size",
            "region": item[0],
            "file_type": "mask",
            "path": item[1],
            "error_type": "WrongMaskSize",
            "message": item[2],
        })

    for item in bad_mask_channel:
        error_rows.append({
            "category": "bad_mask_channel",
            "region": item[0],
            "file_type": "mask",
            "path": item[1],
            "error_type": "WrongMaskChannel",
            "message": item[2],
        })

    for item in bad_mask_values:
        error_rows.append({
            "category": "bad_mask_values",
            "region": item[0],
            "file_type": "mask",
            "path": item[1],
            "error_type": "InvalidMaskValues",
            "message": item[2],
        })

    write_csv(
        ERROR_CSV_PATH,
        error_rows,
        ["category", "region", "file_type", "path", "error_type", "message"],
    )

    summary = {
        "run_time": RUN_TIME,
        "root": str(ROOT),
        "expected_size": EXPECTED_SIZE,
        "total_regions": len(region_stats),
        "total_img": sum(s["img"] for s in region_stats.values()),
        "total_mask": sum(s["mask"] for s in region_stats.values()),
        "total_paired": sum(s["paired"] for s in region_stats.values()),
        "total_success": sum(s["success"] for s in region_stats.values()),
        "bad_open": len(bad_open),
        "bad_pair": len(bad_pair),
        "bad_image_mode": len(bad_image_mode),
        "bad_image_size": len(bad_image_size),
        "bad_mask_size": len(bad_mask_size),
        "bad_mask_channel": len(bad_mask_channel),
        "bad_mask_values": len(bad_mask_values),
        "empty_masks": len(empty_masks),
        "total_elapsed_seconds": round(total_elapsed, 2),
        "log_path": str(LOG_PATH),
        "region_stats_csv": str(REGION_CSV_PATH),
        "summary_json": str(SUMMARY_JSON_PATH),
        "empty_masks_csv": str(EMPTY_MASKS_CSV_PATH),
        "error_records_csv": str(ERROR_CSV_PATH),
        "sample_stats_csv": str(SAMPLE_STATS_CSV_PATH),
    }

    with open(SUMMARY_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return summary


# ============================================================
# 主函数
# ============================================================

def main():
    """主入口函数。"""

    start = time.time()

    log("=" * 70)
    log("Program Name: Remote Sensing Dataset Inspector")
    log("=" * 70)
    log(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Conda Environment: {os.environ.get('CONDA_DEFAULT_ENV', 'unknown')}")
    log(f"Python Version: {sys.version.split()[0]}")
    log(f"Platform: {platform.platform()}")
    log(f"Input Root: {ROOT}")
    log(f"Expected Size: {EXPECTED_SIZE}")
    log(f"Report Interval: {REPORT_INTERVAL}")
    log(f"Log Path: {LOG_PATH}")
    log("=" * 70)

    if not ROOT.exists():
        log(f"[FAIL] Dataset root not found: {ROOT}")
        return

    region_dirs = sorted(p for p in ROOT.iterdir() if p.is_dir())

    if not region_dirs:
        log(f"[FAIL] No region folders found under: {ROOT}")
        return

    log("[INFO] Found Regions:")
    for p in region_dirs:
        log(f"  - {p.name}")

    for region_dir in region_dirs:
        check_region(region_dir)

    total_elapsed = time.time() - start
    summary = save_reports(total_elapsed)

    log("")
    log("=" * 70)
    log("[SUMMARY] REGION STATS")
    log("=" * 70)

    for region, stat in sorted(region_stats.items()):
        empty_ratio = stat["empty_mask"] / stat["paired"] if stat["paired"] else 0

        log(
            f"{region}: "
            f"img={stat['img']}, "
            f"mask={stat['mask']}, "
            f"paired={stat['paired']}, "
            f"empty_mask={stat['empty_mask']}, "
            f"empty_ratio={empty_ratio:.2%}, "
            f"foreground_ratio={stat['foreground_ratio']:.4%}"
        )

    log("")
    log("=" * 70)
    log("[SUMMARY] FINAL SUMMARY")
    log("=" * 70)

    for k, v in summary.items():
        log(f"{k}: {v}")

    has_critical_error = any([
        summary["bad_open"] > 0,
        summary["bad_pair"] > 0,
        summary["bad_image_mode"] > 0,
        summary["bad_image_size"] > 0,
        summary["bad_mask_size"] > 0,
        summary["bad_mask_channel"] > 0,
        summary["bad_mask_values"] > 0,
    ])

    log("")
    log("=" * 70)

    if has_critical_error:
        log("[FAIL] Dataset has critical errors. Please check error_records.csv and log file.")
    else:
        log("[PASS] Dataset integrity check passed. Empty masks are recorded but not treated as critical errors.")

    log("=" * 70)

    log("")
    log("[INFO] Saved Reports:")
    log(f"Log: {LOG_PATH}")
    log(f"Region CSV: {REGION_CSV_PATH}")
    log(f"Summary JSON: {SUMMARY_JSON_PATH}")
    log(f"Empty Masks CSV: {EMPTY_MASKS_CSV_PATH}")
    log(f"Error CSV: {ERROR_CSV_PATH}")
    log(f"Sample Stats CSV: {SAMPLE_STATS_CSV_PATH}")


if __name__ == "__main__":
    main()
