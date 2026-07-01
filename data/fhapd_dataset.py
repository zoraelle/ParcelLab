"""FHAPD 数据集加载工具。

本文件用于把 FHAPD 接入 ParcelLab / RSPBE。

遵循项目扩展数据集规则：
- 文件名：fhapd_dataset.py
- 类名：FhapdDataset
- 命令行：--train_dataset fhapd_dataset

FHAPD 原始目录结构：

FHAPD/
    SC/
        img/
        mask/
    JS/
        img/
        mask/
    ...

Dataset 返回格式与 ftw_dataset.py 保持一致：

    name, image, mask, contour, dist

其中：
- image：RGB影像，ImageNet归一化，shape=[3,H,W]
- mask：二值地块标签，float，shape=[1,H,W]，取值0/1
- contour：由mask在线生成，long，shape=[1,H,W]，取值0/1
- dist：由mask在线生成，float，shape=[1,H,W]，取值0~1

contour 和 dist 的生成严格参考 BsiNet preprocess.py：
- contour = cv2.Canny(im_data, 100, 200)
- dist = cv2.distanceTransform(src=im_data, distanceType=cv2.DIST_L2, maskSize=3)
"""

from __future__ import annotations

import os
from pathlib import Path
from collections.abc import Sequence

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


ALL_REGIONS = "all"


DEFAULT_SPLIT_REGIONS = {
    "train": ["SC", "JS", "HN", "HB", "HuN", "GanSu"],
    "val": ["CQ"],
    "test": ["DB", "XJ"],
}


def _as_list(value: str | Sequence[str]) -> list[str]:
    """把单个区域名或多个区域名统一转换为 list。"""

    if isinstance(value, str):
        return [value]
    return list(value)


def _resolve_data_root(data_root: str) -> Path:
    """解析 FHAPD 数据根目录。

    支持绝对路径，也支持相对于项目根目录的路径。
    """

    path = Path(data_root)

    if path.is_absolute() or path.exists():
        return path

    project_root = Path(__file__).resolve().parents[1]
    project_relative = project_root / path

    if project_relative.exists():
        return project_relative

    return path


def _discover_regions(data_root: Path) -> list[str]:
    """自动发现 FHAPD 根目录下可用区域。

    有效区域必须同时满足：
    - 存在 img 文件夹；
    - 存在 mask 文件夹；
    - img 文件夹中至少有一个 png 文件。
    """

    if not data_root.exists():
        raise FileNotFoundError(f"FHAPD data root not found: {data_root}")

    regions: list[str] = []

    for region_dir in sorted(p for p in data_root.iterdir() if p.is_dir()):
        img_dir = region_dir / "img"
        mask_dir = region_dir / "mask"

        if img_dir.exists() and mask_dir.exists() and any(img_dir.glob("*.png")):
            regions.append(region_dir.name)

    if not regions:
        raise FileNotFoundError(
            f"No FHAPD regions with img/mask were found under {data_root}"
        )

    return regions


def _resolve_regions(
    data_root: Path,
    region: str | Sequence[str],
    split: str,
) -> list[str]:
    """根据 region 和 split 确定当前数据集读取哪些区域。"""

    available_regions = _discover_regions(data_root)
    requested_regions = [r.strip() for r in _as_list(region) if r.strip()]

    if any(r.lower() == ALL_REGIONS for r in requested_regions):
        if len(requested_regions) > 1:
            raise ValueError("Use either 'all' or explicit FHAPD regions, not both.")

        split_regions = DEFAULT_SPLIT_REGIONS.get(split)

        if split_regions is None:
            return available_regions

        selected_regions = [r for r in split_regions if r in available_regions]

        if not selected_regions:
            raise ValueError(
                f"No FHAPD regions selected for split={split}. "
                f"Expected: {split_regions}. Available: {available_regions}."
            )

        return selected_regions

    if not requested_regions:
        raise ValueError("At least one FHAPD region must be provided.")

    missing_regions = [r for r in requested_regions if r not in available_regions]

    if missing_regions:
        raise ValueError(
            f"Requested FHAPD regions not found: {missing_regions}. "
            f"Available regions: {available_regions}"
        )

    return requested_regions


def load_image(image_path: str) -> torch.Tensor:
    """读取 RGB 影像并转换为模型输入张量。"""

    image = Image.open(image_path).convert("RGB")

    data_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    return data_transforms(image)


def read_mask_array(mask_path: str) -> np.ndarray:
    """读取 FHAPD mask，并统一转换为 0/255 uint8。

    这里返回 0/255，而不是 0/1，是为了严格对齐 BsiNet preprocess.py：
    其 distanceTransform 和 Canny 都直接作用于 im_data。
    """

    mask = np.array(Image.open(mask_path))

    if mask.ndim == 3:
        mask = mask[:, :, 0]

    mask = (mask > 0).astype(np.uint8) * 255

    return mask


def mask_to_tensor(mask_255: np.ndarray) -> torch.Tensor:
    """把 0/255 mask 转换为 0/1 float Tensor。"""

    mask = (mask_255 > 0).astype(np.float32)
    mask = np.expand_dims(mask, axis=0)

    return torch.from_numpy(mask).float()


def make_contour_bsinet(mask_255: np.ndarray) -> torch.Tensor:
    """按 BsiNet preprocess.py 生成 contour / boundary。

    BsiNet 原始写法：
        boundary = cv2.Canny(im_data, 100, 200)

    输入：
        mask_255: ndarray, shape=[H,W], dtype=uint8, value=0/255

    返回：
        Tensor, shape=[1,H,W], dtype=long, value=0/1
    """

    boundary = cv2.Canny(mask_255, 100, 200)

    boundary = (boundary > 0).astype(np.int64)
    boundary = np.expand_dims(boundary, axis=0)

    return torch.from_numpy(boundary).long()


def make_dist_bsinet(mask_255: np.ndarray) -> torch.Tensor:
    """按 BsiNet preprocess.py 生成 distance map。

    BsiNet 原始写法：
        result = cv2.distanceTransform(
            src=im_data,
            distanceType=cv2.DIST_L2,
            maskSize=3
        )
        scaled_image = ((result - min_value) / (max_value - min_value)) * 255
        result = scaled_image.astype(np.uint8)

    这里为了适配 ftw_dataset.py 的训练接口，最后再除以255，转为0~1 float。

    输入：
        mask_255: ndarray, shape=[H,W], dtype=uint8, value=0/255

    返回：
        Tensor, shape=[1,H,W], dtype=float32, value=0~1
    """

    result = cv2.distanceTransform(
        src=mask_255,
        distanceType=cv2.DIST_L2,
        maskSize=3,
    )

    min_value = np.min(result)
    max_value = np.max(result)

    if max_value > min_value:
        scaled_image = ((result - min_value) / (max_value - min_value)) * 255
    else:
        scaled_image = np.zeros_like(result, dtype=np.float32)

    result = scaled_image.astype(np.uint8)
    result = result.astype(np.float32) / 255.0
    result = np.expand_dims(result, axis=0)

    return torch.from_numpy(result).float()


class FhapdDataset(Dataset):
    """FHAPD 数据集类。

    该类直接读取 FHAPD 原始 img/mask 目录，不复制、不重组数据。
    boundary 和 distance map 在读取样本时由 mask 在线生成。
    """

    def __init__(
        self,
        data_root: str = os.environ.get("FHAPD_ROOT", "FHAPD"),
        region: str | Sequence[str] = ALL_REGIONS,
        split: str = "train",
        file_names: list[str] | None = None,
    ) -> None:
        self.data_root = _resolve_data_root(data_root)
        self.split = split

        self.regions = _resolve_regions(
            data_root=self.data_root,
            region=region,
            split=split,
        )

        self.multiple_regions = len(self.regions) > 1
        self.samples: list[tuple[str, str]] = []

        for region_name in self.regions:
            img_dir = self.data_root / region_name / "img"
            mask_dir = self.data_root / region_name / "mask"

            if not img_dir.exists():
                raise FileNotFoundError(f"FHAPD image directory not found: {img_dir}")

            if not mask_dir.exists():
                raise FileNotFoundError(f"FHAPD mask directory not found: {mask_dir}")

            if file_names is None:
                image_names = sorted(path.stem for path in img_dir.glob("*.png"))
            else:
                image_names = list(file_names)

            for image_name in image_names:
                image_path = img_dir / f"{image_name}.png"
                mask_path = mask_dir / f"{image_name}.png"

                if not image_path.exists():
                    raise FileNotFoundError(f"Missing FHAPD image: {image_path}")

                if not mask_path.exists():
                    raise FileNotFoundError(f"Missing FHAPD mask: {mask_path}")

                self.samples.append((region_name, image_name))

        if not self.samples:
            raise RuntimeError(
                f"No FHAPD image-mask pairs found. "
                f"split={self.split}, regions={self.regions}"
            )

        self.file_names = [
            self._display_name(region_name, image_name)
            for region_name, image_name in self.samples
        ]

    def _display_name(self, region_name: str, image_name: str) -> str:
        """生成样本显示名。
          为避免不同区域存在重名patch，所有split统一采用：region/image_name
        
        """
        return f"{region_name}/{image_name}"

    def __len__(self) -> int:
        """返回样本数量。"""

        return len(self.samples)

    def _image_path(self, region_name: str, image_name: str) -> Path:
        """生成 image 路径。"""

        return self.data_root / region_name / "img" / f"{image_name}.png"

    def _mask_path(self, region_name: str, image_name: str) -> Path:
        """生成 mask 路径。"""

        return self.data_root / region_name / "mask" / f"{image_name}.png"

    def __getitem__(self, idx: int):
        """读取一个 FHAPD 样本。

        返回格式严格对齐 ftw_dataset.py：

            name, image, mask, contour, dist
        """

        region_name, image_name = self.samples[idx]

        image_path = self._image_path(region_name, image_name)
        mask_path = self._mask_path(region_name, image_name)

        name = self._display_name(region_name, image_name)

        image = load_image(str(image_path))

        mask_255 = read_mask_array(str(mask_path))
        mask = mask_to_tensor(mask_255)
        contour = make_contour_bsinet(mask_255)
        dist = make_dist_bsinet(mask_255)

        return name, image, mask, contour, dist


class FhapdPredictionDataset(FhapdDataset):
    """FHAPD 推理数据集。

    预测阶段只返回 name 和 image。
    """

    def __getitem__(self, idx: int):
        region_name, image_name = self.samples[idx]

        image_path = self._image_path(region_name, image_name)

        name = self._display_name(region_name, image_name)
        image = load_image(str(image_path))

        return name, image
