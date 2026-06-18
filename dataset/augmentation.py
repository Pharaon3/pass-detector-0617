"""Training-time video augmentations (grayscale, rotate, hue, zoom)."""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class AugmentSpec:
    rotation_deg: float = 0.0
    grayscale: bool = False
    zoom: float = 1.0
    hue_deg: float = 0.0

    def tag(self) -> str:
        parts = []
        if self.rotation_deg != 0:
            parts.append(f"rot{self.rotation_deg:+.0f}")
        if self.grayscale:
            parts.append("gray")
        if self.zoom != 1.0:
            parts.append(f"zoom{self.zoom:.1f}")
        if self.hue_deg != 0:
            parts.append(f"hue{self.hue_deg:+.0f}")
        return "_".join(parts) if parts else "orig"


def build_augment_specs(cfg: dict) -> list[AugmentSpec]:
    """
    Cartesian product of augmentation options.

    Default grid size: 3 × 2 × 2 × hue_count
      rotation_deg  (3): 0, -5, +5
      grayscale     (2): color, gray
      zoom          (2): 1.0, 1.1
      hue_deg       (N): e.g. [0, 15, -15]
    """
    rotations = cfg.get("rotation_deg", [0, -5, 5])
    grays = cfg.get("grayscale", [False, True])
    zooms = cfg.get("zoom", [1.0, 1.1])
    hues = cfg.get("hue_deg", [0, 15, -15])

    specs = [
        AugmentSpec(
            rotation_deg=float(r),
            grayscale=bool(g),
            zoom=float(z),
            hue_deg=float(h),
        )
        for r, g, z, h in itertools.product(rotations, grays, zooms, hues)
    ]
    return specs


def zoom_center_crop_rgb(img: np.ndarray, zoom: float) -> np.ndarray:
    if abs(zoom - 1.0) < 1e-6:
        return img
    h, w = img.shape[:2]
    sw = max(w, int(math.ceil(w * zoom)))
    sh = max(h, int(math.ceil(h * zoom)))
    scaled = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_LINEAR)
    x0 = max(0, (sw - w) // 2)
    y0 = max(0, (sh - h) // 2)
    return scaled[y0 : y0 + h, x0 : x0 + w]


def rotate_zoom_crop_rgb(img: np.ndarray, angle_deg: float) -> np.ndarray:
    if abs(angle_deg) < 1e-6:
        return img
    h, w = img.shape[:2]
    center = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    rotated = cv2.warpAffine(
        img,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValues=(0, 0, 0),
    )

    a = math.radians(abs(angle_deg))
    zoom_w = 1.0 / (math.cos(a) - math.sin(a) * h / w)
    zoom_h = 1.0 / (math.cos(a) - math.sin(a) * w / h)
    zoom = max(zoom_w, zoom_h) * 1.02
    return zoom_center_crop_rgb(rotated, zoom)


def to_grayscale_rgb(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)


def hue_shift_rgb(img: np.ndarray, hue_deg: float) -> np.ndarray:
    if abs(hue_deg) < 1e-6:
        return img
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.int16)
    shift = int(round(hue_deg * 179.0 / 360.0))
    hsv[:, :, 0] = (hsv[:, :, 0] + shift) % 180
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def apply_augmentation(frames: np.ndarray, spec: AugmentSpec) -> np.ndarray:
    """
    Apply augmentations to (T, H, W, 3) uint8 RGB frames.
    Order: zoom → rotate → grayscale → hue
    """
    if (
        spec.rotation_deg == 0
        and not spec.grayscale
        and spec.zoom == 1.0
        and spec.hue_deg == 0
    ):
        return frames

    out = []
    for frame in frames:
        img = frame
        img = zoom_center_crop_rgb(img, spec.zoom)
        img = rotate_zoom_crop_rgb(img, spec.rotation_deg)
        if spec.grayscale:
            img = to_grayscale_rgb(img)
        img = hue_shift_rgb(img, spec.hue_deg)
        out.append(img)
    return np.stack(out, axis=0)


class AugmentedPassDataset(Dataset):
    """Expand a base dataset by every AugmentSpec (train only)."""

    def __init__(self, base_dataset: Dataset, subset_indices: list[int], specs: list[AugmentSpec]) -> None:
        self.base_dataset = base_dataset
        self.subset_indices = subset_indices
        self.specs = specs

    def __len__(self) -> int:
        return len(self.subset_indices) * len(self.specs)

    def __getitem__(self, idx: int) -> dict:
        base_pos = self.subset_indices[idx // len(self.specs)]
        spec = self.specs[idx % len(self.specs)]
        item = self.base_dataset.get_item(base_pos, augment_spec=spec)
        item["aug_tag"] = spec.tag()
        return item
