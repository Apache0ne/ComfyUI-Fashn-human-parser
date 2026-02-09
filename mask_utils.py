from __future__ import annotations

import colorsys
from typing import Dict, Iterable, List

import numpy as np

IDS_TO_LABELS: Dict[int, str] = {
    0: "background",
    1: "face",
    2: "hair",
    3: "top",
    4: "dress",
    5: "skirt",
    6: "pants",
    7: "belt",
    8: "bag",
    9: "hat",
    10: "scarf",
    11: "glasses",
    12: "arms",
    13: "hands",
    14: "legs",
    15: "feet",
    16: "torso",
    17: "jewelry",
}
LABELS_TO_IDS: Dict[str, int] = {label: idx for idx, label in IDS_TO_LABELS.items()}

MAX_LABEL_ID = max(IDS_TO_LABELS.keys())
CLASS_LABELS = [IDS_TO_LABELS[idx] for idx in sorted(IDS_TO_LABELS.keys())]


def get_palette(num_classes: int) -> List[int]:
    palette = [0] * (256 * 3)
    palette[0:3] = [0, 0, 0]
    for class_index in range(1, num_classes):
        hue = (class_index - 1) / max(num_classes - 1, 1)
        saturation = 1.0
        value = 1.0 if class_index % 2 == 0 else 0.5
        red, green, blue = [int(channel * 255) for channel in colorsys.hsv_to_rgb(hue, saturation, value)]
        palette[class_index * 3 : class_index * 3 + 3] = [red, green, blue]
    return palette


PALETTE_COLORS = np.array(get_palette(len(IDS_TO_LABELS)), dtype=np.uint8).reshape(-1, 3)


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    clipped = np.clip(mask.astype(np.int32, copy=False), 0, PALETTE_COLORS.shape[0] - 1)
    return PALETTE_COLORS[clipped]


def overlay_on_image(image_rgb: np.ndarray, mask: np.ndarray, alpha: float) -> np.ndarray:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    overlay = colorize_mask(mask).astype(np.float32)
    image_f = image_rgb.astype(np.float32)
    blended = image_f * (1.0 - alpha) + overlay * alpha
    return np.clip(blended, 0, 255).astype(np.uint8)


def selected_label_ids_from_switches(class_switches: Dict[str, bool]) -> List[int]:
    selected = []
    for label in CLASS_LABELS:
        if bool(class_switches.get(label, False)):
            selected.append(LABELS_TO_IDS[label])
    return selected


def build_selected_mask(mask: np.ndarray, selected_label_ids: Iterable[int]) -> np.ndarray:
    selected_label_ids = list(selected_label_ids)
    if not selected_label_ids:
        return np.zeros(mask.shape, dtype=np.float32)
    selected = np.isin(mask, np.asarray(selected_label_ids, dtype=np.int32))
    return selected.astype(np.float32)


def normalize_class_id_mask(mask: np.ndarray) -> np.ndarray:
    return np.clip(mask.astype(np.float32) / float(MAX_LABEL_ID), 0.0, 1.0)
