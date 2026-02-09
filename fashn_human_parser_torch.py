"""PyTorch safetensors implementation of FASHN Human Parser with official preprocessing."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Union

import numpy as np

try:
    import cv2
except Exception:
    cv2 = None

try:
    import torch
except Exception:
    torch = None

try:
    from PIL import Image, ImageOps
except Exception:
    Image = None
    ImageOps = None

try:
    from safetensors.torch import load_file as safetensors_load_file
except Exception:
    safetensors_load_file = None

try:
    from transformers import SegformerConfig, SegformerForSemanticSegmentation
except Exception:
    SegformerConfig = None
    SegformerForSemanticSegmentation = None

try:
    from .mask_utils import IDS_TO_LABELS
except Exception:
    from mask_utils import IDS_TO_LABELS

# ImageNet normalization constants (same as training)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Model input size (matches training)
INPUT_HEIGHT = 576
INPUT_WIDTH = 384


def _require_runtime() -> None:
    missing = []
    if cv2 is None:
        missing.append("opencv-python")
    if torch is None:
        missing.append("torch")
    if Image is None or ImageOps is None:
        missing.append("Pillow")
    if safetensors_load_file is None:
        missing.append("safetensors")
    if SegformerConfig is None or SegformerForSemanticSegmentation is None:
        missing.append("transformers")
    if missing:
        raise RuntimeError(
            "Missing runtime dependencies for safetensors parser: "
            + ", ".join(missing)
            + "."
        )


def _inference_mode_decorator():
    if torch is None:
        def decorator(function):
            return function

        return decorator
    return torch.inference_mode()


def _load_weights_into_model(model, state_dict: Dict) -> None:
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if not missing_keys and not unexpected_keys:
        return

    if all(key.startswith("model.") for key in state_dict.keys()):
        stripped = {key[len("model.") :]: value for key, value in state_dict.items()}
        missing_keys, unexpected_keys = model.load_state_dict(stripped, strict=False)
        if not missing_keys and not unexpected_keys:
            return

    missing_preview = ", ".join(missing_keys[:8]) if missing_keys else "none"
    unexpected_preview = ", ".join(unexpected_keys[:8]) if unexpected_keys else "none"
    raise RuntimeError(
        "Failed loading safetensors weights into SegFormer model. "
        f"Missing keys: {missing_preview}. Unexpected keys: {unexpected_preview}."
    )


class FashnHumanParserTorch:
    """PyTorch-based human parser using local `model.safetensors` + `config.json`."""

    def __init__(self, model_dir: str, precision: str = "fp16", device: str = "auto"):
        _require_runtime()

        model_dir_path = Path(model_dir).expanduser().resolve()
        config_path = model_dir_path / "config.json"
        safetensors_path = model_dir_path / "model.safetensors"

        if not config_path.exists():
            raise FileNotFoundError(f"Missing config file: {config_path}")
        if not safetensors_path.exists():
            raise FileNotFoundError(f"Missing safetensors file: {safetensors_path}")

        if device == "auto":
            resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            resolved_device = device
        if resolved_device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but torch.cuda.is_available() is False.")
        self.device = torch.device(resolved_device)

        precision = precision.lower()
        if precision not in {"fp16", "fp32"}:
            raise ValueError(f"Unsupported precision: {precision}")

        if precision == "fp16" and self.device.type != "cuda":
            # CPU fp16 path is usually unsupported/slow; keep parser robust.
            self.dtype = torch.float32
        else:
            self.dtype = torch.float16 if precision == "fp16" else torch.float32
        self.precision = "fp16" if self.dtype == torch.float16 else "fp32"

        config = SegformerConfig.from_json_file(str(config_path))
        model = SegformerForSemanticSegmentation(config)
        state_dict = safetensors_load_file(str(safetensors_path))
        _load_weights_into_model(model, state_dict)

        if self.dtype == torch.float16:
            model = model.half()
        self.model = model.to(self.device).eval()
        self.model_dir = str(model_dir_path)

    def _preprocess_single(self, image: np.ndarray) -> np.ndarray:
        resized = cv2.resize(image, (INPUT_WIDTH, INPUT_HEIGHT), interpolation=cv2.INTER_AREA)
        normalized = resized.astype(np.float32) / 255.0
        normalized = (normalized - IMAGENET_MEAN) / IMAGENET_STD
        return normalized.transpose(2, 0, 1)

    def _to_numpy(self, image: Union["Image.Image", np.ndarray, str]) -> np.ndarray:
        if isinstance(image, str):
            with Image.open(image) as pil_img:
                pil_img = ImageOps.exif_transpose(pil_img)
                image = np.array(pil_img.convert("RGB"))
        elif Image is not None and isinstance(image, Image.Image):
            image = np.array(image.convert("RGB"))

        if isinstance(image, np.ndarray):
            if image.ndim == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.ndim == 3:
                if image.shape[2] == 4:
                    image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
                elif image.shape[2] != 3:
                    raise ValueError(f"Expected 3 channels, got {image.shape[2]}")
            else:
                raise ValueError(f"Expected 2D/3D image, got {image.ndim}D")

            if image.dtype in (np.float32, np.float64):
                image = (image * 255.0).clip(0, 255).astype(np.uint8)
        else:
            raise TypeError(
                f"Unsupported image type: {type(image).__name__}. "
                "Expected PIL Image, numpy array, or file path string."
            )
        return image

    @_inference_mode_decorator()
    def predict(
        self,
        image: Union["Image.Image", np.ndarray, str, List],
        return_logits: bool = False,
    ):
        is_batch = isinstance(image, list)
        images = image if is_batch else [image]
        if len(images) == 0:
            return []

        images_np = [self._to_numpy(img) for img in images]
        original_sizes = [(img.shape[0], img.shape[1]) for img in images_np]
        batch_tensors = [self._preprocess_single(img) for img in images_np]

        input_tensor = torch.from_numpy(np.stack(batch_tensors)).to(self.device)
        input_tensor = input_tensor.to(self.dtype)

        logits = self.model(pixel_values=input_tensor).logits
        logits = logits.float()

        results = []
        for index, size in enumerate(original_sizes):
            image_logits = logits[index : index + 1]
            upsampled = torch.nn.functional.interpolate(
                image_logits,
                size=size,
                mode="bilinear",
                align_corners=False,
            )
            if return_logits:
                results.append(upsampled.detach().cpu())
            else:
                pred_seg = upsampled.argmax(dim=1).squeeze(0).detach().cpu().numpy()
                results.append(pred_seg.astype(np.int32))

        return results if is_batch else results[0]

    @staticmethod
    def get_label_name(label_id: int) -> str:
        return IDS_TO_LABELS.get(label_id, "unknown")

    @staticmethod
    def get_labels() -> dict:
        return IDS_TO_LABELS.copy()
