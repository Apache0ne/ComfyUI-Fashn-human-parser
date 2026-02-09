"""TensorRT implementation of FASHN Human Parser with official preprocessing."""

from __future__ import annotations

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
    import tensorrt as trt
except Exception:
    trt = None

try:
    from .mask_utils import IDS_TO_LABELS, LABELS_TO_IDS
except Exception:
    from mask_utils import IDS_TO_LABELS, LABELS_TO_IDS

# ImageNet normalization constants (same as training)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Model input size (matches training)
INPUT_HEIGHT = 576
INPUT_WIDTH = 384

# Body coverage mappings for virtual try-on
CATEGORY_TO_BODY_COVERAGE: Dict[str, str] = {
    "tops": "upper",
    "bottoms": "lower",
    "one-pieces": "full",
}

BODY_COVERAGE_TO_LABELS: Dict[str, List[str]] = {
    "upper": ["top", "dress", "scarf"],
    "lower": ["skirt", "pants", "belt"],
    "full": ["top", "dress", "scarf", "skirt", "pants", "belt"],
}

# Labels typically preserved during virtual try-on
IDENTITY_LABELS: List[str] = ["face", "hair", "jewelry", "bag", "glasses", "hat"]


def _trt_dtype_to_np(trt_dtype):
    if trt_dtype == trt.float16:
        return np.float16
    if trt_dtype == trt.float32:
        return np.float32
    if trt_dtype == trt.int8:
        return np.int8
    return np.float32


def _np_dtype_to_torch(np_dtype):
    if torch is None:
        raise RuntimeError("torch is required for TensorRT runtime.")

    dtype = np.dtype(np_dtype)
    if dtype == np.float16:
        return torch.float16
    if dtype == np.float32:
        return torch.float32
    if dtype == np.int8:
        return torch.int8
    return torch.float32


def _require_runtime() -> None:
    missing = []
    if cv2 is None:
        missing.append("opencv-python")
    if torch is None:
        missing.append("torch")
    elif not torch.cuda.is_available():
        missing.append("CUDA-enabled torch")
    if Image is None or ImageOps is None:
        missing.append("Pillow")
    if trt is None:
        missing.append("tensorrt")
    if missing:
        raise RuntimeError(
            "Missing runtime dependencies for TensorRT parser: "
            + ", ".join(missing)
            + "."
        )


def _inference_mode_decorator():
    if torch is None:
        def decorator(function):
            return function

        return decorator
    return torch.inference_mode()


class FashnHumanParserTRT:
    """TensorRT-based human parser with official preprocessing."""

    def __init__(
        self,
        engine_path: str,
        input_name: str = "pixel_values",
        output_name: str = "logits",
    ):
        _require_runtime()

        self.engine_path = engine_path
        self.input_name = input_name
        self.output_name = output_name

        logger = trt.Logger(trt.Logger.INFO)
        with open(engine_path, "rb") as handle, trt.Runtime(logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(handle.read())
            if self.engine is None:
                raise RuntimeError(f"Failed to deserialize TensorRT engine: {engine_path}")

        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Failed to create TensorRT execution context")

        self.input_dtype = _trt_dtype_to_np(self.engine.get_tensor_dtype(self.input_name))
        self.output_dtype = _trt_dtype_to_np(self.engine.get_tensor_dtype(self.output_name))
        self.max_batch = self._get_max_batch()

    def _get_max_batch(self) -> int:
        try:
            _, _, max_shape = self.engine.get_profile_shape(0, self.input_name)
            return int(max_shape[0])
        except Exception:
            return 1

    def _preprocess_single(self, image: np.ndarray) -> np.ndarray:
        resized = cv2.resize(image, (INPUT_WIDTH, INPUT_HEIGHT), interpolation=cv2.INTER_AREA)
        normalized = resized.astype(np.float32) / 255.0
        normalized = (normalized - IMAGENET_MEAN) / IMAGENET_STD
        return normalized.transpose(2, 0, 1)

    def _to_numpy(self, image: Union["Image.Image", np.ndarray, str]) -> np.ndarray:
        if isinstance(image, str):
            if Image is None or ImageOps is None:
                raise RuntimeError("Pillow is required to load image paths.")
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

    def _infer(self, pixel_values: np.ndarray) -> np.ndarray:
        pixel_values = np.ascontiguousarray(pixel_values)
        self.context.set_input_shape(self.input_name, pixel_values.shape)

        input_dtype_torch = _np_dtype_to_torch(self.input_dtype)
        output_dtype_torch = _np_dtype_to_torch(self.output_dtype)
        input_tensor = torch.from_numpy(pixel_values).to(device="cuda", dtype=input_dtype_torch).contiguous()

        out_shape = (
            pixel_values.shape[0],
            len(IDS_TO_LABELS),
            pixel_values.shape[2] // 4,
            pixel_values.shape[3] // 4,
        )
        output_tensor = torch.empty(out_shape, dtype=output_dtype_torch, device="cuda")

        self.context.set_tensor_address(self.input_name, int(input_tensor.data_ptr()))
        self.context.set_tensor_address(self.output_name, int(output_tensor.data_ptr()))

        stream = torch.cuda.current_stream(device=input_tensor.device)
        success = self.context.execute_async_v3(stream_handle=stream.cuda_stream)
        if not success:
            raise RuntimeError("TensorRT execution failed.")
        stream.synchronize()

        return output_tensor.detach().cpu().numpy()

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

        results = []
        start = 0
        while start < len(batch_tensors):
            end = min(start + self.max_batch, len(batch_tensors))
            batch = np.stack(batch_tensors[start:end]).astype(self.input_dtype)
            logits = self._infer(batch)

            for index, size in enumerate(original_sizes[start:end]):
                image_logits = torch.from_numpy(logits[index : index + 1].astype(np.float32))
                upsampled = torch.nn.functional.interpolate(
                    image_logits,
                    size=size,
                    mode="bilinear",
                    align_corners=False,
                )
                if return_logits:
                    results.append(upsampled)
                else:
                    pred_seg = upsampled.argmax(dim=1).squeeze(0).cpu().numpy()
                    results.append(pred_seg)
            start = end

        return results if is_batch else results[0]

    @staticmethod
    def get_label_name(label_id: int) -> str:
        return IDS_TO_LABELS.get(label_id, "unknown")

    @staticmethod
    def get_labels() -> dict:
        return IDS_TO_LABELS.copy()
