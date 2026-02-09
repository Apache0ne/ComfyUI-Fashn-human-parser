from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

try:
    from .engine_builder import prepare_artifacts
    from .fashn_human_parser_torch import FashnHumanParserTorch
    from .fashn_human_parser_trt import FashnHumanParserTRT
    from .mask_utils import (
        CLASS_LABELS,
        build_selected_mask,
        colorize_mask,
        overlay_on_image,
        selected_label_ids_from_switches,
    )
    from .model_store import (
        DEFAULT_CONFIG_URL,
        DEFAULT_SAFETENSORS_URL,
        ensure_model_files,
        ensure_fashn_models_root,
        register_model_folder,
    )
except Exception:
    from engine_builder import prepare_artifacts
    from fashn_human_parser_torch import FashnHumanParserTorch
    from fashn_human_parser_trt import FashnHumanParserTRT
    from mask_utils import (
        CLASS_LABELS,
        build_selected_mask,
        colorize_mask,
        overlay_on_image,
        selected_label_ids_from_switches,
    )
    from model_store import (
        DEFAULT_CONFIG_URL,
        DEFAULT_SAFETENSORS_URL,
        ensure_model_files,
        ensure_fashn_models_root,
        register_model_folder,
    )


ensure_fashn_models_root()
register_model_folder()

DEFAULT_MODEL_NAME = "fashn-human-parser"
NO_ENGINE_OPTION = "<no .engine files found in ComfyUI/models/Fashn-parsers>"


def _scan_engine_options() -> Dict[str, str]:
    models_root = ensure_fashn_models_root().resolve()
    options: Dict[str, str] = {}
    for engine_file in sorted(models_root.rglob("*.engine"), key=lambda path: str(path).lower()):
        if not engine_file.is_file():
            continue
        resolved = engine_file.resolve()
        try:
            label = resolved.relative_to(models_root).as_posix()
        except Exception:
            label = str(resolved)
        if label in options:
            label = str(resolved)
        options[label] = str(resolved)

    if not options:
        options[NO_ENGINE_OPTION] = ""
    return options


def _to_numpy_image_batch(image: torch.Tensor) -> List[np.ndarray]:
    if isinstance(image, torch.Tensor):
        array = image.detach().cpu().numpy()
    else:
        array = np.asarray(image)

    if array.ndim == 3:
        array = array[np.newaxis, ...]
    if array.ndim != 4:
        raise ValueError(f"Expected image tensor with shape [B,H,W,C], got {array.shape}")
    if array.shape[-1] < 3:
        raise ValueError(f"Expected at least 3 channels in input image, got shape {array.shape}")

    array = array[..., :3]
    if np.issubdtype(array.dtype, np.floating):
        array = np.clip(array, 0.0, 1.0) * 255.0
    array = np.clip(array, 0, 255).astype(np.uint8)
    return [np.ascontiguousarray(array[idx]) for idx in range(array.shape[0])]


def _to_comfy_image(image_batch_uint8: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(image_batch_uint8.astype(np.float32) / 255.0)


def _to_comfy_mask(mask_batch_float: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(mask_batch_float.astype(np.float32))


def _build_model_handle(backend: str, runner, source: str) -> Dict:
    return {"backend": backend, "runner": runner, "source": source}


class FashnParserLoadTRT:
    _CACHE: Dict[str, FashnHumanParserTRT] = {}
    _ENGINE_OPTIONS: Dict[str, str] = {}

    @classmethod
    def _refresh_engine_choices(cls) -> List[str]:
        cls._ENGINE_OPTIONS = _scan_engine_options()
        return list(cls._ENGINE_OPTIONS.keys())

    @classmethod
    def INPUT_TYPES(cls):
        engine_choices = cls._refresh_engine_choices()
        return {
            "required": {
                "engine_path": (engine_choices, {"default": engine_choices[0]}),
            }
        }

    RETURN_TYPES = ("FASHN_PARSER_MODEL",)
    RETURN_NAMES = ("parser_model",)
    FUNCTION = "load"
    CATEGORY = "FASHN/Parser"

    def load(self, engine_path: str):
        engine_options = _scan_engine_options()
        selected = (engine_path or "").strip()
        normalized = self._ENGINE_OPTIONS.get(selected) or engine_options.get(selected) or selected
        if not normalized:
            raise FileNotFoundError(NO_ENGINE_OPTION)
        normalized = str(Path(normalized).expanduser().resolve())
        if not Path(normalized).exists():
            raise FileNotFoundError(f"Engine file not found: {normalized}")

        parser = self._CACHE.get(normalized)
        if parser is None:
            parser = FashnHumanParserTRT(engine_path=normalized)
            self._CACHE[normalized] = parser
        return (_build_model_handle("trt", parser, normalized),)


class FashnParserLoadSafetensors:
    _CACHE: Dict[str, FashnHumanParserTorch] = {}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "precision": (["fp16", "fp32"], {"default": "fp16"}),
                "device": (["auto", "cuda", "cpu"], {"default": "auto"}),
            }
        }

    RETURN_TYPES = ("FASHN_PARSER_MODEL",)
    RETURN_NAMES = ("parser_model",)
    FUNCTION = "load"
    CATEGORY = "FASHN/Parser"

    def load(
        self,
        precision: str,
        device: str,
    ):
        model_dir, _, _ = ensure_model_files(
            model_name=DEFAULT_MODEL_NAME,
            auto_download=True,
            safetensors_url=DEFAULT_SAFETENSORS_URL,
            config_url=DEFAULT_CONFIG_URL,
        )

        cache_key = f"{model_dir}|{precision}|{device}"
        parser = self._CACHE.get(cache_key)
        if parser is None:
            parser = FashnHumanParserTorch(
                model_dir=str(model_dir),
                precision=precision,
                device=device,
            )
            self._CACHE[cache_key] = parser
        return (_build_model_handle("torch", parser, str(model_dir)),)


class FashnParserBuildEngine:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "precision": (["fp16", "fp32"], {"default": "fp16"}),
                "opset": ("INT", {"default": 18, "min": 11, "max": 20, "step": 1}),
                "min_h": ("INT", {"default": 256, "min": 64, "max": 4096, "step": 1}),
                "min_w": ("INT", {"default": 256, "min": 64, "max": 4096, "step": 1}),
                "opt_h": ("INT", {"default": 576, "min": 64, "max": 4096, "step": 1}),
                "opt_w": ("INT", {"default": 384, "min": 64, "max": 4096, "step": 1}),
                "max_h": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 1}),
                "max_w": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 1}),
                "workspace_gb": ("FLOAT", {"default": 4.0, "min": 0.5, "max": 64.0, "step": 0.5}),
                "force_reexport": ("BOOLEAN", {"default": False}),
                "force_rebuild": ("BOOLEAN", {"default": False}),
                "keep_onnx": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("engine_path",)
    FUNCTION = "build"
    CATEGORY = "FASHN/Parser"

    def build(
        self,
        precision: str,
        opset: int,
        min_h: int,
        min_w: int,
        opt_h: int,
        opt_w: int,
        max_h: int,
        max_w: int,
        workspace_gb: float,
        force_reexport: bool,
        force_rebuild: bool,
        keep_onnx: bool,
    ):
        artifacts = prepare_artifacts(
            model_name=DEFAULT_MODEL_NAME,
            auto_download=True,
            safetensors_url=DEFAULT_SAFETENSORS_URL,
            config_url=DEFAULT_CONFIG_URL,
            precision=precision,
            opset=opset,
            min_h=min_h,
            min_w=min_w,
            opt_h=opt_h,
            opt_w=opt_w,
            max_h=max_h,
            max_w=max_w,
            workspace_gb=workspace_gb,
            force_reexport=force_reexport,
            force_rebuild=force_rebuild,
        )

        if not keep_onnx:
            onnx_path = Path(artifacts.onnx_path)
            onnx_path.unlink(missing_ok=True)
            onnx_path.with_suffix(onnx_path.suffix + ".data").unlink(missing_ok=True)

        return (str(artifacts.engine_path),)


class FashnParserRun:
    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "image": ("IMAGE",),
            "parser_model": ("FASHN_PARSER_MODEL",),
            "overlay_alpha": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
        }
        for label in CLASS_LABELS:
            required[label] = ("BOOLEAN", {"default": label != "background"})
        return {"required": required}

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "MASK")
    RETURN_NAMES = (
        "overlay_image",
        "segmentation_image",
        "selected_overlay_image",
        "selected_segmentation_image",
        "selected_mask",
    )
    FUNCTION = "run"
    CATEGORY = "FASHN/Parser"

    def _resolve_runner(self, parser_model: Dict):
        runner = parser_model.get("runner") if isinstance(parser_model, dict) else None
        if runner is None or not hasattr(runner, "predict"):
            raise ValueError(
                "Invalid parser_model input. Connect output from FashnParserLoadTRT "
                "or FashnParserLoadSafetensors."
            )
        return runner

    def run(
        self,
        image,
        parser_model: Dict,
        overlay_alpha: float,
        background: bool,
        face: bool,
        hair: bool,
        top: bool,
        dress: bool,
        skirt: bool,
        pants: bool,
        belt: bool,
        bag: bool,
        hat: bool,
        scarf: bool,
        glasses: bool,
        arms: bool,
        hands: bool,
        legs: bool,
        feet: bool,
        torso: bool,
        jewelry: bool,
    ):
        runner = self._resolve_runner(parser_model)
        images_np = _to_numpy_image_batch(image)
        masks = runner.predict(images_np)
        if not isinstance(masks, list):
            masks = [masks]
        if len(masks) != len(images_np):
            raise RuntimeError(
                f"Parser output batch mismatch. Expected {len(images_np)} masks, got {len(masks)}."
            )

        class_switches = {
            "background": background,
            "face": face,
            "hair": hair,
            "top": top,
            "dress": dress,
            "skirt": skirt,
            "pants": pants,
            "belt": belt,
            "bag": bag,
            "hat": hat,
            "scarf": scarf,
            "glasses": glasses,
            "arms": arms,
            "hands": hands,
            "legs": legs,
            "feet": feet,
            "torso": torso,
            "jewelry": jewelry,
        }
        selected_ids = selected_label_ids_from_switches(class_switches)

        overlay_images: List[np.ndarray] = []
        segmentation_images: List[np.ndarray] = []
        selected_overlay_images: List[np.ndarray] = []
        selected_segmentation_images: List[np.ndarray] = []
        selected_masks: List[np.ndarray] = []

        for image_np, raw_mask in zip(images_np, masks):
            mask = np.asarray(raw_mask).astype(np.int32)
            overlay = overlay_on_image(image_np, mask, overlay_alpha)
            segmentation = colorize_mask(mask)
            selected_mask = build_selected_mask(mask, selected_ids)
            selected_mask_3c = (selected_mask[..., None] > 0.5).astype(np.uint8)
            selected_overlay = overlay * selected_mask_3c
            selected_segmentation = segmentation * selected_mask_3c
            overlay_images.append(overlay)
            segmentation_images.append(segmentation)
            selected_overlay_images.append(selected_overlay)
            selected_segmentation_images.append(selected_segmentation)
            selected_masks.append(selected_mask)

        overlay_batch = np.stack(overlay_images, axis=0)
        segmentation_batch = np.stack(segmentation_images, axis=0)
        selected_overlay_batch = np.stack(selected_overlay_images, axis=0)
        selected_segmentation_batch = np.stack(selected_segmentation_images, axis=0)
        selected_batch = np.stack(selected_masks, axis=0)

        return (
            _to_comfy_image(overlay_batch),
            _to_comfy_image(segmentation_batch),
            _to_comfy_image(selected_overlay_batch),
            _to_comfy_image(selected_segmentation_batch),
            _to_comfy_mask(selected_batch),
        )


NODE_CLASS_MAPPINGS = {
    "FashnParserLoadTRT": FashnParserLoadTRT,
    "FashnParserLoadSafetensors": FashnParserLoadSafetensors,
    "FashnParserBuildEngine": FashnParserBuildEngine,
    "FashnParserRun": FashnParserRun,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FashnParserLoadTRT": "FASHN Parser Load TRT",
    "FashnParserLoadSafetensors": "FASHN Parser Load Safetensors",
    "FashnParserBuildEngine": "FASHN Parser Build Engine",
    "FashnParserRun": "FASHN Parser Run",
}
