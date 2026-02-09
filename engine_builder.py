from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

try:
    from .model_store import ensure_model_files, sanitize_model_name
except Exception:
    from model_store import ensure_model_files, sanitize_model_name

VALID_PRECISIONS = {"fp16", "fp32"}


@dataclass(frozen=True)
class ArtifactPaths:
    model_dir: Path
    onnx_path: Path
    engine_path: Path


def _onnx_external_data_path(onnx_path: Path) -> Path:
    return onnx_path.with_suffix(onnx_path.suffix + ".data")


def _onnx_uses_external_data(onnx_path: Path) -> bool:
    try:
        import onnx
    except Exception:
        return False

    try:
        model = onnx.load_model(str(onnx_path), load_external_data=False)
    except Exception:
        return False

    for initializer in model.graph.initializer:
        if initializer.data_location == onnx.TensorProto.EXTERNAL:
            return True
    return False


def validate_trt_profile(
    min_h: int,
    min_w: int,
    opt_h: int,
    opt_w: int,
    max_h: int,
    max_w: int,
) -> None:
    values = [min_h, min_w, opt_h, opt_w, max_h, max_w]
    if any(value <= 0 for value in values):
        raise ValueError("All TensorRT profile dimensions must be positive integers")
    if not (min_h <= opt_h <= max_h):
        raise ValueError(f"Expected min_h <= opt_h <= max_h, got {min_h}, {opt_h}, {max_h}")
    if not (min_w <= opt_w <= max_w):
        raise ValueError(f"Expected min_w <= opt_w <= max_w, got {min_w}, {opt_w}, {max_w}")


def build_profile_signature(
    min_h: int,
    min_w: int,
    opt_h: int,
    opt_w: int,
    max_h: int,
    max_w: int,
) -> str:
    return f"min{min_h}x{min_w}_opt{opt_h}x{opt_w}_max{max_h}x{max_w}"


def _require_export_deps():
    try:
        import torch
        from safetensors.torch import load_file
        from transformers import SegformerConfig, SegformerForSemanticSegmentation
    except Exception as exc:
        raise RuntimeError(
            "ONNX export requires torch, transformers, and safetensors. "
            "Install dependencies in your Comfy environment."
        ) from exc
    return torch, load_file, SegformerConfig, SegformerForSemanticSegmentation


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
        "Failed to load safetensors into SegFormer model. "
        f"Missing keys: {missing_preview}. Unexpected keys: {unexpected_preview}."
    )


def export_onnx_from_local_model(
    model_dir: Path,
    model_name: str,
    precision: str,
    opset: int,
    opt_h: int,
    opt_w: int,
    force_reexport: bool = False,
) -> Path:
    if precision not in VALID_PRECISIONS:
        raise ValueError(f"precision must be one of {sorted(VALID_PRECISIONS)}")

    safe_name = sanitize_model_name(model_name)
    onnx_path = model_dir / f"{safe_name}_{precision}.onnx"
    onnx_data_path = _onnx_external_data_path(onnx_path)
    if onnx_path.exists() and not force_reexport:
        if _onnx_uses_external_data(onnx_path):
            force_reexport = True
        else:
            return onnx_path.resolve()

    if force_reexport:
        onnx_path.unlink(missing_ok=True)
        onnx_data_path.unlink(missing_ok=True)

    torch, load_file, SegformerConfig, SegformerForSemanticSegmentation = _require_export_deps()

    config_path = model_dir / "config.json"
    safetensors_path = model_dir / "model.safetensors"

    config = SegformerConfig.from_json_file(str(config_path))
    model = SegformerForSemanticSegmentation(config)
    state_dict = load_file(str(safetensors_path))
    _load_weights_into_model(model, state_dict)

    if precision == "fp16":
        if not torch.cuda.is_available():
            raise RuntimeError("FP16 ONNX export requires CUDA. Switch to fp32 or enable CUDA.")
        dtype = torch.float16
        device = torch.device("cuda")
        model = model.half()
    else:
        dtype = torch.float32
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device).eval()
    dummy_pixel_values = torch.zeros((1, 3, opt_h, opt_w), dtype=dtype, device=device)

    export_kwargs = {
        "opset_version": int(opset),
        "input_names": ["pixel_values"],
        "output_names": ["logits"],
        "dynamic_axes": {
            "pixel_values": {0: "batch", 2: "height", 3: "width"},
            "logits": {0: "batch", 2: "height", 3: "width"},
        },
    }
    with torch.no_grad():
        export_attempts = (
            {"external_data": False},
            {"use_external_data_format": False},
            {},
        )
        last_type_error = None
        for extra_kwargs in export_attempts:
            try:
                torch.onnx.export(
                    model,
                    (dummy_pixel_values,),
                    str(onnx_path),
                    **export_kwargs,
                    **extra_kwargs,
                )
                last_type_error = None
                break
            except TypeError as exc:
                last_type_error = exc
                continue
        if last_type_error is not None:
            raise last_type_error

    if _onnx_uses_external_data(onnx_path) and not onnx_data_path.exists():
        raise RuntimeError(
            "ONNX export produced an external-data model but the sidecar file is missing: "
            f"{onnx_data_path}"
        )

    return onnx_path.resolve()


def _require_tensorrt_deps():
    try:
        import tensorrt as trt
    except Exception as exc:
        raise RuntimeError("TensorRT engine build requires tensorrt Python bindings.") from exc
    return trt


def _format_parser_errors(parser) -> str:
    errors = []
    for idx in range(parser.num_errors):
        errors.append(str(parser.get_error(idx)))
    return "\n".join(errors) if errors else "unknown parser error"


def build_tensorrt_engine(
    onnx_path: Path,
    engine_path: Path,
    precision: str,
    min_h: int,
    min_w: int,
    opt_h: int,
    opt_w: int,
    max_h: int,
    max_w: int,
    workspace_gb: float,
    force_rebuild: bool = False,
) -> Path:
    if precision not in VALID_PRECISIONS:
        raise ValueError(f"precision must be one of {sorted(VALID_PRECISIONS)}")
    if workspace_gb <= 0:
        raise ValueError("workspace_gb must be > 0")
    validate_trt_profile(min_h=min_h, min_w=min_w, opt_h=opt_h, opt_w=opt_w, max_h=max_h, max_w=max_w)

    if engine_path.exists() and not force_rebuild:
        return engine_path.resolve()

    trt = _require_tensorrt_deps()
    logger = trt.Logger(trt.Logger.INFO)
    workspace_bytes = int(float(workspace_gb) * (1 << 30))

    with (
        trt.Builder(logger) as builder,
        builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network,
        trt.OnnxParser(network, logger) as parser,
    ):
        parse_ok = False
        parse_from_file = getattr(parser, "parse_from_file", None)
        if callable(parse_from_file):
            parse_ok = bool(parse_from_file(str(onnx_path)))
        else:
            with open(onnx_path, "rb") as handle:
                parse_ok = bool(parser.parse(handle.read()))
        if not parse_ok:
            error_text = _format_parser_errors(parser)
            missing_external_data = ""
            onnx_data_path = _onnx_external_data_path(onnx_path)
            if _onnx_uses_external_data(onnx_path) and not onnx_data_path.exists():
                missing_external_data = f"\nMissing ONNX external data file: {onnx_data_path}"
            raise RuntimeError(
                f"Failed to parse ONNX file {onnx_path}:\n{error_text}{missing_external_data}"
            )

        config = builder.create_builder_config()
        if precision == "fp16":
            config.set_flag(trt.BuilderFlag.FP16)
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)

        profile = builder.create_optimization_profile()
        profile.set_shape(
            "pixel_values",
            (1, 3, min_h, min_w),
            (1, 3, opt_h, opt_w),
            (1, 3, max_h, max_w),
        )
        config.add_optimization_profile(profile)

        engine_bytes = builder.build_serialized_network(network, config)
        if engine_bytes is None:
            raise RuntimeError("TensorRT engine build failed (builder returned None).")

        engine_path.parent.mkdir(parents=True, exist_ok=True)
        with open(engine_path, "wb") as handle:
            handle.write(engine_bytes)

    return engine_path.resolve()


def prepare_artifacts(
    model_name: str,
    auto_download: bool,
    safetensors_url: str,
    config_url: str,
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
) -> ArtifactPaths:
    validate_trt_profile(min_h=min_h, min_w=min_w, opt_h=opt_h, opt_w=opt_w, max_h=max_h, max_w=max_w)

    model_dir, _, _ = ensure_model_files(
        model_name=model_name,
        auto_download=auto_download,
        safetensors_url=safetensors_url,
        config_url=config_url,
    )

    onnx_path = export_onnx_from_local_model(
        model_dir=model_dir,
        model_name=model_name,
        precision=precision,
        opset=opset,
        opt_h=opt_h,
        opt_w=opt_w,
        force_reexport=force_reexport,
    )

    safe_name = sanitize_model_name(model_name)
    profile_tag = build_profile_signature(
        min_h=min_h,
        min_w=min_w,
        opt_h=opt_h,
        opt_w=opt_w,
        max_h=max_h,
        max_w=max_w,
    )
    engine_path = model_dir / f"{safe_name}_{precision}_{profile_tag}.engine"

    build_tensorrt_engine(
        onnx_path=onnx_path,
        engine_path=engine_path,
        precision=precision,
        min_h=min_h,
        min_w=min_w,
        opt_h=opt_h,
        opt_w=opt_w,
        max_h=max_h,
        max_w=max_w,
        workspace_gb=workspace_gb,
        force_rebuild=force_rebuild,
    )

    return ArtifactPaths(
        model_dir=model_dir.resolve(),
        onnx_path=onnx_path.resolve(),
        engine_path=engine_path.resolve(),
    )
