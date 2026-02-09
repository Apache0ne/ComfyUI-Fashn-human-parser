# ComfyUI-Fashn-human-parser-

ComfyUI custom nodes for FASHN human parsing with a local conversion pipeline:

- `model.safetensors` + `config.json` -> dynamic `.onnx`
- `.onnx` -> TensorRT `.engine`
- image inference node with segmentation/overlay output and class-switch masks

This implementation uses the local parser logic adapted from:

- `parser-onnx-dump/fashn_human_parser_trt.py`
- `parser-onnx-dump/export_onnx_fp16.py`
- `parser-onnx-dump/build_trt_fp16.py`

It does not rely on the external `fashn-parser` package at runtime.

## Model Storage

All files are stored in:

- `ComfyUI/models/Fashn-parsers/<model_name>/`

Artifacts created by the build node:

- `model.safetensors`
- `config.json`
- `<model_name>_fp16.onnx` or `<model_name>_fp32.onnx`
- `<model_name>_fp16_<profile>.engine` or `<model_name>_fp32_<profile>.engine`

Default download URLs:

- `https://huggingface.co/fashn-ai/fashn-human-parser/resolve/main/model.safetensors`
- `https://huggingface.co/fashn-ai/fashn-human-parser/resolve/main/config.json`

## Nodes

### `FashnParserLoadTRT`

Loads a TensorRT engine from a dropdown populated by scanning:

- `ComfyUI/models/Fashn-parsers/**/*.engine`

### `FashnParserLoadSafetensors`

Loads the default FASHN safetensors model (`fashn-human-parser`) with only:

- `precision` (`fp16`/`fp32`)
- `device` (`auto`/`cuda`/`cpu`)

### `FashnParserBuildEngine`

Builds or reuses ONNX + TensorRT engine from local model files.

Inputs include:

- `precision` (`fp16`/`fp32`), `opset`
- TensorRT dynamic profile (`min/opt/max` HxW)
- `workspace_gb`, `force_reexport`, `force_rebuild`, `keep_onnx`

Outputs:

- `engine_path`

### `FashnParserRun`

Runs human parsing on Comfy `IMAGE` batch.

Inputs:

- `image`, `parser_model` (from TRT or safetensors loader)
- `overlay_alpha`
- per-class switches (`background`, `face`, `hair`, ..., `jewelry`)

Outputs:

- `overlay_image` (`IMAGE`)
- `segmentation_image` (`IMAGE`)
- `selected_overlay_image` (`IMAGE`) selected classes only on overlay output
- `selected_segmentation_image` (`IMAGE`) selected classes only on segmentation output
- `selected_mask` (`MASK`) binary union from enabled class switches

## Install

1. Place this folder in your ComfyUI custom nodes directory.
2. Install dependencies in the same Python environment as ComfyUI:

```bash
pip install -r requirements.txt
```

3. Restart ComfyUI.

## Notes

- FP16 ONNX export requires CUDA.
- TensorRT runtime/build requires TensorRT installed for your CUDA version.
