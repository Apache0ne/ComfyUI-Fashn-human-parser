from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Tuple

DEFAULT_SAFETENSORS_URL = "https://huggingface.co/fashn-ai/fashn-human-parser/resolve/main/model.safetensors"
DEFAULT_CONFIG_URL = "https://huggingface.co/fashn-ai/fashn-human-parser/resolve/main/config.json"

MIN_SAFETENSORS_BYTES = 1_000_000
MIN_CONFIG_BYTES = 200


def sanitize_model_name(model_name: str) -> str:
    name = model_name.strip()
    if not name:
        raise ValueError("model_name cannot be empty")
    sanitized = re.sub(r"[^A-Za-z0-9._-]", "_", name)
    if sanitized in {".", ".."}:
        raise ValueError(f"Invalid model_name: {model_name}")
    return sanitized


def _fallback_models_root() -> Path:
    module_dir = Path(__file__).resolve().parent
    for parent in [module_dir, *module_dir.parents]:
        if parent.name.lower() == "comfyui":
            return parent / "models"
    return module_dir / "ComfyUI" / "models"


def get_models_root() -> Path:
    try:
        import folder_paths

        models_dir = getattr(folder_paths, "models_dir", None)
        if models_dir:
            return Path(models_dir).expanduser().resolve()
    except Exception:
        pass
    return _fallback_models_root().resolve()


def ensure_fashn_models_root() -> Path:
    root = get_models_root() / "Fashn-parsers"
    root.mkdir(parents=True, exist_ok=True)
    return root


def register_model_folder() -> None:
    try:
        import folder_paths
    except Exception:
        return

    root = str(ensure_fashn_models_root())
    for folder_key in ("Fashn-parsers", "fashn_parsers"):
        try:
            folder_paths.add_model_folder_path(folder_key, root)
        except Exception:
            pass


def get_model_dir(model_name: str) -> Path:
    return ensure_fashn_models_root() / sanitize_model_name(model_name)


def _download_file(url: str, output_path: Path, timeout_seconds: int = 180) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    request = urllib.request.Request(url, headers={"User-Agent": "ComfyUI-Fashn-human-parser"})

    bytes_written = 0
    expected_length = None
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            content_length = response.headers.get("Content-Length")
            if content_length is not None:
                try:
                    expected_length = int(content_length)
                except ValueError:
                    expected_length = None

            with open(temp_path, "wb") as handle:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
                    bytes_written += len(chunk)

        if bytes_written <= 0:
            raise RuntimeError(f"Downloaded file is empty: {url}")
        if expected_length is not None and bytes_written != expected_length:
            raise RuntimeError(
                f"Download truncated for {url}. Expected {expected_length} bytes, got {bytes_written}"
            )

        temp_path.replace(output_path)
    except urllib.error.URLError as exc:
        temp_path.unlink(missing_ok=True)
        raise RuntimeError(f"Failed to download {url}: {exc}") from exc
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def _validate_safetensors_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing safetensors file: {path}")
    if path.stat().st_size < MIN_SAFETENSORS_BYTES:
        raise RuntimeError(f"Safetensors file looks invalid (too small): {path}")


def _validate_config_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing config file: {path}")
    if path.stat().st_size < MIN_CONFIG_BYTES:
        raise RuntimeError(f"Config file looks invalid (too small): {path}")

    with open(path, "r", encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict) or "model_type" not in config:
        raise RuntimeError(f"Invalid SegFormer config file: {path}")


def _ensure_single_file(path: Path, validator, auto_download: bool, url: str) -> None:
    if path.exists():
        try:
            validator(path)
            return
        except Exception:
            if not auto_download:
                raise
            path.unlink(missing_ok=True)

    if not auto_download:
        raise FileNotFoundError(
            f"Required file not found: {path}. Enable auto_download to fetch it."
        )

    _download_file(url, path)
    validator(path)


def ensure_model_files(
    model_name: str,
    auto_download: bool = True,
    safetensors_url: str = DEFAULT_SAFETENSORS_URL,
    config_url: str = DEFAULT_CONFIG_URL,
) -> Tuple[Path, Path, Path]:
    model_dir = get_model_dir(model_name)
    model_dir.mkdir(parents=True, exist_ok=True)

    safetensors_path = model_dir / "model.safetensors"
    config_path = model_dir / "config.json"

    _ensure_single_file(
        path=safetensors_path,
        validator=_validate_safetensors_file,
        auto_download=auto_download,
        url=safetensors_url,
    )
    _ensure_single_file(
        path=config_path,
        validator=_validate_config_file,
        auto_download=auto_download,
        url=config_url,
    )

    return model_dir.resolve(), safetensors_path.resolve(), config_path.resolve()
