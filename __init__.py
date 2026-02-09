try:
    from .model_store import ensure_fashn_models_root, register_model_folder
    from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
except Exception:
    from model_store import ensure_fashn_models_root, register_model_folder
    from nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

ensure_fashn_models_root()
register_model_folder()

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
