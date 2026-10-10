from transformers import AutoConfig


def get_vlm_model(config):
    model_type = AutoConfig.from_pretrained(config.framework.qwenvl.base_vlm).model_type
    if model_type == "qwen2_5_vl":
        from .qwen2_5 import _QWen_VL_Interface
        return _QWen_VL_Interface(config)
    if model_type == "qwen3_vl":
        from .qwen3 import _QWen3_VL_Interface
        return _QWen3_VL_Interface(config)
    raise NotImplementedError(f"VLM model type {model_type} not implemented")
