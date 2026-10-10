def has_flash_attn():
    try:
        import torch_npu
        return True
    except ImportError:
        pass
    try:
        import flash_attn
        return True
    except ImportError:
        return False


def auto_get_module_keys(module, max_depth=0, prefix_list=None, current_depth=0, current_prefix=""):
    if current_depth > max_depth:
        return []

    module_keys = []
    for name, sub_module in module.named_children():
        full_name = f"{current_prefix}.{name}" if current_prefix else name
        if prefix_list is None or any(full_name.startswith(prefix) for prefix in prefix_list):
            module_keys.append(full_name)
        module_keys.extend(auto_get_module_keys(sub_module, max_depth, prefix_list, current_depth + 1, full_name))
    return module_keys


def is_module_trainable(module):
    params = list(module.parameters(recurse=False))
    if params:
        return all(p.requires_grad for p in params)
    else:

        return True


def auto_get_trainable_modules(module, prefix="", max_depth=None):
    children = list(module.named_children())

    if (max_depth is not None and max_depth <= 0) or not children:
        return [prefix] if prefix and is_module_trainable(module) else []

    child_keys = []
    all_children_trainable = True
    for name, child in children:
        full_name = f"{prefix}.{name}" if prefix else name

        keys = auto_get_trainable_modules(child, full_name, None if max_depth is None else max_depth - 1)
        if not keys:

            if is_module_trainable(child):
                keys = [full_name]
            else:
                all_children_trainable = False
        else:

            if len(keys) > 1:
                all_children_trainable = False
        child_keys.extend(keys)

    if is_module_trainable(module) and all_children_trainable and child_keys:
        return [prefix] if prefix else child_keys
    else:
        return child_keys


class Registry:
    def __init__(self, name: str):
        self.name = name
        self._registry = {}

    def register(self, key: str):
        def decorator(framework_class):
            if key in self._registry:

                pass
            self._registry[key] = framework_class
            return framework_class

        return decorator

    def __getitem__(self, key):
        return self._registry[key]

    def list(self):
        return {k: v for k, v in self._registry.items()}


POLICY_REGISTRY = Registry("policies")


import numpy as np

from cdp.training.logging import initialize_overwatch


overwatch = initialize_overwatch(__name__)


class FrameworkTools:

    @staticmethod
    def check_unnorm_key(norm_stats: dict, unnorm_key: str | None) -> str:
        if unnorm_key is None:
            assert len(norm_stats) == 1, (
                f"Your model was trained on more than one dataset, "
                f"please pass a `unnorm_key` from the following options to choose the statistics "
                f"used for un-normalizing actions: {norm_stats.keys()}"
            )
            unnorm_key = next(iter(norm_stats.keys()))

        assert unnorm_key in norm_stats, (
            f"The `unnorm_key` you chose is not in the set of available dataset statistics, "
            f"please choose from: {norm_stats.keys()}"
        )
        return unnorm_key

    @staticmethod
    def get_action_stats(norm_stats: dict, unnorm_key: str | None = None) -> dict:
        unnorm_key = FrameworkTools.check_unnorm_key(norm_stats, unnorm_key)
        return norm_stats[unnorm_key]["action"]

    @staticmethod
    def unnormalize_actions(normalized_actions: np.ndarray, action_norm_stats: dict,
                            normalization_mode: str | None = None) -> np.ndarray:
        values = np.asarray(normalized_actions)
        dimension = values.shape[-1]
        if normalization_mode is not None:
            modes = np.full(dimension, normalization_mode, dtype=object)
        else:
            modes = np.asarray(action_norm_stats.get(
                "normalization_modes", ["min_max"] * dimension
            ), dtype=object)
        if "mask" in action_norm_stats:
            modes = modes.copy()
            modes[~np.asarray(action_norm_stats["mask"], dtype=bool)] = "binary"
        actions = values.astype(np.result_type(values.dtype, np.float32), copy=True)
        for mode in set(modes):
            indices = modes == mode
            current = values[..., indices]
            if mode in {"min_max", "q99"}:
                low_key, high_key = ("min", "max") if mode == "min_max" else ("q01", "q99")
                low = np.asarray(action_norm_stats[low_key])[indices]
                high = np.asarray(action_norm_stats[high_key])[indices]
                actions[..., indices] = (current + 1) / 2 * (high - low) + low
            elif mode == "mean_std":
                mean = np.asarray(action_norm_stats["mean"])[indices]
                std = np.asarray(action_norm_stats["std"])[indices]
                actions[..., indices] = current * std + mean
            elif mode == "binary":
                thresholds = np.broadcast_to(action_norm_stats.get("binary_thresholds", 0.5), (dimension,))
                actions[..., indices] = current > thresholds[indices]
            elif mode != "none":
                raise ValueError(f"Unknown normalization mode: {mode}")
        return actions

    @staticmethod
    def get_trainable_module_keys(model, max_depth: int = 1) -> list:
        return auto_get_trainable_modules(model, max_depth=max_depth)
