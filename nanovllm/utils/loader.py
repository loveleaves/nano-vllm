import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    """默认权重加载：直接 copy（不做切分）。"""
    param.data.copy_(loaded_weight)


def load_model(model: nn.Module, path: str):
    """
    从 safetensors 文件加载模型权重，支持 packed 权重名称映射。

    支持模型类属性：
      weight_prefix_to_strip  — 剥离权重名前缀（如 VLM 的 "model.language_model."）
      weight_skip_prefixes    — 跳过匹配前缀的权重（如 "model.visual.", "mtp."）
      packed_modules_mapping  — HF 权重名后缀 → nano-vllm 参数名映射
    """
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    prefix_to_strip = getattr(model, "weight_prefix_to_strip", "")
    skip_prefixes   = getattr(model, "weight_skip_prefixes", ())

    files = sorted(glob(os.path.join(path, "*.safetensors")))
    assert files, f"没有找到 safetensors 文件：{path}"

    for file in files:
        with safe_open(file, framework="pt", device="cpu") as f:
            for weight_name in f.keys():
                # 跳过视觉编码器、MTP 头等无关权重
                if any(weight_name.startswith(p) for p in skip_prefixes):
                    continue

                # 剥离前缀得到参数路径
                param_name = weight_name
                if prefix_to_strip and weight_name.startswith(prefix_to_strip):
                    param_name = weight_name[len(prefix_to_strip):]

                for k in packed_modules_mapping:
                    if k in param_name:
                        v, shard_id = packed_modules_mapping[k]
                        mapped_name = param_name.replace(k, v)
                        try:
                            param = model.get_parameter(mapped_name)
                        except AttributeError:
                            break
                        loader = getattr(param, "weight_loader", default_weight_loader)
                        loader(param, f.get_tensor(weight_name), shard_id)
                        break
                else:
                    try:
                        param = model.get_parameter(param_name)
                    except AttributeError:
                        continue
                    loader = getattr(param, "weight_loader", default_weight_loader)
                    loader(param, f.get_tensor(weight_name))
