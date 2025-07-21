import json
import os
from dataclasses import replace
from typing import List, Optional, Tuple, Union

import einops
import torch
import torch.nn as nn
from accelerate import init_empty_weights
from safetensors import safe_open
from safetensors.torch import load_file
from transformers import CLIPConfig, CLIPTextModel, T5Config, T5EncoderModel

from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)

from jointdit_library import jointdit_model
from jointdit_library.lora_adapter import (
    add_linear_to_double_blocks,
    add_linear_to_single_blocks,
    add_lora_to_attention,
)

from library.utils import load_safetensors
from library.flux_utils import analyze_checkpoint_state

MODEL_VERSION_FLUX_V1 = "flux1"
MODEL_NAME_DEV = "dev"
MODEL_NAME_SCHNELL = "schnell"

def load_empty_flux_model(
    ckpt_path: str, dtype: Optional[torch.dtype], device: Union[str, torch.device], disable_mmap: bool = False, noload = False
) -> Tuple[bool, jointdit_model.JointDiT]:
    is_diffusers, is_schnell, (num_double_blocks, num_single_blocks), ckpt_paths = analyze_checkpoint_state(ckpt_path)
    name = MODEL_NAME_DEV if not is_schnell else MODEL_NAME_SCHNELL

    # build model
    logger.info(f"Building Flux model {name} from {'Diffusers' if is_diffusers else 'BFL'} checkpoint")
    with torch.device("meta"):
        params = jointdit_model.configs[name].params

        # set the number of blocks
        if params.depth != num_double_blocks:
            logger.info(f"Setting the number of double blocks from {params.depth} to {num_double_blocks}")
            params = replace(params, depth=num_double_blocks)
        if params.depth_single_blocks != num_single_blocks:
            logger.info(f"Setting the number of single blocks from {params.depth_single_blocks} to {num_single_blocks}")
            params = replace(params, depth_single_blocks=num_single_blocks)

        model = jointdit_model.JointDiT(params)
        if dtype is not None:
            model = model.to(dtype)

    return is_schnell, model

def setup_jointdit_model(
    model: nn.Module,
    lora_rank: int = 64,
    input_phase_configs: Optional[List[Tuple[str, int, float]]] = None
) -> nn.Module:
    """
    Extend a Flux model with JointDiT adapters:
    - Insert Linear adapters (joint1/joint2) into each block.
    - Attach LoRA modules to input-phase layers and attention layers.
    """
    add_linear_to_double_blocks(model)
    add_linear_to_single_blocks(model)

    if input_phase_configs is None:
        input_phase_configs = [
            ("vector_in.in_layer", 512, 512/2),
            ("vector_in.out_layer", 1024, 1024/2),
            ("txt_in", 1024, 1024/2),
        ]
    for name, r, alpha in input_phase_configs:
        add_lora_to_attention(model, name, r=r, alpha=alpha)

    lora_alpha = lora_rank / 2
    for name in [
        "img_mod.lin", "img_attn.qkv", "txt_mod.lin",
        "txt_attn.qkv", "img_attn.proj", "txt_attn.proj"
    ]:
        add_lora_to_attention(model, name, r=lora_rank, alpha=lora_alpha)

    for name in ["linear1", "modulation.lin"]:
        add_lora_to_attention(model, name, r=lora_rank, alpha=lora_alpha)

    return model