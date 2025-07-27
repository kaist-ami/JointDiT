import argparse
import copy
import glob
import json
import logging
import math
import os
import pdb
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from multiprocessing import Value
from typing import List, Optional, Tuple, Union

import toml
from tqdm import tqdm
import torch
import torch.nn as nn
from accelerate.utils import set_seed
from safetensors.torch import load_file, save_file
from PIL import Image
import numpy as np

# Add local script path for "sd-scripts" utilities
sys.path.append(os.path.join(os.path.dirname(__file__), 'sd-scripts'))
from library import (
    deepspeed_utils,
    strategy_flux,
    strategy_base,
    flux_train_utils,
    flux_utils,
    utils,
)
from library.device_utils import init_ipex, clean_memory_on_device
import library.train_util as train_util
import library.config_util as config_util
from library.config_util import ConfigSanitizer, BlueprintGenerator

# JointDiT-specific utilities
from jointdit_library.jointdit_utils import load_empty_flux_model, setup_jointdit_model
from jointdit_library.inference_pipeline import joint_generation, conditional_generation

init_ipex()
logger = logging.getLogger(__name__)

@lru_cache(maxsize=1)
def load_state_dict_from_cache(pretrained_ckpt: str) -> dict:
    """
    Load and cache a safetensors checkpoint from disk.
    Returns a state_dict mapping param names to tensors.
    """
    print(f"Loading pretrained weights from {pretrained_ckpt}")
    state_dict = load_file(pretrained_ckpt)
    return state_dict

def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser):
    """
    Ensure required arguments are provided based on the chosen generation mode.
    Exits with parser.error() if validation fails.
    """
    if args.gen_type == "joint_generation":
        if args.text_prompt is None:
            parser.error("--text_prompt is required for joint_generation mode.")
    elif args.gen_type == "depth_estimation":
        if args.input_image is None:
            parser.error("--input_image is required for depth_estimation mode.")
    elif args.gen_type == "depth_to_image":
        missing = []
        if args.text_prompt is None:
            missing.append("--text_prompt")
        if args.input_depth is None:
            missing.append("--input_depth")
        if missing:
            msg = ' and '.join(missing)
            parser.error(f"{msg} {'are' if len(missing)>1 else 'is'} required for depth_to_image mode.")
    else:
        parser.error(
            f"Unknown gen_type '{args.gen_type}'. Choose from: joint_generation, depth_estimation, depth_to_image."
        )

def setup_parser() -> argparse.ArgumentParser:
    """
    Define and return the command-line argument parser.
    """
    parser = argparse.ArgumentParser(
        description="Inference script for JointDiT built on top of Flux"
    )

    # Common model/data/configuration arguments
    train_util.add_sd_models_arguments(parser)
    train_util.add_dataset_arguments(parser, True, True, True)
    train_util.add_training_arguments(parser, False)
    config_util.add_config_arguments(parser)
    train_util.add_dit_training_arguments(parser)
    flux_train_utils.add_flux_train_arguments(parser)
    deepspeed_utils.add_deepspeed_arguments(parser)

    # JointDiT-specific arguments
    parser.add_argument(
        "--jointdit_addons_path",
        type=str,
        default="models/jointdit/jointdit_addons.safetensors",
        help=(
            "The path of the pretrained jointdit addons weight"
        )
    )
    parser.add_argument(
        "--gen_type",
        type=str,
        choices=["joint_generation", "depth_estimation", "depth_to_image"],
        required=True,
        help=(
            "Generation mode to run: 'joint_generation', 'depth_estimation', "
            "or 'depth_to_image'."
        )
    )
    parser.add_argument(
        "--text_prompt",
        type=str,
        default=None,
        help="Text prompt for joint_generation and depth_to_image modes."
    )
    parser.add_argument(
        "--output_resolution",
        type=int,
        nargs=2,
        default=[512, 512],
        help=(
            "Output resolution for joint_generation mode, as two integers: "
            "width height, e.g., 512 512 or 1024 1024."
        )
    )
    parser.add_argument(
        "--input_image",
        type=str,
        default=None,
        help="Path to input image for depth_estimation mode."
    )
    parser.add_argument(
        "--input_depth",
        type=str,
        default=None,
        help="Path to input depth map for depth_to_image mode."
    )

    return parser


def inference(args: argparse.Namespace):
    """
    Main inference routine:
      - Load and extend Flux base model to JointDiT
      - Load VAE, set up accelerator
      - Tokenize text, run text encoding
      - Call joint_generation to produce image+depth
    """
    # DeepSpeed & random seed setup
    deepspeed_utils.prepare_deepspeed_args(args)
    if args.seed is not None:
        set_seed(args.seed)

    # Prepare accelerator and dtypes
    accelerator = train_util.prepare_accelerator(args)
    weight_dtype, save_dtype = train_util.prepare_dtype(args)

    # 1) Instantiate an empty JointDiT (Flux) model skeleton
    _, jointdit = load_empty_flux_model(
        args.pretrained_model_name_or_path,
        weight_dtype,
        device="cpu",
        disable_mmap=args.disable_mmap_load_safetensors
    )

    jointdit = jointdit.to_empty(device=accelerator.device)

    # 2) Load base Flux weights only
    base_ckpt = load_state_dict_from_cache(args.pretrained_model_name_or_path)
    jointdit.load_state_dict(base_ckpt, strict=False)

    # 3) Attach JointDiT-specific adapters
    jointdit = setup_jointdit_model(jointdit, lora_rank=64)

    # 4) Load saved adapter weights
    addons = load_state_dict_from_cache(args.jointdit_addons_path)
    jointdit.load_state_dict(addons, strict=False)

    # 5) Final conversion: freeze & cast
    jointdit.requires_grad_(False)
    jointdit.to(weight_dtype)

    # block-swap memory optimization
    if args.blocks_to_swap and args.blocks_to_swap > 0:
        logger.info(f"Enabling block swap: {args.blocks_to_swap} blocks")
        jointdit.enable_block_swap(args.blocks_to_swap, accelerator.device)

    # 6) Load VAE for decoding
    ae = flux_utils.load_ae(args.ae, weight_dtype, device="cpu")
    ae.eval().requires_grad_(False)
    ae.to(accelerator.device, dtype=weight_dtype)
    clean_memory_on_device(accelerator.device)

    # Accelerator prepare (handles placement & optional deepspeed)
    jointdit = accelerator.prepare(jointdit, device_placement=[not args.blocks_to_swap])
    if args.blocks_to_swap and args.blocks_to_swap > 0:
        accelerator.unwrap_model(jointdit).move_to_device_except_swap_blocks(accelerator.device)

    # 7) Text models: CLIP + T5-XXL
    clip_l = flux_utils.load_clip_l(
        args.clip_l, weight_dtype, device="cpu", disable_mmap=args.disable_mmap_load_safetensors
    )
    t5xxl = flux_utils.load_t5xxl(
        args.t5xxl, weight_dtype, device="cpu", disable_mmap=args.disable_mmap_load_safetensors
    )
    for m in (clip_l, t5xxl):
        m.eval().requires_grad_(False)
        m.to(accelerator.device)

    # Set up tokenization & encoding strategies
    t5xxl_max_len = 512
    flux_tok = strategy_flux.FluxTokenizeStrategy(t5xxl_max_len)
    strategy_base.TokenizeStrategy.set_strategy(flux_tok)
    txt_enc = strategy_flux.FluxTextEncodingStrategy(args.apply_t5_attn_mask)
    strategy_base.TextEncodingStrategy.set_strategy(txt_enc)

    # Prepare prompts based on gen_type
    if args.gen_type in ("joint_generation", "depth_to_image"):
        text_prompt = [args.text_prompt]
    else:
        text_prompt = [" "]

    # Text encoding
    with torch.no_grad():
        tokens_and_masks = flux_tok.tokenize(text_prompt)
        inputs = [t.to(accelerator.device) for t in tokens_and_masks]
        conds = txt_enc.encode_tokens(flux_tok, [clip_l, t5xxl], inputs, args.apply_t5_attn_mask)
        if args.full_fp16:
            conds = [c.to(weight_dtype) for c in conds]
        l_pooled, t5_out, txt_ids, attn_mask = conds
        if not args.is_txt_ids_training:
            txt_ids = torch.zeros(t5_out.shape[0], t5_out.shape[1], 3, device=accelerator.device)
        if not args.is_attnmask_training:
            attn_mask = None

    # 8) Conduct joint generation or depth estimtation or depth_to_image
    if args.gen_type == "joint_generation":
        resolution = [args.output_resolution[0],  args.output_resolution[1]]
        image, depth_image, depth_raw = joint_generation( 
            accelerator, args, jointdit, ae,
            [l_pooled, t5_out, txt_ids, attn_mask], weight_dtype, args.output_resolution
        ) # We use the depth raw for 3d lifting visualization
    else:
        # For conditional modes: load and preprocess depth or image
        if args.gen_type == "depth_to_image":
            depth = np.load(args.input_depth)
            depth = (depth - depth.min()) / (depth.max() - depth.min())
            if args.depth_transform == "inverse":
                depth = 1 - depth
            depth_t = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0)
            depth_t = depth_t.repeat(1, 3, 1, 1)
            depth_t = depth_t.to(accelerator.device, dtype=weight_dtype) * 2 - 1
            latent = ae.encode(depth_t)

            resolution = [depth_t.shape[2],  depth_t.shape[3]]

        else:  # depth_estimation
            img = Image.open(args.input_image)
            arr = np.array(img) / 255.0
            img_t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
            img_t = img_t.to(accelerator.device, dtype=weight_dtype) * 2 - 1
            latent = ae.encode(img_t)
        
            resolution = [img_t.shape[2],  img_t.shape[3]]

        image, depth_image, depth_raw = conditional_generation(
            accelerator, args, jointdit, ae,
            latent, [l_pooled, t5_out, txt_ids, attn_mask],
            weight_dtype, resolution, args.gen_type
        ) # We use the depth raw for evaluating depth estimation performance

    # 16) Save outputs
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join("outputs", args.gen_type)
    os.makedirs(out_dir, exist_ok=True)

    image.save(os.path.join(out_dir, f"image_{timestamp}.png"))
    depth_image.save(os.path.join(out_dir, f"depth_{timestamp}.png"))


if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()

    # (1) Compute the path to the .json file that was saved alongside the .safetensors
    json_path = os.path.splitext(args.jointdit_addons_path)[0] + ".json"

    # (2) If the JSON exists, load its flags into args; otherwise set defaults
    if os.path.isfile(json_path):
        with open(json_path, "r") as jf:
            flags = json.load(jf)
        args.is_txt_ids_training   = flags.get("is_txt_ids_training", False)
        args.is_attnmask_training  = flags.get("is_attnmask_training", False)
        args.depth_transform       = flags.get("depth_transform", 'none')
        print(f"Loaded training flags from {json_path}: {flags}")
    else:
        args.is_txt_ids_training   = False
        args.is_attnmask_training  = False
        args.depth_transform       = "none"
        print(f"No JSON flags found at {json_path}, using defaults.")

    # Verify training args & load config file overrides
    train_util.verify_command_line_training_args(args)
    args = train_util.read_config_from_file(args, parser)

    # Validate required args for selected mode
    validate_args(args, parser)

    # Run the main inference pipeline
    inference(args)
