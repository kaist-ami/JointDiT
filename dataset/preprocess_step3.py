import argparse
import copy
import glob
import json
import shutil
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
from tqdm import tqdm

import toml
from tqdm import tqdm
import torch
import torch.nn as nn
from accelerate.utils import set_seed
from safetensors.torch import load_file, save_file
from PIL import Image
import numpy as np

# Add local script path for "sd-scripts" utilities
sys.path.append('./sd-scripts')
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

class PreprocessDataset(torch.utils.data.Dataset):
    def __init__(self, image_folder: str, depth_transform: str):
        self.image_dir = os.path.join(image_folder, "processed_images")
        self.depth_dir = os.path.join(image_folder, "depthmaps")
        self.text_dir = os.path.join(image_folder, "text_prompts")

        self.depth_transform = depth_transform

        # Collect filenames (without extension) that exist in all three folders
        image_paths = glob.glob(os.path.join(self.image_dir, "*.jpg")) + \
                      glob.glob(os.path.join(self.image_dir, "*.png"))
        
        self.filenames = []
        for path in image_paths:
            name = os.path.splitext(os.path.basename(path))[0]
            if os.path.exists(os.path.join(self.depth_dir, name + ".npy")) and \
               os.path.exists(os.path.join(self.text_dir, name + ".txt")):
                self.filenames.append(name)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]

        # Load image and normalize to [-1, 1]
        img_path = os.path.join(self.image_dir, fname + ".jpg")
        if not os.path.exists(img_path):
            img_path = os.path.join(self.image_dir, fname + ".png")
        image = Image.open(img_path).convert("RGB")
        image = np.array(image).astype(np.float32) / 255.0   # Normalize to [0, 1]
        image = torch.from_numpy(image).permute(2, 0, 1)     # Convert to [3, H, W]
        image = image * 2 - 1                                 # Normalize to [-1, 1]

        # Load depth map, normalize to [-1, 1], and convert to 3-channel
        depth = np.load(os.path.join(self.depth_dir, fname + ".npy"))
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)  # Normalize to [0, 1]
        if self.depth_transform == "inverse":
            depth = 1 - depth
        depth = depth * 2 - 1                                               # Normalize to [-1, 1]
        depth = torch.from_numpy(depth).float().unsqueeze(0).repeat(3, 1, 1)  # Convert to [3, H, W]

        # Load text prompt
        with open(os.path.join(self.text_dir, fname + ".txt"), "r") as f:
            prompt = f.read().strip()

        return {
            "image": image,
            "depth": depth,
            "text": prompt,
            "filename": fname,
        }

@lru_cache(maxsize=1)
def load_state_dict_from_cache(pretrained_ckpt: str) -> dict:
    """
    Load and cache a safetensors checkpoint from disk.
    Returns a state_dict mapping param names to tensors.
    """
    print(f"Loading pretrained weights from {pretrained_ckpt}")
    state_dict = load_file(pretrained_ckpt)
    return state_dict

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

    return parser


def preprocess_step3(args: argparse.Namespace):
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

    # 6) Load VAE for decoding
    clean_memory_on_device(accelerator.device)

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

    if args.full_fp16:
        train_util.patch_accelerator_for_fp16_training(accelerator)

    # 1. Load prompts from txt
    prompt_txt_path = "dataset/evaluation_prompts.txt"
    with open(prompt_txt_path, "r") as f:
        prompt_list = [line.strip() for line in f if line.strip()]

    prompt_list.insert(0, " ")

    eval_dir = "evaluation_prompts"
    os.makedirs(eval_dir, exist_ok=True)
    shutil.copy(prompt_txt_path, os.path.join(eval_dir, "evaluation_prompts.txt"))

    # 2. Copy the txt file to the evaluation_prompts directory
    empty_prompt_dir = "empty_prompts"
    os.makedirs(empty_prompt_dir, exist_ok=True)

    # 3. Create subdirectories for text prompt's latents
    latent_dirs = {
        "clip_latents": os.path.join(eval_dir, "clip_latents"),
        "t5_latents": os.path.join(eval_dir, "t5_latents"),
        "txt_ids": os.path.join(eval_dir, "txt_ids"),
        "attn_masks": os.path.join(eval_dir, "attn_masks"),
    }
    for path in latent_dirs.values():
        os.makedirs(path, exist_ok=True)
        
    # Create subdirectories for empty text prompt's latents
    empty_latent_dirs = {
        "clip_latents": os.path.join(empty_prompt_dir, "clip_latents"),
        "t5_latents": os.path.join(empty_prompt_dir, "t5_latents"),
        "txt_ids": os.path.join(empty_prompt_dir, "txt_ids"),
        "attn_masks": os.path.join(empty_prompt_dir, "attn_masks"),
    }
    for path in empty_latent_dirs.values():
        os.makedirs(path, exist_ok=True)

    # 4. Loop over prompts and encode + save
    for idx, text_prompt in enumerate(tqdm(prompt_list, desc="Encoding evaluation prompts")):
        
        file_name = f"{(idx):05d}"  # e.g., 00000, 00001 ...

        with torch.no_grad():
            tokens_and_masks = flux_tok.tokenize(text_prompt)
            inputs = [t.to(accelerator.device) for t in tokens_and_masks]
            conds = txt_enc.encode_tokens(flux_tok, [clip_l, t5xxl], inputs, args.apply_t5_attn_mask)
            if args.full_fp16:
                conds = [c.to(weight_dtype) for c in conds]
            l_pooled, t5_out, txt_ids, attn_mask = conds

        l_pooled = l_pooled[0]     # [768]
        t5_out = t5_out[0]         # [512, 4096]
        txt_ids = txt_ids[0]       # [512, 3]
        attn_mask = attn_mask[0]   # [512]

        # Save tensors
        if idx == 0:
            torch.save(l_pooled, os.path.join(empty_latent_dirs["clip_latents"], "empty.pt"))
            torch.save(t5_out, os.path.join(empty_latent_dirs["t5_latents"], "empty.pt"))
            torch.save(txt_ids, os.path.join(empty_latent_dirs["txt_ids"], "empty.pt"))
            torch.save(attn_mask.cpu().int(), os.path.join(empty_latent_dirs["attn_masks"], "empty.pt"))
        else:
            torch.save(l_pooled, os.path.join(latent_dirs["clip_latents"], file_name + ".pt"))
            torch.save(t5_out, os.path.join(latent_dirs["t5_latents"], file_name + ".pt"))
            torch.save(txt_ids, os.path.join(latent_dirs["txt_ids"], file_name + ".pt"))
            torch.save(attn_mask.cpu().int(), os.path.join(latent_dirs["attn_masks"], file_name + ".pt"))

if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()

    # Verify training args & load config file overrides
    train_util.verify_command_line_training_args(args)
    args = train_util.read_config_from_file(args, parser)

    # Run the main inference pipeline
    preprocess_step3(args)
