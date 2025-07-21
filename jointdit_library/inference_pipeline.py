import argparse
import math
import os
import numpy as np
import toml
import json
import time
from typing import Callable, Dict, List, Optional, Tuple, Union
from glob import glob 
import cv2

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import time


from accelerate import Accelerator, PartialState
from transformers import CLIPTextModel
from tqdm import tqdm
from PIL import Image
from safetensors.torch import save_file

from library import flux_models, flux_utils, strategy_base, train_util
from jointdit_library import jointdit_model
from library.device_utils import init_ipex, clean_memory_on_device
from library.flux_train_utils import get_schedule

init_ipex()

from library.utils import setup_logging, mem_eff_save_file

setup_logging()
import logging

logger = logging.getLogger(__name__)

def joint_generation(
    accelerator: Accelerator,
    args: argparse.Namespace,
    jointdit: jointdit_model.JointDiT,
    ae: flux_models.AutoEncoder,
    text_latents,
    weight_dtype,
    resolution,
):

    # negative_prompt = prompt_dict.get("negative_prompt")
    sample_steps = 20
    width = resolution[1]
    height = resolution[0]
    scale = args.guidance_scale
    seed = None
    controlnet_image = None
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
    else:
        # True random sample image generation
        torch.seed()
        torch.cuda.seed()

    height = max(64, height - height % 16)  # round to divisible by 16
    width = max(64, width - width % 16)  # round to divisible by 16
    logger.info(f"height: {height}")
    logger.info(f"width: {width}")
    logger.info(f"sample_steps: {sample_steps}")
    logger.info(f"scale: {scale}")
    if seed is not None:
        logger.info(f"seed: {seed}")

    # sample noise for a batch of size 2 (paired image + depth)
    packed_latent_height = height // 16
    packed_latent_width = width // 16

    noise = torch.randn(
        2,
        packed_latent_height * packed_latent_width,
        16 * 2 * 2,
        device=accelerator.device,
        dtype=weight_dtype,
        generator=torch.Generator(device=accelerator.device).manual_seed(seed) if seed is not None else None,
    )

    timesteps = get_schedule(sample_steps, noise.shape[1], shift=True)  # FLUX.1 dev -> shift=True
    img_ids = flux_utils.prepare_img_ids(2, packed_latent_height, packed_latent_width).to(accelerator.device, weight_dtype)

    # unpack text latents
    l_pooled = text_latents[0]
    t5_out = text_latents[1]
    txt_ids = text_latents[2]
    t5_attn_mask = text_latents[3]

    if t5_attn_mask is not None:
        t5_attn_mask = t5_attn_mask.to(accelerator.device)

    # duplicate for paired batch
    l_pooled = torch.cat([l_pooled, l_pooled], 0)
    t5_out = torch.cat([t5_out, t5_out], 0)
    txt_ids = torch.cat([txt_ids, txt_ids], 0)
   
    controlnet = None
    controlnet_image = None

    # denoising
    with accelerator.autocast(), torch.no_grad():
        x = denoise(jointdit, noise, img_ids, t5_out, txt_ids, l_pooled, args, timesteps=timesteps, guidance=scale, t5_attn_mask=t5_attn_mask, controlnet=controlnet, controlnet_img=controlnet_image)

    x = flux_utils.unpack_latents(x, packed_latent_height, packed_latent_width) 
    
    # unpack latents and decode
    with accelerator.autocast(), torch.no_grad():
        depth_x = ae.decode(x[1:])
        x = ae.decode(x[:1])

    # convert image and depth to PNG file (visulization)
    x = x.clamp(-1, 1)
    x = x.permute(0, 2, 3, 1)
    image = Image.fromarray((127.5 * (x + 1.0)).float().cpu().numpy().astype(np.uint8)[0])

    depth_x = depth_x.clamp(-1, 1)
    depth_x = depth_x.permute(0, 2, 3, 1)
    
    depth_image = Image.fromarray((127.5 * (depth_x + 1.0)).float().cpu().numpy().mean(-1).astype(np.uint8)[0], "L")

    return image, depth_image, depth_x


def denoise(
    model: jointdit_model.JointDiT,
    img: torch.Tensor,
    img_ids: torch.Tensor,
    txt: torch.Tensor,
    txt_ids: torch.Tensor,
    vec: torch.Tensor,
    args: Optional[argparse.Namespace],
    timesteps: list[float],
    guidance: float = 4.0,
    t5_attn_mask: Optional[torch.Tensor] = None,
    controlnet: Optional[flux_models.ControlNetFlux] = None,
    controlnet_img: Optional[torch.Tensor] = None,
):
    # this is ignored for schnell
    guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)

    for t_curr, t_prev in zip(tqdm(timesteps[:-1]), timesteps[1:]):
        t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
        model.prepare_block_swap_before_forward()
      
        block_samples = None
        block_single_samples = None

        # model forward
        pred = model(
            img=img,
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            timesteps=t_vec,
            y=vec,
            block_controlnet_hidden_states=block_samples,
            block_controlnet_single_hidden_states=block_single_samples,
            guidance=guidance_vec,
            txt_attention_mask=t5_attn_mask,
        )

        img = img + (t_prev - t_curr) * pred

    model.prepare_block_swap_before_forward()
    return img


def conditional_generation(
    accelerator: Accelerator,
    args: argparse.Namespace,
    jointdit: jointdit_model.JointDiT,
    ae: flux_models.AutoEncoder,
    condition_latent,
    text_latents,
    weight_dtype,
    resolution,
    gen_type,
):

    # negative_prompt = prompt_dict.get("negative_prompt")
    sample_steps = 20
    width = resolution[1]
    height = resolution[0]
    scale = args.guidance_scale
    seed = None
    controlnet_image = None
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
    else:
        # True random sample image generation
        torch.seed()
        torch.cuda.seed()

    height = max(64, height - height % 16)  # round to divisible by 16
    width = max(64, width - width % 16)  # round to divisible by 16
    logger.info(f"height: {height}")
    logger.info(f"width: {width}")
    logger.info(f"sample_steps: {sample_steps}")
    logger.info(f"scale: {scale}")
    if seed is not None:
        logger.info(f"seed: {seed}")

    # sample noise for a batch of size 2 (paired image + depth)
    packed_latent_height = height // 16
    packed_latent_width = width // 16

    noise = torch.randn(
        2,
        packed_latent_height * packed_latent_width,
        16 * 2 * 2,
        device=accelerator.device,
        dtype=weight_dtype,
        generator=torch.Generator(device=accelerator.device).manual_seed(seed) if seed is not None else None,
    )

    timesteps = get_schedule(sample_steps, noise.shape[1], shift=True)  # FLUX.1 dev -> shift=True
    img_ids = flux_utils.prepare_img_ids(2, packed_latent_height, packed_latent_width).to(accelerator.device, weight_dtype)
    t5_attn_mask = t5_attn_mask.to(accelerator.device) if args.apply_t5_attn_mask else None

    # unpack text latents
    l_pooled = text_latents[0]
    t5_out = text_latents[1]
    txt_ids = text_latents[2]
    t5_attn_mask = text_latents[3]

    if t5_attn_mask is not None:
        t5_attn_mask = t5_attn_mask.to(accelerator.device)

    # duplicate for paired batch
    l_pooled = torch.cat([l_pooled, l_pooled], 0)
    t5_out = torch.cat([t5_out, t5_out], 0)
    txt_ids = torch.cat([txt_ids, txt_ids], 0)
   
    controlnet = None
    controlnet_image = None

    # denoising
    with accelerator.autocast(), torch.no_grad():
        x = conditional_denoise(args, gen_type, jointdit, noise, img_ids, t5_out, txt_ids, l_pooled, condition_latent,  timesteps=timesteps, guidance=scale, t5_attn_mask=t5_attn_mask, controlnet=controlnet, controlnet_img=controlnet_image)
    
    x = flux_utils.unpack_latents(x, packed_latent_height, packed_latent_width) 
    
    # unpack latents and decode
    with accelerator.autocast(), torch.no_grad():
        depth_x = ae.decode(x[1:])
        x = ae.decode(x[:1])

    # convert image and depth to PNG file (visulization)
    x = x.clamp(-1, 1)
    x = x.permute(0, 2, 3, 1)
    image = Image.fromarray((127.5 * (x + 1.0)).float().cpu().numpy().astype(np.uint8)[0])

    depth_x = depth_x.clamp(-1, 1)
    depth_x = depth_x.permute(0, 2, 3, 1)
    
    depth_image = Image.fromarray((127.5 * (depth_x + 1.0)).float().cpu().numpy().mean(-1).astype(np.uint8)[0], "L")

    return image, depth_image, depth_x


def conditional_denoise(
    args,
    gen_type, 
    model: jointdit_model.JointDiT,
    img: torch.Tensor,
    img_ids: torch.Tensor,
    txt: torch.Tensor,
    txt_ids: torch.Tensor,
    vec: torch.Tensor,
    condition_latent: torch.Tensor,
    timesteps: list[float],
    guidance: float = 4.0,
    t5_attn_mask: Optional[torch.Tensor] = None,
    controlnet: Optional[flux_models.ControlNetFlux] = None,
    controlnet_img: Optional[torch.Tensor] = None,
):  

    # this is ignored for schnell
    guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)

    # conditional noise
    conditional_noise = torch.randn_like(condition_latent)

    t_cond = timesteps[-1]
    
    conditional_noisy_latent = (1 - t_cond) * condition_latent + t_cond * conditional_noise
    packed_conditional_noisy_latent = flux_utils.pack_latents(conditional_noisy_latent) # this is clean latent of conditional input (image or depth)

    for t_curr, t_prev in zip(tqdm(timesteps[:-1]), timesteps[1:]):        
        t_vec = torch.full((1,), t_curr, dtype=img.dtype, device=img.device)

        if gen_type == "depth_to_image":
            t_cond_vec = torch.full((1,), t_cond, dtype=img.dtype, device=img.device)
            t_vec = torch.cat([t_vec, t_cond_vec], dim=0)
            img[1:] = packed_conditional_noisy_latent # replace the conditional latent as clean latent

        else: # depth_estimation
            t_cond_vec = torch.full((1,), t_cond, dtype=img.dtype, device=img.device)
            t_vec = torch.cat([t_cond_vec, t_vec], dim=0)        
            img[:1] = packed_conditional_noisy_latent # replace the conditional latent as clean latent

        model.prepare_block_swap_before_forward()
    
        block_samples = None
        block_single_samples = None
        
        # model forward
        pred = model(
            img=img,
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            timesteps=t_vec,
            y=vec,
            block_controlnet_hidden_states=block_samples,
            block_controlnet_single_hidden_states=block_single_samples,
            guidance=guidance_vec,
            txt_attention_mask=t5_attn_mask,
        )

        img = img + (t_prev - t_curr) * pred

    model.prepare_block_swap_before_forward()
    return img