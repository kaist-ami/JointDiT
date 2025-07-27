# training with captions

# Swap blocks between CPU and GPU:
# This implementation is inspired by and based on the work of 2kpr.
# Many thanks to 2kpr for the original concept and implementation of memory-efficient offloading.
# The original idea has been adapted and extended to fit the current project's needs.

# Key features:
# - CPU offloading during forward and backward passes
# - Use of fused optimizer and grad_hook for efficient gradient processing
# - Per-block fused optimizer instances

import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import math
import os
import sys
from multiprocessing import Value
import time
from typing import List, Optional, Tuple, Union
import toml

sys.path.append(os.path.join(os.path.dirname(__file__), 'sd-scripts'))

from tqdm import tqdm

import torch
import torch.nn as nn
from library import utils
from library.device_utils import init_ipex, clean_memory_on_device

init_ipex()

from accelerate.utils import set_seed
from library import deepspeed_utils, flux_train_utils, flux_utils, strategy_base, strategy_flux
from library.sd3_train_utils import FlowMatchEulerDiscreteScheduler

import library.train_util as train_util

from library.utils import setup_logging, add_logging_arguments

setup_logging()
import logging

logger = logging.getLogger(__name__)

import library.config_util as config_util

# import library.sdxl_train_util as sdxl_train_util
from library.config_util import (
    ConfigSanitizer,
    BlueprintGenerator,
)
from library.custom_train_functions import apply_masked_loss, add_custom_train_arguments
from dataset.jointdit_dataset import JointDataset
from safetensors.torch import load_file
from jointdit_library import inference_pipeline
from jointdit_library import jointdit_utils
from jointdit_library.jointdit_utils import load_empty_flux_model, setup_jointdit_model, save_added_params

import pdb
import json

def load_state_dict_from_cache(pretrained_ckpt):
    print("Loading model...") 
    pretrained_weight_path = pretrained_ckpt
    print("Loading pretrained weights from %s" % pretrained_weight_path)
    state_dict = load_file(pretrained_weight_path)
    return state_dict


class ForkedPdb(pdb.Pdb):
    """A Pdb subclass that may be used
    from a forked multiprocessing child

    """
    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin

def train(args):

    args.output_dir = os.path.join(args.output_dir, args.exp_name)
    os.makedirs(args.output_dir, exist_ok=True)
    
    train_util.verify_training_args(args)
    train_util.prepare_dataset_args(args, True)
    deepspeed_utils.prepare_deepspeed_args(args)
    setup_logging(args, reset=True)

    if args.cpu_offload_checkpointing and not args.gradient_checkpointing:
        logger.warning(
            "cpu_offload_checkpointing is enabled, so gradient_checkpointing is also enabled / cpu_offload_checkpointingが有効になっているため、gradient_checkpointingも有効になります"
        )
        args.gradient_checkpointing = True

    assert (
        args.blocks_to_swap is None or args.blocks_to_swap == 0
    ) or not args.cpu_offload_checkpointing, (
        "blocks_to_swap is not supported with cpu_offload_checkpointing / blocks_to_swapはcpu_offload_checkpointingと併用できません"
    )

    current_epoch = Value("i", 0)
    current_step = Value("i", 0)

    _, is_schnell, _, _ = flux_utils.analyze_checkpoint_state(args.pretrained_model_name_or_path)
   
    # acceleratorを準備する
    logger.info("prepare accelerator")
    accelerator = train_util.prepare_accelerator(args)

    # mixed precisionに対応した型を用意しておき適宜castする
    weight_dtype, save_dtype = train_util.prepare_dtype(args)

    # load FLUX
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
    jointdit.requires_grad_(False)

    # 3) Attach JointDiT-specific adapters
    jointdit = setup_jointdit_model(jointdit, lora_rank=64)
    
    if args.gradient_checkpointing:
        jointdit.enable_gradient_checkpointing(cpu_offload=args.cpu_offload_checkpointing)

    # backward compatibility
    if args.blocks_to_swap is None:
        blocks_to_swap = args.double_blocks_to_swap or 0
        if args.single_blocks_to_swap is not None:
            blocks_to_swap += args.single_blocks_to_swap // 2
        if blocks_to_swap > 0:
            logger.warning(
                "double_blocks_to_swap and single_blocks_to_swap are deprecated. Use blocks_to_swap instead."
                " / double_blocks_to_swapとsingle_blocks_to_swapは非推奨です。blocks_to_swapを使ってください。"
            )
            logger.info(
                f"double_blocks_to_swap={args.double_blocks_to_swap} and single_blocks_to_swap={args.single_blocks_to_swap} are converted to blocks_to_swap={blocks_to_swap}."
            )
            args.blocks_to_swap = blocks_to_swap
        del blocks_to_swap

    is_swapping_blocks = args.blocks_to_swap is not None and args.blocks_to_swap > 0

    if is_swapping_blocks:
        # Swap blocks between CPU and GPU to reduce memory usage, in forward and backward passes.
        # This idea is based on 2kpr's great work. Thank you!
        logger.info(f"enable block swap: blocks_to_swap={args.blocks_to_swap}")
        jointdit.enable_block_swap(args.blocks_to_swap, accelerator.device)

    # load VAE here if not cached
    ae = None
    ae = flux_utils.load_ae(args.ae, weight_dtype, "cpu")
    ae.requires_grad_(False)
    ae.eval()
    ae.to(accelerator.device, dtype=weight_dtype)

    training_models = []
    params_to_optimize = []
    training_models.append(jointdit)
    name_and_params = list(jointdit.named_parameters())
    params_to_optimize.append({
        "params": [p for _, p in name_and_params if p.requires_grad],  # requires_grad 조건 추가
        "lr": args.learning_rate
    })  
    param_names = [[n for n, p in name_and_params if p.requires_grad]]

    # calculate number of trainable parameters
    n_params = 0
    for group in params_to_optimize:
        for p in group["params"]:
            n_params += p.numel()

    accelerator.print(f"number of trainable parameters: {n_params}")    
    accelerator.print("prepare optimizer, data loader etc.")

    if args.blockwise_fused_optimizers:
        # fused backward pass: https://pytorch.org/tutorials/intermediate/optimizer_step_in_backward_tutorial.html
        # Instead of creating an optimizer for all parameters as in the tutorial, we create an optimizer for each block of parameters.
        # This balances memory usage and management complexity.
        # split params into groups. currently different learning rates are not supported
        grouped_params = []
        param_group = {}
        for group in params_to_optimize:
            named_parameters = list(jointdit.named_parameters())
            assert len(named_parameters) == len(group["params"]), "number of parameters does not match"
            for p, np in zip(group["params"], named_parameters):
                # determine target layer and block index for each parameter
                block_type = "other"  # double, single or other
                if np[0].startswith("double_blocks"):
                    block_index = int(np[0].split(".")[1])
                    block_type = "double"
                elif np[0].startswith("single_blocks"):
                    block_index = int(np[0].split(".")[1])
                    block_type = "single"
                else:
                    block_index = -1

                param_group_key = (block_type, block_index)
                if param_group_key not in param_group:
                    param_group[param_group_key] = []
                param_group[param_group_key].append(p)

        block_types_and_indices = []
        for param_group_key, param_group in param_group.items():
            block_types_and_indices.append(param_group_key)
            grouped_params.append({"params": param_group, "lr": args.learning_rate})

            num_params = 0
            for p in param_group:
                num_params += p.numel()
            accelerator.print(f"block {param_group_key}: {num_params} parameters")

        # prepare optimizers for each group
        optimizers = []
        for group in grouped_params:
            _, _, optimizer = train_util.get_optimizer(args, trainable_params=[group])
            optimizers.append(optimizer)
        optimizer = optimizers[0]  # avoid error in the following code

        logger.info(f"using {len(optimizers)} optimizers for blockwise fused optimizers")

        if train_util.is_schedulefree_optimizer(optimizers[0], args):
            raise ValueError("Schedule-free optimizer is not supported with blockwise fused optimizers")
        optimizer_train_fn = lambda: None  # dummy function
        optimizer_eval_fn = lambda: None  # dummy function
    else:
        _, _, optimizer = train_util.get_optimizer(args, trainable_params=params_to_optimize)
        optimizer_train_fn, optimizer_eval_fn = train_util.get_optimizer_train_eval_fn(optimizer, args)

    # prepare dataloader
    train_dataset_group = JointDataset(args=args, image_folder=args.image_folder, drop_text=0.10, depth_transform=args.depth_transform)    

    # DataLoaderのプロセス数：0 は persistent_workers が使えないので注意
    n_workers = os.cpu_count() // 8 # cpu_count or max_data_loader_n_workers
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset_group,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=n_workers,
        pin_memory=True, 
        prefetch_factor=4,
        persistent_workers=args.persistent_data_loader_workers,
    )

    # 学習ステップ数を計算する
    if args.max_train_epochs is not None:
        args.max_train_steps = args.max_train_epochs * math.ceil(
            len(train_dataloader) / accelerator.num_processes / args.gradient_accumulation_steps
        )
        accelerator.print(
            f"override steps. steps for {args.max_train_epochs} epochs is / 指定エポックまでのステップ数: {args.max_train_steps}"
        )

    # データセット側にも学習ステップを送信
    if args.blockwise_fused_optimizers:
        # prepare lr schedulers for each optimizer
        lr_schedulers = [train_util.get_scheduler_fix(args, optimizer, accelerator.num_processes) for optimizer in optimizers]
        lr_scheduler = lr_schedulers[0]  # avoid error in the following code
    else:
        lr_scheduler = train_util.get_scheduler_fix(args, optimizer, accelerator.num_processes)

    # 実験的機能：勾配も含めたfp16/bf16学習を行う　モデル全体をfp16/bf16にする
    if args.full_fp16:
        assert (
            args.mixed_precision == "fp16"
        ), "full_fp16 requires mixed precision='fp16' / full_fp16を使う場合はmixed_precision='fp16'を指定してください。"
        accelerator.print("enable full fp16 training.")
        jointdit.to(weight_dtype)
    elif args.full_bf16:
        assert (
            args.mixed_precision == "bf16"
        ), "full_bf16 requires mixed precision='bf16' / full_bf16を使う場合はmixed_precision='bf16'を指定してください。"
        accelerator.print("enable full bf16 training.")
        jointdit.to(weight_dtype)

    clean_memory_on_device(accelerator.device)

    if args.deepspeed:
        ds_model = deepspeed_utils.prepare_deepspeed_model(args, mmdit=jointdit)
        # most of ZeRO stage uses optimizer partitioning, so we have to prepare optimizer and ds_model at the same time. # pull/1139#issuecomment-1986790007
        ds_model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            ds_model, optimizer, train_dataloader, lr_scheduler
        )
        training_models = [ds_model]
    else:
        # accelerator does some magic
        # if we doesn't swap blocks, we can move the model to device
        jointdit = accelerator.prepare(jointdit, device_placement=[not is_swapping_blocks])
        if is_swapping_blocks:
            accelerator.unwrap_model(jointdit).move_to_device_except_swap_blocks(accelerator.device)  # reduce peak memory usage
        optimizer, train_dataloader, lr_scheduler = accelerator.prepare(optimizer, train_dataloader, lr_scheduler)

    # 実験的機能：勾配も含めたfp16学習を行う　PyTorchにパッチを当ててfp16でのgrad scaleを有効にする
    if args.full_fp16:
        # During deepseed training, accelerate not handles fp16/bf16|mixed precision directly via scaler. Let deepspeed engine do.
        # -> But we think it's ok to patch accelerator even if deepspeed is enabled.
        train_util.patch_accelerator_for_fp16_training(accelerator)

    # resumeする
    train_util.resume_from_local_or_hf_if_specified(accelerator, args)

    if args.fused_backward_pass:
        # use fused optimizer for backward pass: other optimizers will be supported in the future
        import library.adafactor_fused

        library.adafactor_fused.patch_adafactor_fused(optimizer)

        for param_group, param_name_group in zip(optimizer.param_groups, param_names):
            for parameter, param_name in zip(param_group["params"], param_name_group):
                if parameter.requires_grad:

                    def create_grad_hook(p_name, p_group):
                        def grad_hook(tensor: torch.Tensor):
                            if accelerator.sync_gradients and args.max_grad_norm != 0.0:
                                accelerator.clip_grad_norm_(tensor, args.max_grad_norm)
                            optimizer.step_param(tensor, p_group)
                            tensor.grad = None

                        return grad_hook

                    parameter.register_post_accumulate_grad_hook(create_grad_hook(param_name, param_group))

    elif args.blockwise_fused_optimizers:
        # prepare for additional optimizers and lr schedulers
        for i in range(1, len(optimizers)):
            optimizers[i] = accelerator.prepare(optimizers[i])
            lr_schedulers[i] = accelerator.prepare(lr_schedulers[i])

        # counters are used to determine when to step the optimizer
        global optimizer_hooked_count
        global num_parameters_per_group
        global parameter_optimizer_map

        optimizer_hooked_count = {}
        num_parameters_per_group = [0] * len(optimizers)
        parameter_optimizer_map = {}

        for opt_idx, optimizer in enumerate(optimizers):
            for param_group in optimizer.param_groups:
                for parameter in param_group["params"]:
                    if parameter.requires_grad:

                        def grad_hook(parameter: torch.Tensor):
                            if accelerator.sync_gradients and args.max_grad_norm != 0.0:
                                accelerator.clip_grad_norm_(parameter, args.max_grad_norm)

                            i = parameter_optimizer_map[parameter]
                            optimizer_hooked_count[i] += 1
                            if optimizer_hooked_count[i] == num_parameters_per_group[i]:
                                optimizers[i].step()
                                optimizers[i].zero_grad(set_to_none=True)

                        parameter.register_post_accumulate_grad_hook(grad_hook)
                        parameter_optimizer_map[parameter] = opt_idx
                        num_parameters_per_group[opt_idx] += 1

    # epoch数を計算する
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    if (args.save_n_epoch_ratio is not None) and (args.save_n_epoch_ratio > 0):
        args.save_every_n_epochs = math.floor(num_train_epochs / args.save_n_epoch_ratio) or 1

    # 学習する
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    accelerator.print("running training / 学習開始")
    accelerator.print("Total batch size: %d" % total_batch_size)
    accelerator.print(f"  num examples / サンプル数: {train_dataset_group.__len__()}")
    accelerator.print(f"  num batches per epoch / 1epochのバッチ数: {len(train_dataloader)}")
    accelerator.print(f"  num epochs / epoch数: {num_train_epochs}")
    accelerator.print(f"  gradient accumulation steps / 勾配を合計するステップ数 = {args.gradient_accumulation_steps}")
    accelerator.print(f"  total optimization steps / 学習ステップ数: {args.max_train_steps}")

    progress_bar = tqdm(range(args.max_train_steps), smoothing=0, disable=not accelerator.is_local_main_process, desc="steps")

    global_step = 0

    noise_scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=args.discrete_flow_shift)
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)

    if accelerator.is_main_process:
        init_kwargs = {}
        if args.wandb_run_name:
            init_kwargs["wandb"] = {"name": args.wandb_run_name}
        if args.log_tracker_config is not None:
            init_kwargs = toml.load(args.log_tracker_config)
        accelerator.init_trackers(
            "finetuning" if args.log_tracker_name is None else args.log_tracker_name,
            config=train_util.get_sanitized_config_or_none(args),
            init_kwargs=init_kwargs,
        )

    if is_swapping_blocks:
        accelerator.unwrap_model(jointdit).prepare_block_swap_before_forward()

    if args.is_latent_training:        
        clip_l = None
        t5xxl = None
    else:
        clip_l = flux_utils.load_clip_l(
            args.clip_l, weight_dtype, device="cpu", disable_mmap=args.disable_mmap_load_safetensors
        )
        t5xxl = flux_utils.load_t5xxl(
            args.t5xxl, weight_dtype, device="cpu", disable_mmap=args.disable_mmap_load_safetensors
        )
        for m in (clip_l, t5xxl):
            m.eval().requires_grad_(False)
            m.to(accelerator.device)

        t5xxl_max_len = 512
        flux_tok = strategy_flux.FluxTokenizeStrategy(t5xxl_max_len)
        strategy_base.TokenizeStrategy.set_strategy(flux_tok)
        txt_enc = strategy_flux.FluxTextEncodingStrategy(args.apply_t5_attn_mask)
        strategy_base.TextEncodingStrategy.set_strategy(txt_enc)

    # For --sample_at_first
    optimizer_eval_fn()
    inference_pipeline.sample_images(accelerator, args, 0, global_step, jointdit, ae, [clip_l, t5xxl], weight_dtype)
    optimizer_train_fn()

    if len(accelerator.trackers) > 0:
        # log empty object to commit the sample images to wandb
        accelerator.log({}, step=0)

    loss_recorder = train_util.LossRecorder()
    epoch = 0  # avoid error when max_train_steps is 0

    for epoch in range(num_train_epochs):
        accelerator.print(f"\nepoch {epoch+1}/{num_train_epochs}")
        current_epoch.value = epoch + 1

        for m in training_models:
            m.train()

        for step, batch in enumerate(train_dataloader):
            current_step.value = global_step

            if args.blockwise_fused_optimizers:
                optimizer_hooked_count = {i: 0 for i in range(len(optimizers))}  # reset counter for each step

            with accelerator.accumulate(*training_models):
                
                if args.is_latent_training:
                    rgb_latents = batch["image_latent"].to(accelerator.device, dtype=weight_dtype)  
                    depth_latents = batch["depth_latent"].to(accelerator.device, dtype=weight_dtype) 

                    l_pooled = batch["clip_latent"].to(accelerator.device, dtype=weight_dtype) 
                    t5_out = batch["t5_latents"].to(accelerator.device, dtype=weight_dtype) 

                    if args.is_txt_ids_training:
                        txt_ids = batch["txt_ids"].to(accelerator.device, dtype=weight_dtype) 
                    else:
                        txt_ids = torch.zeros(t5_out.shape[0], t5_out.shape[1], 3, device=accelerator.device)

                    l_pooled = torch.cat([l_pooled, l_pooled], dim=0)
                    t5_out = torch.cat([t5_out, t5_out], dim=0)
                    txt_ids = torch.cat([txt_ids, txt_ids], dim= 0)

                    if args.is_attnmask_training:
                        t5_attn_mask = batch["attn_mask"].to(accelerator.device)
                        t5_attn_mask = torch.cat([t5_attn_mask, t5_attn_mask], 0)
                    else:
                        t5_attn_mask = None
                        
                    if torch.any(torch.isnan(rgb_latents)):
                        accelerator.print("NaN found in latents, replacing with zeros")
                        rgb_latents = torch.nan_to_num(rgb_latents, 0, out=rgb_latents)

                    if torch.any(torch.isnan(depth_latents)):
                        accelerator.print("NaN found in depth latents, replacing with zeros")
                        depth_latents = torch.nan_to_num(depth_latents, 0, out=depth_latents)

                else:
                    image = batch["image"].to(accelerator.device, dtype=weight_dtype)  
                    depth = batch["depth"].to(accelerator.device, dtype=weight_dtype)  
                    text_prompt = batch["text"]

                    with torch.no_grad():
                        rgb_latents = ae.encode(image)
                        depth_latents = ae.encode(depth)

                        tokens_and_masks = flux_tok.tokenize(text_prompt)
                        inputs = [t.to(accelerator.device) for t in tokens_and_masks]
                        conds = txt_enc.encode_tokens(flux_tok, [clip_l, t5xxl], inputs, args.apply_t5_attn_mask)
                        if args.full_fp16:
                            conds = [c.to(weight_dtype) for c in conds]
                        l_pooled, t5_out, txt_ids, t5_attn_mask = conds

                        l_pooled = torch.cat([l_pooled, l_pooled], dim=0)
                        t5_out = torch.cat([t5_out, t5_out], dim=0)
                        txt_ids = torch.cat([txt_ids, txt_ids], dim= 0)
                        
                        if not args.is_txt_ids_training:
                            txt_ids = torch.zeros(t5_out.shape[0], t5_out.shape[1], 3, device=accelerator.device)
                            
                        if not args.is_attnmask_training:
                            t5_attn_mask = None
                        else:
                            t5_attn_mask = t5_attn_mask.repeat(2, 1)

                """ Merge RGB and Depth inputs """
                latents = torch.cat([rgb_latents, depth_latents], dim=0)
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]

                """ Adopt the unbalanced timestep sampling strategy that is proposed JointDiT """
                scenario_indices = torch.randint(0, 4, (bsz//2,), device=accelerator.device) # Generate random indices with values 0, 1, or 2 of size bsz_half (1/3 probability for each scenario)
                small_steps = jointdit_utils.get_small_timesteps(
                    args, noise_scheduler_copy, latents[:bsz//2], noise[:bsz//2], accelerator.device, weight_dtype)
                normal_steps = jointdit_utils.get_normal_timesteps(
                    args, noise_scheduler_copy, latents[:bsz//2], noise[:bsz//2], accelerator.device, weight_dtype)

                timesteps_rgb = torch.zeros_like(small_steps)
                timesteps_dep = torch.zeros_like(small_steps)

                # Set timesteps based on each scenario
                for i in range(bsz//2):
                    scenario = scenario_indices[i].item()

                    if scenario == 0 or scenario == 1: # Case 1: Use the same timestep for both RGB and Depth
                        timesteps_rgb[i] = normal_steps[i]
                        timesteps_dep[i] = normal_steps[i]
                    elif scenario == 2: # Case 2: Use a random timestep for RGB and a small timestep for Depth
                        timesteps_rgb[i] = normal_steps[i]
                        timesteps_dep[i] = small_steps[i]
                    else: # Case 3: Use a small timestep for RGB and a random timestep for Depth
                        timesteps_rgb[i] = small_steps[i]
                        timesteps_dep[i] = normal_steps[i]                

                # Create model input data
                noisy_model_input_rgb, timesteps_rgb, sigmas = jointdit_utils.get_noisy_model_input_and_timesteps(
                    args, noise_scheduler_copy, latents[:bsz//2], noise[:bsz//2], accelerator.device, weight_dtype, timesteps=timesteps_rgb
                )

                noisy_model_input_dep, timesteps_dep, sigmas_dep = jointdit_utils.get_noisy_model_input_and_timesteps(
                    args, noise_scheduler_copy, latents[bsz//2:], noise[bsz//2:], accelerator.device, weight_dtype, timesteps=timesteps_dep
                )
                
                """ Merge RGB and Depth inputs """
                noisy_model_input = torch.cat([noisy_model_input_rgb, noisy_model_input_dep], dim=0)
                timesteps = torch.cat([timesteps_rgb, timesteps_dep], dim=0)

                # pack latents and get img_ids
                packed_noisy_model_input = flux_utils.pack_latents(noisy_model_input)  # b, c, h*2, w*2 -> b, h*w, c*4
                packed_latent_height, packed_latent_width = noisy_model_input.shape[2] // 2, noisy_model_input.shape[3] // 2
                img_ids = flux_utils.prepare_img_ids(bsz, packed_latent_height, packed_latent_width).to(device=accelerator.device)

                # get guidance: ensure args.guidance_scale is float
                guidance_vec = torch.full((bsz,), float(args.guidance_scale), device=accelerator.device)

                with accelerator.autocast():
                    # YiYi notes: divide it by 1000 for now because we scale it by 1000 in the transformer model (we should not keep it but I want to keep the inputs same for the model for testing)
                    model_pred = jointdit(
                        img=packed_noisy_model_input,
                        img_ids=img_ids,
                        txt=t5_out,
                        txt_ids=txt_ids,
                        y=l_pooled,
                        timesteps=timesteps / 1000,
                        guidance=guidance_vec,
                        txt_attention_mask=t5_attn_mask,
                    )

                # unpack latents
                model_pred = flux_utils.unpack_latents(model_pred, packed_latent_height, packed_latent_width)

                # apply model prediction type
                model_pred, weighting = flux_train_utils.apply_model_prediction_type(args, model_pred, noisy_model_input, sigmas)

                # flow matching loss: this is different from SD3
                target = noise - latents

                # calculate loss
                huber_c = train_util.get_huber_threshold_if_needed(args, timesteps, noise_scheduler)
                loss = train_util.conditional_loss(model_pred.float(), target.float(), args.loss_type, "none", huber_c)
                if weighting is not None:
                    loss = loss * weighting
                if args.masked_loss or ("alpha_masks" in batch and batch["alpha_masks"] is not None):
                    loss = apply_masked_loss(loss, batch)
                loss = loss.mean([1, 2, 3])

                # loss_weights = batch["loss_weights"]  # 各sampleごとのweight
                loss_weights = 1.0  # 各sampleごとのweight

                loss = loss * loss_weights
                loss = loss.mean()

                # backward
                accelerator.backward(loss)
                
                if not (args.fused_backward_pass or args.blockwise_fused_optimizers):
                    if accelerator.sync_gradients and args.max_grad_norm != 0.0:
                        params_to_clip = []
                        for m in training_models:
                            # params_to_clip.extend(m.parameters())
                            params_to_clip.extend(p for p in m.parameters() if p.requires_grad)
                        accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                else:
                    # optimizer.step() and optimizer.zero_grad() are called in the optimizer hook
                    lr_scheduler.step()
                    if args.blockwise_fused_optimizers:
                        for i in range(1, len(optimizers)):
                            lr_schedulers[i].step()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                optimizer_eval_fn()
                inference_pipeline.sample_images(
                    accelerator, args, None, global_step, jointdit, ae, [clip_l, t5xxl], weight_dtype
                )

                # 指定ステップごとにモデルを保存
                if global_step % args.save_every_n_steps == 0:
                    accelerator.wait_for_everyone()

                    save_path = os.path.join(args.output_dir, f"jointdit_addons_step_{global_step:06d}.safetensors")
                    save_added_params(jointdit, save_path, dtype=torch.bfloat16)

                    # now dump the two flags into a JSON alongside it
                    flags = {
                        "is_txt_ids_training": args.is_txt_ids_training,
                        "is_attnmask_training": args.is_attnmask_training,
                        "depth_transform": args.depth_transform,
                    }
                    json_path = os.path.join(
                        args.output_dir,
                        f"jointdit_addons_step_{global_step:06d}.json"
                    )
                    with open(json_path, "w") as jf:
                        json.dump(flags, jf, indent=4)

                    print(f"Saved training flags to {json_path}")

                optimizer_train_fn()

            current_loss = loss.detach().item()  # 平均なのでbatch sizeは関係ないはず
            if len(accelerator.trackers) > 0:
                logs = {"loss": current_loss}
                train_util.append_lr_to_logs(logs, lr_scheduler, args.optimizer_type, including_unet=True)

                accelerator.log(logs, step=global_step)

            loss_recorder.add(epoch=epoch, step=step, loss=current_loss)
            avr_loss: float = loss_recorder.moving_average
            logs = {"avr_loss": avr_loss}  # , "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

        if len(accelerator.trackers) > 0:
            logs = {"loss/epoch": loss_recorder.moving_average}
            accelerator.log(logs, step=epoch + 1)

        accelerator.wait_for_everyone()

        optimizer_eval_fn()
        optimizer_train_fn()

    is_main_process = accelerator.is_main_process
    # if is_main_process:
    jointdit = accelerator.unwrap_model(jointdit)

    accelerator.end_training()
    optimizer_eval_fn()

    if args.save_state or args.save_state_on_train_end:
        train_util.save_state_on_train_end(args, accelerator)

    del accelerator  # この後メモリを使うのでこれは消す

    if is_main_process:
        flux_train_utils.save_flux_model_on_train_end(args, save_dtype, epoch, global_step, jointdit)
        logger.info("model saved.")


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    add_logging_arguments(parser)
    train_util.add_sd_models_arguments(parser)  # TODO split this
    train_util.add_dataset_arguments(parser, True, True, True)
    train_util.add_training_arguments(parser, False)
    train_util.add_masked_loss_arguments(parser)
    deepspeed_utils.add_deepspeed_arguments(parser)
    train_util.add_sd_saving_arguments(parser)
    train_util.add_optimizer_arguments(parser)
    config_util.add_config_arguments(parser)
    add_custom_train_arguments(parser)  # TODO remove this from here
    train_util.add_dit_training_arguments(parser)
    flux_train_utils.add_flux_train_arguments(parser)

    parser.add_argument(
        "--mem_eff_save",
        action="store_true",
        help="[EXPERIMENTAL] use memory efficient custom model saving method / メモリ効率の良い独自のモデル保存方法を使う",
    )

    parser.add_argument(
        "--fused_optimizer_groups",
        type=int,
        default=None,
        help="**this option is not working** will be removed in the future / このオプションは動作しません。将来削除されます",
    )
    parser.add_argument(
        "--blockwise_fused_optimizers",
        action="store_true",
        help="enable blockwise optimizers for fused backward pass and optimizer step / fused backward passとoptimizer step のためブロック単位のoptimizerを有効にする",
    )
    parser.add_argument(
        "--skip_latents_validity_check",
        action="store_true",
        help="[Deprecated] use 'skip_cache_check' instead / 代わりに 'skip_cache_check' を使用してください",
    )
    parser.add_argument(
        "--double_blocks_to_swap",
        type=int,
        default=None,
        help="[Deprecated] use 'blocks_to_swap' instead / 代わりに 'blocks_to_swap' を使用してください",
    )
    parser.add_argument(
        "--single_blocks_to_swap",
        type=int,
        default=None,
        help="[Deprecated] use 'blocks_to_swap' instead / 代わりに 'blocks_to_swap' を使用してください",
    )
    parser.add_argument(
        "--cpu_offload_checkpointing",
        action="store_true",
        help="[EXPERIMENTAL] enable offloading of tensors to CPU during checkpointing / チェックポイント時にテンソルをCPUにオフロードする",
    )
    parser.add_argument("--image_folder", type=str, required=True)
    parser.add_argument(
        "--is_latent_training",
        action="store_true",
        help="If set, enables latent training mode."
    )
    parser.add_argument(
        "--is_txt_ids_training",
        action="store_true",
        help="If set, use the t5 txt_ids for training (default: false)."
    )
    parser.add_argument(
        "--is_attnmask_training",
        action="store_true",
        help="If set, use the t5 attention mask for training (default: false)."
    )
    parser.add_argument(
        '--depth_transform',
        type=str,
        choices=["none", "inverse"],
        default="none",
        help="Specify how to transform the depth map: 'none' or 'inverse'"
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

    parser.add_argument("--exp_name", type=str, default="training")

    return parser


if __name__ == "__main__":
    parser = setup_parser()

    args = parser.parse_args()
    train_util.verify_command_line_training_args(args)
    args = train_util.read_config_from_file(args, parser)

    # Load depth_transform flag from depth_transform.json in the specified folder
    import os, json
    depth_json = os.path.join(args.image_folder, "depth_transform.json")
    if os.path.isfile(depth_json):
        with open(depth_json, "r") as f:
            data = json.load(f)
        # Attach the JSON value to args, falling back to any existing default
        args.depth_transform = data.get("depth_transform", getattr(args, "depth_transform", "none"))
        print(f"Loaded depth_transform from {depth_json}: {args.depth_transform}")
    else:
        # If no file is found, leave args.depth_transform as it was (or None)
        args.depth_transform = getattr(args, "depth_transform", "none")
        print(f"No depth_transform.json found in {args.image_image_folder}; using {args.depth_transform!r}")

    train(args)
