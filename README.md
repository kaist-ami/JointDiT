# JointDiT: Enhancing RGB-Depth Joint Modeling with Diffusion Transformers (ICCV 2025)
### [Project Page](https://byungki-k.github.io/JointDiT/) | [Paper](https://arxiv.org/abs/2505.00482) | 🤗 [Huggingface Model](https://huggingface.co/byungki-kwon/JointDiT)
This repository contains the official implementation of the ICCV 2025 paper, "JointDiT: Enhancing RGB-Depth Joint Modeling with Diffusion Transformers".

## Acknowledgements

This work builds upon the excellent prior contributions of the following projects:

- **[Flux](https://huggingface.co/black-forest-labs/FLUX.1-dev)** by Black Forest Labs — we use their pretrained diffusion model and autoencoder.
- **[sd-scripts](https://github.com/kohya-ss/sd-scripts)** by kohya-ss — we adopt and extend their fine-tuning infrastructure.

We sincerely thank the authors for making their models and code publicly available.

## 💪To-Do List

- [x] Inference code
- [x] Training code
- [ ] **Extension to FLUX.1-Krea-dev!**
- [ ] 3D lifting code

## Getting started
This code was developed on Ubuntu 24.04 with Python 3.10.18, CUDA 12.1 and PyTorch 2.4.0, using a single NVIDIA H100 (80GB) GPU. 
Later versions should work, but have not been tested.

## Environment setup 

### 1. Clone repository
```bash
git clone -b sd3 https://github.com/kohya-ss/sd-scripts.git
```

### 2. Create and activate conda environment
```bash
conda create -n jointdit python=3.10 -y
conda activate jointdit
```

### 3. Install dependencies
```bash
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
    xformers --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### 4. Prepare model directories
```bash
mkdir -p models/flux models/jointdit
```

### 5. Download Flux model & autoencoder  
> You must be logged in via `huggingface-cli login`
```bash
# Flux model
huggingface-cli download black-forest-labs/FLUX.1-dev \
    flux1-dev.safetensors --local-dir ./models/flux

# Autoencoder
huggingface-cli download black-forest-labs/FLUX.1-dev \
    ae.safetensors --local-dir ./models/flux
```

### 6. Download text encoders
```bash
huggingface-cli download comfyanonymous/flux_text_encoders \
    clip_l.safetensors t5xxl_fp16.safetensors \
    --local-dir ./models/flux
```

### 7. Download JointDiT pretrained weights
```bash
# Replace with actual repo and filename
huggingface-cli download byungki-kwon/JointDiT \
    jointdit.safetensors --local-dir ./models/jointdit
```

## 🚀 Inference

We provide example scripts for three core inference tasks:

- **Joint RGB-Depth generation**
- **Depth estimation from RGB**
- **Depth-to-image synthesis**

Run the following commands accordingly:

```bash
# Joint generation (RGB & Depth)
bash scripts/joint_generation.sh
# → Try changing the `--text_prompt` argument in the script.

# Depth estimation from RGB
bash scripts/depth_estimation.sh
# → Modify the `--input_image` argument to use your own image.

# Depth-to-image synthesis
bash scripts/depth_to_image.sh
# → You can adjust both `--text_prompt` and `--input_depth` in this script.
```

## 🚀 Training

This section explains how to prepare your dataset and run the training code. Please read it carefully.

Since we use a internal dataset for training, we are unable to release the dataset itself.  
Instead, we provide a detailed guide on how to prepare your own dataset using public tools and models.

JointDiT requires **paired data** of `(image, depth map, text prompt)`.  
If you only have raw images (`.png` or `.jpg`), follow the three-step preprocessing pipeline below to generate the required depth and text annotations.

---

### 🧹 Preprocessing Pipeline

The preprocessing is split into **three steps**.  
If you already have `.txt` prompt files with the same names as your images, you can **skip the LLAVA prompt generation**.

This preprocessing pipeline assumes you only have image data in `.png` or `.jpg` format.  
To obtain **relative disparity** and **text prompts**, we use [`Depth-Anything-V2`](https://github.com/DepthAnything/Depth-Anything-V2) and [`LLAVA`](https://github.com/haotian-liu/LLaVA).

Before starting, make sure to:

1. Create a `checkpoints/` folder  
2. Place the **Depth-Anything-V2-Large** model in `checkpoints/`  
3. Upgrade the Transformers library:
   ```bash
   pip install --upgrade transformers
   ```

### 🔹 Step 1: Prepare Raw Images, Depth Maps, and Text prompts

```bash
python dataset/preprocess_step1.py --image_folder "your_image_folder"
```

This script will:

- Move original images to:  
  `your_image_folder/raw_image/`

- Resize and center-crop images to **512×512**, saving them in:  
  `your_image_folder/processed_images/`

- Use **Depth-Anything-V2** to generate relative disparity maps, saved in:  
  `your_image_folder/depthmaps/`

- Use **LLAVA** to generate image captions (text prompts), saved in:  
  `your_image_folder/text_prompts/`

> ⚠️ **LLAVA captioning can be slow.**  
> If you already have prompts, you can skip this step by placing `.txt` files with the same filenames as the images in `your_image_folder/text_prompts/`.

> ⚠️ **Note on resolution.**
> Flux models are capable of learning from arbitrary resolutions during training. However, for simplicity and reproducibility, our paper trains JointDiT using only 512×512 inputs. This may be sub-optimal, especially for datasets where higher resolution details matter. Feel free to modify the preprocessing or training pipeline to accommodate variable resolutions.

### 🔹 (Optional) Step 2: Precompute latents for images, depth maps, and text prompts

To accelerate training and reduce GPU memory usage, you can precompute the latents for the images, depth maps, and text prompts.

Since the T5 text encoder is particularly large, precomputing its outputs is especially helpful in reducing memory consumption during training.

> ℹ️ **Note:**  
> You can set the `--depth_transform` option to either `none` or `inverse`.  
> In our paper, we use the `none` depth_transform option.
> We empirically found that using inverse disparity maps (where nearby regions are dark and distant regions are bright) results in significantly faster convergence compared to using disparity maps (where nearby regions are bright).

#### Example command:
```bash
accelerate launch --config_file default_config.yaml --mixed_precision bf16 dataset/preprocess_step2.py \
  --image_folder "your_image_folder" \
  --depth_transform none \
  --clip_l models/flux/clip_l.safetensors \
  --t5xxl models/flux/t5xxl_fp16.safetensors \
  --ae models/flux/ae.safetensors \
  --sdpa \
  --blocks_to_swap 8 \
  --mixed_precision bf16 \
  --save_precision bf16 \
  --full_bf16 
```

### 🔹 Step 3: Precompute latents for evaluation prompts

This step precomputes the latents of text prompts listed in `dataset/evaluation_prompts.txt`.  
These latents are used during training for periodic evaluation.

✏️ If you have specific text prompts you'd like to use for joint generation evaluation, add them to the `dataset/evaluation_prompts.txt` file (one per line).

#### Example command:
```bash
accelerate launch --config_file default_config.yaml --mixed_precision bf16 dataset/preprocess_step3.py \
  --clip_l models/flux/clip_l.safetensors \
  --t5xxl models/flux/t5xxl_fp16.safetensors \
  --ae models/flux/ae.safetensors \
  --sdpa \
  --blocks_to_swap 8 \
  --mixed_precision bf16 \
  --save_precision bf16 \
  --full_bf16 
```

### 🔹 Step 4: Start the training

Use the command below to start training. The results and model will be saved to `args.output_dir/args.output_name`.

- If you have precomputed latents using **Step 2**, use the `--is_latent_training` flag.  
  Otherwise, remove it to compute latents on-the-fly.
- You can optionally enable T5 text encoder features during training:
  - `--is_txt_ids_training`: use token IDs
  - `--is_attnmask_training`: use attention masks  
  (Note: **These were not used in our paper.**)
- Adjust `--output_resolution` to control the resolution of intermediate samples.

After training is complete, make sure to update the `--jointdit_addons_path` in the inference script to point to your newly trained model checkpoint.

#### Example command:
```bash
accelerate launch --config_file default_config.yaml --main_process_port 12342 --mixed_precision bf16 \
--num_cpu_threads_per_process 8 train.py \
--image_folder "your_image_folder" \
--is_latent_training \
--pretrained_model_name_or_path models/flux/flux1-dev.safetensors \
--clip_l models/flux/clip_l.safetensors \
--t5xxl models/flux/t5xxl_fp16.safetensors \
--ae models/flux/ae.safetensors \
--output_dir experiments \
--exp_name test_training \
--output_resolution 1024 1024 \
--depth_transform none \
--save_model_as safetensors --sdpa \
--persistent_data_loader_workers --max_data_loader_n_workers 16 --seed 42 --gradient_checkpointing --mixed_precision bf16 \
--save_precision bf16 \
--output_name joint \
--learning_rate 1e-5 --max_train_epochs 8  --sdpa --highvram --save_every_n_epochs 1 --optimizer_type adafactor \
--optimizer_args "relative_step=False" "scale_parameter=False" "warmup_init=False" --lr_scheduler constant_with_warmup \
--max_grad_norm 0.0 --timestep_sampling shift --discrete_flow_shift 3.1582 --model_prediction_type raw --guidance_scale 1.0 \
--fused_backward_pass  --blocks_to_swap 8 --full_bf16 \
--sample_every_n_steps 5000 \
--save_every_n_steps 25000 \
--train_batch_size 4 \
--sample_at_first
```


## Citation
If you find our code or paper helps, please consider citing:
````BibTeX
@InProceedings{Byung-Ki_2025_ICCV,
    author    = {Byung-Ki, Kwon and Dai, Qi and Hyoseok, Lee and Luo, Chong and Oh, Tae-Hyun},
    title     = {JointDiT: Enhancing RGB-Depth Joint Modeling with Diffusion Transformers},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2025},
    pages     = {25261-25271}
}
````
