# JointDiT: Enhancing RGB-Depth Joint Modeling with Diffusion Transformers (ICCV 2025)
### [Project Page](https://byungki-k.github.io/JointDiT/) | [Paper](https://arxiv.org/abs/2505.00482)
This repository contains the official implementation of the ICCV 2025 paper, "JointDiT: Enhancing RGB-Depth Joint Modeling with Diffusion Transformers".

## Acknowledgements

This work builds upon the excellent prior contributions of the following projects:

- **[Flux](https://huggingface.co/black-forest-labs/FLUX.1-dev)** by Black Forest Labs — we use their pretrained diffusion model and autoencoder.
- **[sd-scripts](https://github.com/kohya-ss/sd-scripts)** by kohya-ss — we adopt and extend their fine-tuning infrastructure.

We sincerely thank the authors for making their models and code publicly available.

## 💪To-Do List

- [x] Inference code
- [ ] Training code

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
huggingface-cli download <your-username>/<repo> \
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
3. Install the Transformers library:
   ```bash
   pip install --upgrade transformers
   ```

### 🔹 Step 1: Prepare Raw Images, Depth Maps, and Prompts

This is the first of three preprocessing steps.

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