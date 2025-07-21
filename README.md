# JointDiT: Enhancing RGB-Depth Joint Modeling with Diffusion Transformers (ICCV 2025)
### [Project Page](https://byungki-k.github.io/JointDiT/) | [Paper](https://arxiv.org/abs/2505.00482)
This repository contains the official implementation of the ICCV 2025 paper, "JointDiT: Enhancing RGB-Depth Joint Modeling with Diffusion Transformers".

## 💪To-Do List

- [x] Inference code
- [ ] Training code

## Getting started
This code was developed on Ubuntu 24.04 with Python 3.10.18, CUDA 12.1 and PyTorch 2.4.0, using a single NVIDIA H100 (80GB) GPU. 
Later versions should work, but have not been tested.

<details>
<summary>🛠️ Environment setup (click to expand)</summary>

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

</details>
