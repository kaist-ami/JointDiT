import os
import sys
from glob import glob
import argparse
from tqdm import tqdm
import numpy as np
from PIL import Image
import cv2
import torch
import re 

# Add submodule paths
sys.path.append('./Depth-Anything-V2')  # or use absolute path
sys.path.append('./LLaVA')              # or use absolute path

from depth_anything_v2.dpt import DepthAnythingV2
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model
from transformers import AutoProcessor, LlavaForConditionalGeneration
from dataset.dataset_utils import save_text_to_file, clean_text, resize_and_center_crop

def preprocess_step1(args):
    """
    Step 0: Move all .jpg and .png images in image_folder into 'raw_images' subfolder
    """
    raw_folder = os.path.join(args.image_folder, "raw_images")
    os.makedirs(raw_folder, exist_ok=True)

    raw_image_paths = sorted(glob(os.path.join(args.image_folder, "*.jpg")) +
                             glob(os.path.join(args.image_folder, "*.png")))

    for img_path in tqdm(raw_image_paths, desc="move raw images to the raw_images folder"):
        filename = os.path.basename(img_path)
        os.rename(img_path, os.path.join(raw_folder, filename))

    """
    Step 1: Preprocess all .jpg and .png images in the raw_images folder.
    - Resize each image with aspect-ratio preserved
    - Center-crop to 512x512
    - Save to `processed_images` subfolder
    """
    processed_folder = os.path.join(args.image_folder, "processed_images")
    os.makedirs(processed_folder, exist_ok=True)

    image_paths = sorted(glob(os.path.join(raw_folder, "*.jpg")) +
                         glob(os.path.join(raw_folder, "*.png")))

    for img_path in tqdm(image_paths, desc="Processing images"):
        img = cv2.imread(img_path)
        if img is None:
            continue
        processed = resize_and_center_crop(img)
        filename = os.path.basename(img_path)
        cv2.imwrite(os.path.join(processed_folder, filename), processed)

    """
    Step 2: Predict relative depth map using Depth-Anything-V2 (ViT-Large)
    """
    depth_folder = os.path.join(args.image_folder, "depthmaps")
    os.makedirs(depth_folder, exist_ok=True)

    processed_image_paths = sorted(glob(os.path.join(processed_folder, "*.jpg")) +
                                   glob(os.path.join(processed_folder, "*.png")))

    model = DepthAnythingV2(encoder='vitl', features=256, out_channels=[256, 512, 1024, 1024])
    model.load_state_dict(torch.load('checkpoints/depth_anything_v2_vitl.pth', map_location='cpu'))
    model.cuda()
    model.eval()

    for processed_img_path in tqdm(processed_image_paths, desc="Predicting depth maps"):
        raw_img = cv2.imread(processed_img_path)
        depth = model.infer_image(raw_img)
        filename = os.path.splitext(os.path.basename(processed_img_path))[0] + ".npy"
        np.save(os.path.join(depth_folder, filename), depth)

    """
    Step 3: Generate text prompts using LLaVA-1.5 7B model
    """
    text_folder = os.path.join(args.image_folder, "text_prompts")
    os.makedirs(text_folder, exist_ok=True)

    model = LlavaForConditionalGeneration.from_pretrained(
        "llava-hf/llava-1.5-7b-hf", torch_dtype=torch.float16, device_map="auto")
    processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image?"},
                {"type": "image"},
            ],
        },
    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

    for processed_img_path in tqdm(processed_image_paths, desc="Generating captions"):
        raw_img = Image.open(processed_img_path)
        inputs = processor(images=raw_img, text=prompt, return_tensors='pt').to(0, torch.float16)
        output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
        raw_text = processor.decode(output[0][2:], skip_special_tokens=True)
        text = clean_text(raw_text)

        if text is not None:
            filename = os.path.splitext(os.path.basename(processed_img_path))[0] + ".txt"
            save_text_to_file(text, os.path.join(text_folder, filename))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_folder', default=None, help="Path to the folder containing your custom images")
    args = parser.parse_args()

    preprocess_step1(args)
