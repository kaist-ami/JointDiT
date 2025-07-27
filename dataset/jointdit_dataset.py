
import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F
import numpy 
import os
import math
import random
from glob import glob
import os.path as osp
import os 
import torchvision.transforms.functional as TF
import pandas as pd
import csv
import time
from PIL import Image

import sys
import pdb

class JointDataset(data.Dataset):
    def __init__(self, args, image_folder, drop_text, depth_transform):
        
        self.args = args
        self.image_folder = image_folder 
        self.drop_text = drop_text
        self.depth_transform = depth_transform

        self.get_dataset()

    def __getitem__(self, index):
        fname = self.filenames[index]

        if self.args.is_latent_training:
            use_empty_text = np.random.rand() < self.drop_text

            if use_empty_text:
                base_path = "empty_prompts"
                sample = {
                    "image_latent": torch.load(os.path.join(self.image_folder, "image_latents", fname + ".pt"), map_location='cpu'),
                    "depth_latent": torch.load(os.path.join(self.image_folder, "depth_latents", fname + ".pt"), map_location='cpu'),
                    "clip_latent": torch.load(os.path.join(base_path, "clip_latents", "empty.pt"), map_location='cpu'),
                    "t5_latents": torch.load(os.path.join(base_path, "t5_latents", "empty.pt"), map_location='cpu'),
                    "filename": fname,
                }
                if getattr(self.args, "is_txt_ids_training", False):
                    sample["txt_ids"] = torch.load(os.path.join(base_path, "txt_ids", "empty.pt"), map_location='cpu')

                if getattr(self.args, "is_attnmask_training", False):
                    sample["attn_mask"] = torch.load(os.path.join(base_path, "attn_masks", "empty.pt"), map_location='cpu')
            else:
                sample = {
                    "image_latent": torch.load(os.path.join(self.image_folder, "image_latents", fname + ".pt"), map_location='cpu'),
                    "depth_latent": torch.load(os.path.join(self.image_folder, "depth_latents", fname + ".pt"), map_location='cpu'),
                    "clip_latent": torch.load(os.path.join(self.image_folder, "clip_latents", fname + ".pt"), map_location='cpu'),
                    "t5_latents": torch.load(os.path.join(self.image_folder, "t5_latents", fname + ".pt"), map_location='cpu'),
                    "filename": fname,
                }
                if getattr(self.args, "is_txt_ids_training", False):
                    sample["txt_ids"] = torch.load(os.path.join(self.image_folder, "txt_ids", fname + ".pt"), map_location='cpu')

                if getattr(self.args, "is_attnmask_training", False):
                    sample["attn_mask"] = torch.load(os.path.join(self.image_folder, "attn_masks", fname + ".pt"),  map_location='cpu')

        else:
            # Load raw image
            img_path = os.path.join(self.image_folder, "processed_images", fname + ".png")
            if not os.path.exists(img_path):
                img_path = os.path.join(self.image_folder, "processed_images", fname + ".jpg")
            image = Image.open(img_path).convert("RGB")
            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image).permute(2, 0, 1) * 2 - 1  # [-1, 1]

            # Load depth
            depth = np.load(os.path.join(self.image_folder, "depthmaps", fname + ".npy"))
            depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
            if self.depth_transform == "inverse":
                depth = 1 - depth
            depth = torch.from_numpy(depth).float().unsqueeze(0).repeat(3, 1, 1) * 2 - 1  # [3, H, W]

            # Load text
            with open(os.path.join(self.image_folder, "text_prompts", fname + ".txt"), "r") as f:
                text = f.read().strip()

            sample = {
                "image": image,
                "depth": depth,
                "text": text,
                "filename": fname,
            }

        return sample

    def __len__(self):
        return len(self.filenames)

    def get_dataset(self):

        if self.args.is_latent_training:
            # Latent mode: collect .pt files from latent folders
            required_dirs = [
                "image_latents", "depth_latents", "clip_latents", "t5_latents"
            ]
            if getattr(self.args, "is_txt_ids_training", False):
                required_dirs.append("txt_ids")

            if getattr(self.args, "is_attnmask_training", False):
                required_dirs.append("attn_masks")

            file_sets = []
            for subdir in required_dirs:
                path = os.path.join(self.image_folder, subdir)
                files = glob(os.path.join(path, "*.pt"))
                filenames = set(os.path.splitext(os.path.basename(f))[0] for f in files)
                file_sets.append(filenames)

            # Get intersection of all filenames
            common_files = sorted(set.intersection(*file_sets))
            self.filenames = common_files

        else:
            # Non-latent mode: collect images, depths, and text files
            img_dir = os.path.join(self.image_folder, "processed_images")
            depth_dir = os.path.join(self.image_folder, "depthmaps")
            text_dir = os.path.join(self.image_folder, "text_prompts")

            img_files = glob(os.path.join(img_dir, "*.png")) + \
                        glob(os.path.join(img_dir, "*.jpg"))
            depth_files = glob(os.path.join(depth_dir, "*.npy"))
            text_files = glob(os.path.join(text_dir, "*.txt"))

            img_names = set(os.path.splitext(os.path.basename(f))[0] for f in img_files)
            depth_names = set(os.path.splitext(os.path.basename(f))[0] for f in depth_files)
            text_names = set(os.path.splitext(os.path.basename(f))[0] for f in text_files)

            # Get common filenames
            common_files = sorted(img_names & depth_names & text_names)
            self.filenames = common_files
