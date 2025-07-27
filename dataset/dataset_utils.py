import os
import re
import cv2

def save_text_to_file(text, savepath):
    """Save the generated text to a .txt file at the specified path."""
    with open(savepath, 'w') as file:
        file.write(text)

def clean_text(raw_text):
    """
    Extract the assistant's response from the raw output.
    Perform cleaning: remove line breaks, trailing spaces, and final period.
    """
    match = re.search(r"ASSISTANT:(.*)", raw_text, re.DOTALL)
    if match:
        cleaned_text = match.group(1).replace("\n", " ").strip()
        cleaned_text = cleaned_text.strip().rstrip(".")
        cleaned_text = cleaned_text.replace("   ", " ").replace("  ", " ").strip()
        return cleaned_text
    else:
        return None

def resize_and_center_crop(image, size=512):
    """
    Resize the image while preserving aspect ratio.
    The smaller side is resized to `size`, and then the image is center-cropped to `size x size`.
    """
    h, w = image.shape[:2]
    if h < w:
        new_h = size
        new_w = int(w * size / h)
    else:
        new_w = size
        new_h = int(h * size / w)
    image_resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    start_x = (new_w - size) // 2
    start_y = (new_h - size) // 2
    image_cropped = image_resized[start_y:start_y+size, start_x:start_x+size]

    return image_cropped