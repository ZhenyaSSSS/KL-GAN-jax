import os
import shutil
import numpy as np
from PIL import Image
from cleanfid import fid
from tqdm import tqdm

def save_images_to_dir(images_np, directory):
    os.makedirs(directory, exist_ok=True)
    images_np = ((images_np + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    for i, img_arr in enumerate(tqdm(images_np, desc=f"Saving to {directory}")):
        img = Image.fromarray(img_arr)
        img.save(os.path.join(directory, f"{i}.jpg"))

def compute_fid(real_images_np, fake_images_np, tmp_dir="./tmp"):
    real_dir = os.path.join(tmp_dir, "real")
    fake_dir = os.path.join(tmp_dir, "fake")
    
    if os.path.exists(real_dir): shutil.rmtree(real_dir)
    if os.path.exists(fake_dir): shutil.rmtree(fake_dir)
    
    save_images_to_dir(real_images_np, real_dir)
    save_images_to_dir(fake_images_np, fake_dir)
    
    print("Computing FID score...")
    score = fid.compute_fid(real_dir, fake_dir, dataset_name="celeba32", dataset_res=32, dataset_split="custom")
    
    shutil.rmtree(real_dir)
    shutil.rmtree(fake_dir)
    
    return score
