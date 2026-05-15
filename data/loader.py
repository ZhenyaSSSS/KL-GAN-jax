import os
import glob
import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision.datasets import CelebA


def get_celeba_array(data_dir: str = "./data/celeba", image_size: int = 32):
    """Loads CelebA into a single float32 array [N, H, W, C] in [-1, 1]."""
    kaggle_path = "/kaggle/input/celeba-dataset/img_align_celeba/img_align_celeba"
    if os.path.exists(kaggle_path):
        print(f"Found Kaggle dataset at {kaggle_path}")
        image_paths = sorted(glob.glob(os.path.join(kaggle_path, "*.jpg")))
    elif os.path.exists("./data/celeba/img_align_celeba"):
        print("Found local dataset via torchvision structure")
        image_paths = sorted(glob.glob("./data/celeba/img_align_celeba/*.jpg"))
    else:
        print(f"Downloading/Loading CelebA via torchvision to {data_dir}")
        os.makedirs(data_dir, exist_ok=True)
        dataset = CelebA(root=data_dir, split="all", download=True)
        image_paths = [
            os.path.join(data_dir, "celeba", "img_align_celeba", f"{i:06d}.jpg")
            for i in range(1, len(dataset) + 1)
        ]

    print(f"Processing {len(image_paths)} images...")

    def process_image(path):
        with Image.open(path) as img:
            w, h = img.size
            new_w, new_h = 140, 140
            left = (w - new_w) / 2
            top = (h - new_h) / 2
            right = (w + new_w) / 2
            bottom = (h + new_h) / 2
            img = img.crop((left, top, right, bottom))
            img = img.resize((image_size, image_size), Image.Resampling.BILINEAR)
            arr = np.array(img, dtype=np.float32)
            return (arr / 127.5) - 1.0

    images = []
    for path in tqdm(image_paths, desc="Loading images"):
        if os.path.exists(path):
            images.append(process_image(path))

    data_array = np.stack(images, axis=0)
    print(f"Dataset shape: {data_array.shape}, Memory: {data_array.nbytes / (1024**3):.2f} GB")
    return data_array


def load_dataset_replicated(data_dir: str = "./data/celeba", image_size: int = 32):
    """Replicates the full dataset on every local device (each critic sees the same pool)."""
    data_array = get_celeba_array(data_dir, image_size)
    num_samples = int(data_array.shape[0])
    devices = jax.local_devices()
    replicated = jax.device_put_replicated(jnp.asarray(data_array), devices)
    print(
        f"Replicated full dataset ({num_samples} images) to {len(devices)} devices; "
        f"~{data_array.nbytes * len(devices) / (1024**3):.2f} GB total across devices."
    )
    return replicated, num_samples
