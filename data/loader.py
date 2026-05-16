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
    
    kaggle_paths = [
        "/kaggle/input/celeba-dataset/img_align_celeba/img_align_celeba",
        "/kaggle/input/celeba-dataset/img_align_celeba",
        "/kaggle/input/celeba/img_align_celeba/img_align_celeba",
        "/kaggle/input/celeba/img_align_celeba",
        "/kaggle/input/datasets/kushsheth/face-vae/img_align_celeba/img_align_celeba",
        "/kaggle/input/datasets/kushsheth/face-vae/img_align_celeba",
        "/kaggle/input/datasets/kushsheth/face-vae",
        "/kaggle/input/face-vae/img_align_celeba/img_align_celeba",
        "/kaggle/input/face-vae/img_align_celeba",
        "/kaggle/input/face-vae"
    ]
    
    if os.path.exists("/kaggle/input"):
        for d in os.listdir("/kaggle/input"):
            kaggle_paths.append(f"/kaggle/input/{d}/img_align_celeba/img_align_celeba")
            kaggle_paths.append(f"/kaggle/input/{d}/img_align_celeba")
            kaggle_paths.append(f"/kaggle/input/{d}")
    
    image_paths = []
    for path in kaggle_paths:
        if os.path.exists(path):
            found_jpgs = glob.glob(os.path.join(path, "*.jpg"))
            if len(found_jpgs) > 1000:
                print(f"Found Kaggle dataset at {path}")
                image_paths = sorted(found_jpgs)
                break
            
    if not image_paths and os.path.exists("./data/celeba/img_align_celeba"):
        print("Found local dataset via torchvision structure")
        image_paths = sorted(glob.glob("./data/celeba/img_align_celeba/*.jpg"))
        
    if not image_paths:
        print(f"Downloading/Loading CelebA via torchvision to {data_dir}")
        print("WARNING: Google Drive download might fail due to quota limits.")
        print("RECOMMENDED: Add the 'celeba-dataset' to your Kaggle Notebook via 'Add Data' on the right panel!")
        os.makedirs(data_dir, exist_ok=True)
        try:
            dataset = CelebA(root=data_dir, split="all", download=True)
            image_paths = [
                os.path.join(data_dir, "celeba", "img_align_celeba", f"{i:06d}.jpg")
                for i in range(1, len(dataset) + 1)
            ]
        except Exception as e:
            print("Failed to download CelebA via torchvision:", e)
            print("To fix this, click 'Add Data' in Kaggle and search for 'celeba-dataset'.")
            raise RuntimeError("Dataset not found and Google Drive quota exceeded.")

    print(f"Processing {len(image_paths)} images...")

    def process_image(path):
        try:
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
        except Exception:
            return None

    import multiprocessing
    from concurrent.futures import ThreadPoolExecutor

    num_workers = min(64, multiprocessing.cpu_count() * 2)
    print(f"Using {num_workers} threads for ultra-fast image loading...")

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(tqdm(executor.map(process_image, image_paths), total=len(image_paths), desc="Loading images"))

    images = [r for r in results if r is not None]

    data_array = np.stack(images, axis=0)
    print(f"Dataset shape: {data_array.shape}, Memory: {data_array.nbytes / (1024**3):.2f} GB")
    return data_array


def load_dataset_sharded(data_dir: str = "./data/celeba", image_size: int = 32):
    """Shards the full dataset across local devices to save HBM and avoid host-device transfers."""
    data_array = get_celeba_array(data_dir, image_size)
    num_samples = int(data_array.shape[0])
    
    devices = jax.local_devices()
    num_devices = len(devices)
    
    # Pad to make it divisible by num_devices
    pad_size = (num_devices - (num_samples % num_devices)) % num_devices
    if pad_size > 0:
        data_array = np.concatenate([data_array, data_array[:pad_size]], axis=0)
    
    samples_per_device = data_array.shape[0] // num_devices
    
    # Reshape to [num_devices, samples_per_device, H, W, C]
    data_array = data_array.reshape(num_devices, samples_per_device, *data_array.shape[1:])
    
    sharded = jax.device_put_sharded(list(data_array), devices)
    print(f"Sharded full dataset ({num_samples} images + {pad_size} pad) across {num_devices} devices; "
          f"~{data_array[0].nbytes / (1024**2):.2f} MB per device.")
    return sharded, samples_per_device
