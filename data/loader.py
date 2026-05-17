import os
import jax
import numpy as np

def load_latents_sharded(
    npy_path="/kaggle/input/datasets/sautkin/ffhq-sd3-5-vae-repa-e-latents-256/ffhq_latents_32x32x16.npy",
    latent_mean=None,
    latent_std=None,
    clip_value=10.0,
):
    print(f"Loading latents from {npy_path}...")

    if not os.path.exists(npy_path):
        raise FileNotFoundError(f"Latents file not found: {npy_path}")

    data_array = np.load(npy_path).astype(np.float32)
    print(f"Shape: {data_array.shape}, size: {data_array.nbytes / (1024**3):.2f} GB")

    if latent_mean is not None and latent_std is not None:
        print("Per-channel standardization (mean, std)...")
        mean_arr = np.array(latent_mean, dtype=np.float32).reshape(1, 1, 1, -1)
        std_arr = np.array(latent_std, dtype=np.float32).reshape(1, 1, 1, -1)
        data_array = (data_array - mean_arr) / std_arr
        
    if clip_value is not None:
        print(f"Clipping to [{-clip_value}, {clip_value}]")
        data_array = np.clip(data_array, -clip_value, clip_value)

    np.random.shuffle(data_array)

    num_samples = int(data_array.shape[0])
    devices = jax.local_devices()
    num_devices = len(devices)

    pad_size = (num_devices - (num_samples % num_devices)) % num_devices
    if pad_size > 0:
        data_array = np.concatenate([data_array, data_array[:pad_size]], axis=0)
    
    samples_per_device = data_array.shape[0] // num_devices

    data_array = data_array.reshape(num_devices, samples_per_device, *data_array.shape[1:])

    sharded = jax.device_put_sharded(list(data_array), devices)
    print(f"Sharded across {num_devices} devices, ~{data_array[0].nbytes / (1024**2):.2f} MB per shard")
    
    return sharded, samples_per_device
