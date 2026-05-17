import os
import jax
import numpy as np

def load_latents_sharded(
    npy_path="/kaggle/input/datasets/sautkin/ffhq-sd3-5-vae-repa-e-latents-256/ffhq_latents_32x32x16.npy",
    latent_mean=None,
    latent_std=None,
    clip_value=10.0,
):
    print(f"Загрузка латентов из {npy_path} в ОЗУ...")
    
    if not os.path.exists(npy_path):
        raise FileNotFoundError(f"Файл {npy_path} не найден! Сначала сгенерируйте датасет.")
        
    data_array = np.load(npy_path).astype(np.float32)
    print(f"Исходная форма: {data_array.shape}, Память: {data_array.nbytes / (1024**3):.2f} GB")
    
    # 1. Нормализуем дисперсию и среднее поканально
    if latent_mean is not None and latent_std is not None:
        print("Применяем поканальную нормализацию (mean, std)...")
        mean_arr = np.array(latent_mean, dtype=np.float32).reshape(1, 1, 1, -1)
        std_arr = np.array(latent_std, dtype=np.float32).reshape(1, 1, 1, -1)
        data_array = (data_array - mean_arr) / std_arr
        
    if clip_value is not None:
        print(f"Обрезаем значения (clip) до диапазона [{-clip_value}, {clip_value}]")
        data_array = np.clip(data_array, -clip_value, clip_value)
    
    # 2. Перемешиваем
    np.random.shuffle(data_array)
    
    num_samples = int(data_array.shape[0])
    devices = jax.local_devices()
    num_devices = len(devices)
    
    # 3. Паддинг для ровного деления
    pad_size = (num_devices - (num_samples % num_devices)) % num_devices
    if pad_size > 0:
        data_array = np.concatenate([data_array, data_array[:pad_size]], axis=0)
    
    samples_per_device = data_array.shape[0] // num_devices
    
    # Reshape: [num_devices, samples_per_device, 64, 64, 16]
    data_array = data_array.reshape(num_devices, samples_per_device, *data_array.shape[1:])
    
    sharded = jax.device_put_sharded(list(data_array), devices)
    print(f"Датасет распределен по {num_devices} TPU. На каждом TPU: ~{data_array[0].nbytes / (1024**2):.2f} MB")
    
    return sharded, samples_per_device
