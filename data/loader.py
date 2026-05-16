import os
import jax
import numpy as np

def load_latents_sharded(
    npy_path="/kaggle/input/datasets/sautkin/ffhq-sd3-5-vae-repa-e-latents/ffhq_latents_64x64x16.npy",
    scaling_factor=1.0,
):
    print(f"Загрузка латентов из {npy_path} в ОЗУ...")
    
    if not os.path.exists(npy_path):
        raise FileNotFoundError(f"Файл {npy_path} не найден! Сначала сгенерируйте датасет.")
        
    data_array = np.load(npy_path).astype(np.float32)
    print(f"Исходная форма: {data_array.shape}, Память: {data_array.nbytes / (1024**3):.2f} GB")
    
    # 1. Нормализуем дисперсию к ~1.0
    print(f"Применяем scaling_factor: {scaling_factor}")
    data_array = data_array * (1.0 / scaling_factor)
    
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
