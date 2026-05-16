import os
import yaml
from dataclasses import dataclass

import jax.numpy as jnp


def _parse_dtype(name: str):
    if isinstance(name, str):
        n = name.lower()
        if n in ("bf16", "bfloat16"):
            return jnp.bfloat16
        if n in ("f32", "float32"):
            return jnp.float32
    return jnp.float32


@dataclass
class Config:
    seed: int = 42
    
    batch_size_per_device: int = 32
    epochs: int = 500
    lr_gen: float = 0.0001
    lr_disc: float = 0.00005
    beta1: float = 0.5
    beta2: float = 0.999
    latent_dim: int = 128
    lambda_div: float = 0.5
    diversity_temperature: float = 0.5

    image_size: int = 32
    channels: int = 3
    latent_scaling_factor: float = 2.35711
    latent_npy_path: str = "/kaggle/input/datasets/sautkin/ffhq-sd3-5-vae-repa-e-latents/ffhq_latents_64x64x16.npy"
    use_sn: bool = False
    num_kernels_mbd: int = 100
    kernel_dim_mbd: int = 5

    network_dtype: str = "bfloat16"
    jax_matmul_precision: str = "default"

    wandb_project: str = "KL-GAN-TPU"
    wandb_run_name: str = "default_run"
    
    eval_every_epochs: int = 5
    fid_every_epochs: int = 50
    num_fid_samples: int = 10000

    @property
    def compute_dtype(self):
        return _parse_dtype(self.network_dtype)

    @classmethod
    def load(cls, yaml_path: str):
        if not os.path.exists(yaml_path):
            return cls()
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        fields = {f.name for f in cls.__dataclass_fields__.values()}
        kwargs = {k: v for k, v in data.items() if k in fields}
        return cls(**kwargs)

# Global config instance loaded from default.yaml if it exists
config = Config.load(os.path.join(os.path.dirname(__file__), "configs", "default.yaml"))