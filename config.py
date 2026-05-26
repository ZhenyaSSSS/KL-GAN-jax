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
    
    batch_size_per_device: int = 256
    epochs: int = 500
    lr_gen: float = 0.0002
    lr_disc: float = 0.0002
    beta1: float = 0.5
    beta2: float = 0.999
    latent_dim: int = 128
    lambda_div: float = 0.5
    diversity_temperature: float = 0.5

    loss_type: str = "manifold"
    manifold_proj_dim: int = 16
    contrastive_pairing: str = "real_aug"
    contrastive_loss_type: str = "full_yin_yang"
    contrastive_temperature: float = 0.1
    lambda_contrastive: float = 4.0
    lambda_decorr: float = 0.0
    lambda_cov: float = 5.0
    sinkhorn_epsilon: float = 0.05
    sinkhorn_max_iter: int = 15

    image_size: int = 32
    channels: int = 16
    
    latent_mean: tuple = (
        0.022520065307617188, 0.028891298919916153, -0.032047729939222336, -0.02299668826162815,
        0.07542382180690765, 0.005438343621790409, 0.018080536276102066, -0.009380939416587353,
        -0.059318918734788895, -0.005332487169653177, 0.0028201835229992867, 0.001997536513954401,
        -0.059100743383169174, -0.014550937339663506, -0.04712633043527603, 0.020210357382893562
    )
    latent_std: tuple = (
        2.3372809886932373, 2.3508591651916504, 2.346092700958252, 2.3590447902679443,
        2.3411736488342285, 2.3550071716308594, 2.368407964706421, 2.3568196296691895,
        2.332087278366089, 2.373119592666626, 2.3301920890808105, 2.348010778427124,
        2.3711912631988525, 2.364023447036743, 2.3427982330322266, 2.3572182655334473
    )
    latent_clip_value: float = 10.0
    latent_npy_path: str = "/kaggle/input/datasets/sautkin/ffhq-sd3-5-vae-repa-e-latents-256/ffhq_latents_32x32x16.npy"
    use_minibatch_discrimination: bool = False
    num_kernels_mbd: int = 100
    kernel_dim_mbd: int = 5

    disc_base_features: int = 256
    gen_features: int = 384
    gen_depth: int = 16
    gen_mapping_dim: int = 1024

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