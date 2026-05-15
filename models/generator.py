import jax.numpy as jnp
import flax.linen as nn
from models.layers import HybridDiTBlock, UpSamplePixelShuffle

class MappingNetwork(nn.Module):
    features: int = 256
    num_layers: int = 4
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, z):
        # Обучаемые статистики для стиля
        mu = self.param('style_mu', nn.initializers.zeros_init(), z.shape[-1:], jnp.float32)
        log_sigma = self.param('style_log_sigma', nn.initializers.zeros_init(), z.shape[-1:], jnp.float32)
        z = z * jnp.exp(log_sigma) + mu

        w = z
        for _ in range(self.num_layers):
            w = nn.Dense(self.features, dtype=self.dtype)(w)
            w = nn.swish(w)
        return w

class Generator(nn.Module):
    channels: int = 3
    features: int = 128
    depth: int = 6
    patch_size: int = 2
    dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, z, spatial_noise=None):
        # 1. Подготавливаем стиль
        w = MappingNetwork(features=self.features, dtype=jnp.float32)(z).astype(self.dtype)
        
        # 2. Подготавливаем пространственный шум (если не передан, генерим)
        B = z.shape[0]
        if spatial_noise is None:
            raise ValueError("В концепции трубы Generator требует spatial_noise на вход!")
            
        # Обучаемые статистики для пространственного шума
        mu_s = self.param('spatial_mu', nn.initializers.zeros_init(), (1, 1, 1, spatial_noise.shape[-1]), jnp.float32)
        log_sigma_s = self.param('spatial_log_sigma', nn.initializers.zeros_init(), (1, 1, 1, spatial_noise.shape[-1]), jnp.float32)
        
        x = spatial_noise * jnp.exp(log_sigma_s) + mu_s
        x = x.astype(self.dtype)

        # 3. Patchify
        x = nn.Conv(
            self.features, 
            kernel_size=(self.patch_size, self.patch_size), 
            strides=(self.patch_size, self.patch_size),
            padding="VALID",
            dtype=self.dtype
        )(x)

        # 4. Изотропная труба
        for _ in range(self.depth):
            x = HybridDiTBlock(features=self.features, dtype=self.dtype)(x, w)

        # 5. Unpatchify
        x = nn.LayerNorm(dtype=jnp.float32)(x).astype(self.dtype)
        x = UpSamplePixelShuffle(features=self.channels, scale=self.patch_size, dtype=self.dtype)(x)
        
        # Финальная проекция
        x = nn.Conv(self.channels, (3, 3), padding="SAME", dtype=self.dtype)(x)
        return nn.tanh(x)