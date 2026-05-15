import jax
import flax.linen as nn
import jax.numpy as jnp
from models.layers import GlobalAttention

def upsample_bilinear(x, scale=2):
    B, H, W, C = x.shape
    return jax.image.resize(x, (B, H * scale, W * scale, C), method='bilinear')

class MappingNetwork(nn.Module):
    """Z -> W Style mapping network."""
    features: int = 256
    num_layers: int = 4
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, z):
        w = z
        for _ in range(self.num_layers):
            w = nn.Dense(self.features, dtype=self.dtype)(w)
            w = nn.swish(w)
        return w

class ResBlockGen(nn.Module):
    """Style-based ResNet Block with Upsampling."""
    features: int
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x, w):
        shortcut = x
        
        # Block 1
        style1 = nn.Dense(x.shape[-1] * 2, dtype=self.dtype)(w)
        scale1, shift1 = jnp.split(style1, 2, axis=-1)
        x = nn.GroupNorm(num_groups=1, dtype=jnp.float32)(x).astype(self.dtype) # LayerNorm
        x = x * (1.0 + jnp.expand_dims(scale1, (1, 2))) + jnp.expand_dims(shift1, (1, 2))
        x = nn.swish(x)
        
        x = upsample_bilinear(x)
        x = nn.Conv(self.features, (3, 3), padding="SAME", dtype=self.dtype)(x)
        
        # Block 2
        style2 = nn.Dense(self.features * 2, dtype=self.dtype)(w)
        scale2, shift2 = jnp.split(style2, 2, axis=-1)
        x = nn.GroupNorm(num_groups=1, dtype=jnp.float32)(x).astype(self.dtype)
        x = x * (1.0 + jnp.expand_dims(scale2, (1, 2))) + jnp.expand_dims(shift2, (1, 2))
        x = nn.swish(x)
        
        x = nn.Conv(self.features, (3, 3), padding="SAME", dtype=self.dtype)(x)
        
        # Shortcut upsample
        shortcut = upsample_bilinear(shortcut)
        if shortcut.shape[-1] != self.features:
            shortcut = nn.Conv(self.features, (1, 1), padding="SAME", dtype=self.dtype)(shortcut)
            
        return x + shortcut

class Generator(nn.Module):
    """SOTA Style-based Generator."""
    channels: int = 3
    dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, z):
        # 1. Считаем стиль W (в float32 для точности)
        w = MappingNetwork(features=256, dtype=jnp.float32)(z).astype(self.dtype)
        
        # 2. Инициализируем константный или проекционный тензор 4x4
        x = nn.Dense(256 * 4 * 4, dtype=self.dtype)(w)
        x = x.reshape((x.shape[0], 4, 4, 256))
        
        # 3. ResNet блоки с модуляцией стилем и Upsample (вместо ConvTranspose)
        x = ResBlockGen(features=128, dtype=self.dtype)(x, w) # 4 -> 8
        
        # Self-Attention на 8x8, как в топовых GAN (внимание к глобальным деталям)
        x = GlobalAttention(features=128, dtype=self.dtype)(x)
        
        x = ResBlockGen(features=64, dtype=self.dtype)(x, w)  # 8 -> 16
        x = ResBlockGen(features=32, dtype=self.dtype)(x, w)  # 16 -> 32
        
        # 4. Финальная проекция в RGB
        x = nn.swish(x)
        x = nn.Conv(self.channels, (3, 3), padding="SAME", dtype=self.dtype)(x)
        x = nn.tanh(x)
        
        return x
