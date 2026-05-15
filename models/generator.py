import jax
import flax.linen as nn
import jax.numpy as jnp
from models.layers import GlobalAttention, SqueezeExcitation

class UpSamplePixelShuffle(nn.Module):
    """Depth-to-space upsampling (subpixel / pixel shuffle)."""
    features: int
    scale: int = 2
    dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, x):
        B, H, W, _ = x.shape
        x = nn.Conv(self.features * (self.scale ** 2), (3, 3), padding="SAME", dtype=self.dtype)(x)
        x = x.reshape((B, H, W, self.scale, self.scale, self.features))
        x = jnp.transpose(x, (0, 1, 3, 2, 4, 5))
        x = x.reshape((B, H * self.scale, W * self.scale, self.features))
        
        return x

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
    """Style-modulated residual block with pixel-shuffle upsample and SE."""
    features: int
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x, w):
        shortcut = x
        style1 = nn.Dense(x.shape[-1] * 2, dtype=self.dtype)(w)
        scale1, shift1 = jnp.split(style1, 2, axis=-1)
        x = nn.GroupNorm(num_groups=1, dtype=jnp.float32)(x).astype(self.dtype)
        x = x * (1.0 + jnp.expand_dims(scale1, (1, 2))) + jnp.expand_dims(shift1, (1, 2))
        x = nn.swish(x)
        x = UpSamplePixelShuffle(features=self.features, dtype=self.dtype)(x)
        style2 = nn.Dense(self.features * 2, dtype=self.dtype)(w)
        scale2, shift2 = jnp.split(style2, 2, axis=-1)
        x = nn.GroupNorm(num_groups=1, dtype=jnp.float32)(x).astype(self.dtype)
        x = x * (1.0 + jnp.expand_dims(scale2, (1, 2))) + jnp.expand_dims(shift2, (1, 2))
        x = nn.swish(x)
        
        x = nn.Conv(self.features, (3, 3), padding="SAME", dtype=self.dtype)(x)
        x = SqueezeExcitation(features=self.features, dtype=self.dtype)(x)

        B, H, W, C = shortcut.shape
        shortcut = jax.image.resize(
            shortcut, shape=(B, H * 2, W * 2, C), method="nearest"
        )
        if C != self.features:
            shortcut = nn.Conv(
                self.features,
                (1, 1),
                padding="SAME",
                use_bias=False,
                dtype=self.dtype,
            )(shortcut)

        return x + shortcut

class Generator(nn.Module):
    """Style-based generator: mapping network, residual blocks, optional attention."""
    channels: int = 3
    dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, z):
        w = MappingNetwork(features=256, dtype=jnp.float32)(z).astype(self.dtype)
        x = nn.Dense(256 * 4 * 4, dtype=self.dtype)(w)
        x = x.reshape((x.shape[0], 4, 4, 256))
        x = ResBlockGen(features=128, dtype=self.dtype)(x, w)
        x = GlobalAttention(features=128, dtype=self.dtype)(x)
        x = ResBlockGen(features=64, dtype=self.dtype)(x, w)
        x = ResBlockGen(features=32, dtype=self.dtype)(x, w)
        x = nn.swish(x)
        x = nn.Conv(self.channels, (3, 3), padding="SAME", dtype=self.dtype)(x)
        x = nn.tanh(x)
        
        return x
