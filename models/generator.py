import jax.numpy as jnp
import flax.linen as nn
from models.layers import HybridDiTBlock


class MappingNetwork(nn.Module):
    features: int = 256
    num_layers: int = 4
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, z):
        z = z / jnp.sqrt(jnp.mean(jnp.square(z), axis=-1, keepdims=True) + 1e-8)
        w = z
        for _ in range(self.num_layers):
            w = nn.Dense(self.features, dtype=self.dtype)(w)
            w = nn.swish(w)
        return w


class Generator(nn.Module):
    channels: int = 3
    features: int = 128
    mapping_dim: int = 256
    depth: int = 6
    patch_size: int = 2
    image_size: int = 32
    dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, z):
        w = MappingNetwork(features=self.mapping_dim, num_layers=4, dtype=jnp.float32)(z).astype(self.dtype)
        grid_size = self.image_size // self.patch_size
        x = self.param(
            "constant_input",
            nn.initializers.normal(stddev=0.02),
            (1, grid_size, grid_size, self.features),
        )
        B = z.shape[0]
        x = jnp.broadcast_to(x, (B, grid_size, grid_size, self.features)).astype(self.dtype)
        for _ in range(self.depth):
            x = HybridDiTBlock(features=self.features, dtype=self.dtype)(x, w)
        x = nn.LayerNorm(dtype=jnp.float32)(x).astype(self.dtype)
        out_channels = (self.patch_size**2) * self.channels
        x = nn.Dense(out_channels, kernel_init=nn.initializers.zeros_init(), dtype=self.dtype)(x)
        x = x.reshape((B, grid_size, grid_size, self.patch_size, self.patch_size, self.channels))
        x = jnp.transpose(x, (0, 1, 3, 2, 4, 5))
        x = x.reshape((B, self.image_size, self.image_size, self.channels))
        return nn.tanh(x)
