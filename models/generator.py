import jax
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
    image_size: int = 32
    dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, z, noise=None):
        w = MappingNetwork(features=self.mapping_dim, num_layers=4, dtype=jnp.float32)(z).astype(self.dtype)

        x_mean = self.param(
            "constant_input",
            nn.initializers.normal(stddev=0.02),
            (1, self.image_size, self.image_size, self.features),
        )

        x_std = self.param(
            "noise_std",
            nn.initializers.constant(0.01),
            (1, 1, 1, self.features),
        )

        B = z.shape[0]
        x_mean = jnp.broadcast_to(x_mean, (B, self.image_size, self.image_size, self.features))
        
        if noise is None:
            if self.has_variable("rngs", "noise"):
                noise = jax.random.normal(self.make_rng("noise"), x_mean.shape)
            else:
                noise = jnp.zeros_like(x_mean)

        x = x_mean + noise * jax.nn.softplus(x_std)
        x = x.astype(self.dtype)

        for _ in range(self.depth):
            x = nn.remat(HybridDiTBlock)(features=self.features, dtype=self.dtype)(x, w)

        x = nn.LayerNorm(dtype=jnp.float32)(x).astype(self.dtype)
        x = nn.Dense(self.channels, kernel_init=nn.initializers.zeros_init(), dtype=self.dtype)(x)
        return x
