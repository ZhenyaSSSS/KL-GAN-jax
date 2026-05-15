import flax.linen as nn
import jax.numpy as jnp


class Generator(nn.Module):
    channels: int = 3
    dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, z):
        x = nn.Dense(256 * 4 * 4, dtype=self.dtype)(z)
        x = x.reshape((x.shape[0], 4, 4, 256))
        
        # BN in float32 for numerical stability
        x = nn.BatchNorm(use_running_average=False, dtype=jnp.float32)(x)
        x = nn.relu(x)

        x = nn.ConvTranspose(features=128, kernel_size=(4, 4), strides=(2, 2), padding="SAME", dtype=self.dtype)(x)
        x = nn.BatchNorm(use_running_average=False, dtype=jnp.float32)(x)
        x = nn.relu(x)

        x = nn.ConvTranspose(features=64, kernel_size=(4, 4), strides=(2, 2), padding="SAME", dtype=self.dtype)(x)
        x = nn.BatchNorm(use_running_average=False, dtype=jnp.float32)(x)
        x = nn.relu(x)

        x = nn.ConvTranspose(
            features=self.channels, kernel_size=(4, 4), strides=(2, 2), padding="SAME", dtype=self.dtype
        )(x)
        x = nn.tanh(x)
        
        return x
