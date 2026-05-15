import flax.linen as nn
import jax.numpy as jnp

from models.layers import MinibatchDiscrimination


class Discriminator(nn.Module):
    use_sn: bool = False
    num_kernels_mbd: int = 100
    kernel_dim_mbd: int = 5
    dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=64, kernel_size=(4, 4), strides=(2, 2), padding="SAME", dtype=self.dtype)(x)
        x = nn.leaky_relu(x, negative_slope=0.2)

        x = nn.Conv(features=128, kernel_size=(4, 4), strides=(2, 2), padding="SAME", dtype=self.dtype)(x)
        # BN in float32 for numerical stability
        x = nn.BatchNorm(use_running_average=False, dtype=jnp.float32)(x)
        x = nn.leaky_relu(x, negative_slope=0.2)

        x = nn.Conv(features=256, kernel_size=(4, 4), strides=(2, 2), padding="SAME", dtype=self.dtype)(x)
        x = nn.BatchNorm(use_running_average=False, dtype=jnp.float32)(x)
        x = nn.leaky_relu(x, negative_slope=0.2)

        x = x.reshape((x.shape[0], -1))

        x = MinibatchDiscrimination(
            num_kernels=self.num_kernels_mbd,
            kernel_dim=self.kernel_dim_mbd,
            dtype=self.dtype,
        )(x)

        f = nn.Dense(features=256, dtype=self.dtype)(x)
        
        # Гарантируем, что выходные признаки всегда float32
        # Это критически важно для стабильности вычисления KL-дивергенции и дисперсии
        return f.astype(jnp.float32)
