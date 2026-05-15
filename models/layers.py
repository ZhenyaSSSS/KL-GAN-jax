import jax.numpy as jnp
import flax.linen as nn


class MinibatchDiscrimination(nn.Module):
    num_kernels: int = 100
    kernel_dim: int = 5
    dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, x):
        T = nn.Dense(self.num_kernels * self.kernel_dim, use_bias=False, dtype=self.dtype)(x)
        T = T.reshape((x.shape[0], self.num_kernels, self.kernel_dim))

        # Вычисляем разницы и экспоненту в float32, чтобы избежать underflow/overflow
        T_f32 = T.astype(jnp.float32)
        diffs = jnp.expand_dims(T_f32, 1) - jnp.expand_dims(T_f32, 0)
        abs_diffs = jnp.sum(jnp.abs(diffs), axis=-1)

        c_b = jnp.exp(-abs_diffs)
        o_x = jnp.sum(c_b, axis=1) - 1.0

        return jnp.concatenate([x, o_x.astype(x.dtype)], axis=-1)
