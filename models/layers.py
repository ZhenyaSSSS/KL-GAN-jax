import jax
import jax.numpy as jnp
import flax.linen as nn

class BlurPool(nn.Module):
    """Anti-aliasing filter (BlurPool) for downsampling."""
    @nn.compact
    def __call__(self, x):
        # Простой и быстрый 2x2 Average Pool для антиалиасинга перед свертками
        return nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))

class GeGLU(nn.Module):
    """GeGLU activation: splits input in half and applies GELU to one half."""
    @nn.compact
    def __call__(self, x):
        x, gate = jnp.split(x, 2, axis=-1)
        return x * nn.gelu(gate)

class GlobalAttention(nn.Module):
    """Linear Self-Attention for low-resolution feature maps."""
    features: int
    dtype: jnp.dtype = jnp.float32
    
    @nn.compact
    def __call__(self, x):
        B, H, W, C = x.shape
        x_flat = x.reshape((B, H * W, C))
        
        qkv = nn.Dense(self.features * 3, dtype=self.dtype)(x_flat)
        q, k, v = jnp.split(qkv, 3, axis=-1)
        
        # Простейший Scaled Dot-Product Attention
        attn = jnp.einsum('bnc,bmc->bnm', q, k) * (self.features ** -0.5)
        attn = jax.nn.softmax(attn, axis=-1)
        
        out = jnp.einsum('bnm,bmc->bnc', attn, v)
        out = out.reshape((B, H, W, self.features))
        
        out = nn.Dense(C, dtype=self.dtype)(out)
        return x + out

class MinibatchDiscrimination(nn.Module):
    num_kernels: int = 100
    kernel_dim: int = 5
    dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, x):
        T = nn.Dense(self.num_kernels * self.kernel_dim, use_bias=False, dtype=self.dtype)(x)
        T = T.reshape((x.shape[0], self.num_kernels, self.kernel_dim))

        # Вычисляем разницы и экспоненту в float32
        T_f32 = T.astype(jnp.float32)
        diffs = jnp.expand_dims(T_f32, 1) - jnp.expand_dims(T_f32, 0)
        abs_diffs = jnp.sum(jnp.abs(diffs), axis=-1)

        c_b = jnp.exp(-abs_diffs)
        o_x = jnp.sum(c_b, axis=1) - 1.0

        return jnp.concatenate([x, o_x.astype(x.dtype)], axis=-1)
