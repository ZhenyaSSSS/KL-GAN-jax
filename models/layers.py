import jax
import jax.numpy as jnp
import flax.linen as nn

class BlurPool(nn.Module):
    """Anti-aliasing filter (BlurPool) for downsampling."""
    @nn.compact
    def __call__(self, x):
        return nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))

class GeGLU(nn.Module):
    """GeGLU activation: splits input in half and applies GELU to one half."""
    @nn.compact
    def __call__(self, x):
        x, gate = jnp.split(x, 2, axis=-1)
        return x * nn.gelu(gate)


class GRN(nn.Module):
    """Global Response Normalization (ConvNeXt V2): channel competition via spatial L2 energy."""

    dim: int
    dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, x):
        gx = jnp.sqrt(jnp.sum(jnp.square(x.astype(jnp.float32)), axis=(1, 2), keepdims=True) + 1e-6).astype(
            self.dtype
        )
        nx = gx / (jnp.mean(gx.astype(jnp.float32), axis=-1, keepdims=True) + 1e-6).astype(self.dtype)
        gamma = self.param("gamma", nn.initializers.zeros_init(), (1, 1, 1, self.dim), self.dtype)
        beta = self.param("beta", nn.initializers.zeros_init(), (1, 1, 1, self.dim), self.dtype)
        return x * (1.0 + gamma * nx) + beta


class GlobalAttention(nn.Module):
    """Multi-head self-attention with QK GroupNorm and zero-init residual scale (gamma)."""

    features: int
    num_heads: int = 4
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x):
        B, H, W, C = x.shape
        if C != self.features:
            raise ValueError(f"channel dim {C} must match features={self.features}")
        if C % self.num_heads != 0:
            raise ValueError(f"channels {C} must be divisible by num_heads={self.num_heads}")

        head_dim = C // self.num_heads
        n_tokens = H * W
        x_flat = x.reshape((B, n_tokens, C))

        qkv = nn.Dense(C * 3, use_bias=False, dtype=self.dtype)(x_flat)
        q, k, v = jnp.split(qkv, 3, axis=-1)

        q = q.reshape((B, n_tokens, self.num_heads, head_dim))
        k = k.reshape((B, n_tokens, self.num_heads, head_dim))
        v = v.reshape((B, n_tokens, self.num_heads, head_dim))

        q_flat = q.reshape((B, n_tokens, C))
        k_flat = k.reshape((B, n_tokens, C))
        q_flat = nn.GroupNorm(
            num_groups=self.num_heads,
            use_bias=False,
            use_scale=False,
            dtype=jnp.float32,
        )(q_flat).astype(self.dtype)
        k_flat = nn.GroupNorm(
            num_groups=self.num_heads,
            use_bias=False,
            use_scale=False,
            dtype=jnp.float32,
        )(k_flat).astype(self.dtype)
        q = q_flat.reshape((B, n_tokens, self.num_heads, head_dim))
        k = k_flat.reshape((B, n_tokens, self.num_heads, head_dim))

        attn = jnp.einsum("bnhd,bmhd->bhnm", q, k) * (head_dim ** -0.5)
        attn = jax.nn.softmax(attn.astype(jnp.float32), axis=-1).astype(self.dtype)

        out = jnp.einsum("bhnm,bmhd->bnhd", attn, v)
        out = out.reshape((B, n_tokens, C))
        out = nn.Dense(C, dtype=self.dtype)(out)
        out = out.reshape((B, H, W, C))

        gamma = self.param("gamma", nn.initializers.constant(0.0), (1,), self.dtype)
        return x + out * gamma


class MinibatchDiscrimination(nn.Module):
    num_kernels: int = 100
    kernel_dim: int = 5
    dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, x):
        T = nn.Dense(self.num_kernels * self.kernel_dim, use_bias=False, dtype=self.dtype)(x)
        T = T.reshape((x.shape[0], self.num_kernels, self.kernel_dim))

        T_f32 = T.astype(jnp.float32)
        diffs = jnp.expand_dims(T_f32, 1) - jnp.expand_dims(T_f32, 0)
        abs_diffs = jnp.sum(jnp.abs(diffs), axis=-1)

        c_b = jnp.exp(-abs_diffs)
        o_x = jnp.sum(c_b, axis=1) - 1.0

        return jnp.concatenate([x, o_x.astype(x.dtype)], axis=-1)
