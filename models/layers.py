import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Tuple

class LipschitzDense(nn.Module):
    features: int
    use_bias: bool = True
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x):
        in_features = x.shape[-1]
        kernel = self.param(
            "kernel",
            nn.initializers.lecun_normal(),
            (in_features, self.features),
            self.dtype,
        )
        kf = kernel.astype(jnp.float32)
        norm = jnp.sqrt(jnp.sum(jnp.square(kf)) + 1e-8)
        kernel_n = (kf / norm).astype(self.dtype)
        y = jnp.dot(x.astype(kernel_n.dtype), kernel_n)
        if self.use_bias:
            bias = self.param("bias", nn.initializers.zeros_init(), (self.features,), self.dtype)
            y = y + bias
        return y


class LipschitzConv(nn.Module):
    features: int
    kernel_size: Tuple[int, int] = (3, 3)
    strides: Tuple[int, int] = (1, 1)
    padding: str = "SAME"
    feature_group_count: int = 1
    use_bias: bool = True
    dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, x):
        in_features = x.shape[-1]
        kh, kw = self.kernel_size
        rhs = in_features // self.feature_group_count
        kernel_shape = (kh, kw, rhs, self.features)
        kernel = self.param(
            "kernel",
            nn.initializers.lecun_normal(),
            kernel_shape,
            self.dtype,
        )
        kf = kernel.astype(jnp.float32)
        norm = jnp.sqrt(jnp.sum(jnp.square(kf)) + 1e-8)
        kernel_n = (kf / norm).astype(self.dtype)
        y = jax.lax.conv_general_dilated(
            x.astype(kernel_n.dtype),
            kernel_n,
            self.strides,
            self.padding,
            dimension_numbers=("NHWC", "HWIO", "NHWC"),
            feature_group_count=self.feature_group_count,
        )
        if self.use_bias:
            bias = self.param("bias", nn.initializers.zeros_init(), (self.features,), self.dtype)
            y = y + bias
        return y.astype(self.dtype)


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


def apply_rotary_emb(x, cos, sin):
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    x_rot = jnp.concatenate([-x2, x1], axis=-1)
    return (x * cos) + (x_rot * sin)


def get_2d_rope_sin_cos(H, W, head_dim, reduction_ratio=1, dtype=jnp.float32):
    import numpy as np
    if head_dim % 4 != 0:
        raise ValueError(f"head_dim {head_dim} must be divisible by 4 for 2D RoPE")
    half_dim = head_dim // 2
    inv_freq = 1.0 / (10000 ** (np.arange(0, half_dim, 2, dtype=np.float32) / half_dim))
    pos_y = np.arange(0, H * reduction_ratio, reduction_ratio, dtype=np.float32)
    pos_x = np.arange(0, W * reduction_ratio, reduction_ratio, dtype=np.float32)
    freqs_y = np.einsum("i,j->ij", pos_y, inv_freq)
    freqs_x = np.einsum("i,j->ij", pos_x, inv_freq)
    freqs_y = np.broadcast_to(freqs_y[:, None, :], (pos_y.shape[0], pos_x.shape[0], freqs_y.shape[-1]))
    freqs_x = np.broadcast_to(freqs_x[None, :, :], (pos_y.shape[0], pos_x.shape[0], freqs_x.shape[-1]))
    freqs = np.concatenate([freqs_y, freqs_x], axis=-1)
    emb = np.concatenate([freqs, freqs], axis=-1)
    cos = np.cos(emb)
    sin = np.sin(emb)
    # Возвращаем статические константы (jnp.asarray превратит их в frozen constants в графе XLA)
    return jnp.asarray(cos[None, ..., None, :], dtype=dtype), jnp.asarray(sin[None, ..., None, :], dtype=dtype)


class SpatialReductionAttention(nn.Module):
    """Эффективный Attention: Q в полном разрешении, K и V в сжатом; 2D RoPE на Q и K."""
    features: int
    num_heads: int = 4
    reduction_ratio: int = 2
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x):
        B, H, W, C = x.shape
        head_dim = C // self.num_heads
        n_tokens_q = H * W

        cos_q, sin_q = get_2d_rope_sin_cos(H, W, head_dim, reduction_ratio=1, dtype=self.dtype)

        x_flat = x.reshape((B, n_tokens_q, C))
        q = nn.Dense(C, use_bias=False, dtype=self.dtype)(x_flat)
        q = q.reshape((B, H, W, self.num_heads, head_dim))
        q = apply_rotary_emb(q, cos_q, sin_q)
        q = q.reshape((B, n_tokens_q, self.num_heads, head_dim))

        if self.reduction_ratio > 1:
            x_reduced = nn.avg_pool(
                x,
                window_shape=(self.reduction_ratio, self.reduction_ratio),
                strides=(self.reduction_ratio, self.reduction_ratio),
            )
            Hr = H // self.reduction_ratio
            Wr = W // self.reduction_ratio
            cos_k, sin_k = get_2d_rope_sin_cos(Hr, Wr, head_dim, reduction_ratio=self.reduction_ratio, dtype=self.dtype)
        else:
            x_reduced = x
            Hr, Wr = H, W
            cos_k, sin_k = cos_q, sin_q

        n_tokens_kv = Hr * Wr
        x_reduced_flat = x_reduced.reshape((B, n_tokens_kv, C))

        kv = nn.Dense(C * 2, use_bias=False, dtype=self.dtype)(x_reduced_flat)
        k, v = jnp.split(kv, 2, axis=-1)

        k = k.reshape((B, Hr, Wr, self.num_heads, head_dim))
        k = apply_rotary_emb(k, cos_k, sin_k)
        k = k.reshape((B, n_tokens_kv, self.num_heads, head_dim))
        v = v.reshape((B, n_tokens_kv, self.num_heads, head_dim))

        q = nn.GroupNorm(num_groups=self.num_heads, use_scale=False)(q.reshape((B, n_tokens_q, C))).reshape(
            (B, n_tokens_q, self.num_heads, head_dim)
        )
        k = nn.GroupNorm(num_groups=self.num_heads, use_scale=False)(k.reshape((B, n_tokens_kv, C))).reshape(
            (B, n_tokens_kv, self.num_heads, head_dim)
        )

        attn = jnp.einsum("bnhd,bmhd->bhnm", q, k) * (head_dim ** -0.5)
        attn = jax.nn.softmax(attn.astype(jnp.float32), axis=-1).astype(self.dtype)

        out = jnp.einsum("bhnm,bmhd->bnhd", attn, v)
        out = out.reshape((B, n_tokens_q, C))
        
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
        T = LipschitzDense(
            self.num_kernels * self.kernel_dim, use_bias=False, dtype=self.dtype
        )(x)
        T = T.reshape((x.shape[0], self.num_kernels, self.kernel_dim))

        T_f32 = T.astype(jnp.float32)
        diffs = jnp.expand_dims(T_f32, 1) - jnp.expand_dims(T_f32, 0)
        abs_diffs = jnp.sum(jnp.abs(diffs), axis=-1)

        c_b = jnp.exp(-abs_diffs)
        o_x = jnp.sum(c_b, axis=1) - 1.0

        return jnp.concatenate([x, o_x.astype(x.dtype)], axis=-1)

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

class AdaLNZero(nn.Module):
    """SOTA DiT Style Modulation (AdaLN-Zero)"""
    features: int
    dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, x, w):
        x = nn.LayerNorm(use_scale=False, use_bias=False, dtype=jnp.float32)(x).astype(self.dtype)
        
        style_params = nn.Dense(
            self.features * 6, 
            kernel_init=nn.initializers.zeros_init(), 
            bias_init=nn.initializers.constant(0.01),
            dtype=self.dtype
        )(w)
        
        shift_s, scale_s, gate_s, shift_f, scale_f, gate_f = jnp.split(style_params, 6, axis=-1)
        
        shift_s = jnp.expand_dims(shift_s, (1, 2))
        scale_s = jnp.expand_dims(scale_s, (1, 2))
        gate_s = jnp.expand_dims(gate_s, (1, 2))
        shift_f = jnp.expand_dims(shift_f, (1, 2))
        scale_f = jnp.expand_dims(scale_f, (1, 2))
        gate_f = jnp.expand_dims(gate_f, (1, 2))
        
        return x, shift_s, scale_s, gate_s, shift_f, scale_f, gate_f

class HybridDiTBlock(nn.Module):
    features: int
    num_heads: int = 4
    dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, x, w):
        norm_x, shift_s, scale_s, gate_s, shift_f, scale_f, gate_f = AdaLNZero(
            features=self.features, dtype=self.dtype
        )(x, w)

        x_modulated = norm_x * (1.0 + scale_s) + shift_s

        x_local = nn.Conv(
            self.features, (7, 7), padding="SAME", feature_group_count=self.features, dtype=self.dtype
        )(x_modulated)

        attn_out = SpatialReductionAttention(
            features=self.features, 
            num_heads=self.num_heads, 
            reduction_ratio=4,
            dtype=self.dtype
        )(x_local)

        x = x + gate_s * (x_local + attn_out)

        norm_x2 = nn.LayerNorm(use_scale=False, use_bias=False, dtype=jnp.float32)(x).astype(self.dtype)
        x_modulated2 = norm_x2 * (1.0 + scale_f) + shift_f

        ffn = nn.Dense(self.features * 4, dtype=self.dtype)(x_modulated2)
        ffn = GeGLU()(ffn)
        ffn = GRN(dim=self.features * 2, dtype=self.dtype)(ffn)
        ffn = nn.Dense(self.features, dtype=self.dtype)(ffn)

        x = x + gate_f * ffn

        return x