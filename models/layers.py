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
            bias_init=nn.initializers.zeros_init(),
            dtype=self.dtype
        )(w)
        
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = jnp.split(style_params, 6, axis=-1)
        
        shift_msa = jnp.expand_dims(shift_msa, (1, 2))
        scale_msa = jnp.expand_dims(scale_msa, (1, 2))
        gate_msa = jnp.expand_dims(gate_msa, (1, 2))
        shift_mlp = jnp.expand_dims(shift_mlp, (1, 2))
        scale_mlp = jnp.expand_dims(scale_mlp, (1, 2))
        gate_mlp = jnp.expand_dims(gate_mlp, (1, 2))
        
        return x, shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp

class HybridDiTBlock(nn.Module):
    """DiT + ConvNeXt V2 Hybrid Block"""
    features: int
    num_heads: int = 4
    dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, x, w):
        norm_x, shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = AdaLNZero(
            features=self.features, dtype=self.dtype
        )(x, w)
        
        x_modulated = norm_x * (1.0 + scale_msa) + shift_msa
        
        x_local = nn.Conv(
            self.features, (3, 3), padding="SAME", feature_group_count=self.features, dtype=self.dtype
        )(x_modulated)
        
        B, H, W, C = x_local.shape
        x_flat = x_local.reshape((B, H * W, C))
        attn_out = nn.SelfAttention(
            num_heads=self.num_heads, 
            qkv_features=self.features,
            out_features=self.features,
            dtype=self.dtype
        )(x_flat)
        attn_out = attn_out.reshape((B, H, W, C))
        
        x = x + gate_msa * attn_out
        
        norm_x2 = nn.LayerNorm(use_scale=False, use_bias=False, dtype=jnp.float32)(x).astype(self.dtype)
        x_modulated2 = norm_x2 * (1.0 + scale_mlp) + shift_mlp
        
        ffn_out = nn.Dense(self.features * 4, dtype=self.dtype)(x_modulated2)
        ffn_out = GeGLU()(ffn_out)
        ffn_out = nn.Dense(self.features, dtype=self.dtype)(ffn_out)
        
        x = x + gate_mlp * ffn_out
        
        return x