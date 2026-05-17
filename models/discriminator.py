import flax.linen as nn
import jax.numpy as jnp
from models.layers import GeGLU, MinibatchDiscrimination, SpatialReductionAttention

class DiscBlock(nn.Module):
    """ConvNeXt-style block with optional spatial reduction attention (no AdaLN)."""
    features: int
    num_heads: int = 4
    use_attn: bool = True
    attn_reduction: int = 4
    dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, x):
        shortcut = x

        x = nn.LayerNorm(dtype=jnp.float32)(x).astype(self.dtype)
        x_local = nn.Conv(
            self.features,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="SAME",
            feature_group_count=self.features,
            dtype=self.dtype,
        )(x)
        
        if self.use_attn:
            attn_out = SpatialReductionAttention(
                features=self.features, 
                num_heads=self.num_heads, 
                reduction_ratio=self.attn_reduction,
                dtype=self.dtype
            )(x_local)
            x = x_local + attn_out
        else:
            x = x_local

        x = nn.LayerNorm(dtype=jnp.float32)(x).astype(self.dtype)
        x_ffn = nn.Dense(self.features * 4, dtype=self.dtype)(x)
        x_ffn = GeGLU()(x_ffn)
        x_ffn = nn.Dense(self.features, dtype=self.dtype)(x_ffn)
        
        gamma = self.param("gamma", nn.initializers.constant(1e-4), (1, 1, 1, self.features), self.dtype)

        return shortcut + x_ffn * gamma

class Discriminator(nn.Module):
    use_sn: bool = False
    use_mbd: bool = False
    num_kernels_mbd: int = 100
    kernel_dim_mbd: int = 5
    base_features: int = 128
    dtype: jnp.dtype = jnp.bfloat16
    loss_type: str = "manifold"
    manifold_proj_dim: int = 16

    @nn.compact
    def __call__(self, x):
        x = x.astype(self.dtype)

        x = nn.Conv(
            self.base_features,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="SAME",
            dtype=self.dtype,
        )(x)

        for _ in range(2):
            x = nn.remat(DiscBlock)(features=self.base_features, use_attn=True, dtype=self.dtype)(x)

        x = nn.Conv(
            self.base_features * 2,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding="SAME",
            dtype=self.dtype,
        )(x)

        for _ in range(2):
            x = nn.remat(DiscBlock)(features=self.base_features * 2, use_attn=False, dtype=self.dtype)(x)

        x = nn.Conv(
            self.base_features * 4,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding="SAME",
            dtype=self.dtype,
        )(x)

        for _ in range(2):
            x = nn.remat(DiscBlock)(features=self.base_features * 4, use_attn=False, dtype=self.dtype)(x)
            
        x = nn.LayerNorm(dtype=jnp.float32)(x).astype(self.dtype)
        x = nn.swish(x)
        
        x = jnp.mean(x, axis=(1, 2))

        if self.use_mbd:
            x = MinibatchDiscrimination(
                num_kernels=self.num_kernels_mbd,
                kernel_dim=self.kernel_dim_mbd,
                dtype=self.dtype,
            )(x)

        if self.loss_type == "manifold":
            f = nn.Dense(self.manifold_proj_dim, dtype=self.dtype)(x)
            f = nn.tanh(f)
        else:
            f = nn.Dense(256, dtype=self.dtype)(x)
            
        return f.astype(jnp.float32)
