import flax.linen as nn
import jax.numpy as jnp
from models.layers import GeGLU, MinibatchDiscrimination, SpatialReductionAttention, LipschitzDense, LipschitzConv

class DiscBlock(nn.Module):
    """Изотропный блок дискриминатора (ConvNeXt + опционально SRA) без AdaLN"""
    features: int
    num_heads: int = 4
    use_attn: bool = True
    attn_reduction: int = 4
    dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, x):
        shortcut = x
        
        # Local
        x = nn.LayerNorm(dtype=jnp.float32)(x).astype(self.dtype)
        x_local = LipschitzConv(
            self.features,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="SAME",
            feature_group_count=self.features,
            dtype=self.dtype,
        )(x)
        
        if self.use_attn:
            # Global Attn (SRA)
            attn_out = SpatialReductionAttention(
                features=self.features, 
                num_heads=self.num_heads, 
                reduction_ratio=self.attn_reduction,
                dtype=self.dtype
            )(x_local)
            x = x_local + attn_out
        else:
            x = x_local
        
        # FFN
        x = nn.LayerNorm(dtype=jnp.float32)(x).astype(self.dtype)
        x_ffn = LipschitzDense(self.features * 4, dtype=self.dtype)(x)
        x_ffn = GeGLU()(x_ffn)
        x_ffn = LipschitzDense(self.features, dtype=self.dtype)(x_ffn)
        
        # Scale residual
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
        
        # 1. Начальный маппинг в высоком разрешении (32x32)
        x = LipschitzConv(
            self.base_features,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="SAME",
            dtype=self.dtype,
        )(x)
        
        # Stage 1: 32x32 (с Attention)
        for _ in range(2):
            x = nn.remat(DiscBlock)(features=self.base_features, use_attn=True, dtype=self.dtype)(x)
            
        # Downsample: 32x32 -> 16x16
        x = LipschitzConv(
            self.base_features * 2,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding="SAME",
            dtype=self.dtype,
        )(x)
        
        # Stage 2: 16x16 (только локальные текстуры ConvNeXt)
        for _ in range(2):
            x = nn.remat(DiscBlock)(features=self.base_features * 2, use_attn=False, dtype=self.dtype)(x)
            
        # Downsample: 16x16 -> 8x8
        x = LipschitzConv(
            self.base_features * 4,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding="SAME",
            dtype=self.dtype,
        )(x)
        
        # Stage 3: 8x8 (только локальные текстуры ConvNeXt)
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
            f = LipschitzDense(self.manifold_proj_dim, dtype=self.dtype)(x)
            f = nn.tanh(f)
        else:
            f = LipschitzDense(256, dtype=self.dtype)(x)
            
        return f.astype(jnp.float32)
