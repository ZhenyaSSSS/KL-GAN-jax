import flax.linen as nn
import jax.numpy as jnp
from models.layers import GeGLU, MinibatchDiscrimination

class DiscBlock(nn.Module):
    """Изотропный блок дискриминатора (ConvNeXt + Attn) без AdaLN"""
    features: int
    num_heads: int = 4
    dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, x):
        shortcut = x
        
        # Local
        x = nn.LayerNorm(dtype=jnp.float32)(x).astype(self.dtype)
        x = nn.Conv(self.features, (3, 3), padding="SAME", feature_group_count=self.features, dtype=self.dtype)(x)
        
        # Global Attn
        B, H, W, C = x.shape
        x_flat = x.reshape((B, H * W, C))
        attn_out = nn.SelfAttention(num_heads=self.num_heads, dtype=self.dtype)(x_flat)
        x = x + attn_out.reshape((B, H, W, C))
        
        # FFN
        x = nn.LayerNorm(dtype=jnp.float32)(x).astype(self.dtype)
        x_ffn = nn.Dense(self.features * 4, dtype=self.dtype)(x)
        x_ffn = GeGLU()(x_ffn)
        x_ffn = nn.Dense(self.features, dtype=self.dtype)(x_ffn)
        
        # Scale residual
        gamma = self.param("gamma", nn.initializers.constant(1e-4), (1, 1, 1, self.features), self.dtype)
        
        return shortcut + x_ffn * gamma

class Discriminator(nn.Module):
    use_sn: bool = False
    num_kernels_mbd: int = 100
    kernel_dim_mbd: int = 5
    depth: int = 6
    patch_size: int = 2
    features: int = 128
    dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, x):
        x = x.astype(self.dtype)
        
        # 1. Patchify
        x = nn.Conv(
            self.features, 
            kernel_size=(self.patch_size, self.patch_size), 
            strides=(self.patch_size, self.patch_size),
            dtype=self.dtype
        )(x)
        
        # 2. Изотропная труба
        for _ in range(self.depth):
            x = DiscBlock(features=self.features, dtype=self.dtype)(x)
            
        x = nn.LayerNorm(dtype=jnp.float32)(x).astype(self.dtype)
        x = nn.swish(x)
        
        # 3. Global Pooling
        x = jnp.mean(x, axis=(1, 2))
        
        # 4. Minibatch Discrimination
        x = MinibatchDiscrimination(
            num_kernels=self.num_kernels_mbd,
            kernel_dim=self.kernel_dim_mbd,
            dtype=self.dtype,
        )(x)

        f = nn.Dense(256, dtype=self.dtype)(x)
        return f.astype(jnp.float32)