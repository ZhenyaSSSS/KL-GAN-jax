import flax.linen as nn
import jax.numpy as jnp
from models.layers import MinibatchDiscrimination, BlurPool, GeGLU, GlobalAttention

class ConvNeXtBlock(nn.Module):
    """Depthwise ConvNeXt-style block for Discriminator."""
    features: int
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x):
        shortcut = x
        
        # Depthwise Conv + LayerNorm
        x = nn.Conv(x.shape[-1], (7, 7), padding="SAME", feature_group_count=x.shape[-1], dtype=self.dtype)(x)
        x = nn.LayerNorm(dtype=jnp.float32)(x).astype(self.dtype)
        
        # Pointwise Conv + GeGLU
        x = nn.Dense(self.features * 4 * 2, dtype=self.dtype)(x)
        x = GeGLU()(x)
        x = nn.Dense(self.features, dtype=self.dtype)(x)
        
        if shortcut.shape[-1] != self.features:
            shortcut = nn.Dense(self.features, dtype=self.dtype)(shortcut)
            
        return x + shortcut

class Discriminator(nn.Module):
    """SOTA Hybrid Discriminator with ConvNeXt blocks, BlurPool and Attention."""
    use_sn: bool = False # Deprecated, LayerNorm is used instead for stability
    num_kernels_mbd: int = 100
    kernel_dim_mbd: int = 5
    dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, x):
        x = x.astype(self.dtype)
        x = nn.Conv(64, (3, 3), padding="SAME", dtype=self.dtype)(x)
        x = nn.swish(x)
        
        # 32x32 -> 16x16
        x = ConvNeXtBlock(features=128, dtype=self.dtype)(x)
        x = BlurPool()(x)
        
        # 16x16 -> 8x8
        x = ConvNeXtBlock(features=256, dtype=self.dtype)(x)
        x = GlobalAttention(features=256, dtype=self.dtype)(x) # Attention на 8x8
        x = BlurPool()(x)
        
        # 8x8 -> 4x4
        x = ConvNeXtBlock(features=512, dtype=self.dtype)(x)
        x = BlurPool()(x)
        
        x = nn.swish(x)
        x = x.reshape((x.shape[0], -1))

        x = MinibatchDiscrimination(
            num_kernels=self.num_kernels_mbd,
            kernel_dim=self.kernel_dim_mbd,
            dtype=self.dtype,
        )(x)

        f = nn.Dense(256, dtype=self.dtype)(x)
        return f.astype(jnp.float32)
