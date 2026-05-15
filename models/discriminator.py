import flax.linen as nn
import jax.numpy as jnp
from models.layers import MinibatchDiscrimination, BlurPool, GeGLU, GlobalAttention, GRN

class ConvNeXtBlock(nn.Module):
    """Depthwise ConvNeXt V2-style block: dw conv -> LN -> MLP + GeGLU -> GRN -> projection."""
    features: int
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x):
        shortcut = x
        x = nn.Conv(x.shape[-1], (7, 7), padding="SAME", feature_group_count=x.shape[-1], dtype=self.dtype)(x)
        x = nn.LayerNorm(dtype=jnp.float32)(x).astype(self.dtype)
        x = nn.Dense(self.features * 4 * 2, dtype=self.dtype)(x)
        x = GeGLU()(x)
        x = GRN(dim=self.features * 4, dtype=self.dtype)(x)
        x = nn.Dense(self.features, dtype=self.dtype)(x)
        
        if shortcut.shape[-1] != self.features:
            shortcut = nn.Dense(self.features, dtype=self.dtype)(shortcut)
            
        return x + shortcut

class Discriminator(nn.Module):
    """ConvNeXt-style discriminator with BlurPool and multi-scale self-attention (16², 8²)."""
    use_sn: bool = False  # unused; kept for config compatibility
    num_kernels_mbd: int = 100
    kernel_dim_mbd: int = 5
    dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, x):
        x = x.astype(self.dtype)
        x = nn.Conv(64, (3, 3), padding="SAME", dtype=self.dtype)(x)
        x = nn.swish(x)
        x = ConvNeXtBlock(features=128, dtype=self.dtype)(x)
        x = BlurPool()(x)
        x = ConvNeXtBlock(features=256, dtype=self.dtype)(x)
        x = GlobalAttention(features=256, dtype=self.dtype)(x)
        x = BlurPool()(x)
        x = ConvNeXtBlock(features=512, dtype=self.dtype)(x)
        x = GlobalAttention(features=512, dtype=self.dtype)(x)
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
