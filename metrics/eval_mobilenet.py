import jax
import jax.numpy as jnp
import flax.linen as nn
from training.losses import calc_stats, kl_divergence_gaussian

class SimpleFeatureExtractor(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=64, kernel_size=(3, 3), strides=(2, 2))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))
        return x

def calculate_mobilenet_kl(real_images, fake_images, feature_extractor, params):
    f_real = feature_extractor.apply({'params': params}, real_images)
    f_fake = feature_extractor.apply({'params': params}, fake_images)
    
    mu_r, var_r = calc_stats(f_real)
    mu_f, var_f = calc_stats(f_fake)
    
    kl_score = kl_divergence_gaussian(mu_f, var_f, mu_r, var_r)
    return kl_score
