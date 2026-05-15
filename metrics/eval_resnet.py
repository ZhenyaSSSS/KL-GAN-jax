import jax
import jax.numpy as jnp
from functools import partial
from training.losses import calc_stats_stable, kl_divergence_stable
import flaxmodels

def load_pretrained_resnet():
    """ResNet18 with per-layer activations (for feature extraction)."""
    return flaxmodels.ResNet18(output='activations', pretrained='imagenet')


@partial(jax.jit, static_argnums=(0,))
def _extract_batch(resnet_model, params, batch):
    out = resnet_model.apply(params, batch, train=False)
    features = out['block4_1']
    features = jnp.mean(features, axis=(1, 2))
    return features

def calculate_resnet_kl(real_images, fake_images, resnet_model, params, batch_size=128):
    """KL divergence between real and fake batches in ResNet18 feature space."""

    def extract_features_batched(images):
        features = []
        for i in range(0, images.shape[0], batch_size):
            batch = images[i:i+batch_size]
            features.append(_extract_batch(resnet_model, params, batch))
        return jnp.concatenate(features, axis=0)

    f_real = extract_features_batched(real_images)
    f_fake = extract_features_batched(fake_images)

    mu_r, var_r, log_var_r = calc_stats_stable(f_real)
    mu_f, var_f, log_var_f = calc_stats_stable(f_fake)
    kl_score = kl_divergence_stable(mu_f, log_var_f, mu_r, log_var_r, var_f)
    return kl_score
