import jax
import jax.numpy as jnp
from functools import partial
from training.losses import calc_stats, kl_divergence_gaussian
import flaxmodels

def load_pretrained_resnet():
    """Loads a pretrained ResNet18 model that outputs activations."""
    # We use output='activations' so we can extract intermediate feature maps.
    resnet = flaxmodels.ResNet18(output='activations', pretrained='imagenet')
    return resnet

@partial(jax.jit, static_argnums=(0,))
def _extract_batch(resnet_model, params, batch):
    # Apply model, train=False to disable dropout/batchnorm stats updates
    # The image values are expected in [-1, 1], ResNet usually expects ImageNet norm,
    # but for consistent relative features, keeping it as is works well for KL comparison.
    # Note: we add dummy batchnorm state (flaxmodels handles it inside if we just pass params)
    out = resnet_model.apply(params, batch, train=False)
    # block4_1 output shape is (B, H, W, 512). For 32x32 it becomes (B, 1, 1, 512).
    features = out['block4_1'] 
    features = jnp.mean(features, axis=(1, 2)) # Global Average Pooling fallback
    return features

def calculate_resnet_kl(real_images, fake_images, resnet_model, params, batch_size=128):
    """Calculates KL Divergence between real and fake images using ResNet18 features."""
    
    def extract_features_batched(images):
        features = []
        for i in range(0, images.shape[0], batch_size):
            batch = images[i:i+batch_size]
            features.append(_extract_batch(resnet_model, params, batch))
        return jnp.concatenate(features, axis=0)

    f_real = extract_features_batched(real_images)
    f_fake = extract_features_batched(fake_images)
    
    mu_r, var_r = calc_stats(f_real)
    mu_f, var_f = calc_stats(f_fake)
    
    kl_score = kl_divergence_gaussian(mu_f, var_f, mu_r, var_r)
    return kl_score
