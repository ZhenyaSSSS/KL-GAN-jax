import jax
import jax.numpy as jnp
from functools import partial
from training.losses import (
    calc_stats_stable,
    symmetric_kl_loss,
    symmetric_kl_loss_with_fixed_real,
    zero_centered_repulsion_loss,
    sinkhorn_divergence,
    contrastive_info_nce_loss,
    tpu_feature_decorrelation_loss,
    coverage_loss,
)
from config import config

def apply_simple_augmentation(images, rng):
    """SOTA-like augmentation for latents (Noise + Spatial Flip)."""
    rng1, rng2 = jax.random.split(rng, 2)
    
    # 1. Gaussian Noise (0.05 std)
    noise = jax.random.normal(rng1, images.shape) * 0.05
    x = images + noise
    
    # 2. Random Horizontal Flip (safe for spatial VAE latents)
    flip_mask = jax.random.bernoulli(rng2, 0.5, (images.shape[0], 1, 1, 1))
    x_flipped = jnp.flip(x, axis=2)
    x = jnp.where(flip_mask, x_flipped, x)
    
    return x

@partial(jax.pmap, axis_name="tpu_nodes")
def train_step(rng, g_state, d_state, ema_g_params, real_images):
    rng, z_rng, noise_rng, aug_rng = jax.random.split(rng, 4)
    z = jax.random.normal(z_rng, (real_images.shape[0], config.latent_dim))

    if config.loss_type == "manifold":
        def d_loss_fn(d_params):
            proj_real = d_state.apply_fn({"params": d_params}, real_images)
            real_images_aug = apply_simple_augmentation(real_images, aug_rng)
            proj_real_aug = d_state.apply_fn({"params": d_params}, real_images_aug)
            
            fake_images = g_state.apply_fn({"params": g_state.params}, z, rngs={"noise": noise_rng})
            proj_fake = d_state.apply_fn({"params": d_params}, fake_images)
            
            loss_sinkhorn = -sinkhorn_divergence(proj_real, proj_fake, epsilon=config.sinkhorn_epsilon, max_iter=config.sinkhorn_max_iter)
            loss_contrastive = contrastive_info_nce_loss(proj_real, proj_real_aug)
            loss_decorr = tpu_feature_decorrelation_loss(proj_real)
            loss_cov = coverage_loss(proj_real)
            
            loss_D = (loss_sinkhorn + 
                      config.lambda_contrastive * loss_contrastive + 
                      config.lambda_decorr * loss_decorr + 
                      config.lambda_cov * loss_cov)
                      
            return loss_D, (loss_sinkhorn, loss_contrastive, loss_decorr, loss_cov, proj_real)

        grads_d, (loss_sinkhorn, loss_contrastive, loss_decorr, loss_cov, proj_real) = jax.grad(d_loss_fn, has_aux=True)(d_state.params)

        def g_loss_fn(g_params):
            fake_images = g_state.apply_fn({"params": g_params}, z, rngs={"noise": noise_rng})
            proj_fake = d_state.apply_fn({"params": d_state.params}, fake_images)
            
            loss_G = sinkhorn_divergence(proj_real, proj_fake, epsilon=config.sinkhorn_epsilon, max_iter=config.sinkhorn_max_iter)
            return loss_G, loss_G

        grads_g, loss_G = jax.grad(g_loss_fn, has_aux=True)(g_state.params)
        
        metrics = {
            "loss_G": loss_G,
            "loss_D": loss_sinkhorn + config.lambda_contrastive * loss_contrastive + config.lambda_decorr * loss_decorr + config.lambda_cov * loss_cov,
            "Sinkhorn": -loss_sinkhorn, # Log actual positive divergence
            "Contrastive": loss_contrastive,
            "Decorr": loss_decorr,
            "Coverage": loss_cov,
        }

    else:
        def d_loss_fn(d_params):
            f_real = d_state.apply_fn({"params": d_params}, real_images)
            fake_images = g_state.apply_fn({"params": g_state.params}, z, rngs={"noise": noise_rng})
            f_fake = d_state.apply_fn({"params": d_params}, fake_images)

            mu_r, var_r, log_var_r = calc_stats_stable(f_real)
            skl = symmetric_kl_loss(f_real, f_fake)

            all_mu_real = jax.lax.all_gather(mu_r, axis_name="tpu_nodes")
            all_log_var_real = jax.lax.all_gather(log_var_r, axis_name="tpu_nodes")
            div_loss = zero_centered_repulsion_loss(
                mu_r, all_mu_real, log_var_r, all_log_var_real, config.diversity_temperature
            )

            loss_D = -skl + (config.lambda_div * div_loss)
            return loss_D, (skl, div_loss, mu_r, var_r, log_var_r)

        grads_d, (skl, div_loss, mu_r, var_r, log_var_r) = jax.grad(d_loss_fn, has_aux=True)(d_state.params)

        def g_loss_fn(g_params):
            fake_images = g_state.apply_fn({"params": g_params}, z, rngs={"noise": noise_rng})
            f_fake = d_state.apply_fn({"params": d_state.params}, fake_images)

            loss_G = symmetric_kl_loss_with_fixed_real(mu_r, var_r, log_var_r, f_fake)
            return loss_G, loss_G

        grads_g, loss_G = jax.grad(g_loss_fn, has_aux=True)(g_state.params)

        metrics = {
            "loss_G": loss_G,
            "loss_D": -skl + (config.lambda_div * div_loss),
            "SKL": skl,
            "Div_Loss": div_loss,
        }

    grads_g = jax.lax.pmean(grads_g, axis_name="tpu_nodes")

    new_g_state = g_state.apply_gradients(grads=grads_g)
    new_d_state = d_state.apply_gradients(grads=grads_d)

    def update_ema(ema, p):
        return 0.999 * ema + 0.001 * p

    new_ema_g_params = jax.tree_util.tree_map(update_ema, ema_g_params, new_g_state.params)

    return new_g_state, new_d_state, new_ema_g_params, metrics
