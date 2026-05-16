import jax
import jax.numpy as jnp
from functools import partial
from training.losses import (
    calc_stats_stable,
    symmetric_kl_loss,
    symmetric_kl_loss_with_fixed_real,
    zero_centered_repulsion_loss,
)
from config import config


@partial(jax.pmap, axis_name="tpu_nodes")
def train_step(rng, g_state, d_state, ema_g_params, real_images):
    rng, z_rng, noise_rng = jax.random.split(rng, 3)
    z = jax.random.normal(z_rng, (real_images.shape[0], config.latent_dim))

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

    grads_g = jax.lax.pmean(grads_g, axis_name="tpu_nodes")
    grads_d = jax.lax.pmean(grads_d, axis_name="tpu_nodes")

    new_g_state = g_state.apply_gradients(grads=grads_g)
    new_d_state = d_state.apply_gradients(grads=grads_d)

    def update_ema(ema, p):
        return 0.999 * ema + 0.001 * p

    new_ema_g_params = jax.tree_util.tree_map(update_ema, ema_g_params, new_g_state.params)

    metrics = {
        "loss_G": loss_G,
        "loss_D": -skl + (config.lambda_div * div_loss),
        "SKL": skl,
        "Div_Loss": div_loss,
    }

    return new_g_state, new_d_state, new_ema_g_params, metrics
