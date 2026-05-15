import jax
import jax.numpy as jnp
from functools import partial
from training.losses import calc_stats, symmetric_kl_loss, pearson_correlation_squared
from config import config

@partial(jax.pmap, axis_name='tpu_nodes')
def train_step(rng, g_state, d_state, real_images):
    z = jax.random.normal(rng, (real_images.shape[0], config.latent_dim))
    
    def compute_losses(g_params, d_params):
        fake_images = g_state.apply_fn({"params": g_params}, z)
        f_real = d_state.apply_fn({"params": d_params}, real_images)
        f_fake = d_state.apply_fn({"params": d_params}, fake_images)

        mu_r, var_r = calc_stats(f_real)
        mu_f, var_f = calc_stats(f_fake)
        
        skl = symmetric_kl_loss(mu_r, var_r, mu_f, var_f)
        
        all_mu_real = jax.lax.all_gather(mu_r, axis_name='tpu_nodes') 
        
        def calc_corr(mu_other):
            return pearson_correlation_squared(mu_r, mu_other)
            
        correlations = jax.vmap(calc_corr)(all_mu_real)
        div_loss = jnp.sum(correlations)
        
        loss_D = -skl + (config.lambda_div * div_loss)
        loss_G = skl
        
        return loss_D, loss_G, {"loss_G": loss_G, "loss_D": loss_D, "SKL": skl, "Div_Loss": div_loss}

    def g_loss_fn(g_params):
        _, l_g, metrics = compute_losses(g_params, d_state.params)
        return l_g, metrics

    def d_loss_fn(d_params):
        l_d, _, metrics = compute_losses(g_state.params, d_params)
        return l_d, metrics

    grads_g, metrics_g = jax.grad(g_loss_fn, has_aux=True)(g_state.params)
    grads_d, metrics_d = jax.grad(d_loss_fn, has_aux=True)(d_state.params)
    
    grads_g = jax.lax.pmean(grads_g, axis_name='tpu_nodes')
    
    new_g_state = g_state.apply_gradients(grads=grads_g)
    new_d_state = d_state.apply_gradients(grads=grads_d)
    
    return new_g_state, new_d_state, metrics_g
