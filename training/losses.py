import jax
import jax.numpy as jnp

def calc_stats(f, eps=1e-6):
    mu = jnp.mean(f, axis=0)
    var = jnp.mean((f - mu)**2, axis=0) + eps
    return mu, var

def kl_divergence_gaussian(mu_p, var_p, mu_q, var_q):
    kl = jnp.log(jnp.sqrt(var_q) / jnp.sqrt(var_p)) + (var_p + (mu_p - mu_q)**2) / (2 * var_q) - 0.5
    return jnp.sum(kl)

def symmetric_kl_loss(mu_r, var_r, mu_f, var_f):
    kl_r_f = kl_divergence_gaussian(mu_r, var_r, mu_f, var_f)
    kl_f_r = kl_divergence_gaussian(mu_f, var_f, mu_r, var_r)
    return 0.5 * (jnp.log(1 + kl_r_f) + jnp.log(1 + kl_f_r))

def pearson_correlation_squared(x, y, eps=1e-8):
    x_c = x - jnp.mean(x)
    y_c = y - jnp.mean(y)
    cov = jnp.sum(x_c * y_c)
    var_x = jnp.sum(x_c**2)
    var_y = jnp.sum(y_c**2)
    std = jnp.sqrt(var_x * var_y)
    corr = cov / (std + eps)
    return corr**2
