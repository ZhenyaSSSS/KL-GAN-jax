import jax.numpy as jnp


def calc_stats_stable(f, min_var=1e-5):
    """Batch mean, variance, and log-variance with a floor for numerical stability."""
    mu = jnp.mean(f, axis=0)
    var = jnp.mean((f - mu) ** 2, axis=0)
    var_clipped = jnp.clip(var, a_min=min_var)
    log_var = jnp.log(var_clipped)
    return mu, var_clipped, log_var


def calc_stats(f, eps=1e-6):
    """Legacy helper: mean and variance with epsilon added to variance."""
    mu = jnp.mean(f, axis=0)
    var = jnp.mean((f - mu) ** 2, axis=0) + eps
    return mu, var


def kl_divergence_gaussian(mu_p, var_p, mu_q, var_q):
    """Diagonal Gaussian KL(p || q); prefer kl_divergence_stable for training."""
    kl = jnp.log(jnp.sqrt(var_q) / jnp.sqrt(var_p)) + (var_p + (mu_p - mu_q) ** 2) / (2 * var_q) - 0.5
    return jnp.sum(kl)


def kl_divergence_stable(mu_p, log_var_p, mu_q, log_var_q, var_p):
    """KL(p || q) for diagonal Gaussians using log-variances (no division by var_q)."""
    term1 = log_var_q - log_var_p
    mean_diff_sq = (mu_p - mu_q) ** 2
    term2 = (var_p + mean_diff_sq) * jnp.exp(-log_var_q)
    kl = 0.5 * (term1 + term2 - 1.0)
    return jnp.sum(kl)


def symmetric_kl_loss(f_real, f_fake):
    """Symmetric log-KL on batch feature statistics (both directions)."""
    mu_r, var_r, log_var_r = calc_stats_stable(f_real)
    mu_f, var_f, log_var_f = calc_stats_stable(f_fake)
    kl_r_f = kl_divergence_stable(mu_r, log_var_r, mu_f, log_var_f, var_r)
    kl_f_r = kl_divergence_stable(mu_f, log_var_f, mu_r, log_var_r, var_f)
    return 0.5 * (jnp.log(1.0 + kl_r_f) + jnp.log(1.0 + kl_f_r))


def symmetric_kl_loss_with_fixed_real(mu_r, var_r, log_var_r, f_fake):
    """Same as symmetric_kl_loss but real-side stats are fixed (e.g. generator step)."""
    mu_f, var_f, log_var_f = calc_stats_stable(f_fake)
    kl_r_f = kl_divergence_stable(mu_r, log_var_r, mu_f, log_var_f, var_r)
    kl_f_r = kl_divergence_stable(mu_f, log_var_f, mu_r, log_var_r, var_f)
    return 0.5 * (jnp.log(1.0 + kl_r_f) + jnp.log(1.0 + kl_f_r))


def safe_normalize(x, eps=1e-8):
    norm = jnp.sqrt(jnp.sum(jnp.square(x), axis=-1, keepdims=True) + eps)
    return x / norm


def zero_centered_repulsion_loss(mu, all_mu, log_var, all_log_var, temperature=0.5):
    mu_norm = safe_normalize(mu)
    all_mu_norm = safe_normalize(all_mu)
    lv_norm = safe_normalize(log_var)
    all_lv_norm = safe_normalize(all_log_var)
    sim_mu = jnp.dot(all_mu_norm, mu_norm)
    sim_lv = jnp.dot(all_lv_norm, lv_norm)
    sim = 0.5 * (sim_mu + sim_lv)
    sim = jnp.clip(sim, -0.9999, 0.9999)
    t = jnp.maximum(temperature, 1e-8)
    sim_scaled = sim / t
    sum_exp = jnp.sum(jnp.exp(sim_scaled))
    self_sim_exp = jnp.exp(1.0 / t)
    others_exp_sum = jnp.clip(sum_exp - self_sim_exp, a_min=1e-7)
    num_devices = all_mu.shape[0]
    num_others = jnp.maximum(1.0, num_devices - 1.0)
    mean_others_exp = others_exp_sum / num_others
    loss = t * jnp.log(mean_others_exp)
    return jnp.where(num_devices > 1, loss, jnp.asarray(0.0, dtype=loss.dtype))
