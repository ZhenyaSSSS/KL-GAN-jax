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

def contrastive_diversity_loss(mu, all_mu, temperature=0.1):
    """CLIP-style InfoNCE Loss для расталкивания Дискриминаторов."""
    # Нормализуем вектора (L2)
    mu_norm = mu / jnp.clip(jnp.linalg.norm(mu, keepdims=True), a_min=1e-6)
    all_mu_norm = all_mu / jnp.clip(jnp.linalg.norm(all_mu, axis=-1, keepdims=True), a_min=1e-6)
    
    # Считаем косинусные сходства
    sim = jnp.dot(all_mu_norm, mu_norm) / temperature
    
    # Минимизируем exp(sim) со всеми критиками (чем меньше сходство, тем меньше лосс)
    # Вычитаем сходство с самим собой (оно всегда = 1/temp) чтобы не штрафовать за него
    self_sim = 1.0 / temperature
    return jnp.log(jnp.clip(jnp.sum(jnp.exp(sim)) - jnp.exp(self_sim), a_min=1e-6))
