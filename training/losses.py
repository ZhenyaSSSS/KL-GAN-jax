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

# --- Manifold-GAN Losses ---

def compute_cost_matrix(X, Y):
    dist = jnp.sum(X**2, axis=1, keepdims=True) + jnp.sum(Y**2, axis=1) - 2 * jnp.dot(X, Y.T)
    dist = jnp.maximum(dist, 0.0)
    d = X.shape[-1]
    return dist / jnp.maximum(jnp.asarray(d, dtype=dist.dtype), jnp.asarray(1.0, dtype=dist.dtype))

def sinkhorn_ot_primal(C, epsilon=0.05, max_iter=15):
    import jax
    n, m = C.shape
    log_mu = jnp.full((n,), -jnp.log(n))
    log_nu = jnp.full((m,), -jnp.log(m))
    log_K = -C / epsilon

    def scan_body(carry, _):
        f, g = carry
        f = log_mu - jax.nn.logsumexp(log_K + g[None, :], axis=1)
        g = log_nu - jax.nn.logsumexp(log_K.T + f[None, :], axis=1)
        return (f, g), None

    f_init = jnp.zeros((n,))
    g_init = jnp.zeros((m,))
    (f_final, g_final), _ = jax.lax.scan(scan_body, (f_init, g_init), None, length=max_iter)

    log_P = f_final[:, None] + log_K + g_final[None, :]
    P = jnp.exp(log_P)
    
    linear_cost = jnp.sum(P * C)
    kl_cost = jnp.sum(P * jnp.log(jnp.maximum(P, 1e-30))) + jnp.log(n * m)
    return linear_cost + epsilon * kl_cost

def sinkhorn_divergence(X, Y, epsilon=0.05, max_iter=15):
    C_xy = compute_cost_matrix(X, Y)
    C_xx = compute_cost_matrix(X, X)
    C_yy = compute_cost_matrix(Y, Y)
    
    ot_xy = sinkhorn_ot_primal(C_xy, epsilon, max_iter)
    ot_xx = sinkhorn_ot_primal(C_xx, epsilon, max_iter)
    ot_yy = sinkhorn_ot_primal(C_yy, epsilon, max_iter)
    
    return ot_xy - 0.5 * ot_xx - 0.5 * ot_yy

def contrastive_info_nce_loss(z1, z2, temperature=0.1):
    """Decoupled Contrastive Learning (DCL) loss."""
    import jax
    import jax.scipy.special
    
    N = z1.shape[0]

    # 1. L2-нормализация эмбеддингов
    z1 = z1 / jnp.linalg.norm(z1, axis=-1, keepdims=True)
    z2 = z2 / jnp.linalg.norm(z2, axis=-1, keepdims=True)

    # 2. Собираем все признаки в один батч размером (2N, D)
    z = jnp.concatenate([z1, z2], axis=0)

    # 3. Считаем матрицу косинусного сходства "Всех со Всеми"
    sim_matrix = jnp.dot(z, z.T) / temperature

    # 4. Индексы для позитивных пар
    labels = jnp.arange(2 * N)
    pos_indices = (labels + N) % (2 * N)
    pos_sim = sim_matrix[labels, pos_indices]

    # 5. Маска для негативных примеров (выкидываем диагональ и позитивные пары)
    mask = jnp.ones((2 * N, 2 * N))
    mask = mask.at[labels, labels].set(0.0)
    mask = mask.at[labels, pos_indices].set(0.0)

    # 6. Считаем негативную часть через logsumexp
    neg_logsumexp = jax.scipy.special.logsumexp(sim_matrix, axis=1, b=mask)

    # 7. Итоговая формула DCL
    loss_per_sample = -pos_sim + neg_logsumexp

    return jnp.mean(loss_per_sample)

def tpu_feature_decorrelation_loss(local_proj):
    import jax
    all_projs = jax.lax.all_gather(local_proj, axis_name="tpu_nodes")
    num_tpus, B, D = all_projs.shape
    
    if num_tpus == 1:
        return jnp.asarray(0.0)

    mean_proj = jnp.mean(all_projs, axis=1, keepdims=True)
    std_proj = jnp.std(all_projs, axis=1, keepdims=True) + 1e-6
    z_all = (all_projs - mean_proj) / std_proj
    
    z_all_transposed = jnp.transpose(z_all, (0, 2, 1)) # [num_tpus, D, B]
    z_flat = z_all_transposed.reshape(num_tpus * D, B)
    cov_matrix = jnp.dot(z_flat, z_flat.T) / (B - 1)
    
    import jax.scipy.linalg
    mask = 1.0 - jax.scipy.linalg.block_diag(*[jnp.ones((D, D)) for _ in range(num_tpus)])
    
    off_diag_cov = cov_matrix * mask
    loss = jnp.sum(off_diag_cov**2) / (num_tpus * (num_tpus - 1) * D * D)
    
    return loss

def coverage_loss(proj):
    target_var = 0.33
    variances = jnp.var(proj, axis=0)
    return jnp.mean((variances - target_var)**2)
