import jax.numpy as jnp


def calc_stats_stable(f, min_var=1e-5):
    """Batch mean, variance, and log-variance with a floor for numerical stability."""
    mu = jnp.mean(f, axis=0)
    var = jnp.mean((f - mu) ** 2, axis=0)
    var_clipped = jnp.clip(var, a_min=min_var)
    log_var = jnp.log(var_clipped)
    return mu, var_clipped, log_var


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

# Manifold / KL-GAN losses


def compute_cost_matrix(X, Y):
    """Squared L2 pairwise cost; divide by D so scale is O(1) for tanh-bounded D-dim features."""
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

def _l2_normalize_rows(x, eps=1e-8):
    return safe_normalize(x.astype(jnp.float32), eps=eps)


def _contrastive_sim_matrix(z1, z2, temperature):
    N = z1.shape[0]
    z1 = _l2_normalize_rows(z1)
    z2 = _l2_normalize_rows(z2)
    z = jnp.concatenate([z1, z2], axis=0)
    t = jnp.asarray(temperature, dtype=jnp.float32)
    sim_matrix = jnp.dot(z, z.T) / t
    labels = jnp.arange(2 * N)
    pos_indices = (labels + N) % (2 * N)
    pos_sim = sim_matrix[labels, pos_indices]
    return sim_matrix, labels, pos_indices, pos_sim


def contrastive_info_nce_loss(z1, z2, temperature=0.1):
    """SimCLR InfoNCE: -log(exp(S+) / sum_{j!=i} exp(S_ij)), positive in denominator."""
    import jax.scipy.special

    sim_matrix, labels, pos_indices, pos_sim = _contrastive_sim_matrix(z1, z2, temperature)
    mask = jnp.ones(sim_matrix.shape)
    mask = mask.at[labels, labels].set(0.0)
    denom_logsumexp = jax.scipy.special.logsumexp(sim_matrix, axis=1, b=mask)
    return jnp.mean(-pos_sim + denom_logsumexp)


def contrastive_dcl_loss(z1, z2, temperature=0.1):
    """DCL: -S+ + logsumexp(S-) only; exp(S+) excluded from denominator."""
    import jax.scipy.special

    sim_matrix, labels, pos_indices, pos_sim = _contrastive_sim_matrix(z1, z2, temperature)
    mask = jnp.ones(sim_matrix.shape)
    mask = mask.at[labels, labels].set(0.0)
    mask = mask.at[labels, pos_indices].set(0.0)
    neg_logsumexp = jax.scipy.special.logsumexp(sim_matrix, axis=1, b=mask)
    return jnp.mean(-pos_sim + neg_logsumexp)


def _four_view_yin_yang_sim(z_real, z_real_aug, z_fake, z_fake_aug, temperature):
    N = z_real.shape[0]
    z = jnp.concatenate(
        [
            _l2_normalize_rows(z_real),
            _l2_normalize_rows(z_real_aug),
            _l2_normalize_rows(z_fake),
            _l2_normalize_rows(z_fake_aug),
        ],
        axis=0,
    )
    t = jnp.asarray(temperature, dtype=jnp.float32)
    sim_matrix = jnp.dot(z, z.T) / t
    labels = jnp.arange(4 * N)
    pos_indices = jnp.concatenate(
        [
            jnp.arange(N, 2 * N),
            jnp.arange(0, N),
            jnp.arange(3 * N, 4 * N),
            jnp.arange(2 * N, 3 * N),
        ]
    )
    pos_sim = sim_matrix[labels, pos_indices]
    return sim_matrix, labels, pos_indices, pos_sim, N


def full_yin_yang_contrastive_loss(z_real, z_real_aug, z_fake, z_fake_aug, temperature=0.1):
    sim_matrix, labels, pos_indices, pos_sim, N = _four_view_yin_yang_sim(
        z_real, z_real_aug, z_fake, z_fake_aug, temperature
    )
    import jax.scipy.special

    mask = jnp.ones((4 * N, 4 * N), dtype=jnp.float32)
    mask = mask.at[labels, labels].set(0.0)
    mask = mask.at[labels, pos_indices].set(0.0)
    neg_logsumexp = jax.scipy.special.logsumexp(sim_matrix, axis=1, b=mask)
    return jnp.mean(-pos_sim + neg_logsumexp)


def asymmetric_yin_yang_contrastive_loss(
    z_real, z_real_aug, z_fake, z_fake_aug, temperature=0.1, repulsion_beta=5.0
):
    sim_matrix, labels, pos_indices, pos_sim, N = _four_view_yin_yang_sim(
        z_real, z_real_aug, z_fake, z_fake_aug, temperature
    )
    import jax.scipy.special

    beta = jnp.asarray(repulsion_beta, dtype=jnp.float32)
    mask = jnp.ones((4 * N, 4 * N), dtype=jnp.float32)
    mask = mask.at[N : 2 * N, 2 * N : 4 * N].set(beta)
    mask = mask.at[2 * N : 4 * N, N : 2 * N].set(beta)
    mask = mask.at[labels, labels].set(0.0)
    mask = mask.at[labels, pos_indices].set(0.0)
    neg_logsumexp = jax.scipy.special.logsumexp(sim_matrix, axis=1, b=mask)
    return jnp.mean(-pos_sim + neg_logsumexp)


def yin_yang_contrastive_loss(z_real, z_real_aug, z_fake, temperature=0.1):
    """DCL on real/aug anchors; fake projections act as extra negatives (no extra forward)."""
    import jax.scipy.special

    N = z_real.shape[0]
    z_r = _l2_normalize_rows(z_real)
    z_ra = _l2_normalize_rows(z_real_aug)
    z_f = _l2_normalize_rows(z_fake)
    z = jnp.concatenate([z_r, z_ra, z_f], axis=0)
    t = jnp.asarray(temperature, dtype=jnp.float32)
    sim_matrix = jnp.dot(z, z.T) / t
    anchors_sim = sim_matrix[: 2 * N, :]
    labels = jnp.arange(2 * N)
    pos_indices = (labels + N) % (2 * N)
    pos_sim = anchors_sim[labels, pos_indices]
    mask = jnp.ones((2 * N, 3 * N))
    mask = mask.at[labels, labels].set(0.0)
    mask = mask.at[labels, pos_indices].set(0.0)
    neg_logsumexp = jax.scipy.special.logsumexp(anchors_sim, axis=1, b=mask)
    return jnp.mean(-pos_sim + neg_logsumexp)


def contrastive_loss(
    z1,
    z2,
    loss_type="asymmetric_yin_yang",
    temperature=0.1,
    z_fake=None,
    z_fake_aug=None,
    repulsion_beta=5.0,
):
    if loss_type == "asymmetric_yin_yang":
        if z_fake is None or z_fake_aug is None:
            raise ValueError("asymmetric_yin_yang contrastive loss requires z_fake and z_fake_aug")
        return asymmetric_yin_yang_contrastive_loss(
            z1, z2, z_fake, z_fake_aug, temperature, repulsion_beta=repulsion_beta
        )
    if loss_type == "full_yin_yang":
        if z_fake is None or z_fake_aug is None:
            raise ValueError("full_yin_yang contrastive loss requires z_fake and z_fake_aug")
        return full_yin_yang_contrastive_loss(z1, z2, z_fake, z_fake_aug, temperature)
    if loss_type == "yin_yang":
        if z_fake is None:
            raise ValueError("yin_yang contrastive loss requires z_fake")
        return yin_yang_contrastive_loss(z1, z2, z_fake, temperature)
    if loss_type == "dcl":
        return contrastive_dcl_loss(z1, z2, temperature)
    return contrastive_info_nce_loss(z1, z2, temperature)

def tpu_feature_decorrelation_loss(local_proj):
    import jax
    all_projs = jax.lax.all_gather(local_proj, axis_name="tpu_nodes")
    num_tpus, B, D = all_projs.shape
    
    if num_tpus == 1:
        return jnp.asarray(0.0)

    mean_proj = jnp.mean(all_projs, axis=1, keepdims=True)
    std_proj = jnp.std(all_projs, axis=1, keepdims=True) + 1e-6
    z_all = (all_projs - mean_proj) / std_proj
    
    z_all_transposed = jnp.transpose(z_all, (0, 2, 1))
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
