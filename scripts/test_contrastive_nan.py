"""Synthetic stability checks for contrastive / Yin-Yang losses (run on CPU)."""
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training.losses import (
    contrastive_dcl_loss,
    contrastive_info_nce_loss,
    contrastive_loss,
    full_yin_yang_contrastive_loss,
    yin_yang_contrastive_loss,
)

N = 256
D = 16
TAU = 0.1


def _tanh_proj(key, shape):
    return jnp.tanh(jax.random.normal(key, shape) * 0.5)


def _report(name, loss, grads_ok=False):
    loss_f = float(loss)
    nan = bool(jnp.isnan(loss))
    inf = bool(jnp.isinf(loss))
    print(f"  {name}: loss={loss_f:.6f}  nan={nan}  inf={inf}  grad_ok={grads_ok}")


def _grad_finite(fn, *args):
    loss, grad = jax.value_and_grad(fn)(*args)
    leaves = jax.tree_util.tree_leaves(grad)
    ok = all(jnp.all(jnp.isfinite(g)) for g in leaves) if leaves else True
    return loss, ok


def scenario_normal():
    k = jax.random.PRNGKey(0)
    k, kr, kra, kf, kfa = jax.random.split(k, 5)
    z_r = _tanh_proj(kr, (N, D))
    z_ra = _tanh_proj(kra, (N, D))
    z_f = _tanh_proj(kf, (N, D))
    z_fa = _tanh_proj(kfa, (N, D))
    return z_r, z_ra, z_f, z_fa


def scenario_near_zero():
    return tuple(jnp.ones((N, D), dtype=jnp.float32) * 1e-12 for _ in range(4))


def scenario_collapsed_fake():
    k = jax.random.PRNGKey(1)
    z_r = _tanh_proj(k, (N, D))
    z_ra = _tanh_proj(jax.random.fold_in(k, 1), (N, D))
    z_f = jnp.broadcast_to(jnp.ones((1, D)) * 0.01, (N, D))
    z_fa = jnp.broadcast_to(jnp.ones((1, D)) * 0.02, (N, D))
    return z_r, z_ra, z_f, z_fa


def scenario_bf16():
    z_r, z_ra, z_f, z_fa = scenario_normal()
    return (
        z_r.astype(jnp.bfloat16),
        z_ra.astype(jnp.bfloat16),
        z_f.astype(jnp.bfloat16),
        z_fa.astype(jnp.bfloat16),
    )


def scenario_opposite_real_aug():
    k = jax.random.PRNGKey(2)
    z_r = _tanh_proj(k, (N, D))
    z_ra = -z_r
    z_f = _tanh_proj(jax.random.fold_in(k, 1), (N, D))
    z_fa = _tanh_proj(jax.random.fold_in(k, 2), (N, D))
    return z_r, z_ra, z_f, z_fa


def diagnose_full_yy(z_r, z_ra, z_f, z_fa):
    norms = [float(jnp.min(jnp.linalg.norm(z, axis=-1))) for z in (z_r, z_ra, z_f, z_fa)]
    z_parts = [
        z_r / jnp.linalg.norm(z_r, axis=-1, keepdims=True),
        z_ra / jnp.linalg.norm(z_ra, axis=-1, keepdims=True),
        z_f / jnp.linalg.norm(z_f, axis=-1, keepdims=True),
        z_fa / jnp.linalg.norm(z_fa, axis=-1, keepdims=True),
    ]
    z = jnp.concatenate(z_parts, axis=0)
    sim = jnp.dot(z.astype(jnp.float32), z.astype(jnp.float32).T) / TAU
    nan_sim = int(jnp.sum(jnp.isnan(sim)))
    inf_sim = int(jnp.sum(jnp.isinf(sim)))
    print(f"    norms_min={norms}  sim_nan={nan_sim}  sim_inf={inf_sim}  sim_max={float(jnp.max(sim)):.3f}")


def scenario_exact_zeros():
    z = jnp.zeros((N, D), dtype=jnp.float32)
    return z, z, z, z


def run_scenario(label, maker):
    print(f"\n=== {label} ===")
    z_r, z_ra, z_f, z_fa = maker()
    diagnose_full_yy(z_r, z_ra, z_f, z_fa)

    _report("full_yin_yang", full_yin_yang_contrastive_loss(z_r, z_ra, z_f, z_fa, TAU))
    _report("yin_yang", yin_yang_contrastive_loss(z_r, z_ra, z_f, TAU))
    _report("infonce", contrastive_info_nce_loss(z_r, z_ra, TAU))
    _report("dcl", contrastive_dcl_loss(z_r, z_ra, TAU))

    def f(zr, zra, zf, zfa):
        return full_yin_yang_contrastive_loss(zr, zra, zf, zfa, TAU)

    loss, ok = _grad_finite(f, z_r, z_ra, z_f, z_fa)
    _report("full_yin_yang+grad", loss, ok)


def main():
    print(f"JAX devices: {jax.devices()}")
    print(f"N={N} D={D} tau={TAU}")
    run_scenario("normal tanh projections", scenario_normal)
    run_scenario("exact zeros (was NaN before fix)", scenario_exact_zeros)
    run_scenario("near-zero vectors", scenario_near_zero)
    run_scenario("collapsed fake cluster", scenario_collapsed_fake)
    run_scenario("bf16 inputs", scenario_bf16)
    run_scenario("real_aug = -real (strong aug)", scenario_opposite_real_aug)

    z_r, z_ra, z_f, z_fa = scenario_normal()
    _report(
        "wrapper full_yin_yang",
        contrastive_loss(z_r, z_ra, "full_yin_yang", TAU, z_f, z_fa),
    )
    failed = []
    for maker in (
        scenario_normal,
        scenario_exact_zeros,
        scenario_near_zero,
        scenario_collapsed_fake,
        scenario_bf16,
        scenario_opposite_real_aug,
    ):
        z_r, z_ra, z_f, z_fa = maker()
        loss = full_yin_yang_contrastive_loss(z_r, z_ra, z_f, z_fa, TAU)
        if jnp.isnan(loss) or jnp.isinf(loss):
            failed.append(maker.__name__)
    if failed:
        print(f"\nFAIL: NaN/Inf in {failed}")
        sys.exit(1)
    print("\nAll scenarios finite. Done.")


if __name__ == "__main__":
    main()
