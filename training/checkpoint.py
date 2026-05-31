import os
import pickle
from typing import Any, Optional

import jax
import jax.numpy as jnp
import numpy as np
from flax import jax_utils
from flax.training import train_state

CHECKPOINT_VERSION = 1


def _to_numpy(pytree):
    return jax.tree_util.tree_map(lambda x: np.asarray(jax.device_get(x)), pytree)


def _train_state_to_dict(ts: train_state.TrainState) -> dict:
    return {
        "step": int(np.asarray(ts.step)),
        "params": _to_numpy(ts.params),
        "opt_state": _to_numpy(ts.opt_state),
    }


def _train_state_from_dict(
    d: dict, apply_fn, tx, devices: Optional[list] = None
) -> train_state.TrainState:
    params = d["params"]
    opt_state = d["opt_state"]
    if devices is not None and devices:

        def _maybe_shard(x):
            x = jnp.asarray(x)
            if x.ndim > 0 and x.shape[0] == len(devices):
                return jax.device_put_sharded([x[i] for i in range(len(devices))], devices)
            return jax.device_put(x)

        params = jax.tree_util.tree_map(_maybe_shard, params)
        opt_state = jax.tree_util.tree_map(_maybe_shard, opt_state)

    ts = train_state.TrainState.create(apply_fn=apply_fn, params=params, tx=tx)
    return ts.replace(step=jnp.asarray(d["step"]), opt_state=opt_state)


def resolve_checkpoint_path(checkpoint_dir: str, checkpoint_file: str, resume_from: str) -> Optional[str]:
    if resume_from in (None, "", "false", "False", "0"):
        return None
    if resume_from in ("auto", "latest", "true", "True", "1"):
        path = os.path.join(checkpoint_dir, checkpoint_file)
        return path if os.path.isfile(path) else None
    path = resume_from
    return path if os.path.isfile(path) else None


def save_training_checkpoint(
    path: str,
    *,
    g_state: train_state.TrainState,
    d_state: train_state.TrainState,
    ema_g_params,
    rng,
    global_step: int,
    epoch: int,
    wandb_run_id: Optional[str],
    config_dict: dict,
    num_devices: int,
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "version": CHECKPOINT_VERSION,
        "global_step": int(global_step),
        "epoch": int(epoch),
        "num_devices": int(num_devices),
        "rng": np.asarray(jax.device_get(rng)),
        "wandb_run_id": wandb_run_id,
        "config": config_dict,
        "g_state": _train_state_to_dict(jax_utils.unreplicate(g_state)),
        "d_state": _train_state_to_dict(d_state),
        "ema_g_params": _to_numpy(jax_utils.unreplicate(ema_g_params)),
    }
    tmp = f"{path}.tmp"
    with open(tmp, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def load_training_checkpoint(path: str) -> dict[str, Any]:
    with open(path, "rb") as f:
        payload = pickle.load(f)
    if payload.get("version") != CHECKPOINT_VERSION:
        raise ValueError(f"Unsupported checkpoint version: {payload.get('version')}")
    return payload


def restore_from_checkpoint(
    payload: dict,
    *,
    g_apply_fn,
    d_apply_fn,
    tx_g,
    tx_d,
    devices,
) -> tuple[train_state.TrainState, train_state.TrainState, Any, Any, int, int]:
    if payload["num_devices"] != len(devices):
        raise ValueError(
            f"Checkpoint num_devices={payload['num_devices']} != current {len(devices)}"
        )
    g_state = _train_state_from_dict(payload["g_state"], g_apply_fn, tx_g, devices=None)
    g_state = jax_utils.replicate(jax.device_put(g_state))
    d_state = _train_state_from_dict(payload["d_state"], d_apply_fn, tx_d, devices=devices)
    ema_g_params = jax_utils.replicate(jax.device_put(payload["ema_g_params"]))
    rng = jnp.asarray(payload["rng"], dtype=jnp.uint32)
    return (
        g_state,
        d_state,
        ema_g_params,
        rng,
        int(payload["global_step"]),
        int(payload["epoch"]),
    )
