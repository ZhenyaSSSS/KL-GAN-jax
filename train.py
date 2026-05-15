import os
import time

import jax
import jax.numpy as jnp
import flax
from flax.training import train_state
import optax
import wandb
from tqdm import tqdm
import numpy as np

from config import config
from data.loader import load_dataset_replicated
from models.generator import Generator
from models.discriminator import Discriminator
from training.step import train_step
from metrics.eval_resnet import load_pretrained_resnet, calculate_resnet_kl

# Fixed latent batch size for @jax.jit gen_batch (avoids XLA recompile on shape changes).
GEN_BATCH_SIZE = 256
LOG_EVERY_STEPS = 20


@jax.pmap
def pmap_get_batch(dataset, indices):
    return jnp.take(dataset, indices, axis=0)


def init_tpu():
    try:
        from jax.tools import colab_tpu
        colab_tpu.setup_tpu()
        print("TPU initialized successfully!")
    except Exception as e:
        print("Not running on Kaggle/Colab TPU or already initialized.", e)
    print(f"JAX device count: {jax.device_count()}")


def create_image_grid(images, grid_size=(8, 8)):
    images = np.asarray(images, dtype=np.float32)
    images = ((images + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    h, w, c = images.shape[1:]
    grid_h, grid_w = grid_size
    grid = np.zeros((grid_h * h, grid_w * w, c), dtype=np.uint8)

    for idx, img in enumerate(images):
        if idx >= grid_h * grid_w:
            break
        i = idx // grid_w
        j = idx % grid_w
        grid[i * h : (i + 1) * h, j * w : (j + 1) * w, :] = img

    return grid


def main():
    jax.config.update("jax_default_matmul_precision", config.jax_matmul_precision)

    init_tpu()
    num_devices = jax.device_count()
    if num_devices < 1:
        raise RuntimeError("No JAX devices found.")

    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") is not None:
        try:
            from kaggle_secrets import UserSecretsClient

            user_secrets = UserSecretsClient()
            wandb_api_key = user_secrets.get_secret("WANDB_API_KEY")
            wandb.login(key=wandb_api_key)
        except Exception as e:
            print("Failed to load W&B key from Kaggle secrets:", e)

    wandb.init(project=config.wandb_project, name=config.wandb_run_name)
    wandb.config.update(config.__dict__)

    rng = jax.random.PRNGKey(config.seed)

    print("Loading and replicating full dataset to all devices...")
    dataset_replicated, num_samples = load_dataset_replicated(image_size=config.image_size)
    if num_samples < config.batch_size_per_device:
        raise ValueError("Dataset smaller than per-device batch size.")
    steps_per_epoch = num_samples // config.batch_size_per_device

    dt = config.compute_dtype
    g_model = Generator(channels=config.channels, image_size=config.image_size, dtype=dt)
    d_model = Discriminator(
        use_sn=config.use_sn,
        num_kernels_mbd=config.num_kernels_mbd,
        kernel_dim_mbd=config.kernel_dim_mbd,
        dtype=dt,
    )

    tx_g = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate=config.lr_gen, b1=config.beta1, b2=config.beta2),
    )
    tx_d = optax.adam(learning_rate=config.lr_disc, b1=config.beta1, b2=config.beta2)

    rng, init_g_rng = jax.random.split(rng)
    dummy_z = jnp.ones((1, config.latent_dim), dtype=jnp.float32)
    g_params = g_model.init(init_g_rng, dummy_z)["params"]
    g_state = train_state.TrainState.create(apply_fn=g_model.apply, params=g_params, tx=tx_g)
    g_state = flax.jax_utils.replicate(g_state)

    rng, init_d_rng = jax.random.split(rng)
    d_keys = jax.random.split(init_d_rng, num_devices)
    dummy_img = jnp.ones((1, config.image_size, config.image_size, config.channels), dtype=jnp.float32)

    @jax.pmap
    def init_d_state(key):
        variables = d_model.init(key, dummy_img)
        return train_state.TrainState.create(
            apply_fn=d_model.apply, 
            params=variables["params"], 
            tx=tx_d
        )

    d_state = init_d_state(d_keys)

    ema_g_params = g_state.params

    rng, test_z_rng = jax.random.split(rng)
    test_z = jax.random.normal(test_z_rng, (GEN_BATCH_SIZE, config.latent_dim), dtype=jnp.float32)

    try:
        resnet_model = load_pretrained_resnet()
        rng, resnet_rng = jax.random.split(rng)
        resnet_params = resnet_model.init(resnet_rng, jnp.ones((1, 32, 32, 3)))
        print("Successfully loaded pre-trained ResNet18 for KL divergence.")
    except Exception as e:
        print(f"Failed to load ResNet18: {e}")
        resnet_model = None
        resnet_params = None

    # Replicated layout may be [num_devices, N, H, W, C]; slice one shard before host copy
    # to avoid pulling all device copies into CPU RAM (device_get(X)[0] would fetch 8×).
    on_host = np.asarray(jax.device_get(dataset_replicated[0]))

    @jax.jit
    def gen_batch(params, z):
        return g_model.apply({"params": params}, z)

    def gen_batch_pad(params, z, n_take):
        """Always compile-friendly GEN_BATCH_SIZE; returns host numpy of first n_take rows."""
        n = int(z.shape[0])
        if n > GEN_BATCH_SIZE:
            raise ValueError(f"z batch {n} exceeds GEN_BATCH_SIZE={GEN_BATCH_SIZE}")
        if n == GEN_BATCH_SIZE:
            out = gen_batch(params, z)
            return np.asarray(out[:n_take])
        pad = GEN_BATCH_SIZE - n
        z_pad = jnp.concatenate([z, jnp.zeros((pad, z.shape[1]), dtype=z.dtype)], axis=0)
        out = gen_batch(params, z_pad)
        return np.asarray(out[:n_take])

    print("Starting compilation...")
    compile_start = time.time()
    global_step = 0

    for epoch in range(1, config.epochs + 1):
        epoch_loss_g, epoch_loss_d, epoch_skl, epoch_div = 0.0, 0.0, 0.0, 0.0

        with tqdm(total=steps_per_epoch, desc=f"Epoch {epoch}/{config.epochs}") as pbar:
            for _ in range(steps_per_epoch):
                step_start = time.time()
                rng, idx_rng = jax.random.split(rng)
                idx = jax.random.randint(
                    idx_rng,
                    (num_devices, config.batch_size_per_device),
                    0,
                    num_samples,
                    dtype=jnp.int32,
                )
                real_batch = pmap_get_batch(dataset_replicated, idx)

                rng, *train_rngs = jax.random.split(rng, num_devices + 1)
                train_rngs = jnp.array(train_rngs)

                g_state, d_state, ema_g_params, metrics = train_step(train_rngs, g_state, d_state, ema_g_params, real_batch)

                if epoch == 1 and _ == 0:
                    print(f"Compilation finished in {time.time() - compile_start:.2f} s.")

                loss_g_val = float(jnp.mean(metrics["loss_G"]))
                loss_d_val = float(jnp.mean(metrics["loss_D"]))
                skl_val = float(jnp.mean(metrics["SKL"]))
                div_val = float(jnp.mean(metrics["Div_Loss"]))

                epoch_loss_g += loss_g_val
                epoch_loss_d += loss_d_val
                epoch_skl += skl_val
                epoch_div += div_val

                step_time = time.time() - step_start
                global_step += 1
                imgs_per_step = num_devices * config.batch_size_per_device

                if global_step % LOG_EVERY_STEPS == 0:
                    mem_gb = 0.0
                    try:
                        dev = jax.local_devices()[0]
                        if hasattr(dev, "memory_stats"):
                            mem_gb = dev.memory_stats().get("bytes_in_use", 0) / (1024**3)
                    except Exception:
                        pass
                    wandb.log(
                        {
                            "Step/Loss_G": loss_g_val,
                            "Step/Loss_D": loss_d_val,
                            "Step/SKL": skl_val,
                            "Step/Div": div_val,
                            "Perf/Step_Time_sec": step_time,
                            "Perf/Images_Per_Sec": imgs_per_step / max(step_time, 1e-9),
                            "Perf/Memory_Used_GB": mem_gb,
                        },
                        step=global_step,
                    )

                pbar.set_postfix(Loss_D=f"{loss_d_val:.3f}", Loss_G=f"{loss_g_val:.3f}")
                pbar.update(1)

        wandb.log(
            {
                "Epoch/Loss_G": epoch_loss_g / steps_per_epoch,
                "Epoch/Loss_D": epoch_loss_d / steps_per_epoch,
                "Epoch/SKL": epoch_skl / steps_per_epoch,
                "Epoch/Diversity": epoch_div / steps_per_epoch,
            },
            step=global_step,
        )

        ema_g_params_cpu = jax.tree_util.tree_map(lambda x: x[0], ema_g_params)
        gen_start = time.time()
        fake_test_images = gen_batch(ema_g_params_cpu, test_z)
        fake_test_images = np.asarray(fake_test_images)[:64]
        print(f"Gen preview & device sync: {time.time() - gen_start:.2f}s")
        grid = create_image_grid(fake_test_images)
        wandb.log({"Generated Images": wandb.Image(grid)}, step=global_step)

        if epoch % config.eval_every_epochs == 0 and resnet_model is not None:
            real_eval = on_host[:2048]
            rng, z_rng = jax.random.split(rng)
            z_eval = jax.random.normal(z_rng, (2048, config.latent_dim), dtype=jnp.float32)

            fake_eval = []
            for i in range(0, 2048, GEN_BATCH_SIZE):
                z_chunk = z_eval[i : i + GEN_BATCH_SIZE]
                fake_eval.append(gen_batch(ema_g_params_cpu, z_chunk))
            fake_eval = jnp.concatenate(fake_eval, axis=0)

            kl_score = calculate_resnet_kl(real_eval, fake_eval, resnet_model, resnet_params)
            wandb.log({"Val/ResNet18_KL": float(kl_score)}, step=global_step)
            print(f"Epoch {epoch} | ResNet18_KL: {float(kl_score):.4f}")

        if epoch % config.fid_every_epochs == 0 or epoch == config.epochs:
            fake_images_fid = []
            for i in tqdm(range(0, config.num_fid_samples, GEN_BATCH_SIZE), desc="Generating FID samples"):
                need = min(GEN_BATCH_SIZE, config.num_fid_samples - i)
                rng, z_rng = jax.random.split(rng)
                z_small = jax.random.normal(z_rng, (need, config.latent_dim), dtype=jnp.float32)
                fake_images_fid.append(gen_batch_pad(ema_g_params_cpu, z_small, need))
            fake_images_fid = np.concatenate(fake_images_fid, axis=0)

            os.makedirs("/kaggle/working/fid_samples", exist_ok=True)
            np.save(f"/kaggle/working/fid_samples/fake_images_epoch_{epoch}.npy", fake_images_fid)
            try:
                from flax.training import checkpoints
                checkpoints.save_checkpoint(
                    ckpt_dir="/kaggle/working/checkpoints",
                    target=ema_g_params_cpu,
                    step=epoch,
                    prefix="g_ema_",
                    keep=3,
                )
                print(f"Saved checkpoint and FID samples to /kaggle/working/ for epoch {epoch}")
            except ImportError:
                import pickle

                os.makedirs("/kaggle/working/checkpoints", exist_ok=True)
                with open(f"/kaggle/working/checkpoints/g_ema_{epoch}.pkl", "wb") as f:
                    pickle.dump(ema_g_params_cpu, f)
                print(f"Saved weights (pickle) and FID samples to /kaggle/working/ for epoch {epoch}")

    wandb.finish()
    print("Training complete.")


if __name__ == "__main__":
    main()
