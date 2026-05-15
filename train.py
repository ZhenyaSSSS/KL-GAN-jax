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
from metrics.eval_mobilenet import SimpleFeatureExtractor, calculate_mobilenet_kl
from metrics.fid_score import compute_fid


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
    import jax
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
    g_model = Generator(channels=config.channels, dtype=dt)
    d_model = Discriminator(
        use_sn=config.use_sn,
        num_kernels_mbd=config.num_kernels_mbd,
        kernel_dim_mbd=config.kernel_dim_mbd,
        dtype=dt,
    )

    tx_g = optax.adam(learning_rate=config.lr_gen, b1=config.beta1, b2=config.beta2)
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

    # Initialize EMA params for Generator
    ema_g_params = g_state.params

    rng, test_z_rng = jax.random.split(rng)
    test_z = jax.random.normal(test_z_rng, (64, config.latent_dim), dtype=jnp.float32)

    rng, fe_rng = jax.random.split(rng)
    fe_model = SimpleFeatureExtractor()
    fe_params = fe_model.init(fe_rng, dummy_img)["params"]

    on_host = np.asarray(jax.device_get(dataset_replicated))

    print("Starting compilation...")
    compile_start = time.time()

    for epoch in range(1, config.epochs + 1):
        epoch_loss_g, epoch_loss_d, epoch_skl, epoch_div = 0.0, 0.0, 0.0, 0.0

        with tqdm(total=steps_per_epoch, desc=f"Epoch {epoch}/{config.epochs}") as pbar:
            for _ in range(steps_per_epoch):
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

                epoch_loss_g += float(jnp.mean(metrics["loss_G"]))
                epoch_loss_d += float(jnp.mean(metrics["loss_D"]))
                epoch_skl += float(jnp.mean(metrics["SKL"]))
                epoch_div += float(jnp.mean(metrics["Div_Loss"]))

                pbar.update(1)

        avg_metrics = {
            "Train/Loss_G": epoch_loss_g / steps_per_epoch,
            "Train/Loss_D": epoch_loss_d / steps_per_epoch,
            "Train/SKL": epoch_skl / steps_per_epoch,
            "Train/Diversity": epoch_div / steps_per_epoch,
        }
        wandb.log(avg_metrics, step=epoch)

        # Evaluate using EMA params!
        ema_g_params_cpu = jax.tree_util.tree_map(lambda x: x[0], ema_g_params)
        fake_test_images = g_model.apply({"params": ema_g_params_cpu}, test_z)
        grid = create_image_grid(np.array(fake_test_images))
        wandb.log({"Generated Images": wandb.Image(grid)}, step=epoch)

        if epoch % config.eval_every_epochs == 0:
            real_eval = on_host[:1024]
            rng, z_rng = jax.random.split(rng)
            z_eval = jax.random.normal(z_rng, (1024, config.latent_dim), dtype=jnp.float32)

            @jax.jit
            def gen_batch(z):
                return g_model.apply({"params": ema_g_params_cpu}, z)

            fake_eval = []
            for i in range(0, 1024, 128):
                fake_eval.append(gen_batch(z_eval[i : i + 128]))
            fake_eval = jnp.concatenate(fake_eval, axis=0)

            kl_score = calculate_mobilenet_kl(real_eval, fake_eval, fe_model, fe_params)
            wandb.log({"Val/MobileNet_KL": float(kl_score)}, step=epoch)
            print(f"Epoch {epoch} | MobileNet_KL: {float(kl_score):.4f}")

        if epoch % config.fid_every_epochs == 0 or epoch == config.epochs:
            fake_images_fid = []
            for i in tqdm(range(0, config.num_fid_samples, 256), desc="Generating FID samples"):
                rng, z_rng = jax.random.split(rng)
                z_batch = jax.random.normal(
                    z_rng,
                    (min(256, config.num_fid_samples - i), config.latent_dim),
                    dtype=jnp.float32,
                )
                fake_images_fid.append(np.array(gen_batch(z_batch)))
            fake_images_fid = np.concatenate(fake_images_fid, axis=0)

            real_images_fid = on_host[: config.num_fid_samples]
            fid_score = compute_fid(real_images_fid, fake_images_fid)
            wandb.log({"Val/FID": fid_score}, step=epoch)
            print(f"Epoch {epoch} | FID: {fid_score:.4f}")

    wandb.finish()
    print("Training complete.")


if __name__ == "__main__":
    main()
