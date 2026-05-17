import os
import time
import threading
from functools import partial

import torch
from diffusers import AutoencoderKL

import jax
import jax.numpy as jnp
import flax
from flax.training import train_state
import optax
import wandb
from tqdm import tqdm
import numpy as np

from config import config
from data.loader import load_latents_sharded
from models.generator import Generator
from models.discriminator import Discriminator
from training.step import train_step
from training.losses import calc_stats_stable, kl_divergence_stable

# Fixed latent batch size for @jax.jit gen_batch (avoids XLA recompile on shape changes).
GEN_BATCH_SIZE = 256
LOG_EVERY_STEPS = 20


@jax.jit
def calculate_latent_kl(real_latents, fake_latents):
    f_real = real_latents.reshape((real_latents.shape[0], -1))
    f_fake = fake_latents.reshape((fake_latents.shape[0], -1))

    mu_r, var_r, log_var_r = calc_stats_stable(f_real)
    mu_f, var_f, log_var_f = calc_stats_stable(f_fake)
    
    kl_score = kl_divergence_stable(mu_f, log_var_f, mu_r, log_var_r, var_f)
    return kl_score / f_real.shape[1]


@jax.pmap
def pmap_get_batch(dataset_shard, indices):
    return jnp.take(dataset_shard, indices, axis=0)


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

    print("Загрузка латентного датасета...")
    dataset_sharded, samples_per_device = load_latents_sharded(
        npy_path=config.latent_npy_path,
        latent_mean=config.latent_mean,
        latent_std=config.latent_std,
        clip_value=config.latent_clip_value,
    )
    num_samples = samples_per_device * num_devices
    if num_samples < config.batch_size_per_device:
        raise ValueError("Dataset smaller than per-device batch size.")
    steps_per_epoch = num_samples // config.batch_size_per_device

    print("Загрузка PyTorch VAE на CPU для декодирования (JAX заберет TPU)...")
    vae = AutoencoderKL.from_pretrained("REPA-E/e2e-sd3.5-vae").to("cpu")
    vae.eval()
    vae.requires_grad_(False)

    def decode_latents_to_rgb(jax_latents):
        latents_np = np.asarray(jax_latents)
        if config.latent_clip_value is not None:
            latents_np = np.clip(latents_np, -config.latent_clip_value, config.latent_clip_value)
            
        mean_arr = np.array(config.latent_mean, dtype=np.float32).reshape(1, 1, 1, -1)
        std_arr = np.array(config.latent_std, dtype=np.float32).reshape(1, 1, 1, -1)
        latents_np = latents_np * std_arr + mean_arr
        
        latents_pt = torch.tensor(latents_np).permute(0, 3, 1, 2).float()
        
        decoded_images = []
        with torch.no_grad():
            for i in range(0, len(latents_pt), 8):
                chunk = latents_pt[i:i+8]
                img = vae.decode(chunk).sample
                decoded_images.append(img.numpy())
                
        decoded_images = np.concatenate(decoded_images, axis=0)
        decoded_images = np.transpose(decoded_images, (0, 2, 3, 1))
        return decoded_images

    dt = config.compute_dtype
    g_model = Generator(channels=config.channels, image_size=config.image_size, dtype=dt)
    d_model = Discriminator(
        use_sn=config.use_sn,
        num_kernels_mbd=config.num_kernels_mbd,
        kernel_dim_mbd=config.kernel_dim_mbd,
        dtype=dt,
        loss_type=config.loss_type,
        manifold_proj_dim=config.manifold_proj_dim,
    )

    tx_g = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(
            learning_rate=config.lr_gen,
            b1=config.beta1,
            b2=config.beta2,
            weight_decay=1e-4,
        ),
    )
    tx_d = optax.adamw(
        learning_rate=config.lr_disc,
        b1=config.beta1,
        b2=config.beta2,
        weight_decay=1e-4,
    )

    rng, init_g_rng, init_noise_rng = jax.random.split(rng, 3)
    dummy_z = jnp.ones((1, config.latent_dim), dtype=jnp.float32)
    g_params = g_model.init({'params': init_g_rng, 'noise': init_noise_rng}, dummy_z)["params"]
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

    on_host = np.asarray(jax.device_get(dataset_sharded[0]))

    # Compile the ppermute logic to rotate shards across TPU cores
    perm = [(i, (i + 1) % num_devices) for i in range(num_devices)]
    
    @partial(jax.pmap, axis_name="tpu_nodes")
    def rotate_shards(dataset_shard):
        return jax.lax.ppermute(dataset_shard, axis_name="tpu_nodes", perm=perm)

    @jax.jit
    def gen_batch(params, z, noise_rng):
        return g_model.apply({"params": params}, z, rngs={"noise": noise_rng})

    def gen_batch_pad(params, z, n_take, noise_rng):
        """Always compile-friendly GEN_BATCH_SIZE; returns host numpy of first n_take rows."""
        n = int(z.shape[0])
        if n > GEN_BATCH_SIZE:
            raise ValueError(f"z batch {n} exceeds GEN_BATCH_SIZE={GEN_BATCH_SIZE}")
        if n == GEN_BATCH_SIZE:
            out = gen_batch(params, z, noise_rng)
            return np.asarray(out[:n_take])
        pad = GEN_BATCH_SIZE - n
        z_pad = jnp.concatenate([z, jnp.zeros((pad, z.shape[1]), dtype=z.dtype)], axis=0)
        out = gen_batch(params, z_pad, noise_rng)
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
                    samples_per_device,
                    dtype=jnp.int32,
                )
                real_batch = pmap_get_batch(dataset_sharded, idx)

                rng, *train_rngs = jax.random.split(rng, num_devices + 1)
                train_rngs = jnp.array(train_rngs)

                g_state, d_state, ema_g_params, metrics = train_step(train_rngs, g_state, d_state, ema_g_params, real_batch)

                if epoch == 1 and _ == 0:
                    print(f"Compilation finished in {time.time() - compile_start:.2f} s.")

                loss_g_val = float(jnp.mean(metrics["loss_G"]))
                loss_d_val = float(jnp.mean(metrics["loss_D"]))
                
                if config.loss_type == "manifold":
                    sinkhorn_val = float(jnp.mean(metrics.get("Sinkhorn", 0.0)))
                    contrastive_val = float(jnp.mean(metrics.get("Contrastive", 0.0)))
                    decorr_val = float(jnp.mean(metrics.get("Decorr", 0.0)))
                    cov_val = float(jnp.mean(metrics.get("Coverage", 0.0)))
                    
                    epoch_skl += sinkhorn_val # Reuse epoch_skl for sinkhorn
                    epoch_div += decorr_val   # Reuse epoch_div for decorr
                else:
                    skl_val = float(jnp.mean(metrics.get("SKL", 0.0)))
                    div_val = float(jnp.mean(metrics.get("Div_Loss", 0.0)))
                    epoch_skl += skl_val
                    epoch_div += div_val

                epoch_loss_g += loss_g_val
                epoch_loss_d += loss_d_val

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
                        
                    log_dict = {
                        "Step/Loss_G": loss_g_val,
                        "Step/Loss_D": loss_d_val,
                        "Perf/Step_Time_sec": step_time,
                        "Perf/Images_Per_Sec": imgs_per_step / max(step_time, 1e-9),
                        "Perf/Memory_Used_GB": mem_gb,
                    }
                    
                    if config.loss_type == "manifold":
                        log_dict.update({
                            "Step/Sinkhorn": sinkhorn_val,
                            "Step/Contrastive": contrastive_val,
                            "Step/Decorr": decorr_val,
                            "Step/Coverage": cov_val,
                        })
                    else:
                        log_dict.update({
                            "Step/SKL": skl_val,
                            "Step/Div": div_val,
                        })
                        
                    wandb.log(log_dict, step=global_step)

                pbar.set_postfix(Loss_D=f"{loss_d_val:.3f}", Loss_G=f"{loss_g_val:.3f}")
                pbar.update(1)

        epoch_log_dict = {
            "Epoch/Loss_G": epoch_loss_g / steps_per_epoch,
            "Epoch/Loss_D": epoch_loss_d / steps_per_epoch,
        }
        
        if config.loss_type == "manifold":
            epoch_log_dict.update({
                "Epoch/Sinkhorn": epoch_skl / steps_per_epoch,
                "Epoch/Decorr": epoch_div / steps_per_epoch,
            })
        else:
            epoch_log_dict.update({
                "Epoch/SKL": epoch_skl / steps_per_epoch,
                "Epoch/Diversity": epoch_div / steps_per_epoch,
            })
            
        wandb.log(epoch_log_dict, step=global_step)

        ema_g_params_cpu = jax.tree_util.tree_map(lambda x: x[0], ema_g_params)
        rng, preview_noise_rng = jax.random.split(rng)
        fake_test_latents = gen_batch(ema_g_params_cpu, test_z[:16], preview_noise_rng) 
        
        # --- АСИНХРОННАЯ ОТПРАВКА В W&B ---
        def decode_and_log(latents, step):
            rgb_images = decode_latents_to_rgb(latents)
            grid = create_image_grid(rgb_images, grid_size=(4, 4))
            wandb.log({"Generated Images (512x512)": wandb.Image(grid)}, step=step)

        latents_np = np.asarray(fake_test_latents).copy()
        t = threading.Thread(target=decode_and_log, args=(latents_np, global_step))
        t.start()

        # --- ЭВАЛЮАЦИЯ (Latent KL) ---
        if epoch % config.eval_every_epochs == 0:
            real_eval = on_host[:1024] 
            rng, z_rng = jax.random.split(rng)
            z_eval = jax.random.normal(z_rng, (1024, config.latent_dim), dtype=jnp.float32)

            fake_eval = []
            rng, eval_noise_rng = jax.random.split(rng)
            for i in range(0, 1024, GEN_BATCH_SIZE):
                eval_noise_rng, step_noise_rng = jax.random.split(eval_noise_rng)
                z_chunk = z_eval[i : i + GEN_BATCH_SIZE]
                fake_eval.append(gen_batch(ema_g_params_cpu, z_chunk, step_noise_rng))
            fake_eval = jnp.concatenate(fake_eval, axis=0)

            kl_score = calculate_latent_kl(real_eval, fake_eval)
            wandb.log({"Val/Latent_KL": float(kl_score)}, step=global_step)
            print(f"Epoch {epoch} | Latent KL: {float(kl_score):.5f}")

        if epoch % config.fid_every_epochs == 0 or epoch == config.epochs:
            fake_latents_fid = []
            rng, fid_noise_rng = jax.random.split(rng)
            for i in tqdm(range(0, config.num_fid_samples, GEN_BATCH_SIZE), desc="Gen FID Latents"):
                need = min(GEN_BATCH_SIZE, config.num_fid_samples - i)
                rng, z_rng = jax.random.split(rng)
                fid_noise_rng, step_noise_rng = jax.random.split(fid_noise_rng)
                z_small = jax.random.normal(z_rng, (need, config.latent_dim), dtype=jnp.float32)
                fake_latents_fid.append(gen_batch_pad(ema_g_params_cpu, z_small, need, step_noise_rng))
                
            fake_latents_fid = np.concatenate(fake_latents_fid, axis=0)
            
            # Демасштабируем обратно
            if config.latent_clip_value is not None:
                fake_latents_fid = np.clip(fake_latents_fid, -config.latent_clip_value, config.latent_clip_value)
            mean_arr = np.array(config.latent_mean, dtype=np.float32).reshape(1, 1, 1, -1)
            std_arr = np.array(config.latent_std, dtype=np.float32).reshape(1, 1, 1, -1)
            fake_latents_fid = fake_latents_fid * std_arr + mean_arr

            os.makedirs("/kaggle/working/fid_samples", exist_ok=True)
            np.save(f"/kaggle/working/fid_samples/fake_latents_epoch_{epoch}.npy", fake_latents_fid)
            print(f"Saved 10k LATENTS to disk. Decode them later on a GPU!")

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

        # Rotate shards across TPU cores over the interconnect so every D sees the whole dataset over 8 epochs
        dataset_sharded = rotate_shards(dataset_sharded)

    wandb.finish()
    print("Training complete.")


if __name__ == "__main__":
    main()
