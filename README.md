# KL-GAN TPU (Latent GAN via JAX)

A highly optimized Latent Generative Adversarial Network implemented in JAX/Flax, designed to train natively on Google TPU architectures (e.g., Kaggle TPU v3-8). It learns the latent space of a pre-trained VAE (REPA-E / SD 3.5 VAE) to synthesize high-resolution images (512x512) extremely fast.

## Features

- **Latent GAN Architecture:** Operates directly on `64x64x16` VAE latents instead of raw pixels, bypassing spatial and memory bottlenecks.
- **TPU Acceleration:** End-to-end JAX compilation, multi-device sharding (`jax.device_put_sharded`), and high-speed ICI interconnect communication via `jax.lax.ppermute`.
- **Hybrid DiT:** Employs a customized Hybrid Diffusion Transformer architecture for the Generator and Discriminator.
- **Asynchronous W&B Logging:** Decodes generated latents back to RGB via PyTorch on CPU in a background thread. This allows the TPU to instantly proceed to the next epoch without idling.
- **Latent KL Evaluation:** Blazing fast Fréchet-like divergence calculation directly in the latent space on TPU without the need to decode images or run a heavy ResNet evaluator.

## Installation

```bash
pip install -r requirements.txt
```

*Note: JAX on TPU environments (like Kaggle) is typically pre-installed. The script leverages `colab_tpu` to automatically initialize the TPU cluster.*

## Dataset

The model expects a pre-computed `.npy` file containing VAE latents. For example, FFHQ dataset processed into `64x64x16` latents. 
If running on Kaggle, simply attach the dataset containing your `.npy` file and update the `latent_npy_path` in the configuration.

## Configuration

Edit `configs/default.yaml` and `config.py` to tune hyperparameters. Key parameters include:
- `latent_scaling_factor`: Empirical standard deviation of your latent dataset to normalize the variance to ~1.0.
- `batch_size_per_device`: Adjusted for TPU memory (e.g., `64` per core gives a total batch size of `512` on 8 cores).
- `latent_dim`: Size of the input noise vector $z$.
- `image_size` & `channels`: Spatial dimensions of the latents (e.g., `64` and `16`).

## Training

To launch the training pipeline on a Kaggle TPU:

```bash
python train.py
```

The script will automatically shard the dataset across all available TPU cores, compile the computational graphs with XLA, and log metrics/images to Weights & Biases.

## Evaluation (FID)

To maximize TPU utilization, FID score computation is decoupled from the main training loop:
1. During training, the model periodically generates and saves 10,000 raw latents to disk (`fake_latents_epoch_X.npy`).
2. After training, you can download these latents to a GPU-enabled environment.
3. Decode the latents to RGB via the VAE and compute the standard `clean-fid` against your real dataset.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
