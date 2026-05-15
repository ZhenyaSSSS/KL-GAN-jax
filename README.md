# KL-GAN TPU (JAX / Flax)

Distributed ensemble KL-GAN for CelebA 32×32 on multi-host TPU (e.g. Kaggle TPU v3-8): one generator, eight discriminators, KL + Pearson diversity, W&B logging, optional FID.

## Setup

```bash
cd kl_gan_tpu
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate   # Linux / macOS
pip install -r requirements.txt
```

Edit `configs/default.yaml` as needed. On Kaggle, add CelebA input and set `WANDB_API_KEY` in Secrets.

## Run

From this directory (so imports resolve):

```bash
python train.py
```

## Layout

- `configs/default.yaml` — hyperparameters  
- `data/loader.py` — CelebA load + replicate dataset to devices  
- `models/` — generator, discriminator, MBD  
- `training/` — losses, `pmap` train step  
- `metrics/` — ResNet feature KL (training), FID helper for offline use  
- `train.py` — entrypoint  

## License

Use and modify as you wish; add your own `LICENSE` if you need a formal one.
