#!/bin/bash -l

set -e


nohup python scripts/train_gmvae.py --latent_dim 8 &
nohup python scripts/train_gmvae.py --latent_dim 16 &
nohup python scripts/train_gmvae.py --latent_dim 32 &
nohup python scripts/train_gmvae.py --latent_dim 64 &
nohup python scripts/train_gmvae.py --latent_dim 128 &

wait

nohup python scripts/train_gmvae.py --latent_dim 256 &
nohup python scripts/train_gmvae.py --latent_dim 512 &

wait

nohup python scripts/train_gmvae.py --latent_dim 8 --phase test --cuda 0 &
nohup python scripts/train_gmvae.py --latent_dim 16 --phase test --cuda 0 &
nohup python scripts/train_gmvae.py --latent_dim 32 --phase test --cuda 0 &
nohup python scripts/train_gmvae.py --latent_dim 64 --phase test --cuda 0 &
nohup python scripts/train_gmvae.py --latent_dim 128 --phase test --cuda 0 &
nohup python scripts/train_gmvae.py --latent_dim 256 --phase test --cuda 0 &
nohup python scripts/train_gmvae.py --latent_dim 512 --phase test --cuda 0 &
