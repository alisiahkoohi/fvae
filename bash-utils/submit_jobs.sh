#!/bin/bash -l

# set -e

python scripts/train_gmvae.py --latent_dim 2 --ncluster 5 &
python scripts/train_gmvae.py --latent_dim 4 --ncluster 5 &
python scripts/train_gmvae.py --latent_dim 8 --ncluster 5 &

wait

nohup python scripts/train_gmvae.py --latent_dim 2 --ncluster 5 --phase test --cuda 0 &
nohup python scripts/train_gmvae.py --latent_dim 4 --ncluster 5 --phase test --cuda 0 &
nohup python scripts/train_gmvae.py --latent_dim 8 --ncluster 5 --phase test --cuda 0 &

python scripts/train_gmvae.py --latent_dim 4 --ncluster 5 &
python scripts/train_gmvae.py --latent_dim 4 --ncluster 10 &
python scripts/train_gmvae.py --latent_dim 4 --ncluster 15 &

wait

python scripts/train_gmvae.py --latent_dim 4 --ncluster 5 --phase test --cuda 0 &
python scripts/train_gmvae.py --latent_dim 4 --ncluster 10 --phase test --cuda 0 &
python scripts/train_gmvae.py --latent_dim 4 --ncluster 15 --phase test --cuda 0 &

python scripts/train_gmvae.py --latent_dim 16 --ncluster 5 &
python scripts/train_gmvae.py --latent_dim 32 --ncluster 5 &
python scripts/train_gmvae.py --latent_dim 64 --ncluster 5 &

wait

python scripts/train_gmvae.py --latent_dim 16 --ncluster 5 --phase test --cuda 0 &
python scripts/train_gmvae.py --latent_dim 32 --ncluster 5 --phase test --cuda 0 &
python scripts/train_gmvae.py --latent_dim 64 --ncluster 5 --phase test --cuda 0 &

python scripts/train_gmvae.py --latent_dim 16 --ncluster 10 &
python scripts/train_gmvae.py --latent_dim 32 --ncluster 10 &
python scripts/train_gmvae.py --latent_dim 64 --ncluster 10 &

wait

python scripts/train_gmvae.py --latent_dim 16 --ncluster 10 --phase test --cuda 0 &
python scripts/train_gmvae.py --latent_dim 32 --ncluster 10 --phase test --cuda 0 &
python scripts/train_gmvae.py --latent_dim 64 --ncluster 10 --phase test --cuda 0 &
