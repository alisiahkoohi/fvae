#!/bin/bash -l

set -e

nohup python scripts/train_gmvae.py --latent_dim 2 > logs/train_gmvae_2.log &

wait

nohup python scripts/train_gmvae.py --latent_dim 2 --phase test --cuda 0 > logs/train_gmvae_2_test.log &
nohup python scripts/train_gmvae.py --latent_dim 4 > logs/train_gmvae_4.log &

wait

nohup python scripts/train_gmvae.py --latent_dim 4 --phase test --cuda 0 > logs/train_gmvae_4_test.log &
nohup python scripts/train_gmvae.py --latent_dim 64 > logs/train_gmvae_64.log &

wait

nohup python scripts/train_gmvae.py --latent_dim 64 --phase test --cuda 0 > logs/train_gmvae_64_test.log &
nohup python scripts/train_gmvae.py --latent_dim 4 --ncluster 5 > logs/train_gmvae_4_5.log &

wait

nohup python scripts/train_gmvae.py --latent_dim 4 --ncluster 5 --phase test --cuda 0 > logs/train_gmvae_4_5_test.log &
nohup python scripts/train_gmvae.py --latent_dim 4 --ncluster 20 > logs/train_gmvae_4_20.log &

wait

nohup python scripts/train_gmvae.py --latent_dim 4 --ncluster 20 --phase test --cuda 0 > logs/train_gmvae_4_20_test.log &
nohup python scripts/train_gmvae.py --latent_dim 4 --hidden_dim 1024 > logs/train_gmvae_4_1024.log &

wait

nohup python scripts/train_gmvae.py --latent_dim 4 --hidden_dim 1024 --phase test --cuda 0 > logs/train_gmvae_4_1024_test.log &
nohup python scripts/train_gmvae.py --latent_dim 4 --hidden_dim 64 > logs/train_gmvae_4_64.log &
