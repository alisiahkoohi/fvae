filename1="3c_2020_window_size-2048_q-6-2-2_j-6-7-7_use_day_data-1_avg_pool-0_model_type-scat_filter_key-true.h5"
filename2="3c_2020_window_size-2048_q-6-2_j-6-7_use_day_data-1_avg_pool-0_model_type-scat+cov_filter_key-true.h5"
filename3="3c_2020_window_size-2048_q-6-2_j-6-7_use_day_data-1_avg_pool-0_model_type-scat_filter_key-true.h5"


CUDA_VISIBLE_DEVICES=3 python scripts/train_gmvae.py --h5_filename $filename1 --filter_key '2020' --w_rec 0.05 --max_epoch 1000  --load_to_memory 1 --window_size 2048 --hidden_dim 1024 --nlayer 4 --experiment_name bssa_3c_full-2020-scat --ncluster 9 --latent_dim 128 --batchsize 16384 --lr_final 0.001

python scripts/train_gmvae.py --h5_filename $filename1 --filter_key '2020' --w_rec 0.05 --max_epoch 1000  --load_to_memory 1 --window_size 2048 --hidden_dim 1024 --nlayer 4 --experiment_name bssa_3c_full-2020-scat --ncluster 9 --latent_dim 128 --batchsize 16384 --lr_final 0.001 --phase test --cuda 0


CUDA_VISIBLE_DEVICES=3 python scripts/train_gmvae.py --h5_filename $filename2 --filter_key '2020' --w_rec 0.01 --max_epoch 1000  --load_to_memory 1 --window_size 2048 --hidden_dim 1024 --nlayer 4 --experiment_name bssa_3c_full-2020-62-scatcov --ncluster 9 --latent_dim 128 --batchsize 16384 --lr_final 0.001

python scripts/train_gmvae.py --h5_filename $filename2 --filter_key '2020' --w_rec 0.01 --max_epoch 1000  --load_to_memory 1 --window_size 2048 --hidden_dim 1024 --nlayer 4 --experiment_name bssa_3c_full-2020-62-scatcov --ncluster 9 --latent_dim 128 --batchsize 16384 --lr_final 0.001 --phase test


CUDA_VISIBLE_DEVICES=3 python scripts/train_gmvae.py --h5_filename $filename3 --filter_key '2020' --w_rec 0.05 --max_epoch 1000  --load_to_memory 1 --window_size 2048 --hidden_dim 1024 --nlayer 4 --experiment_name bssa_3c_full-2020-62-scat --ncluster 9 --latent_dim 128 --batchsize 16384 --lr_final 0.001

python scripts/train_gmvae.py --h5_filename $filename3 --filter_key '2020' --w_rec 0.05 --max_epoch 1000  --load_to_memory 1 --window_size 2048 --hidden_dim 1024 --nlayer 4 --experiment_name bssa_3c_full-2020-62-scat --ncluster 9 --latent_dim 128 --batchsize 16384 --lr_final 0.001 --phase test
