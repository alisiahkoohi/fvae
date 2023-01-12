python scripts/train_gmvae.py \
    --h5_filename 3c_raw_window_size-2048_q-6-2-2_j-7-7-7_use_day_data-1_avg_pool-0_model_type-scat_filter_key-true.h5 \
    --type scat_cov_pca_25,cov_pca_25 \
    --experiment_name 622-62_raw \
    --normalize 0 \
    --latent_dim 32 \
    --cuda 0 \
    --ncluster 9 \
    --nlayer 8 \
    --phase test &

python scripts/train_gmvae.py \
    --h5_filename 3c_raw_window_size-2048_q-6-2-2_j-7-7-7_use_day_data-1_avg_pool-0_model_type-scat_filter_key-true.h5 \
    --type scat_cov_pca_25 \
    --experiment_name 622-62_raw \
    --normalize 0 \
    --latent_dim 32 \
    --cuda 0 \
    --ncluster 9 \
    --nlayer 8 \
    --phase test &

python scripts/train_gmvae.py \
    --h5_filename 3c_raw_window_size-2048_q-6-2-2_j-7-7-7_use_day_data-1_avg_pool-0_model_type-scat_filter_key-true.h5 \
    --type cov_pca_25 \
    --experiment_name 622-62_raw \
    --normalize 0 \
    --latent_dim 32 \
    --cuda 0 \
    --ncluster 9 \
    --nlayer 8 \
    --phase test &

wait

python scripts/train_gmvae.py \
    --h5_filename 3c_raw_window_size-2048_q-6-2-2_j-7-7-7_use_day_data-1_avg_pool-0_model_type-scat_filter_key-true.h5 \
    --type scat_cov_pca_25,cov_pca_25 \
    --experiment_name 622-62_raw \
    --normalize 0 \
    --latent_dim 32 \
    --cuda 0 \
    --ncluster 9 \
    --nlayer 8 \
    --filter_key "2020-JUN-03,2020-JUN-04,2020-JUN-05,2020-JUN-06,2020-JUN-07,2020-JUN-08,2020-JUN-09,2020-JUN-10,2020-JUN-11" \
    --phase test &

python scripts/train_gmvae.py \
    --h5_filename 3c_raw_window_size-2048_q-6-2-2_j-7-7-7_use_day_data-1_avg_pool-0_model_type-scat_filter_key-true.h5 \
    --type scat_cov_pca_25 \
    --experiment_name 622-62_raw \
    --normalize 0 \
    --latent_dim 32 \
    --cuda 0 \
    --ncluster 9 \
    --nlayer 8 \
    --filter_key "2020-JUN-03,2020-JUN-04,2020-JUN-05,2020-JUN-06,2020-JUN-07,2020-JUN-08,2020-JUN-09,2020-JUN-10,2020-JUN-11" \
    --phase test &

python scripts/train_gmvae.py \
    --h5_filename 3c_raw_window_size-2048_q-6-2-2_j-7-7-7_use_day_data-1_avg_pool-0_model_type-scat_filter_key-true.h5 \
    --type cov_pca_25 \
    --experiment_name 622-62_raw \
    --normalize 0 \
    --latent_dim 32 \
    --cuda 0 \
    --ncluster 9 \
    --nlayer 8 \
    --filter_key "2020-JUN-03,2020-JUN-04,2020-JUN-05,2020-JUN-06,2020-JUN-07,2020-JUN-08,2020-JUN-09,2020-JUN-10,2020-JUN-11" \
    --phase test &

wait

python scripts/train_gmvae.py \
    --h5_filename 3c_raw_window_size-2048_q-6-2-2_j-7-7-7_use_day_data-1_avg_pool-0_model_type-scat_filter_key-true.h5 \
    --type scat_cov_pca_25,cov_pca_25 \
    --experiment_name 622-62_raw \
    --normalize 0 \
    --latent_dim 32 \
    --cuda 0 \
    --ncluster 9 \
    --nlayer 8 \
    --filter_key 2020 \
    --phase test &

python scripts/train_gmvae.py \
    --h5_filename 3c_raw_window_size-2048_q-6-2-2_j-7-7-7_use_day_data-1_avg_pool-0_model_type-scat_filter_key-true.h5 \
    --type scat_cov_pca_25 \
    --experiment_name 622-62_raw \
    --normalize 0 \
    --latent_dim 32 \
    --cuda 0 \
    --ncluster 9 \
    --nlayer 8 \
    --filter_key 2020 \
    --phase test &

python scripts/train_gmvae.py \
    --h5_filename 3c_raw_window_size-2048_q-6-2-2_j-7-7-7_use_day_data-1_avg_pool-0_model_type-scat_filter_key-true.h5 \
    --type cov_pca_25 \
    --experiment_name 622-62_raw \
    --normalize 0 \
    --latent_dim 32 \
    --cuda 0 \
    --ncluster 9 \
    --nlayer 8 \
    --filter_key 2020 \
    --phase test &