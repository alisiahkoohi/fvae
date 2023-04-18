CUDA_VISIBLE_DEVICES=3 python scripts/train_gmvae.py \
    --h5_filename 3c_raw_window_size-2048_q-6-2-2_j-7-7-7_use_day_data-1_avg_pool-0_model_type-scat_filter_key-true.h5 \
    --type scat_cov_pca,cov_pca \
    --experiment_name 622-62_raw \
    --normalize 0 \
    --latent_dim 32 \
    --cuda 1 \
    --ncluster 9 \
    --filter_key '' \
    --nlayer 8
