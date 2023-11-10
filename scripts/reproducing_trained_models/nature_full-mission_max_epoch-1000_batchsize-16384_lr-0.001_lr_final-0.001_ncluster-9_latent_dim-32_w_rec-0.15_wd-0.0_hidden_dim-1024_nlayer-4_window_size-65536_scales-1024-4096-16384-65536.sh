git_root_path=$(git rev-parse --show-toplevel)

CUDA_VISIBLE_DEVICES=0 python $git_root_path/scripts/train_facvae.py \
    --cuda 1 \
    --h5_filename "pyramid_full-mission_window_size-65536_q-1-1_j-8-8_use_day_data-1_avgpool_base-4_avgpool_exp-5-6-7-8_model_type-scat+cov_filter_key-true.h5" \
    --experiment_name nature_full-mission \
    --batchsize 16384 \
    --hidden_dim 1024 \
    --ncluster 9 \
    --latent_dim 32 \
    --nlayer 4 \
    --filter_key "" &

