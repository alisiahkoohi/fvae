git_root_path=$(git rev-parse --show-toplevel)



for i in {20..40..2}
do
    j=$((i + 1))

    CUDA_VISIBLE_DEVICES=0 python $git_root_path/scripts/train_facvae.py \
        --cuda 1 \
        --h5_filename "pyramid_full-mission_window_size-65536_q-1-1_j-8-8_use_day_data-1_avgpool_base-4_avgpool_exp-5-6-7-8_model_type-scat+cov_filter_key-true.h5" \
        --experiment_name nature_full-mission \
        --batchsize 16384 \
        --hidden_dim 1024 \
        --ncluster 9 \
        --latent_dim 32 \
        --nlayer 4 \
        --seed $i \
        --filter_key "" &

    CUDA_VISIBLE_DEVICES=2 python $git_root_path/scripts/train_facvae.py \
        --cuda 1 \
        --h5_filename "pyramid_full-mission_window_size-65536_q-1-1_j-8-8_use_day_data-1_avgpool_base-4_avgpool_exp-5-6-7-8_model_type-scat+cov_filter_key-true.h5" \
        --experiment_name nature_full-mission \
        --batchsize 16384 \
        --hidden_dim 1024 \
        --ncluster 9 \
        --latent_dim 32 \
        --nlayer 4 \
        -- seed $j \
        --filter_key "" &

    wait

done