git_root_path=$(git rev-parse --show-toplevel)


# Spring 1: Sol 116-305
CUDA_VISIBLE_DEVICES=2 python $git_root_path/scripts/train_facvae.py \
    --cuda 1 \
    --h5_filename "pyramid_full-mission_window_size-65536_q-1-1_j-8-8_use_day_data-1_avgpool_base-4_avgpool_exp-5-6-7-8_model_type-scat+cov_filter_key-true.h5" \
    --experiment_name nature_full-mission \
    --batchsize 16384 \
    --hidden_dim 1024 \
    --ncluster 9 \
    --latent_dim 32 \
    --nlayer 4 \
    --filter_key "2019-JUN,2019-JUL,2019-AUG,2019-SEP" \
    --extension full_spring1 \
    --phase test &

# Summer 1: Sol 306-483
CUDA_VISIBLE_DEVICES=2 python $git_root_path/scripts/train_facvae.py \
    --cuda 1 \
    --h5_filename "pyramid_full-mission_window_size-65536_q-1-1_j-8-8_use_day_data-1_avgpool_base-4_avgpool_exp-5-6-7-8_model_type-scat+cov_filter_key-true.h5" \
    --experiment_name nature_full-mission \
    --batchsize 16384 \
    --hidden_dim 1024 \
    --ncluster 9 \
    --latent_dim 32 \
    --nlayer 4 \
    --filter_key "2019-OCT,2019-NOV,2019-DEC,2020-JAN,2020-FEB,2020-MARCH" \
    --extension full_summer1 \
    --phase test &

wait

# Fall 1: Sol 484-624
CUDA_VISIBLE_DEVICES=2 python $git_root_path/scripts/train_facvae.py \
    --cuda 1 \
    --h5_filename "pyramid_full-mission_window_size-65536_q-1-1_j-8-8_use_day_data-1_avgpool_base-4_avgpool_exp-5-6-7-8_model_type-scat+cov_filter_key-true.h5" \
    --experiment_name nature_full-mission \
    --batchsize 16384 \
    --hidden_dim 1024 \
    --ncluster 9 \
    --latent_dim 32 \
    --nlayer 4 \
    --filter_key "2020-APRIL,2020-MAY,2020-JUN,2020-JUL,2020-AUG" \
    --extension full_fall1 \
    --phase test &

# Winter 1: Sol 625-782
CUDA_VISIBLE_DEVICES=2 python $git_root_path/scripts/train_facvae.py \
    --cuda 1 \
    --h5_filename "pyramid_full-mission_window_size-65536_q-1-1_j-8-8_use_day_data-1_avgpool_base-4_avgpool_exp-5-6-7-8_model_type-scat+cov_filter_key-true.h5" \
    --experiment_name nature_full-mission \
    --batchsize 16384 \
    --hidden_dim 1024 \
    --ncluster 9 \
    --latent_dim 32 \
    --nlayer 4 \
    --filter_key "2020-SEP,2020-OCT,2020-NOV,2020-DEC,2021-JAN" \
    --extension full_winter1 \
    --phase test &

wait

# Spring 2: Sol 783-978
CUDA_VISIBLE_DEVICES=2 python $git_root_path/scripts/train_facvae.py \
    --cuda 1 \
    --h5_filename "pyramid_full-mission_window_size-65536_q-1-1_j-8-8_use_day_data-1_avgpool_base-4_avgpool_exp-5-6-7-8_model_type-scat+cov_filter_key-true.h5" \
    --experiment_name nature_full-mission \
    --batchsize 16384 \
    --hidden_dim 1024 \
    --ncluster 9 \
    --latent_dim 32 \
    --nlayer 4 \
    --filter_key "2021-FEB,2021-MARCH,2021-APRIL,2021-MAY,2021-JUN,2021-JUL,2021-AUG" \
    --extension full_spring2 \
    --phase test &

# Summer 2: Sol 979-1155
CUDA_VISIBLE_DEVICES=2 python $git_root_path/scripts/train_facvae.py \
    --cuda 1 \
    --h5_filename "pyramid_full-mission_window_size-65536_q-1-1_j-8-8_use_day_data-1_avgpool_base-4_avgpool_exp-5-6-7-8_model_type-scat+cov_filter_key-true.h5" \
    --experiment_name nature_full-mission \
    --batchsize 16384 \
    --hidden_dim 1024 \
    --ncluster 9 \
    --latent_dim 32 \
    --nlayer 4 \
    --filter_key "2021-SEP,2021-OCT,2021-NOV,2021-DEC,2022-JAN,2022-FEB" \
    --extension full_summer2 \
    --phase test &

wait

# Fall 2: Sol 1156-
CUDA_VISIBLE_DEVICES=2 python $git_root_path/scripts/train_facvae.py \
    --cuda 1 \
    --h5_filename "pyramid_full-mission_window_size-65536_q-1-1_j-8-8_use_day_data-1_avgpool_base-4_avgpool_exp-5-6-7-8_model_type-scat+cov_filter_key-true.h5" \
    --experiment_name nature_full-mission \
    --batchsize 16384 \
    --hidden_dim 1024 \
    --ncluster 9 \
    --latent_dim 32 \
    --nlayer 4 \
    --filter_key "2022-MARCH,2022-APRIL,2022-MAY" \
    --extension full_fall2 \
    --phase test
