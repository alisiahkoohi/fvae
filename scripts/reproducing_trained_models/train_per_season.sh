git_root_path=$(git rev-parse --show-toplevel)

# Spring 1: Sol 116-305
CUDA_VISIBLE_DEVICES=3 python $git_root_path/scripts/train_facvae.py \
    --h5_filename "pyramid_full-mission_window_size-65536_q-1-1_j-8-8_use_day_data-1_avgpool_base-4_avgpool_exp-5-6-7-8_model_type-scat+cov_filter_key-true_detrend-1.h5" \
    --experiment_name pyramid_spring-1-detrend \
    --batchsize 16384 \
    --hidden_dim 1024 \
    --ncluster 9 \
    --latent_dim 32 \
    --filter_key "2019-JUN,2019-JUL,2019-AUG,2019-SEP"

CUDA_VISIBLE_DEVICES=3 python $git_root_path/scripts/train_facvae.py \
    --h5_filename "pyramid_full-mission_window_size-65536_q-1-1_j-8-8_use_day_data-1_avgpool_base-4_avgpool_exp-5-6-7-8_model_type-scat+cov_filter_key-true_detrend-1.h5" \
    --experiment_name pyramid_spring-1-detrend \
    --batchsize 16384 \
    --hidden_dim 1024 \
    --ncluster 9 \
    --latent_dim 32 \
    --phase test

# Summer 1: Sol 306-483
CUDA_VISIBLE_DEVICES=3 python $git_root_path/scripts/train_facvae.py \
    --h5_filename "pyramid_full-mission_window_size-65536_q-1-1_j-8-8_use_day_data-1_avgpool_base-4_avgpool_exp-5-6-7-8_model_type-scat+cov_filter_key-true_detrend-1.h5" \
    --experiment_name pyramid_summer-1-detrend \
    --batchsize 16384 \
    --hidden_dim 1024 \
    --ncluster 9 \
    --latent_dim 32 \
    --filter_key "2019-OCT,2019-NOV,2019-DEC,2020-JAN,2020-FEB,2020-MARCH"

CUDA_VISIBLE_DEVICES=3 python $git_root_path/scripts/train_facvae.py \
    --h5_filename "pyramid_full-mission_window_size-65536_q-1-1_j-8-8_use_day_data-1_avgpool_base-4_avgpool_exp-5-6-7-8_model_type-scat+cov_filter_key-true_detrend-1.h5" \
    --experiment_name pyramid_summer-1-detrend \
    --batchsize 16384 \
    --hidden_dim 1024 \
    --ncluster 9 \
    --latent_dim 32 \
    --phase test

# Fall 1: Sol 484-624
CUDA_VISIBLE_DEVICES=3 python $git_root_path/scripts/train_facvae.py \
    --h5_filename "pyramid_full-mission_window_size-65536_q-1-1_j-8-8_use_day_data-1_avgpool_base-4_avgpool_exp-5-6-7-8_model_type-scat+cov_filter_key-true_detrend-1.h5" \
    --experiment_name pyramid_fall-1-detrend \
    --batchsize 16384 \
    --hidden_dim 1024 \
    --ncluster 9 \
    --latent_dim 32 \
    --filter_key "2020-APRIL,2020-MAY,2020-JUN,2020-JUL,2020-AUG"

CUDA_VISIBLE_DEVICES=3 python $git_root_path/scripts/train_facvae.py \
    --h5_filename "pyramid_full-mission_window_size-65536_q-1-1_j-8-8_use_day_data-1_avgpool_base-4_avgpool_exp-5-6-7-8_model_type-scat+cov_filter_key-true_detrend-1.h5" \
    --experiment_name pyramid_fall-1-detrend \
    --batchsize 16384 \
    --hidden_dim 1024 \
    --ncluster 9 \
    --latent_dim 32 \
    --phase test

# Winter 1: Sol 625-782
CUDA_VISIBLE_DEVICES=3 python $git_root_path/scripts/train_facvae.py \
    --h5_filename "pyramid_full-mission_window_size-65536_q-1-1_j-8-8_use_day_data-1_avgpool_base-4_avgpool_exp-5-6-7-8_model_type-scat+cov_filter_key-true_detrend-1.h5" \
    --experiment_name pyramid_winter-1-detrend \
    --batchsize 16384 \
    --hidden_dim 1024 \
    --ncluster 9 \
    --latent_dim 32 \
    --filter_key "2020-SEP,2020-OCT,2020-NOV,2020-DEC,2021-JAN"

CUDA_VISIBLE_DEVICES=3 python $git_root_path/scripts/train_facvae.py \
    --h5_filename "pyramid_full-mission_window_size-65536_q-1-1_j-8-8_use_day_data-1_avgpool_base-4_avgpool_exp-5-6-7-8_model_type-scat+cov_filter_key-true_detrend-1.h5" \
    --experiment_name pyramid_winter-1-detrend \
    --batchsize 16384 \
    --hidden_dim 1024 \
    --ncluster 9 \
    --latent_dim 32 \
    --phase test

# Spring 2: Sol 783-978
CUDA_VISIBLE_DEVICES=3 python $git_root_path/scripts/train_facvae.py \
    --h5_filename "pyramid_full-mission_window_size-65536_q-1-1_j-8-8_use_day_data-1_avgpool_base-4_avgpool_exp-5-6-7-8_model_type-scat+cov_filter_key-true_detrend-1.h5" \
    --experiment_name pyramid_spring-2-detrend \
    --batchsize 16384 \
    --hidden_dim 1024 \
    --ncluster 9 \
    --latent_dim 32 \
    --filter_key "2021-FEB,2021-MARCH,2021-APRIL,2021-MAY,2021-JUN,2021-JUL,2021-AUG"

CUDA_VISIBLE_DEVICES=3 python $git_root_path/scripts/train_facvae.py \
    --h5_filename "pyramid_full-mission_window_size-65536_q-1-1_j-8-8_use_day_data-1_avgpool_base-4_avgpool_exp-5-6-7-8_model_type-scat+cov_filter_key-true_detrend-1.h5" \
    --experiment_name pyramid_spring-2-detrend \
    --batchsize 16384 \
    --hidden_dim 1024 \
    --ncluster 9 \
    --latent_dim 32 \
    --phase test

# Summer 2: Sol 979-1155
CUDA_VISIBLE_DEVICES=3 python $git_root_path/scripts/train_facvae.py \
    --h5_filename "pyramid_full-mission_window_size-65536_q-1-1_j-8-8_use_day_data-1_avgpool_base-4_avgpool_exp-5-6-7-8_model_type-scat+cov_filter_key-true_detrend-1.h5" \
    --experiment_name pyramid_summer-2-detrend \
    --batchsize 16384 \
    --hidden_dim 1024 \
    --ncluster 9 \
    --latent_dim 32 \
    --filter_key "2021-SEP,2021-OCT,2021-NOV,2021-DEC,2022-JAN,2022-FEB"

CUDA_VISIBLE_DEVICES=3 python $git_root_path/scripts/train_facvae.py \
    --h5_filename "pyramid_full-mission_window_size-65536_q-1-1_j-8-8_use_day_data-1_avgpool_base-4_avgpool_exp-5-6-7-8_model_type-scat+cov_filter_key-true_detrend-1.h5" \
    --experiment_name pyramid_summer-2-detrend \
    --batchsize 16384 \
    --hidden_dim 1024 \
    --ncluster 9 \
    --latent_dim 32 \
    --phase test

# Fall 2: Sol 1156-
CUDA_VISIBLE_DEVICES=3 python $git_root_path/scripts/train_facvae.py \
    --h5_filename "pyramid_full-mission_window_size-65536_q-1-1_j-8-8_use_day_data-1_avgpool_base-4_avgpool_exp-5-6-7-8_model_type-scat+cov_filter_key-true_detrend-1.h5" \
    --experiment_name pyramid_fall-2-detrend \
    --batchsize 16384 \
    --hidden_dim 1024 \
    --ncluster 9 \
    --latent_dim 32 \
    --filter_key "2022-MARCH,2022-APRIL,2022-MAY"

CUDA_VISIBLE_DEVICES=3 python $git_root_path/scripts/train_facvae.py \
    --h5_filename "pyramid_full-mission_window_size-65536_q-1-1_j-8-8_use_day_data-1_avgpool_base-4_avgpool_exp-5-6-7-8_model_type-scat+cov_filter_key-true_detrend-1.h5" \
    --experiment_name pyramid_fall-2-detrend \
    --batchsize 16384 \
    --hidden_dim 1024 \
    --ncluster 9 \
    --latent_dim 32 \
    --phase test