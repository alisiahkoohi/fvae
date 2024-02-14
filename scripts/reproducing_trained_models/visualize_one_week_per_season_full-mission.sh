git_root_path=$(git rev-parse --show-toplevel)

# Spring 1: Sol 116-305
CUDA_VISIBLE_DEVICES=3 python $git_root_path/scripts/train_facvae.py \
    --cuda 1 \
    --h5_filename "pyramid_full-mission_window_size-65536_q-1-1_j-8-8_use_day_data-1_avgpool_base-4_avgpool_exp-5-6-7-8_model_type-scat+cov_filter_key-true.h5" \
    --experiment_name nature_full-mission \
    --batchsize 16384 \
    --hidden_dim 1024 \
    --ncluster 9 \
    --latent_dim 32 \
    --nlayer 4 \
    --seed 29 \
    --filter_key "2019-JUN-03,2019-JUN-04,2019-JUN-05,2019-JUN-06,2019-JUN-07,2019-JUN-08,2019-JUN-09,2019-JUN-10,2019-JUN-11" \
    --extension spring1_2019-JUN-03 \
    --phase test &

# Spring 2: Sol 783-978
CUDA_VISIBLE_DEVICES=3 python $git_root_path/scripts/train_facvae.py \
    --cuda 1 \
    --h5_filename "pyramid_full-mission_window_size-65536_q-1-1_j-8-8_use_day_data-1_avgpool_base-4_avgpool_exp-5-6-7-8_model_type-scat+cov_filter_key-true.h5" \
    --experiment_name nature_full-mission \
    --batchsize 16384 \
    --hidden_dim 1024 \
    --ncluster 9 \
    --latent_dim 32 \
    --nlayer 4 \
    --seed 29 \
    --filter_key "2021-MAY-20,2021-MAY-21,2021-MAY-22,2021-MAY-23,2021-MAY-24,2021-MAY-25,2021-MAY-26,2021-MAY-27,2021-MAY-28" \
    --extension spring2_2021-MAY-20 \
    --phase test &

wait

# Summer 1: Sol 306-483
CUDA_VISIBLE_DEVICES=3 python $git_root_path/scripts/train_facvae.py \
    --cuda 1 \
    --h5_filename "pyramid_full-mission_window_size-65536_q-1-1_j-8-8_use_day_data-1_avgpool_base-4_avgpool_exp-5-6-7-8_model_type-scat+cov_filter_key-true.h5" \
    --experiment_name nature_full-mission \
    --batchsize 16384 \
    --hidden_dim 1024 \
    --ncluster 9 \
    --latent_dim 32 \
    --nlayer 4 \
    --seed 29 \
    --filter_key "2019-DEC-03,2019-DEC-04,2019-DEC-05,2019-DEC-06,2019-DEC-07,2019-DEC-08,2019-DEC-09,2019-DEC-10,2019-DEC-11" \
    --extension summer1_2020-JAN-03 \
    --phase test &

# Summer 2: Sol 979-1155
CUDA_VISIBLE_DEVICES=3 python $git_root_path/scripts/train_facvae.py \
    --cuda 1 \
    --h5_filename "pyramid_full-mission_window_size-65536_q-1-1_j-8-8_use_day_data-1_avgpool_base-4_avgpool_exp-5-6-7-8_model_type-scat+cov_filter_key-true.h5" \
    --experiment_name nature_full-mission \
    --batchsize 16384 \
    --hidden_dim 1024 \
    --ncluster 9 \
    --latent_dim 32 \
    --nlayer 4 \
    --seed 29 \
    --filter_key "2021-NOV-20,2021-NOV-21,2021-NOV-22,2021-NOV-23,2021-NOV-24,2021-NOV-25,2021-NOV-26,2021-NOV-27,2021-NOV-28" \
    --extension summer2_2021-NOV-20 \
    --phase test &

wait

# Fall 1: Sol 484-624
CUDA_VISIBLE_DEVICES=3 python $git_root_path/scripts/train_facvae.py \
    --cuda 1 \
    --h5_filename "pyramid_full-mission_window_size-65536_q-1-1_j-8-8_use_day_data-1_avgpool_base-4_avgpool_exp-5-6-7-8_model_type-scat+cov_filter_key-true.h5" \
    --experiment_name nature_full-mission \
    --batchsize 16384 \
    --hidden_dim 1024 \
    --ncluster 9 \
    --latent_dim 32 \
    --nlayer 4 \
    --seed 29 \
    --filter_key "2020-MAY-03,2020-MAY-04,2020-MAY-05,2020-MAY-06,2020-MAY-07,2020-MAY-08,2020-MAY-09,2020-MAY-10,2020-MAY-11" \
    --extension fall1_2020-JUN-03 \
    --phase test &

# Fall 2: Sol 1156-
CUDA_VISIBLE_DEVICES=3 python $git_root_path/scripts/train_facvae.py \
    --cuda 1 \
    --h5_filename "pyramid_full-mission_window_size-65536_q-1-1_j-8-8_use_day_data-1_avgpool_base-4_avgpool_exp-5-6-7-8_model_type-scat+cov_filter_key-true.h5" \
    --experiment_name nature_full-mission \
    --batchsize 16384 \
    --hidden_dim 1024 \
    --ncluster 9 \
    --latent_dim 32 \
    --nlayer 4 \
    --seed 29 \
    --filter_key "2021-APRIL-20,2021-APRIL-21,2021-APRIL-22,2021-APRIL-23,2021-APRIL-24,2021-APRIL-25,2021-APRIL-26,2021-APRIL-27,2021-APRIL-28" \
    --extension fall2_2021-APRIL-20 \
    --phase test &

wait

# Winter 1: Sol 625-782
CUDA_VISIBLE_DEVICES=3 python $git_root_path/scripts/train_facvae.py \
    --cuda 1 \
    --h5_filename "pyramid_full-mission_window_size-65536_q-1-1_j-8-8_use_day_data-1_avgpool_base-4_avgpool_exp-5-6-7-8_model_type-scat+cov_filter_key-true.h5" \
    --experiment_name nature_full-mission \
    --batchsize 16384 \
    --hidden_dim 1024 \
    --ncluster 9 \
    --latent_dim 32 \
    --nlayer 4 \
    --seed 29 \
    --filter_key "2020-NOV-20,2020-NOV-21,2020-NOV-22,2020-NOV-23,2020-NOV-24,2020-NOV-25,2020-NOV-26,2020-NOV-27,2020-NOV-28" \
    --extension winter1_2020-NOV-20 \
    --phase test