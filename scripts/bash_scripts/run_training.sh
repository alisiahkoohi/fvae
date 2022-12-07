
for input_args in $*;
do
    IFS=":"
    set -- $input_args
    CUDA_VISIBLE_DEVICES=0 python scripts/train_gmvae.py \
        --h5_filename "$1" \
        --filter_key "2019-JUN-03,2019-JUN-04,2019-JUN-05,2019-JUN-06,2019-JUN-07,2019-JUN-08,2019-JUN-09,2019-JUN-10,2019-JUN-11" \
        --w_rec 0.05 \
        --max_epoch 1000  \
        --load_to_memory 1 \
        --window_size 2048 \
        --hidden_dim 1024 \
        --nlayer 4 \
        --experiment_name glitch_q-"$2" \
        --ncluster 9 \
        --latent_dim 128 \
        --batchsize 16384 \
        --lr_final 0.001

    CUDA_VISIBLE_DEVICES=0 python scripts/train_gmvae.py \
        --h5_filename "$1" \
        --filter_key "2019-JUN-03,2019-JUN-04,2019-JUN-05,2019-JUN-06,2019-JUN-07,2019-JUN-08,2019-JUN-09,2019-JUN-10,2019-JUN-11" \
        --w_rec 0.05 \
        --max_epoch 1000  \
        --load_to_memory 1 \
        --window_size 2048 \
        --hidden_dim 1024 \
        --nlayer 4 \
        --experiment_name glitch_q-"$2" \
        --ncluster 9 \
        --latent_dim 128 \
        --batchsize 16384 \
        --lr_final 0.001 \
        --phase test
done