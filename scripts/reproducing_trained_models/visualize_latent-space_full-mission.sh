git_root_path=$(git rev-parse --show-toplevel)


for umap_n_neighbors in 300
    do
    for umap_min_dist in 1.0
        do
        CUDA_VISIBLE_DEVICES=0,1,2,3 python $git_root_path/scripts/train_facvae.py \
            --cuda 1 \
            --h5_filename "pyramid_full-mission_window_size-65536_q-1-1_j-8-8_use_day_data-1_avgpool_base-4_avgpool_exp-5-6-7-8_model_type-scat+cov_filter_key-true.h5" \
            --experiment_name nature_full-mission \
            --batchsize 16384 \
            --hidden_dim 1024 \
            --ncluster 9 \
            --latent_dim 32 \
            --nlayer 4 \
            --filter_key "" \
            --extension full-mission_ep-20000_dist-${umap_min_dist}_neigh-${umap_n_neighbors} \
            --event_quality "A,B" \
            --seed 29 \
            --umap_n_neighbors $umap_n_neighbors \
            --umap_min_dist $umap_min_dist \
            --umap_n_epochs 20000 \
            --phase test

        killall python
    done
done
