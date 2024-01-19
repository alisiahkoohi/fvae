git_root_path=$(git rev-parse --show-toplevel)

CUDA_VISIBLE_DEVICES=$1 python $git_root_path/facvae/utils/calculate_umap.py $2 $3 $4 $5
