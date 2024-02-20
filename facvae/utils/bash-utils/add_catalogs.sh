
git_root_path=$(git rev-parse --show-toplevel)

for filename in $*;
do
    python $git_root_path/facvae/utils/catalog-utils/insert_catalog_to_h5.py \
        --h5_filename "$filename" \
        --h5_dataset_name labels  \
        --catalog_filename events_InSIght_v14.pkl \
        --n_workers 40 \
        --target_column_name type

    python $git_root_path/facvae/utils/catalog-utils/insert_catalog_to_h5.py \
        --h5_filename "$filename" \
        --h5_dataset_name pressure  \
        --catalog_filename pressure_drops_InSIght.pkl \
        --n_workers 40 \
        --target_column_name drop

done


