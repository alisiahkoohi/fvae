
for filename in $*;
do
    python scripts/insert_catalog_to_h5.py \
        --h5_filename "$filename" \
        --h5_dataset_name labels  \
        --catalog_filename events_InSIght.pkl \
        --window_size 4096 \
        --n_workers 4 \
        --target_column_name type

    python scripts/insert_catalog_to_h5.py \
        --h5_filename "$filename" \
        --h5_dataset_name pressure  \
        --catalog_filename pressure_drops_InSIght.pkl \
        --window_size 4096 \
        --n_workers 4 \
        --target_column_name drop
    
    python scripts/insert_catalog_to_h5.py \
        --h5_filename "$filename" \
        --h5_dataset_name glitches  \
        --catalog_filename Salma_glitches_InSIght.pkl \
        --window_size 4096 \
        --n_workers 4 \
        --target_column_name glitch 
done


