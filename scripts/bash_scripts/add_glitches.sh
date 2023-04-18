filename=3c_window_size-4096_q-1-1_j-8-8_use_day_data-1_avg_pool-0_model_type-scat+cov_filter_key-true.h5
python scripts/insert_catalog_to_h5.py --h5_filename $filename --h5_dataset_name glitches  --catalog_filename Salma_glitches_InSIght.pkl --window_size 4096 --n_workers 30 --target_column_name glitch
