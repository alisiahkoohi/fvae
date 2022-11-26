#filename=3c_2019_window_size-2048_q-6-2_j-6-7_use_day_data-1_avg_pool-0_model_type-scat+cov_filter_key-true.h5
#python scripts/insert_catalog_to_h5.py --h5_filename $filename --h5_dataset_name glitches  --catalog_filename Salma_glitches_InSIght.pkl --window_size 2048 --n_workers 30 --target_column_name glitch

#filename=3c_window_size-2048_q-6-2-2_j-6-7-7_use_day_data-1_avg_pool-0_model_type-scat_filter_key-true.h5
#python scripts/insert_catalog_to_h5.py --h5_filename $filename --h5_dataset_name glitches  --catalog_filename Salma_glitches_InSIght.pkl --window_size 2048 --n_workers 30 --target_column_name glitch

filename=3c_window_size-2048_q-6-2-2_r-2_J-7_use_day_data-1_model_type-scat.h5
python scripts/insert_catalog_to_h5.py --h5_filename $filename --h5_dataset_name glitches  --catalog_filename Salma_glitches_InSIght.pkl --window_size 2048 --n_workers 30 --target_column_name glitch

filename=3c_window_size-2048_q-6-2_j-6-7_use_day_data-1_avg_pool-0_model_type-scat+cov_filter_key-true.h5
python scripts/insert_catalog_to_h5.py --h5_filename $filename --h5_dataset_name glitches  --catalog_filename Salma_glitches_InSIght.pkl --window_size 2048 --n_workers 30 --target_column_name glitch

filename=3c_window_size-2048_q-6-2_j-6-7_use_day_data-1_avg_pool-0_model_type-scat_filter_key-true.h5
python scripts/insert_catalog_to_h5.py --h5_filename $filename --h5_dataset_name glitches  --catalog_filename Salma_glitches_InSIght.pkl --window_size 2048 --n_workers 30 --target_column_name glitch









