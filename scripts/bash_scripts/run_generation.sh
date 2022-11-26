CUDA_VISIBLE_DEVICES=2 python scripts/generate_scattering_cov.py --window_size 2048 --q '6,2,2' --j '6,7,7' --cuda 1 --nchunks 256 --use_day_data 1 --avg_pool 0 --model_type scat --filter_key '2020' --filename 3c_2020

CUDA_VISIBLE_DEVICES=2 python scripts/generate_scattering_cov.py --window_size 2048 --q '6,2' --j '6,7' --cuda 1 --nchunks 256 --use_day_data 1 --avg_pool 0 --model_type scat --filter_key '2020' --filename 3c_2020

CUDA_VISIBLE_DEVICES=2 python scripts/generate_scattering_cov.py --window_size 2048 --q '6,2' --j '6,7' --cuda 1 --nchunks 256 --use_day_data 1 --avg_pool 0 --model_type scat+cov --filter_key '2020' --filename 3c_2020
