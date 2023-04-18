
# 1,3 1,4 1,5 2,1 2,3 2,5
for Q in   1,1
do  
    IFS=","
    set -- $Q
    CUDA_VISIBLE_DEVICES=2 python scripts/generate_scattering_cov.py \
        --window_size 4096 --q $1,$2 --j '8,8' --cuda 1 --nchunks 16 \
        --use_day_data 1 --avg_pool 8 --model_type scat+cov \
        --filter_key "" \
        --filename 3c
done
