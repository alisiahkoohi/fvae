
# 1,3 1,4 1,5 2,1 2,3 2,5
for Q in   3,1 3,3 3,5 4,1 4,3 4,5
do  
    IFS=","
    set -- $Q
    CUDA_VISIBLE_DEVICES=0 python scripts/generate_scattering_cov.py \
        --window_size 2048 --q $1,$2 --j '7,7' --cuda 0 --nchunks 256 \
        --use_day_data 1 --avg_pool 0 --model_type scat \
        --filter_key "2019-JUN-03,2019-JUN-04,2019-JUN-05,2019-JUN-06,2019-JUN-07,2019-JUN-08,2019-JUN-09,2019-JUN-10,2019-JUN-11" \
        --filename 3c_2019-JUNE
done
