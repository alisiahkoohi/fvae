
# 1,3 1,4 1,5 2,1 2,3 2,5
for Q in 1,1
do
    IFS=","
    set -- $Q
    CUDA_VISIBLE_DEVICES=3 python scripts/generate_scattering_cov.py \
        --q $1,$2 \
        --filter_key "2020" \
        --filename neurips_2020
done
