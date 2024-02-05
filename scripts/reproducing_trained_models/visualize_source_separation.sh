git_root_path=$(git rev-parse --show-toplevel)
source_separation_visualization_script=$git_root_path/scripts/visualize_source_separation.py

for cluster in 1 3 6
do
    python $source_separation_visualization_script \
        --cluster_g $cluster \
        --scale_g "16384" \
        --cluster_n "0,3,8" \
        --scale_n "4096,4096,4096" \
        --R 50 \
        --filter_key "2019-JUN-03,2019-JUN-04,2019-JUN-05,2019-JUN-06,2019-JUN-07,2019-JUN-08,2019-JUN-09,2019-JUN-10,2019-JUN-11"
done

for cluster in 3 1 4 7
do
    python $source_separation_visualization_script \
        --cluster_g $cluster \
        --scale_g "65536" \
        --cluster_n "0,3,8" \
        --scale_n "4096,4096,4096" \
        --R 50 \
        --filter_key "2019-JUN-03,2019-JUN-04,2019-JUN-05,2019-JUN-06,2019-JUN-07,2019-JUN-08,2019-JUN-09,2019-JUN-10,2019-JUN-11"
done
