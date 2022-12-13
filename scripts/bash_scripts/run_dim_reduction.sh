
git_root_path=$(git rev-parse --show-toplevel)

for filename in $*;
do
    python $git_root_path/scripts/run_dim_reduction.py "$filename"
done



