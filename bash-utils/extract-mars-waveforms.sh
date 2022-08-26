repo_root=$(git rev-parse --show-toplevel)
cd ${repo_root}/data/mars/
mkdir ${repo_root}/data/mars/waveform/

for i in $(find . -type f -name "*UVW*ACC.mseed"); do
    new_name=${i//\//\-}
    rclone copyto $i waveform/${new_name:2}
done
