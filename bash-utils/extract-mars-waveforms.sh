repo_root=$(git rev-parse --show-toplevel)
cd ${repo_root}/data/mars/raw/
waveform_path=${repo_root}/data/mars/waveforms
mkdir $waveform_path

for i in $(find . -type f -name "*UVW*ACC.mseed"); do
    echo $i
    new_name=${i//\//\-}
    new_name=${new_name:2}
    prefix=${new_name::-23}
    postfix=${new_name: -20}
    new_name=${prefix}${postfix}
    rclone copyto $i  $waveform_path/$new_name
done
