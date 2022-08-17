for i in $(find . -type f -name "*UVW*ACC.mseed"); do
    new_name=${i//\//\-}
    rclone copyto $i waveform/${new_name:2}
done
