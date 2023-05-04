git_root_path=$(git rev-parse --show-toplevel)


# Spring 1: Sol 116-305
python $git_root_path/scripts/train_facvae.py \
    --phase test  \
    --experiment_name pyramid_full-mission  \
    --cuda 0 \
    --filter_key "2019-JUN,2019-JUL,2019-AUG,2019-SEP" \
    --ncluster 9


# Summer 1: Sol 306-483
python $git_root_path/scripts/train_facvae.py \
    --phase test  \
    --experiment_name pyramid_full-mission  \
    --cuda 0 \
    --filter_key "2019-OCT,2019-NOV,2019-DEC,2020-JAN,2020-FEB,2020-MARCH" \
    --ncluster 9

# Fall 1: Sol 484-624
python $git_root_path/scripts/train_facvae.py \
    --phase test  \
    --experiment_name pyramid_full-mission  \
    --cuda 0 \
    --filter_key "2020-APRIL,2020-MAY,2020-JUN,2020-JUL,2020-AUG" \
    --ncluster 9

# Winter 1: Sol 625-782
python $git_root_path/scripts/train_facvae.py \
    --phase test  \
    --experiment_name pyramid_full-mission  \
    --cuda 0 \
    --filter_key "2020-SEP,2020-OCT,2020-NOV,2020-DEC,2021-JAN" \
    --ncluster 9

# Spring 2: Sol 783-978
python $git_root_path/scripts/train_facvae.py \
    --phase test  \
    --experiment_name pyramid_full-mission  \
    --cuda 0 \
    --filter_key "2021-FEB,2021-MARCH,2021-APRIL,2021-MAY,2021-JUN,2021-JUL,2021-AUG" \
    --ncluster 9

# Summer 2: Sol 979-1155
python $git_root_path/scripts/train_facvae.py \
    --phase test  \
    --experiment_name pyramid_full-mission  \
    --cuda 0 \
    --filter_key "2021-SEP,2021-OCT,2021-NOV,2021-DEC,2022-JAN,2022-FEB" \
    --ncluster 9

# Fall 2: Sol 1156-
python $git_root_path/scripts/train_facvae.py \
    --phase test  \
    --experiment_name pyramid_full-mission  \
    --cuda 0 \
    --filter_key "2022-MARCH,2022-APRIL,2022-MAY" \
    --ncluster 9