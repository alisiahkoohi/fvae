import os
from cuml.manifold import UMAP
import numpy as np
import gc
import h5py

import sys

from facvae.utils import datadir


def calculate_umap(scale, n_neighbors, min_dist, n_epochs):
    filename = os.path.join(datadir('tmp'),
                            'latent_features_' + str(scale) + '.h5')
    file = h5py.File(filename, 'r')
    latent_features = file['latent_features'][:]
    file.close()
    umap_features = {}
    gc.collect()
    print(
        'start computing UMAP features for scale {}'.format(scale),
        flush=True,
    )
    umap_class = UMAP(
        n_neighbors=int(n_neighbors),
        min_dist=float(min_dist),
        metric='euclidean',
        verbose=True,
        n_epochs=int(n_epochs),
    )
    gc.collect()
    umap_features[scale] = umap_class.fit_transform(latent_features)
    gc.collect()

    os.remove(filename)

    filename = os.path.join(datadir('tmp'),
                            'umap_features_' + str(scale) + '.h5')
    file = h5py.File(filename, 'a')
    if 'umap_features' in file.keys():
        del file['umap_features']
    file.create_dataset('umap_features', data=np.array(umap_features[scale]))
    file.close()


if __name__ == '__main__':
    calculate_umap(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
