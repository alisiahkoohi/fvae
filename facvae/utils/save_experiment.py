import os
import h5py


def save_exp_to_h5(path, args, **kwargs):
    """
    Setting up an HDF5 file to write scattering covariances.
    """
    # Check if the file already exists. If it does, delete it.
    if os.path.exists(path):
        os.remove(path)

    # HDF5 File.
    file = h5py.File(path, 'a')
    for key, value in vars(args).items():
        # Add the key and value to the file.
        file[key] = value
    for key, value in kwargs.items():
        # Add the key and value to the file.
        file[key] = value
    file.close()
