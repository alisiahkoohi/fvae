"""Write Mars .mseed data into h5 format. """
import h5py
import obspy
import numpy as np
import os
from tqdm import tqdm

from facvae.utils import datadir

MARS_PATH = datadir('mars')
RAW_WAV_FILENAME = 'day-long-raw-waveforms.h5'


def setup_hdf5_file(path):
    """
    Setting up an HDF5 file to write scattering covariances.
    """
    # Path to the file.
    file_path = os.path.join(path, RAW_WAV_FILENAME)

    # Horizons file
    file = h5py.File(file_path, 'a')
    file.close()


def update_hdf5_file(path, filename, trace):
    """
    Update the HDF5 file by writing new data.
    """
    # Path to the file.
    file_path = os.path.join(path, RAW_WAV_FILENAME)

    # HDF5 file.
    file = h5py.File(file_path, 'r+')

    # Group for the given file.
    file_group = file.create_group(filename)

    # HDF5 dataset for waveform.
    file_group.create_dataset('waveform', data=trace.data, dtype=np.float32)
    file_group.create_dataset('start-time',
                              data=str(trace.stats.starttime)[:-1],
                              dtype=h5py.string_dtype())
    file_group.create_dataset('end-time',
                              data=str(trace.stats.endtime)[:-1],
                              dtype=h5py.string_dtype())

    file.close()


def extract_day_long_waveforms():

    waveform_path = datadir(os.path.join(MARS_PATH, 'waveforms'))
    raw_wav_path = datadir(os.path.join(MARS_PATH, 'waveforms_h5'))
    raw_data_files = os.listdir(waveform_path)

    setup_hdf5_file(raw_wav_path)
    discarded_files = 0
    with tqdm(raw_data_files,
              unit='file',
              colour='#B5F2A9',
              dynamic_ncols=True) as pb:
        for i, file in enumerate(pb):
            # Read data into a stream format.
            data_stream = obspy.read(os.path.join(waveform_path, file))

            # Only keep files that do not have gaps.
            if len(data_stream.get_gaps()) == 0:

                # The following line although will not do interpolation—because
                # there are not gaps—but will combine different streams into
                # one.
                trace = data_stream.merge(method=1,
                                          fill_value="interpolate")[0]

                update_hdf5_file(raw_wav_path, file, trace)

                pb.set_postfix({
                    'shape': trace.data.shape,
                    'discarded': f'{discarded_files/(i + 1):.4f}'
                })

            else:
                discarded_files += 1


if __name__ == "__main__":

    extract_day_long_waveforms()
