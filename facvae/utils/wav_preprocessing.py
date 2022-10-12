import scipy
import numpy as np
import os

SAMPLING_RATE = 16000
MAX_LENGTH = 16000


# Find all the wave files in subfolders of a folder.
def find_wav_files(folder):
    files = []
    for root, dirs, filenames in os.walk(folder):
        for filename in filenames:
            if filename.endswith('.wav'):
                files.append(os.path.join(root, filename))
    return files


# Read wave file and resample it to 16kHz.
def read_wav_file(filename):
    # Read wave file.
    sr, wave = scipy.io.wavfile.read(filename)
    # Resample to 16kHz.
    wave = scipy.signal.resample(wave, int(len(wave) * SAMPLING_RATE / sr))
    return wave.astype(np.float32)


# Remove leading and trailing silence from a wave file.
def trim_and_add_silence(wave, threshold=0.1):
    for i in range(len(wave)):
        if abs(wave[i]) > threshold:
            break
    for j in range(len(wave) - 1, 0, -1):
        if abs(wave[j]) > threshold:
            break
    wave = wave[i:j]
    if len(wave) < MAX_LENGTH:
        wave = np.pad(wave, ((MAX_LENGTH - len(wave)) // 2,
                             (MAX_LENGTH - len(wave)) // 2 + 1), 'constant')
        wave = wave[:MAX_LENGTH]
    return wave


# Play wave file.
def play_wav_file(wave):
    scipy.io.wavfile.write('temp.wav', SAMPLING_RATE, wave.astype(np.int16))
    os.system('aplay temp.wav')
    os.remove('temp.wav')


# Creare HDF5 dataset with all wav files in one big matrix.
def create_hdf5_dataset(folder, filename):
    import h5py
    files = find_wav_files(folder)
    f = h5py.File(filename, 'w')
    dataset = f.create_dataset('dataset',
                               shape=(len(files), MAX_LENGTH),
                               dtype=np.float32)
    label_dataset = f.create_dataset('label',
                                     shape=(len(files)),
                                     dtype=np.int)
    file_dataset = f.create_dataset('filename',
                                    shape=len(files),
                                    dtype=h5py.string_dtype())
    for i, file in enumerate(files):
        print('Reading file %s' % file)
        wave = read_wav_file(file)
        wave = trim_and_add_silence(wave)
        dataset[i, :] = wave
        label_dataset[i] = int(file.split('/')[-1][0])
        file_dataset[i] = file
    f.close()
