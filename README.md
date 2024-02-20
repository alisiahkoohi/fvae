<h1 align="center">Martian time-series unraveled: A multi-scale nested approach with factorial variational autoencoders</h1>

## Installation

Run the commands below to install the required packages.

```bash
git clone https://github.com/alisiahkoohi/facvae
cd facvae/
conda env create -f environment.yml
source activate facvae
pip install -e .
```

Also add the following to your `~/.bashrc`:

```bash
export MARSCONVERTER=/PATH_TO_REPO/facvae/facvae/marsconverter
```

After the above steps, you can run the example scripts by just
activating the environment, i.e., `conda activate facvae`, the
following times.

## Data

### Training data

Data required for training---i.e., the pyramidal scattering spectra, can be downloaded with the following command:


```bash
mkdir -p data/mars/scat_covs_h5/
wget -O "data/mars/scat_covs_h5/pyramid_full-mission_window_size-65536_q-1-1_j-8-8_use_day_data-1_avgpool_base-4_avgpool_exp-5-6-7-8_model_type-scat-cov_filter_key-true.h5" "https://www.dropbox.com/scl/fi/pwv4hwf0mu43b256dvt0q/pyramid_full-mission_window_size-65536_q-1-1_j-8-8_use_day_data-1_avgpool_base-4_avgpool_exp-5-6-7-8_model_type-scat-cov_filter_key-true.h5?rlkey=f3g0q2y5vrnpj6oaz68edf813&dl=0" --no-check-certificate
```

### Raw data for visualization

In order to visualize the results, including the aligned waveforms, time histograms, and latent space, the raw data is also required. The raw data can be downloaded from [here](https://www.dropbox.com/scl/fo/38tr0k9kghtben1mwv3qs/h?rlkey=tlccygf71nutreqakq9p54a0w&dl=0) and it must be placed in the `data/mars/raw/` directory.

### Pretrained model

The pretrained model can be downloaded with the following command. Note that for the visualization and source separation scripts to use this model, the default values in associated configuration json files must be used.

```bash
mkdir -p "data/checkpoints/nature_full-mission_max_epoch-1000_batchsize-16384_lr-0.001_lr_final-0.001_ncluster-9_latent_dim-32_w_rec-0.15_wd-0.0_hidden_dim-1024_nlayer-4_window_size-65536_scales-1024-4096-16384-65536_seed-29/"
wget -O "data/checkpoints/nature_full-mission_max_epoch-1000_batchsize-16384_lr-0.001_lr_final-0.001_ncluster-9_latent_dim-32_w_rec-0.15_wd-0.0_hidden_dim-1024_nlayer-4_window_size-65536_scales-1024-4096-16384-65536_seed-29/checkpoint_999.pth" "https://www.dropbox.com/scl/fi/7v7zjgzjn67t2ukp27ilr/checkpoint_999.pth?rlkey=nh6tap4xsc6p9e5b37660btpb&dl=0" --no-check-certificate
```


## Usage

To run the example script, you can use the following commands. The list of command line arguments and their default values can be found in the configuration json files in `configs/`.

### Training the fVAE on the full mission data.

For a full list of command line arguments, see `configs/facvae_full-mission.json`.

```bash
python scripts/train_facvae.py
```

### Visualizing the fVAE: aligned waveforms, time histograms, and latent space.

```bash
python scripts/train_facvae.py --phase test

```

### Source separation using the trained fVAE.

For a full list of command line arguments, see `configs/source_separation.json`. Resutls will be saved in `plots/` directory. Note that the variables `cluster_n` and `scale_n` are based on the pretrained model and should be set accordingly when a new model is trained.

**Glitch example:**

```bash
python scripts/separate_facvae.py --cluster_n "5" --cluster_g "4" --scale_n "1024" --scale_g "65536"
```

**Wind example:**

```bash
python scripts/separate_facvae.py --cluster_n "1,6" --cluster_g "3" --scale_n "1024,1024" --scale_g "65536"
```

## Questions

Please contact alisk@rice.edu for questions.

## Author

Ali Siahkoohi



