# factorialVAE

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
