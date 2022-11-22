# factorialVAE

## Installation

Run the commands below to install the required packages.

```bash
git clone https://github.com/alisiahkoohi/factorialVAE
cd factorialVAE/
conda env create -f environment.yml
source activate factorialVAE
pip install -e .
```

Also add the following to your `~/.bashrc`:

```bash
export MARSCONVERTER=/PATH_TO_REPO/factorialVAE/facvae/marsconverter
```

After the above steps, you can run the example scripts by just
activating the environment, i.e., `conda activate factorialVAE`, the
following times.
