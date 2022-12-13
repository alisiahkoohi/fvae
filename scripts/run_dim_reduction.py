import sys
from facvae.utils import MarsDataset

H5_FILE_PATH = sys.argv[1]

if __name__ == "__main__":
    dataset = MarsDataset(H5_FILE_PATH,
                          0.99,
                          data_types=['scat_cov'],
                          load_to_memory=False,
                          normalize_data=False)
    dataset.pca_dim_reduction()
    
