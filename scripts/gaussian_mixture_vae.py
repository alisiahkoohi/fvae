from ast import Yield
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import numpy as np
import os
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from facvae.utils import checkpointsdir, CustomLRScheduler, logsdir, plotsdir
from facvae.vae import GMVAENetwork, LossFunctions, Metrics

sns.set_style("whitegrid")
font = {'family': 'serif', 'style': 'normal', 'size': 12}
matplotlib.rc('font', **font)
sfmt = matplotlib.ticker.ScalarFormatter(useMathText=True)
sfmt.set_powerlimits((0, 0))
matplotlib.use("Agg")


class GaussianMixtureVAE(object):
    """Class training a Gaussian mixture variational autoencoder model.
    """

    def __init__(self, args, mars_dataset, device):
        self.w_rec = args.w_rec
        self.w_gauss = args.w_gauss
        self.w_cat = args.w_cat
        self.mars_dataset = mars_dataset
        self.device = device

        # Network architecture.
        self.network = GMVAENetwork(args.input_size, args.latent_dim,
                                    args.ncluster, args.init_temp,
                                    args.hard_gumbel).to(self.device)

        # Tensorboard writer.
        self.writer = SummaryWriter(log_dir=logsdir(args.experiment))

        self.train_log = {
            'rec': [],
            'gauss': [],
            'cat': [],
            'vae': [],
        }
        self.val_log = {key: [] for key in self.train_log}

        # Loss functions and metrics.
        self.losses = LossFunctions()
        self.metrics = Metrics()

    def compute_loss(self, data, out_net):
        """Method defining the loss functions derived from the variational lower bound
        Args:
            data: (array) corresponding array containing the input data
            out_net: (dict) contains the graph operations or nodes of the network output

        Returns:
            loss_dic: (dict) contains the values of each loss function and predictions
        """
        # obtain network variables
        z, data_recon = out_net['gaussian'], out_net['x_rec']
        logits, prob_cat = out_net['logits'], out_net['prob_cat']
        y_mu, y_var = out_net['y_mean'], out_net['y_var']
        mu, var = out_net['mean'], out_net['var']

        # Reconstruction loss
        rec_loss = self.losses.reconstruction_loss(data, data_recon, 'mse')

        # Gaussian loss.
        gauss_loss = self.losses.gaussian_loss(z, mu, var, y_mu, y_var)

        # Categorical loss.
        cat_loss = -self.losses.entropy(logits, prob_cat) - np.log(0.1)

        # Total loss.
        vae_loss = (self.w_rec * rec_loss + self.w_gauss * gauss_loss +
                    self.w_cat * cat_loss)

        # Obtain predictions.
        _, clusters = torch.max(logits, dim=1)

        return {
            'vae': vae_loss,
            'rec': rec_loss,
            'gauss': gauss_loss,
            'cat': cat_loss,
            'clusters': clusters
        }

    def train(self, args, train_loader, val_loader):
        """Train the model

        Args:
            train_loader: (DataLoader) corresponding loader containing the training data
            val_loader: (DataLoader) corresponding loader containing the validation data

        Returns:
            output: (dict) contains the history of train/val loss
        """
        # Optimizer.
        optim = torch.optim.Adam(self.network.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.wd)

        # Setup the learning rate scheduler.
        scheduler = CustomLRScheduler(optim, args.lr, args.lr_final,
                                      args.max_epoch)

        self.steps_per_epoch = len(train_loader)
        print(f'Number of steps per epoch: {self.steps_per_epoch}')

        # Training loop, run for `args.max_epoch` epochs.
        with tqdm(range(args.max_epoch),
                  unit='epoch',
                  colour='#B5F2A9',
                  dynamic_ncols=True) as pb:
            for epoch in pb:
                # iterate over the dataset
                for idx in train_loader:
                    # Reset gradient attributes.
                    optim.zero_grad()
                    # Update learning rate.
                    scheduler.step()

                    # Load data batch.
                    x = self.mars_dataset.sample_data(idx)
                    x = x.to(self.device)
                    # Forward call.
                    y = self.network(x)
                    # Compute loss.
                    train_loss = self.compute_loss(x, y)
                    # Compute gradients.
                    train_loss['vae'].backward()
                    # Update parameters.
                    optim.step()

                # Log progress.
                if epoch % 100 == 0:
                    with torch.no_grad():
                        x_val = self.mars_dataset.sample_data(
                            next(iter(val_loader)))
                        x_val = x_val.to(self.device)
                        y_val = self.network(x_val)
                        val_loss = self.compute_loss(x_val, y_val)

                    self.log_progress(args, pb, epoch, train_loss, val_loss)

                # Decay gumbel temperature
                if args.decay_temp == 1:
                    self.network.gumbel_temp = np.maximum(
                        args.init_temp * np.exp(-args.temp_decay * epoch),
                        args.min_temp)

            torch.save(
                {
                    'model_state_dict': self.network.state_dict(),
                    'optim_state_dict': optim.state_dict(),
                    'epoch': epoch,
                    'train_log': self.train_log,
                    'val_log': self.val_log
                },
                os.path.join(checkpointsdir(args.experiment),
                             f'checkpoint_{epoch}.pth'))

    def log_progress(self, args, pb, epoch, train_loss, val_loss):
        """Log progress of training."""
        # Bookkeeping.
        progress_bar_dict = {}
        for key, item in train_loss.items():
            if key != 'clusters':
                self.train_log[key].append(item.item())
                progress_bar_dict[key] = f'{item.item():2.2f}'
        for key, item in val_loss.items():
            if key != 'clusters':
                self.val_log[key].append(item.item())

        # Progress bar.
        pb.set_postfix(progress_bar_dict)

        self.writer.add_scalars(
            'classes_train', {
                str(i): (train_loss['clusters']
                         == i).cpu().numpy().astype(float).mean()
                for i in range(args.ncluster)
            }, epoch)

        self.writer.add_scalars(
            'classes_val', {
                str(i):
                (val_loss['clusters'] == i).cpu().numpy().astype(float).mean()
                for i in range(args.ncluster)
            }, epoch)

        self.writer.add_scalars('vae_loss', {
            'train': train_loss['vae'],
            'val': val_loss['vae']
        }, epoch)

        self.writer.add_scalars('rec_loss', {
            'train': train_loss['rec'],
            'val': val_loss['rec']
        }, epoch)

        self.writer.add_scalars('gauss_loss', {
            'train': train_loss['gauss'],
            'val': val_loss['gauss']
        }, epoch)

        self.writer.add_scalars('cat_loss', {
            'train': train_loss['cat'],
            'val': val_loss['cat']
        }, epoch)

    def latent_features(self, args, data_loader):
        """Obtain latent features learnt by the model

        Args:
            data_loader: (DataLoader) loader containing the data
            return_labels: (boolean) whether to return true labels or not

        Returns:
           features: (array) array containing the features from the data
        """
        N = len(data_loader.dataset)
        features = np.zeros([N, args.latent_dim])
        clusters = np.zeros([N])
        counter = 0
        with torch.no_grad():
            for idx in data_loader:
                # Load data batch.
                x = self.mars_dataset.sample_data(idx)

                # flatten data
                x = x.view(x.size(0), -1)
                y = self.network.inference(x)
                latent_feat = y['mean']
                cluster_membership = y['logits'].argmax(axis=1)

                features[counter:counter +
                         x.size(0), :] = latent_feat.cpu().detach().numpy()[
                             ...]
                clusters[counter:counter + x.size(0)] = cluster_membership.cpu(
                ).detach().numpy()[...]

                counter += x.shape[0]

        return features, clusters

    def plot_waveforms(self, args, data_loader, sample_size=5):
        """Plot waveforms.
        """
        # Sample random data from loader
        x = self.mars_dataset.sample_data(range(len(data_loader.dataset)),
                                          type='scat_cov')
        x = x.to(self.device)

        # Obtain reconstructed data.
        cluster_membership = []
        with torch.no_grad():
            y = self.network(x)
            confident_idxs = y['prob_cat'].max(axis=-1)[0].sort()[1]
            cluster_membership = y['logits'][confident_idxs, :].argmax(axis=1)

        colors = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
            '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
        ]
        fig, ax = plt.subplots(sample_size,
                               args.ncluster,
                               figsize=(4 * args.ncluster, 4 * args.ncluster))
        fig_sp, ax_sp = plt.subplots(sample_size,
                                     args.ncluster,
                                     figsize=(4 * args.ncluster,
                                              4 * args.ncluster))
        fig_scat, ax_scat = plt.subplots(sample_size,
                                         args.ncluster,
                                         figsize=(4 * args.ncluster,
                                                  4 * args.ncluster))
        for i in range(args.ncluster):

            cluster_idxs = confident_idxs[np.where(
                cluster_membership == i)[0]][-sample_size:, ...]
            waveforms = self.mars_dataset.sample_data(cluster_idxs,
                                                      type='waveform')
            x = self.mars_dataset.sample_data(cluster_idxs, type='scat_cov')

            for j in range(sample_size):
                ax_sp[j, i].specgram(waveforms[j, :],
                                     Fs=20.0,
                                     mode='magnitude', cmap='inferno')
                ax_sp[j, i].set_title("Spectogram from cluster " + str(i))
                ax_sp[j, i].grid(False)

                ax[j, i].plot(waveforms[j, :],
                              color=colors[i],
                              lw=1.2,
                              alpha=0.8)
                ax[j, i].set_title("Waveform from cluster " + str(i))

                ax_scat[j, i].plot(x[j, :], color=colors[i], lw=1.2, alpha=0.8)
                ax_scat[j, i].set_title("Scat covs from cluster " + str(i))

        fig.savefig(os.path.join(plotsdir(args.experiment),
                                 'waveform_samples.png'),
                    format="png",
                    bbox_inches="tight",
                    dpi=300,
                    pad_inches=.05)
        plt.close(fig)

        fig_sp.savefig(os.path.join(plotsdir(args.experiment),
                                    'waveform_spectograms.png'),
                       format="png",
                       bbox_inches="tight",
                       dpi=300,
                       pad_inches=.05)
        plt.close(fig_sp)

        fig_scat.savefig(os.path.join(plotsdir(args.experiment),
                                      'scatcov_samples.png'),
                         format="png",
                         bbox_inches="tight",
                         dpi=300,
                         pad_inches=.05)
        plt.close(fig_scat)

    def reconstruct_data(self, args, data_loader, sample_size=5):
        """Reconstruct Data

        Args:
            data_loader: (DataLoader) loader containing the data
            sample_size: (int) size of random data to consider from data_loader

        Returns:
            reconstructed: (array) array containing the reconstructed data
        """
        # Sample random data from loader
        x = self.mars_dataset.sample_data(next(iter(data_loader)))
        indices = np.random.randint(0, x.shape[0], size=sample_size)
        x = x[indices, ...]
        x = x.to(self.device)

        # Obtain reconstructed data.
        with torch.no_grad():
            y = self.network(x)
            x_rec = y['x_rec']

        fig, ax = plt.subplots(1, sample_size, figsize=(25, 5))
        for i in range(sample_size):
            ax[i].plot(x[i, :], lw=.8, alpha=1, color='k', label='original')
            ax[i].plot(x_rec[i, :],
                       lw=.8,
                       alpha=0.5,
                       color='r',
                       label='reconstructed')
        plt.legend()
        plt.savefig(os.path.join(plotsdir(args.experiment), 'rec.png'),
                    format="png",
                    bbox_inches="tight",
                    dpi=300,
                    pad_inches=.05)
        plt.close(fig)

    def plot_latent_space(self, args, data_loader, save=False):
        """Plot the latent space learnt by the model

        Args:
            data: (array) corresponding array containing the data
            labels: (array) corresponding array containing the labels
            save: (bool) whether to save the latent space plot

        Returns:
            fig: (figure) plot of the latent space
        """
        # obtain the latent features
        features, clusters = self.latent_features(args, data_loader)
        features_tsne = TSNE(n_components=2,
                             learning_rate='auto',
                             init='pca',
                             early_exaggeration=10,
                             perplexity=200).fit_transform(features)
        features_pca = PCA(n_components=2).fit_transform(features)
        # plot only the first 2 dimensions
        colors = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
            '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
        ]
        # cmap = plt.cm.get_cmap('hsv', args.ncluster)
        label_colors = {i: colors[i] for i in range(args.ncluster)}
        colors = [label_colors[int(i)] for i in clusters]
        fig = plt.figure(figsize=(8, 6))
        plt.scatter(features_pca[:, 0],
                    features_pca[:, 1],
                    marker='o',
                    c=colors,
                    edgecolor='none',
                    cmap=plt.cm.get_cmap('jet', 10),
                    s=10)
        plt.title("Two dimensional PCA of the latent samples")
        plt.savefig(os.path.join(plotsdir(args.experiment),
                                 'pca_latent_space.png'),
                    format="png",
                    bbox_inches="tight",
                    dpi=300,
                    pad_inches=.05)
        plt.close(fig)

        fig = plt.figure(figsize=(8, 6))
        plt.scatter(features_tsne[:, 0],
                    features_tsne[:, 1],
                    marker='o',
                    c=colors,
                    edgecolor='none',
                    cmap=plt.cm.get_cmap('jet', 10),
                    s=10)
        plt.title("T-SNE visualization of the latent samples")
        plt.savefig(os.path.join(plotsdir(args.experiment),
                                 'latent_space_tsne.png'),
                    format="png",
                    bbox_inches="tight",
                    dpi=300,
                    pad_inches=.05)
        plt.close(fig)

    def random_generation(self, args, data_loader, num_elements=3):
        """Random generation for each category

        Args:
            num_elements: (int) number of elements to generate

        Returns:
            generated data according to num_elements
        """
        colors = ["#6e6e6e", "#cccccc", "#a3a3a3", "#4a4a4a", "#000000"]

        # categories for each element
        arr = np.array([])
        for i in range(args.ncluster):
            arr = np.hstack([arr, np.ones(num_elements) * i])
        indices = arr.astype(int).tolist()

        categorical = torch.nn.functional.one_hot(torch.tensor(indices),
                                                  args.ncluster).float()

        categorical = categorical.to(self.device)

        # infer the gaussian distribution according to the category
        mean, var = self.network.generative.pzy(categorical)

        # gaussian random sample by using the mean and variance
        noise = torch.randn_like(var)
        std = torch.sqrt(var)
        gaussian = mean + noise * std

        # generate new samples with the given gaussian
        samples = self.network.generative.pxz(gaussian).cpu().detach().numpy()
        colors = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
            '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
        ]
        fig, ax = plt.subplots(num_elements,
                               args.ncluster,
                               figsize=(4 * args.ncluster, 4 * args.ncluster))
        for i in range(args.ncluster):
            for j in range(num_elements):
                ax[j, i].plot(samples[i * num_elements + j, :],
                              color=colors[i],
                              lw=1.2,
                              alpha=0.8)
                ax[j, i].set_title("Sample from cluster " + str(i))
            # ax[i].axis('off')
        plt.savefig(os.path.join(plotsdir(args.experiment),
                                 'joint_samples.png'),
                    format="png",
                    bbox_inches="tight",
                    dpi=300,
                    pad_inches=.05)
        plt.close(fig)

        x = self.mars_dataset.sample_data(next(iter(data_loader)))
        indices = np.random.randint(0,
                                    x.shape[0],
                                    size=args.ncluster * num_elements)
        x = x[indices, ...]

        fig, ax = plt.subplots(num_elements, num_elements, figsize=(12, 12))
        for i in range(num_elements):
            for j in range(num_elements):
                ax[j, i].plot(x[i * num_elements + j, :],
                              color="k",
                              lw=1.2,
                              alpha=0.7)
                # ax[i].axis('off')
                ax[j, i].set_title("Sample from testing dataset")
        plt.savefig(os.path.join(plotsdir(args.experiment),
                                 'test_samples.png'),
                    format="png",
                    bbox_inches="tight",
                    dpi=300,
                    pad_inches=.05)
        plt.close(fig)

    def load(self, args, epoch):
        file_to_load = os.path.join(checkpointsdir(args.experiment),
                                    'checkpoint_' + str(epoch) + '.pth')
        if os.path.isfile(file_to_load):
            if self.device == torch.device(type='cpu'):
                checkpoint = torch.load(file_to_load, map_location='cpu')
            else:
                checkpoint = torch.load(file_to_load)

            self.network.load_state_dict(checkpoint['model_state_dict'])

            assert epoch == checkpoint["epoch"]
