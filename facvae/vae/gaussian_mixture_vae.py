import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import numpy as np
import os
import torch
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from facvae.utils import checkpointsdir, CustomLRScheduler, logsdir, plotsdir
from facvae.vae import GMVAENetwork, LossFunctions, Metrics

sns.set_style("whitegrid")
font = {'family': 'serif', 'style': 'normal', 'size': 10}
matplotlib.rc('font', **font)
sfmt = matplotlib.ticker.ScalarFormatter(useMathText=True)
sfmt.set_powerlimits((0, 0))
matplotlib.use("Agg")


class GaussianMixtureVAE(object):
    """Class training a Gaussian mixture variational autoencoder model.
    """

    def __init__(self, args, mars_dataset):
        self.w_rec, self.w_gauss, self.w_cat = args.weights
        self.mars_dataset = mars_dataset

        # Network architecture.
        self.network = GMVAENetwork(args.input_size, args.latent_dim,
                                    args.ncluster, args.init_temp,
                                    args.hard_gumbel).to(args.device)

        # Tensorboard writer.
        self.writer = SummaryWriter(log_dir=logsdir(args.experiment))

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
        loss_rec = self.losses.reconstruction_loss(data, data_recon, 'mse')

        # Gaussian loss.
        loss_gauss = self.losses.gaussian_loss(z, mu, var, y_mu, y_var)

        # Categorical loss.
        loss_cat = -self.losses.entropy(logits, prob_cat) - np.log(0.1)

        # Total loss.
        loss_total = (self.w_rec * loss_rec + self.w_gauss * loss_gauss +
                      self.w_cat * loss_cat)

        # Obtain predictions.
        _, predicted_labels = torch.max(logits, dim=1)

        return loss_total, loss_rec, loss_gauss, loss_cat, predicted_labels

    def test(self, data_loader, epoch):
        """Test the model with new data

        Args:
            data_loader: (DataLoader) corresponding loader containing the test/validation data
            return_loss: (boolean) whether to return the average loss values

        Return:
            accuracy and nmi for the given test data

        """
        # TODO: needed?
        self.network.eval()
        total_loss = 0.
        recon_loss = 0.
        cat_loss = 0.
        gauss_loss = 0.

        num_batches = 0.

        predicted_labels_list = []

        with torch.no_grad():
            for idx in data_loader:
                data = self.mars_dataset.sample_data(idx)
                if self.cuda == 1:
                    data = data.cuda()

                # flatten data
                data = data.view(data.size(0), -1)

                # forward call
                out_net = self.network(data)
                loss_dict = self.compute_loss(data, out_net)

                # accumulate values
                total_loss += loss_dict['total'].item()
                recon_loss += loss_dict['reconstruction'].item()
                gauss_loss += loss_dict['gaussian'].item()
                cat_loss += loss_dict['categorical'].item()

                # save predicted and true labels
                predicted = loss_dict['predicted_labels']
                predicted_labels_list.append(predicted)

                num_batches += 1.

                if num_batches % 10 == 9:
                    self.writer.add_scalars(
                        'test loss', {
                            'rec': recon_loss / num_batches,
                            'gauss': gauss_loss / num_batches,
                            'cat': cat_loss / num_batches,
                            'total': total_loss / num_batches
                        },
                        epoch * len(data_loader) + num_batches)

        predicted_labels = torch.cat(predicted_labels_list,
                                     dim=0).cpu().numpy()

        self.writer.add_scalars('classes', {
            str(i): (predicted_labels == i).mean()
            for i in range(self.ncluster)
        }, epoch)

        return total_loss / num_batches

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

        train_log = {
            'rec_loss': [],
            'gauss_loss': [],
            'cat_loss': [],
            'vae_loss': [],
        }
        # val_log = {key: [] for key in train_log}

        # Training loop, run for `args.max_epoch` epochs.
        with tqdm(range(args.max_epoch),
                  unit='epoch',
                  colour='#B5F2A9',
                  dynamic_ncols=True) as pb:
            for epoch in pb:
                # (train_loss, train_rec, train_gauss, train_cat, train_acc,
                #  train_nmi) = self.train_epoch(optim, scheduler, train_loader,
                #                                epoch, mars_dataset)
                # iterate over the dataset
                for itr, idx in enumerate(train_loader):
                    # Reset gradient attributes.
                    optim.zero_grad()
                    # Update learning rate.
                    scheduler.step()

                    # Load data batch.
                    x = self.mars_dataset.sample_data(idx)
                    x = x.to(args.device)
                    # Flatten data.
                    x = x.view(x.size(0), -1)
                    # Forward call.
                    y = self.network(x)
                    # Compute loss.
                    (vae_loss, rec_loss, gauss_loss, cat_loss,
                     clusters) = self.compute_loss(x, y)
                    # Compute gradients.
                    vae_loss.backward()
                    # Update parameters.
                    optim.step()

                    # Log progress.
                    if itr % 50 == 0:
                        self.log_progress(args, pb, epoch, itr, train_log,
                                          rec_loss, gauss_loss, cat_loss,
                                          vae_loss, clusters)

                # val_loss = self.test(val_loader, epoch, mars_dataset)

                # Decay gumbel temperature
                if args.decay_temp == 1:
                    self.network.gumbel_temp = np.maximum(
                        args.init_temp * np.exp(-args.temp_decay * epoch),
                        args.min_temp)

                if epoch % int(args.max_epoch /
                               5) == 0 or epoch == args.max_epoch - 1:
                    torch.save(
                        {
                            'model_state_dict': self.network.state_dict(),
                            'optim_state_dict': optim.state_dict(),
                            'epoch': epoch,
                            'train_log': train_log
                        },
                        os.path.join(checkpointsdir(args.experiment),
                                     f'checkpoint_{epoch}.pth'))

    def log_progress(self, args, pb, epoch, itr, train_log, rec_loss,
                     gauss_loss, cat_loss, vae_loss, clusters):
        """Log progress of training."""
        # Bookkeeping.
        train_log['vae_loss'].append(vae_loss.item())
        train_log['rec_loss'].append(rec_loss.item())
        train_log['gauss_loss'].append(gauss_loss.item())
        train_log['cat_loss'].append(cat_loss.item())

        # Progress bar.
        pb.set_postfix({
            'itr': f'{itr + 1:3d}',
            'vae_loss': f'{vae_loss.item():2.2f}',
            'rec_loss': f'{rec_loss.item():2.2f}',
            'gauss_loss': f'{gauss_loss.item():2.2f}',
            'cat_loss': f'{cat_loss.item():2.2f}',
        })

        self.writer.add_scalars(
            'classes', {
                str(i): (clusters == i).cpu().numpy().astype(float).mean()
                for i in range(args.ncluster)
            }, epoch * self.steps_per_epoch + itr)

        self.writer.add_scalar('vae_loss', vae_loss.item(),
                               epoch * self.steps_per_epoch + itr)
        self.writer.add_scalar('rec_loss', rec_loss.item(),
                               epoch * self.steps_per_epoch + itr)
        self.writer.add_scalar('gauss_loss', gauss_loss.item(),
                               epoch * self.steps_per_epoch + itr)
        self.writer.add_scalar('cat_loss', cat_loss.item(),
                               epoch * self.steps_per_epoch + itr)

    def latent_features(self, args, data_loader):
        """Obtain latent features learnt by the model

        Args:
            data_loader: (DataLoader) loader containing the data
            return_labels: (boolean) whether to return true labels or not

        Returns:
           features: (array) array containing the features from the data
        """
        # TODO: necessary?
        # self.network.eval()

        N = len(data_loader.dataset)
        features = np.zeros((N, args.latent_dim))
        clusters = np.zeros((N))
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

    def reconstruct_data(self, data_loader, sample_size=-1):
        """Reconstruct Data

        Args:
            data_loader: (DataLoader) loader containing the data
            sample_size: (int) size of random data to consider from data_loader

        Returns:
            reconstructed: (array) array containing the reconstructed data
        """
        # TODO: necessary?
        # self.network.eval()

        # sample random data from loader
        indices = np.random.randint(0,
                                    len(data_loader.dataset),
                                    size=sample_size)
        test_random_loader = torch.utils.data.DataLoader(
            data_loader.dataset,
            batchsize=sample_size,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(indices))

        # obtain values
        it = iter(test_random_loader)
        test_batch_data, _ = it.next()
        original = test_batch_data.data.numpy()
        if self.cuda:
            test_batch_data = test_batch_data.cuda()

            # obtain reconstructed data
        out = self.network(test_batch_data)
        reconstructed = out['x_rec']
        return original, reconstructed.data.cpu().numpy()

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

        # plot only the first 2 dimensions
        label_colors = {
            0: 'r',
            1: 'g',
            2: 'b',
            3: 'c',
            4: 'm',
        }
        colors = [label_colors[int(i)] for i in clusters]
        fig = plt.figure(figsize=(8, 6))
        plt.scatter(features[:, 0],
                    features[:, 1],
                    marker='o',
                    c=colors,
                    edgecolor='none',
                    cmap=plt.cm.get_cmap('jet', 10),
                    s=10)
        plt.colorbar()
        plt.savefig(os.path.join(plotsdir(args.experiment),
                                 'latent_space.png'),
                    format="png",
                    bbox_inches="tight",
                    dpi=300,
                    pad_inches=.05)
        plt.close(fig)

    def random_generation(self, args, num_elements=10):
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

        categorical = F.one_hot(torch.tensor(indices), args.ncluster).float()

        categorical = categorical.to(args.device)

        # infer the gaussian distribution according to the category
        mean, var = self.network.generative.pzy(categorical)

        # gaussian random sample by using the mean and variance
        noise = torch.randn_like(var)
        std = torch.sqrt(var)
        gaussian = mean + noise * std

        # generate new samples with the given gaussian
        samples = self.network.generative.pxz(gaussian).cpu().detach().numpy()

        fig, ax = plt.subplots(1, args.ncluster, figsize=(25, 5))
        for i in range(args.ncluster):
            for j in range(num_elements):
                ax[i].plot(samples[i * num_elements + j, :], lw=.8, alpha=0.3)
            # ax[i].axis('off')
        plt.savefig(os.path.join(plotsdir(args.experiment), 'error_log.png'),
                    format="png",
                    bbox_inches="tight",
                    dpi=300,
                    pad_inches=.05)
        plt.close(fig)

    def load(self, args, epoch):
        file_to_load = os.path.join(checkpointsdir(args.experiment),
                                    'checkpoint_' + str(epoch) + '.pth')
        if os.path.isfile(file_to_load):
            if args.device == torch.device(type='cpu'):
                checkpoint = torch.load(file_to_load, map_location='cpu')
            else:
                checkpoint = torch.load(file_to_load)

            self.network.load_state_dict(checkpoint['model_state_dict'])

            assert epoch == checkpoint["epoch"]
