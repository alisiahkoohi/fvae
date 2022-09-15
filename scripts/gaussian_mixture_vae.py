import numpy as np
import os
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from facvae.utils import checkpointsdir, CustomLRScheduler, logsdir
from facvae.vae import GMVAENetwork, LossFunctions, Metrics


class GaussianMixtureVAE(object):
    """Class training a Gaussian mixture variational autoencoder model.
    """

    def __init__(self, args, dataset, device):
        self.w_rec = args.w_rec
        self.w_gauss = args.w_gauss
        self.w_cat = args.w_cat
        self.dataset = dataset
        self.device = device

        # Network architecture.
        self.network = GMVAENetwork(args.input_size,
                                    args.latent_dim,
                                    args.ncluster,
                                    args.init_temp,
                                    hidden_dim=args.hidden_dim,
                                    nlayer=args.nlayer).to(self.device)

        # Tensorboard writer.
        if args.phase == 'train':
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
                    x = self.dataset.sample_data(idx)
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
                        x_val = self.dataset.sample_data(next(
                            iter(val_loader)))
                        x_val = x_val.to(self.device)
                        y_val = self.network(x_val)
                        val_loss = self.compute_loss(x_val, y_val)

                    self.log_progress(args, pb, epoch, train_loss, val_loss)

                # Decay gumbel temperature
                if args.temp_decay > 0:
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

    def load_checkpoint(self, args, epoch):
        file_to_load = os.path.join(checkpointsdir(args.experiment),
                                    'checkpoint_' + str(epoch) + '.pth')
        if os.path.isfile(file_to_load):
            if self.device == torch.device(type='cpu'):
                checkpoint = torch.load(file_to_load, map_location='cpu')
            else:
                checkpoint = torch.load(file_to_load)

            self.network.load_state_dict(checkpoint['model_state_dict'])

            if not epoch == checkpoint["epoch"]:
                raise ValueError(
                    'Inconsistent filename and loaded checkpoint.')
        return self.network
