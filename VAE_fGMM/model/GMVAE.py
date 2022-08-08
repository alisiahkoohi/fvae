import os
from time import time
from datetime import datetime
from tqdm import tqdm
from pathlib import Path
from torch import optim
from torch.utils.data.sampler import SubsetRandomSampler
from VAE_fGMM.networks.Networks import *
from VAE_fGMM.losses.LossFunctions import *
from VAE_fGMM.metrics.Metrics import *
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter


class GMVAE:

    def __init__(self, args):
        self.num_epochs = args.epochs
        self.cuda = args.cuda
        self.verbose = args.verbose

        self.batch_size = args.batch_size
        self.batch_size_val = args.batch_size_val
        self.learning_rate = args.learning_rate
        self.decay_epoch = args.decay_epoch
        self.lr_decay = args.lr_decay
        self.w_cat = args.w_categ
        self.w_gauss = args.w_gauss
        self.w_rec = args.w_rec
        self.rec_type = args.rec_type
        self.compute_acc = args.dataset == 'mnist'

        self.num_classes = args.num_classes
        self.gaussian_size = args.gaussian_size
        self.input_size = args.input_size

        output_path = Path(__file__).parents[2] / 'output'
        dir_name = datetime.now().strftime("%d_%m_%Y_H%H_M%M")
        self.exp_path = output_path / dir_name
        self.exp_path.mkdir(exist_ok=True)
        (self.exp_path / 'checkpoints').mkdir(exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.exp_path / 'ts_logs'))

        # gumbel
        self.init_temp = args.init_temp
        self.decay_temp = args.decay_temp
        self.min_temp = args.min_temp
        self.decay_temp_rate = args.decay_temp_rate

        self.network = GMVAENet(self.input_size, self.gaussian_size,
                                self.num_classes, args.init_temp,
                                args.hard_gumbel)
        self.losses = LossFunctions()
        self.metrics = Metrics()

        if self.cuda:
            self.network = self.network.cuda()

    def unlabeled_loss(self, data, out_net):
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

        # reconstruction loss
        loss_rec = self.losses.reconstruction_loss(data, data_recon,
                                                   self.rec_type)

        # gaussian loss
        loss_gauss = self.losses.gaussian_loss(z, mu, var, y_mu, y_var)

        # categorical loss
        loss_cat = -self.losses.entropy(logits, prob_cat) - np.log(0.1)

        # total loss
        loss_total = self.w_rec * loss_rec + self.w_gauss * loss_gauss + self.w_cat * loss_cat

        # obtain predictions
        _, predicted_labels = torch.max(logits, dim=1)

        loss_dic = {
            'total': loss_total,
            'predicted_labels': predicted_labels,
            'reconstruction': loss_rec,
            'gaussian': loss_gauss,
            'categorical': loss_cat
        }
        return loss_dic

    def train_epoch(self, optimizer, data_loader, epoch):
        """Train the model for one epoch

        Args:
            optimizer: (Optim) optimizer to use in backpropagation
            data_loader: (DataLoader) corresponding loader containing the training data

        Returns:
            average of all loss values, accuracy, nmi
        """
        self.network.train()
        total_loss = 0.
        recon_loss = 0.
        cat_loss = 0.
        gauss_loss = 0.

        accuracy = 0.
        nmi = 0.
        num_batches = 0.

        true_labels_list = []
        predicted_labels_list = []

        # iterate over the dataset
        for data, labels in tqdm(data_loader):
            if self.cuda == 1:
                data = data.cuda()

            optimizer.zero_grad()

            # flatten data
            data = data.view(data.size(0), -1)

            # forward call
            out_net = self.network(data)
            unlab_loss_dic = self.unlabeled_loss(data, out_net)
            total = unlab_loss_dic['total']

            # accumulate values
            total_loss += total.item()
            recon_loss += unlab_loss_dic['reconstruction'].item()
            gauss_loss += unlab_loss_dic['gaussian'].item()
            cat_loss += unlab_loss_dic['categorical'].item()

            # perform backpropagation
            total.backward()
            optimizer.step()

            # save predicted and true labels
            predicted = unlab_loss_dic['predicted_labels']
            true_labels_list.append(labels)
            predicted_labels_list.append(predicted)

            num_batches += 1.

            if num_batches % 100 == 99:
                self.writer.add_scalar('train loss', total_loss / num_batches,
                                       epoch * len(data_loader) + num_batches)

            # average per batch
        total_loss /= num_batches
        recon_loss /= num_batches
        gauss_loss /= num_batches
        cat_loss /= num_batches

        # compute metrics
        if self.compute_acc:
            # concat all true and predicted labels
            true_labels = torch.cat(true_labels_list, dim=0).cpu().numpy()
            predicted_labels = torch.cat(predicted_labels_list,
                                         dim=0).cpu().numpy()

            accuracy = 100.0 * self.metrics.cluster_acc(
                predicted_labels, true_labels)
            nmi = 100.0 * self.metrics.nmi(predicted_labels, true_labels)
        else:
            accuracy, nmi = -10.0, -10.0

        return total_loss, recon_loss, gauss_loss, cat_loss, accuracy, nmi

    def test(self, data_loader, epoch=None, return_loss=False):
        """Test the model with new data

        Args:
            data_loader: (DataLoader) corresponding loader containing the test/validation data
            return_loss: (boolean) whether to return the average loss values

        Return:
            accuracy and nmi for the given test data

        """
        self.network.eval()
        total_loss = 0.
        recon_loss = 0.
        cat_loss = 0.
        gauss_loss = 0.

        accuracy = 0.
        nmi = 0.
        num_batches = 0.

        true_labels_list = []
        predicted_labels_list = []

        with torch.no_grad():
            for data, labels in data_loader:
                if self.cuda == 1:
                    data = data.cuda()

                # flatten data
                data = data.view(data.size(0), -1)

                # forward call
                out_net = self.network(data)
                unlab_loss_dic = self.unlabeled_loss(data, out_net)

                # accumulate values
                total_loss += unlab_loss_dic['total'].item()
                recon_loss += unlab_loss_dic['reconstruction'].item()
                gauss_loss += unlab_loss_dic['gaussian'].item()
                cat_loss += unlab_loss_dic['categorical'].item()

                # save predicted and true labels
                predicted = unlab_loss_dic['predicted_labels']
                true_labels_list.append(labels)
                predicted_labels_list.append(predicted)

                num_batches += 1.

                if num_batches % 10 == 9 and epoch is not None:
                    self.writer.add_scalars(
                        'test loss', {
                            'rec': recon_loss / num_batches,
                            'gauss': gauss_loss / num_batches,
                            'cat': cat_loss / num_batches,
                            'total': total_loss / num_batches
                        },
                        epoch * len(data_loader) + num_batches)

        # average per batch
        if return_loss:
            total_loss /= num_batches
            recon_loss /= num_batches
            gauss_loss /= num_batches
            cat_loss /= num_batches

        # concat all true and predicted labels
        true_labels = torch.cat(true_labels_list, dim=0).cpu().numpy()
        predicted_labels = torch.cat(predicted_labels_list,
                                     dim=0).cpu().numpy()

        if epoch is not None:
            self.writer.add_scalars(
                'classes', {
                    str(i): (predicted_labels == i).mean()
                    for i in range(self.num_classes)
                }, epoch)

        # compute metrics
        if self.compute_acc:
            accuracy = 100.0 * self.metrics.cluster_acc(
                predicted_labels, true_labels)
            nmi = 100.0 * self.metrics.nmi(predicted_labels, true_labels)
        else:
            accuracy, nmi = -10.0, -10.0

        if return_loss:
            return total_loss, recon_loss, gauss_loss, cat_loss, accuracy, nmi
        else:
            return accuracy, nmi

    def train(self, train_loader, val_loader):
        """Train the model

        Args:
            train_loader: (DataLoader) corresponding loader containing the training data
            val_loader: (DataLoader) corresponding loader containing the validation data

        Returns:
            output: (dict) contains the history of train/val loss
        """
        optimizer = optim.Adam(self.network.parameters(),
                               lr=self.learning_rate)
        train_history_acc, val_history_acc = [], []
        train_history_nmi, val_history_nmi = [], []

        for epoch in range(1, self.num_epochs + 1):
            t1 = time()
            train_loss, train_rec, train_gauss, train_cat, train_acc, train_nmi = \
                self.train_epoch(optimizer, train_loader, epoch)
            t2 = time() - t1

            val_loss, val_rec, val_gauss, val_cat, val_acc, val_nmi = self.test(
                val_loader, epoch, True)

            # if verbose then print specific information about training
            if self.verbose:
                print("(Epoch %d / %d) t: %.1lf" %
                      (epoch, self.num_epochs, t2))
                print("Train - REC: %.3lf;  Gauss: %.3lf;  Cat: %.3lf;" %
                      (train_rec, train_gauss, train_cat))
                print("Valid - REC: %.3lf;  Gauss: %.3lf;  Cat: %.3lf;" %
                      (val_rec, val_gauss, val_cat))
                print(
                    "Accuracy=Train: %.3lf; Val: %.3lf   NMI=Train: %.3lf; Val: %.3lf   Total Loss=Train: %.3lf; Val: %.3lf" % \
                    (train_acc, val_acc, train_nmi, val_nmi, train_loss, val_loss))
            else:
                print(
                    '(Epoch %d / %d) t: %.1lf Train_Loss: %.3lf; Val_Loss: %.3lf   Train_ACC: %.3lf; Val_ACC: %.3lf   Train_NMI: %.3lf; Val_NMI: %.3lf' % \
                    (epoch, self.num_epochs, t2, train_loss, val_loss, train_acc, val_acc, train_nmi, val_nmi))

            # decay gumbel temperature
            if self.decay_temp == 1:
                self.network.gumbel_temp = np.maximum(
                    self.init_temp * np.exp(-self.decay_temp_rate * epoch),
                    self.min_temp)
                if self.verbose:
                    print("Gumbel Temperature: %.3lf" %
                          self.network.gumbel_temp)

            train_history_acc.append(train_acc)
            val_history_acc.append(val_acc)
            train_history_nmi.append(train_nmi)
            val_history_nmi.append(val_nmi)

            if epoch % 10 == 9:
                torch.save(self.network.cpu(),
                           self.exp_path / 'checkpoints' / f'epoch{epoch}.pth')
                if self.cuda == 1:
                    self.network.cuda()

        torch.save(self.network.cpu(),
                   self.exp_path / 'checkpoints' / f'final.pth')
        if self.cuda == 1:
            self.network.cuda()

        return {
            'train_history_nmi': train_history_nmi,
            'val_history_nmi': val_history_nmi,
            'train_history_acc': train_history_acc,
            'val_history_acc': val_history_acc
        }

    def latent_features(self, data_loader, return_labels=False):
        """Obtain latent features learnt by the model

        Args:
            data_loader: (DataLoader) loader containing the data
            return_labels: (boolean) whether to return true labels or not

        Returns:
           features: (array) array containing the features from the data
        """
        self.network.eval()
        N = len(data_loader.dataset)
        features = np.zeros((N, self.gaussian_size))
        if return_labels:
            true_labels = np.zeros(N, dtype=np.int64)
        start_ind = 0
        with torch.no_grad():
            for (data, labels) in data_loader:
                if self.cuda == 1:
                    data = data.cuda()
                # flatten data
                data = data.view(data.size(0), -1)
                out = self.network.inference(data)
                latent_feat = out['mean']
                end_ind = min(start_ind + data.size(0), N + 1)

                # return true labels
                if return_labels:
                    true_labels[start_ind:end_ind] = labels.cpu().numpy()
                features[start_ind:end_ind] = latent_feat.cpu().detach().numpy(
                )
                start_ind += data.size(0)
        if return_labels:
            return features, true_labels
        return features

    def reconstruct_data(self, data_loader, sample_size=-1):
        """Reconstruct Data

        Args:
            data_loader: (DataLoader) loader containing the data
            sample_size: (int) size of random data to consider from data_loader

        Returns:
            reconstructed: (array) array containing the reconstructed data
        """
        self.network.eval()

        # sample random data from loader
        indices = np.random.randint(0,
                                    len(data_loader.dataset),
                                    size=sample_size)
        test_random_loader = torch.utils.data.DataLoader(
            data_loader.dataset,
            batch_size=sample_size,
            sampler=SubsetRandomSampler(indices))

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

    def plot_latent_space(self, data_loader, save=False):
        """Plot the latent space learnt by the model

        Args:
            data: (array) corresponding array containing the data
            labels: (array) corresponding array containing the labels
            save: (bool) whether to save the latent space plot

        Returns:
            fig: (figure) plot of the latent space
        """
        # obtain the latent features
        features = self.latent_features(data_loader)

        # plot only the first 2 dimensions
        fig = plt.figure(figsize=(8, 6))
        plt.scatter(features[:, 0],
                    features[:, 1],
                    c=labels,
                    marker='o',
                    edgecolor='none',
                    cmap=plt.cm.get_cmap('jet', 10),
                    s=10)
        plt.colorbar()
        if (save):
            fig.savefig('latent_space.png')
        return fig

    def random_generation(self, num_elements=1):
        """Random generation for each category

        Args:
            num_elements: (int) number of elements to generate

        Returns:
            generated data according to num_elements
        """
        # categories for each element
        arr = np.array([])
        for i in range(self.num_classes):
            arr = np.hstack([arr, np.ones(num_elements) * i])
        indices = arr.astype(int).tolist()

        categorical = F.one_hot(torch.tensor(indices),
                                self.num_classes).float()

        if self.cuda:
            categorical = categorical.cuda()

        # infer the gaussian distribution according to the category
        mean, var = self.network.generative.pzy(categorical)

        # gaussian random sample by using the mean and variance
        noise = torch.randn_like(var)
        std = torch.sqrt(var)
        gaussian = mean + noise * std

        # generate new samples with the given gaussian
        generated = self.network.generative.pxz(gaussian)

        return generated.cpu().detach().numpy()
