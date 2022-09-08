import numpy as np
import torch
from torch.nn import functional as F

from facvae.vae.layers import GumbelSoftmax, Gaussian

hidden_dim = 512


# Inference Network
class InferenceNet(torch.nn.Module):

    def __init__(self, x_dim, z_dim, y_dim, hidden_dim=512):
        super(InferenceNet, self).__init__()

        # q(y|x)
        self.inference_qyx = torch.nn.ModuleList([
            torch.nn.Linear(x_dim, hidden_dim),
            torch.nn.LeakyReLU(negative_slope=0.2),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.LeakyReLU(negative_slope=0.2),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.LeakyReLU(negative_slope=0.2),
            GumbelSoftmax(hidden_dim, y_dim)
        ])

        # q(z|y,x)
        self.inference_qzyx = torch.nn.ModuleList([
            torch.nn.Linear(x_dim + y_dim, hidden_dim),
            torch.nn.LeakyReLU(negative_slope=0.2),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.LeakyReLU(negative_slope=0.2),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.LeakyReLU(negative_slope=0.2),
            Gaussian(hidden_dim, z_dim)
        ])

    # q(y|x)
    def qyx(self, x, temperature, hard):
        num_layers = len(self.inference_qyx)
        for i, layer in enumerate(self.inference_qyx):
            if i == num_layers - 1:
                # last layer is gumbel softmax
                x = layer(x, temperature, hard)
            else:
                x = layer(x)
        return x

    # q(z|x,y)
    def qzxy(self, x, y):
        concat = torch.cat((x, y), dim=1)
        for layer in self.inference_qzyx:
            concat = layer(concat)
        return concat

    def forward(self, x, temperature=1.0, hard=0):
        # x = Flatten(x)

        # q(y|x)
        logits, prob, y = self.qyx(x, temperature, hard)

        # q(z|x,y)
        mu, var, z = self.qzxy(x, y)

        output = {
            'mean': mu,
            'var': var,
            'gaussian': z,
            'logits': logits,
            'prob_cat': prob,
            'categorical': y
        }
        return output


class Tanh(torch.nn.Module):

    def __init__(self, scale=1.0):
        super(Tanh, self).__init__()
        self.scale = scale

    def forward(self, x):
        return self.scale * torch.tanh(x)


# Generative Network
class GenerativeNet(torch.nn.Module):

    def __init__(self, x_dim, z_dim, y_dim, hidden_dim=512):
        super(GenerativeNet, self).__init__()

        # p(z|y)
        self.y_mu = torch.nn.Linear(y_dim, z_dim)
        self.y_var = torch.nn.Linear(y_dim, z_dim)

        # p(x|z)
        self.generative_pxz = torch.nn.ModuleList([
            torch.nn.Linear(z_dim, hidden_dim),
            torch.nn.LeakyReLU(negative_slope=0.2),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.LeakyReLU(negative_slope=0.2),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.LeakyReLU(negative_slope=0.2),
            torch.nn.Linear(hidden_dim, x_dim),
            # torch.nn.Sigmoid(),
            # nn.LeakyReLU(negative_slope=0.2),
            # Tanh(scale=2 * np.pi),
        ])

    # p(z|y)
    def pzy(self, y):
        y_mu = self.y_mu(y)
        y_var = F.softplus(self.y_var(y))
        return y_mu, y_var

    # p(x|z)
    def pxz(self, z):
        for layer in self.generative_pxz:
            z = layer(z)
        return z

    def forward(self, z, y):
        # p(z|y)
        y_mu, y_var = self.pzy(y)

        # p(x|z)
        x_rec = self.pxz(z)

        output = {'y_mean': y_mu, 'y_var': y_var, 'x_rec': x_rec}
        return output


# GMVAE Network
class GMVAENetwork(torch.nn.Module):

    def __init__(self, x_dim, z_dim, y_dim, init_temp, hard_gumbel):
        super(GMVAENetwork, self).__init__()

        self.inference = InferenceNet(x_dim, z_dim, y_dim)
        self.generative = GenerativeNet(x_dim, z_dim, y_dim)

        self.gumbel_temp = init_temp
        self.hard_gumbel = hard_gumbel

        # weight initialization
        for m in self.modules():
            if type(m) == torch.nn.Linear or type(
                    m) == torch.nn.Conv2d or type(
                        m) == torch.nn.ConvTranspose2d:
                torch.nn.init.xavier_normal_(m.weight)
                if m.bias.data is not None:
                    torch.nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        out_inf = self.inference(x, self.gumbel_temp, self.hard_gumbel)
        z, y = out_inf['gaussian'], out_inf['categorical']
        out_gen = self.generative(z, y)

        # merge output
        output = out_inf
        for key, value in out_gen.items():
            output[key] = value
        return output
