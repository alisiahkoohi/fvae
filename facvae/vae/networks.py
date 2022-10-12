import torch
from torch.nn import functional as F

from facvae.vae.layers import GumbelSoftmax, Gaussian


class View(torch.nn.Module):
    """A module to create a view of an existing torch.Tensor (avoid copying).
    Attributes:
        shape: A tuple containing the desired shape of the view.
    """

    def __init__(self, *shape):
        """Initializes a Concat module.
        Args:
            shape: A tuple containing the desired shape of the view.
        """
        super().__init__()
        self.shape = shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Creates a view of an input.
        Args:
            x: A torch.Tensor object.
        Returns:
            A torch.Tensor containing the view of the input with given
                dimensions.
        """
        return x.view(*self.shape)


# Inference Network
class InferenceNet(torch.nn.Module):

    def __init__(self, x_dim, z_dim, y_dim, hidden_dim, nlayer):
        super(InferenceNet, self).__init__()

        # q(y|x)
        self.inference_qyx = torch.nn.ModuleList([
            torch.nn.Linear(x_dim, hidden_dim, bias=False),
            torch.nn.BatchNorm1d(hidden_dim),
            torch.nn.LeakyReLU(negative_slope=0.2)
        ])
        for i in range(1, nlayer):
            self.inference_qyx.append(
                torch.nn.Linear(hidden_dim, hidden_dim, bias=False))
            self.inference_qyx.append(torch.nn.BatchNorm1d(hidden_dim))
            self.inference_qyx.append(torch.nn.LeakyReLU(negative_slope=0.2))
        self.inference_qyx.append(GumbelSoftmax(hidden_dim, y_dim))
        self.inference_qyx = torch.nn.Sequential(*self.inference_qyx)

        # q(z|y,x)
        self.inference_qzyx = torch.nn.ModuleList([
            torch.nn.Linear(x_dim + y_dim, hidden_dim, bias=False),
            torch.nn.BatchNorm1d(hidden_dim),
            torch.nn.LeakyReLU(negative_slope=0.2)
        ])
        for i in range(1, nlayer):
            self.inference_qzyx.append(
                torch.nn.Linear(hidden_dim, hidden_dim, bias=False))
            self.inference_qzyx.append(torch.nn.BatchNorm1d(hidden_dim))
            self.inference_qzyx.append(torch.nn.LeakyReLU(negative_slope=0.2))
        self.inference_qzyx.append(Gaussian(hidden_dim, z_dim))
        self.inference_qzyx = torch.nn.Sequential(*self.inference_qzyx)

    # q(y|x)
    def qyx(self, x, temperature, hard):
        nlayer = len(self.inference_qyx)
        for i, layer in enumerate(self.inference_qyx):
            if i == nlayer - 1:
                # last layer is gumbel softmax
                x = layer(x, temperature, hard)
            else:
                x = layer(x)
        return x

    # q(z|x,y)
    def qzxy(self, x, y):
        concat = torch.cat((x, y), dim=1)
        return self.inference_qzyx(concat)

    def forward(self, x, temperature=1.0, hard=0):
        # from IPython import embed; embed()

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


# Generative Network
class GenerativeNet(torch.nn.Module):

    def __init__(self, x_dim, z_dim, y_dim, hidden_dim, nlayer):
        super(GenerativeNet, self).__init__()

        # p(z|y)
        self.y_mu = torch.nn.Linear(y_dim, z_dim)
        self.y_var = torch.nn.Linear(y_dim, z_dim)

        # p(x|z)
        self.generative_pxz = torch.nn.ModuleList([
            torch.nn.Linear(z_dim, hidden_dim, bias=False),
            torch.nn.BatchNorm1d(hidden_dim),
            torch.nn.LeakyReLU(negative_slope=0.2)
        ])
        for i in range(1, nlayer):
            self.generative_pxz.append(
                torch.nn.Linear(hidden_dim, hidden_dim, bias=False))
            self.generative_pxz.append(torch.nn.BatchNorm1d(hidden_dim))
            self.generative_pxz.append(torch.nn.LeakyReLU(negative_slope=0.2))
        self.generative_pxz.append(torch.nn.Linear(hidden_dim, x_dim))
        self.generative_pxz = torch.nn.Sequential(*self.generative_pxz)

    # p(z|y)
    def pzy(self, y):
        y_mu = self.y_mu(y)
        y_var = F.softplus(self.y_var(y))
        return y_mu, y_var

    # p(x|z)
    def pxz(self, z):
        return self.generative_pxz(z)

    def forward(self, z, y):
        # p(z|y)
        y_mu, y_var = self.pzy(y)

        # p(x|z)
        x_rec = self.pxz(z)

        output = {'y_mean': y_mu, 'y_var': y_var, 'x_rec': x_rec}
        return output


# GMVAE Network
class GMVAENetwork(torch.nn.Module):

    def __init__(self,
                 x_dim,
                 z_dim,
                 y_dim,
                 init_temp,
                 hard_gumbel=0,
                 hidden_dim=512,
                 nlayer=3,
                 batchnorm=0):
        super(GMVAENetwork, self).__init__()

        self.inference = InferenceNet(x_dim, z_dim, y_dim, hidden_dim, nlayer)
        self.generative = GenerativeNet(x_dim, z_dim, y_dim, hidden_dim,
                                        nlayer)
        if batchnorm:
            self.bn = torch.nn.BatchNorm1d(x_dim,
                                           affine=True,
                                           track_running_stats=True,
                                           eps=1e-5)

        self.gumbel_temp = init_temp
        self.hard_gumbel = hard_gumbel

    def forward(self, x):
        x = self.bn(x) if hasattr(self, 'bn') else x
        out_inf = self.inference(x, self.gumbel_temp, self.hard_gumbel)
        z, y = out_inf['gaussian'], out_inf['categorical']
        out_gen = self.generative(z, y)

        # merge output
        output = out_inf
        for key, value in out_gen.items():
            output[key] = value
        return output
