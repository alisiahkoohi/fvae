import torch
from torch.nn import functional as F


# Flatten layer
class Flatten(torch.nn.Module):

    def forward(self, x):
        return x.view(x.size(0), -1)


# Reshape layer
class Reshape(torch.nn.Module):

    def __init__(self, outer_shape):
        super(Reshape, self).__init__()
        self.outer_shape = outer_shape

    def forward(self, x):
        return x.view(x.size(0), *self.outer_shape)


# Sample from the Gumbel-Softmax distribution and optionally discretize.
class GumbelSoftmax(torch.nn.Module):

    def __init__(self, f_dim, c_dim):
        super(GumbelSoftmax, self).__init__()
        self.logits = torch.nn.Linear(f_dim, c_dim)
        self.f_dim = f_dim
        self.c_dim = c_dim

    def sample_gumbel(self, shape, is_cuda=False, eps=1e-20):
        U = torch.rand(shape)
        if is_cuda:
            U = U.cuda()
        return -torch.log(-torch.log(U + eps) + eps)

    def gumbel_softmax_sample(self, logits, temperature):
        y = logits + self.sample_gumbel(logits.size(), logits.is_cuda)
        return F.softmax(y / temperature, dim=-1)

    def gumbel_softmax(self, logits, temperature, hard=False):
        """
        ST-gumple-softmax
        input: [*, n_class]
        return: flatten --> [*, n_class] an one-hot vector
        """
        # categorical_dim = 10
        y = self.gumbel_softmax_sample(logits, temperature)

        if not hard:
            return y

        shape = y.size()
        _, ind = y.max(dim=-1)
        y_hard = torch.zeros_like(y).view(-1, shape[-1])
        y_hard.scatter_(1, ind.view(-1, 1), 1)
        y_hard = y_hard.view(*shape)
        # Set gradients w.r.t. y_hard gradients w.r.t. y
        y_hard = (y_hard - y).detach() + y
        return y_hard

    def forward(self, x, temperature=1.0, hard=False):
        logits = self.logits(x).view(-1, self.c_dim)
        prob = F.softmax(logits, dim=-1)
        y = self.gumbel_softmax(logits, temperature, hard)
        return logits, prob, y


# Sample from a Gaussian distribution
class Gaussian(torch.nn.Module):

    def __init__(self, in_dim, z_dim):
        super(Gaussian, self).__init__()
        self.mu = torch.nn.Linear(in_dim, z_dim, bias=False)
        self.var = torch.nn.Linear(in_dim, z_dim, bias=False)

    def reparameterize(self, mu, var):
        std = torch.sqrt(var + 1e-10)
        noise = torch.randn_like(std)
        z = mu + noise * std
        return z

    def forward(self, x):
        mu = self.mu(x)
        var = F.softplus(self.var(x))
        z = self.reparameterize(mu, var)
        return mu, var, z


class ResidualUnit(torch.nn.Module):

    def __init__(self, n_channels, dilation=1):
        super().__init__()

        self.dilation = dilation

        self.layers = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=n_channels,
                            out_channels=n_channels,
                            kernel_size=21,
                            dilation=dilation,
                            padding='same',
                            padding_mode='reflect'), torch.nn.ELU(),
            torch.nn.Conv1d(in_channels=n_channels,
                            out_channels=n_channels,
                            kernel_size=1,
                            padding='same',
                            padding_mode='reflect'))

    def forward(self, x):
        return x + self.layers(x)


class EncoderBlock(torch.nn.Module):

    def __init__(self, in_channels, hidden_channels, out_channels, stride):
        super().__init__()

        self.layers = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=in_channels,
                            out_channels=hidden_channels,
                            kernel_size=21,
                            stride=stride), ResidualUnit(hidden_channels),
            torch.nn.ELU(), ResidualUnit(hidden_channels), torch.nn.ELU(),
            ResidualUnit(hidden_channels), torch.nn.ELU(),
            torch.nn.Conv1d(in_channels=hidden_channels,
                            out_channels=out_channels,
                            kernel_size=2 * stride,
                            stride=stride))

    def forward(self, x):
        return self.layers(x)
