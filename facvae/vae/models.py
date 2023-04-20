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
class Encoder(torch.nn.Module):
    def __init__(self, x_shape, z_dim, y_dim, hidden_dim, nlayer):
        super(Encoder, self).__init__()

        # q(y|x)
        self.inference_qyx = torch.nn.ModuleDict({
            scale: torch.nn.ModuleList([
                torch.nn.Linear(x_shape[scale][1], hidden_dim, bias=False),
                torch.nn.BatchNorm1d(x_shape[scale][0]),
                torch.nn.LeakyReLU(negative_slope=0.2),
                View((-1, x_shape[scale][0] * hidden_dim)),
                torch.nn.Linear(x_shape[scale][0] * hidden_dim,
                                hidden_dim,
                                bias=False),
                torch.nn.BatchNorm1d(hidden_dim),
                torch.nn.LeakyReLU(negative_slope=0.2),
            ])
            for scale in x_shape.keys()
        })
        for scale in x_shape.keys():
            for i in range(1, nlayer):
                self.inference_qyx[scale].append(
                    torch.nn.Linear(hidden_dim, hidden_dim, bias=False))
                self.inference_qyx[scale].append(
                    torch.nn.BatchNorm1d(hidden_dim))
                self.inference_qyx[scale].append(
                    torch.nn.LeakyReLU(negative_slope=0.2))
            self.inference_qyx[scale] = torch.nn.Sequential(
                *self.inference_qyx[scale])

        self.gumbel_softmax = torch.nn.ModuleDict({
            scale: GumbelSoftmax(hidden_dim, y_dim)
            for scale in x_shape.keys()
        })

        # q(z|y,x)
        self.inference_qzyx = torch.nn.ModuleDict({
            scale: torch.nn.ModuleList([
                torch.nn.Linear(x_shape[scale][1] + y_dim,
                                hidden_dim,
                                bias=False),
                torch.nn.BatchNorm1d(x_shape[scale][0]),
                torch.nn.LeakyReLU(negative_slope=0.2),
                View((-1, x_shape[scale][0] * hidden_dim)),
                torch.nn.Linear(x_shape[scale][0] * hidden_dim,
                                hidden_dim,
                                bias=False),
                torch.nn.BatchNorm1d(hidden_dim),
                torch.nn.LeakyReLU(negative_slope=0.2)
            ])
            for scale in x_shape.keys()
        })
        for scale in x_shape.keys():
            for i in range(1, nlayer):
                self.inference_qzyx[scale].append(
                    torch.nn.Linear(hidden_dim, hidden_dim, bias=False))
                self.inference_qzyx[scale].append(
                    torch.nn.BatchNorm1d(hidden_dim))
                self.inference_qzyx[scale].append(
                    torch.nn.LeakyReLU(negative_slope=0.2))
            self.inference_qzyx[scale] = torch.nn.Sequential(
                *self.inference_qzyx[scale])

        self.gaussian = torch.nn.ModuleDict(
            {scale: Gaussian(hidden_dim, z_dim)
             for scale in x_shape.keys()})

    # q(y|x)
    def qyx(self, x, temperature, hard):
        ys = {scale: self.inference_qyx[scale](x[scale]) for scale in x.keys()}
        ys = {
            scale: self.gumbel_softmax[scale](ys[scale], temperature, hard)
            for scale in ys.keys()
        }
        return ys

    # q(z|x,y)
    def qzxy(self, x, y):
        xy = {
            scale: torch.cat((x[scale], y[scale].unsqueeze(1).repeat(
                1, x[scale].shape[1], 1)),
                             dim=2)
            for scale in x.keys()
        }
        xy = {
            scale: self.inference_qzyx[scale](xy[scale])
            for scale in xy.keys()
        }
        xy = {scale: self.gaussian[scale](xy[scale]) for scale in xy.keys()}
        return xy

    def forward(self, x, temperature=1.0, hard=0):
        # q(y|x)
        # logits, prob, y = self.qyx(x, temperature, hard)
        qyx = self.qyx(x, temperature, hard)
        logits = {scale: qyx[scale][0] for scale in qyx.keys()}
        prob = {scale: qyx[scale][1] for scale in qyx.keys()}
        y = {scale: qyx[scale][2] for scale in qyx.keys()}

        # # Release the GPU memory of qyx.
        # del qyx
        # torch.cuda.empty_cache()

        # q(z|x,y)
        qzxy = self.qzxy(x, y)
        mu = {scale: qzxy[scale][0] for scale in qzxy.keys()}
        var = {scale: qzxy[scale][1] for scale in qzxy.keys()}
        z = {scale: qzxy[scale][2] for scale in qzxy.keys()}

        # # Release the GPU memory of qzxy.
        # del qzxy
        # torch.cuda.empty_cache()

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
class Decoder(torch.nn.Module):
    def __init__(self, x_shape, z_dim, y_dim, hidden_dim, nlayer):
        super(Decoder, self).__init__()

        # p(z|y): GMMs for each scale.
        self.y_mu = torch.nn.ModuleDict(
            {scale: torch.nn.Linear(y_dim, z_dim)
             for scale in x_shape.keys()})
        self.y_var = torch.nn.ModuleDict(
            {scale: torch.nn.Linear(y_dim, z_dim)
             for scale in x_shape.keys()})

        # Latent variable nonlinear combination.
        self.generative_mix_z = torch.nn.Sequential(
            torch.nn.Linear(len(x_shape) * z_dim,
                            len(x_shape) * z_dim,
                            bias=False),
            torch.nn.BatchNorm1d(len(x_shape) * z_dim),
            torch.nn.LeakyReLU(negative_slope=0.2),
            torch.nn.Linear(len(x_shape) * z_dim,
                            len(x_shape) * z_dim,
                            bias=False),
            torch.nn.BatchNorm1d(len(x_shape) * z_dim),
            torch.nn.LeakyReLU(negative_slope=0.2),
            torch.nn.Linear(len(x_shape) * z_dim,
                            len(x_shape) * z_dim,
                            bias=False),
            torch.nn.BatchNorm1d(len(x_shape) * z_dim),
            torch.nn.LeakyReLU(negative_slope=0.2),
        )

        # p(x|z)
        self.generative_pxz = torch.nn.ModuleDict({
            scale: torch.nn.ModuleList([
                torch.nn.Linear(z_dim, hidden_dim, bias=False),
                torch.nn.BatchNorm1d(hidden_dim),
                torch.nn.LeakyReLU(negative_slope=0.2),
                torch.nn.Linear(hidden_dim,
                                x_shape[scale][0] * hidden_dim,
                                bias=False),
                torch.nn.BatchNorm1d(x_shape[scale][0] * hidden_dim),
                torch.nn.LeakyReLU(negative_slope=0.2),
            ])
            for scale in x_shape.keys()
        })
        for scale in x_shape.keys():
            for i in range(1, nlayer):
                self.generative_pxz[scale].append(
                    torch.nn.Linear(x_shape[scale][0] * hidden_dim,
                                    x_shape[scale][0] * hidden_dim,
                                    bias=False))
                self.generative_pxz[scale].append(
                    torch.nn.BatchNorm1d(x_shape[scale][0] * hidden_dim))
                self.generative_pxz[scale].append(
                    torch.nn.LeakyReLU(negative_slope=0.2))
            self.generative_pxz[scale].append(
                View((-1, x_shape[scale][0], hidden_dim)))
            self.generative_pxz[scale].append(
                torch.nn.Linear(hidden_dim, x_shape[scale][1]))
            self.generative_pxz[scale] = torch.nn.Sequential(
                *self.generative_pxz[scale])

    # p(z|y)
    def pzy(self, y):
        y_mu = {scale: self.y_mu[scale](y[scale]) for scale in y.keys()}
        y_var = {
            scale: F.softplus(self.y_var[scale](y[scale])) + 1.0
            for scale in y.keys()
        }
        return y_mu, y_var

    def nonlinear_z_mix(self, z):
        split_sizes = [z[scale].shape[1] for scale in z.keys()]
        nb = {scale: z[scale].shape[0] for scale in z.keys()}
        nrepeat = {scale: max(nb.values()) // nb[scale] for scale in z.keys()}
        repeat_idx = {
            scale: torch.arange(z[scale].shape[0],
                                device=z[scale].device).repeat_interleave(
                                    nrepeat[scale])
            for scale in z.keys()
        }
        z_out = {
            scale: torch.index_select(z[scale], 0, repeat_idx[scale])
            for scale in z.keys()
        }

        z_out = self.generative_mix_z(
            torch.cat([z_out[scale] for scale in z.keys()], dim=1))

        z_out = torch.split(z_out, split_sizes, dim=1)
        z_out = {scale: z_out[i] for i, scale in enumerate(z.keys())}
        z_out = {
            scale: torch.mean(z_out[scale].view(-1, nrepeat[scale],
                                                z[scale].shape[-1]),
                              dim=1)
            for scale in z.keys()
        }
        return z_out

    # p(x|z)
    def pxz(self, z):
        pxz = {
            scale: self.generative_pxz[scale](z[scale])
            for scale in z.keys()
        }
        return pxz

    def forward(self, z, y):
        # p(z|y)
        y_mu, y_var = self.pzy(y)

        # p(x|z)
        z = self.nonlinear_z_mix(z)
        x_rec = self.pxz(z)

        output = {'y_mean': y_mu, 'y_var': y_var, 'x_rec': x_rec}
        return output


class FactorialVAE(torch.nn.Module):
    def __init__(
        self,
        x_shape,
        z_dim,
        y_dim,
        init_temp,
        hard_gumbel=0,
        hidden_dim=512,
        nlayer=3,
    ):
        super(FactorialVAE, self).__init__()

        self.inference = Encoder(x_shape, z_dim, y_dim, hidden_dim, nlayer)
        self.generative = Decoder(x_shape, z_dim, y_dim, hidden_dim, nlayer)
        self.gumbel_temp = init_temp
        self.hard_gumbel = hard_gumbel

    def forward(self, x):
        out_inf = self.inference(x, self.gumbel_temp, self.hard_gumbel)
        z, y = out_inf['gaussian'], out_inf['categorical']
        out_gen = self.generative(z, y)

        # merge output. out_inf.update(out_gen)?
        output = out_inf
        for key, value in out_gen.items():
            output[key] = value
        return output
