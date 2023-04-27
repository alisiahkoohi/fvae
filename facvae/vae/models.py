from typing import Tuple, Dict

import torch
from torch.nn import functional as F

from facvae.vae.layers import GumbelSoftmax, Gaussian


class SkipConnection(torch.nn.Module):

    def __init__(self, module):
        super(SkipConnection, self).__init__()
        self.module = module

    def forward(self, x):
        return x + self.module(x)


class View(torch.nn.Module):
    """
    A module to create a view of an existing torch.Tensor (avoids copying).
    Attributes:
        shape: A tuple containing the desired shape of the view.
    """

    def __init__(self, *shape: int) -> None:
        """
        Initializes a Concat module.

        Args:
            shape (Tuple[int]): A tuple containing the desired shape of the
                view.
        """
        super().__init__()
        self.shape = shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Creates a view of an input.
        Args:
            x (torch.Tensor): A tensor of shape (batch_size, *input_shape).
        Returns:
            torch.Tensor: A tensor of shape (batch_size, *self.shape).
        """
        # Use the `view` method of the input tensor to create a view with the
        # desired shape.
        return x.view(*self.shape)


class Encoder(torch.nn.Module):
    """
    A module to create an encoder network for the FactorialVAE.
    """

    def __init__(
        self,
        x_shape: dict,
        z_dim: int,
        y_dim: int,
        hidden_dim: int,
        nlayer: int,
    ) -> None:
        """
        Initializes the Encoder module.

        Args:
            x_shape (dict): Dictionary containing the shapes of input tensors
                at different scales. Each key is a string representing the
                scale of the tensor, and each value is a tuple of integers
                representing the shape of the tensor at that scale.
            z_dim (int): Dimensionality of the latent variable z.
            y_dim (int): Dimensionality of the categorical variable y.
            hidden_dim (int): Dimensionality of the hidden layer.
            nlayer (int): Number of layers in the neural network.

        Returns:
            None
        """
        super(Encoder, self).__init__()

        # Common component multiscale data representation
        self.common = torch.nn.ModuleList([
            SkipConnection(
                torch.nn.Sequential(
                    torch.nn.Linear(sum(
                        [x_shape[scale][1] for scale in x_shape.keys()]),
                                    hidden_dim,
                                    bias=False),
                    torch.nn.BatchNorm1d(list(x_shape.values())[0][0]),
                    torch.nn.LeakyReLU(negative_slope=0.2),
                    torch.nn.Linear(
                        hidden_dim,
                        sum([x_shape[scale][1] for scale in x_shape.keys()]),
                        bias=False))) for _ in range(nlayer)
        ])
        self.common = torch.nn.Sequential(*self.common)

        # q(y|x)
        self.inference_qyx = torch.nn.ModuleDict({
            # Iterate over each scale in x_shape dictionary
            scale:
            torch.nn.ModuleList([
                # Create linear layer to map x to hidden_dim features
                torch.nn.Linear(x_shape[scale][1], hidden_dim, bias=False),
                # Batch normalization layer to normalize the output of previous
                # layer
                torch.nn.BatchNorm1d(x_shape[scale][0]),
                # Leaky ReLU activation function with negative slope of 0.2
                torch.nn.LeakyReLU(negative_slope=0.2),
                # Reshape the output of previous layer into a 2D tensor
                View((-1, x_shape[scale][0] * hidden_dim)),
                # Linear layer to map the 2D tensor to hidden_dim features
                torch.nn.Linear(x_shape[scale][0] * hidden_dim,
                                hidden_dim,
                                bias=False),
                # Batch normalization layer to normalize the output of previous
                # layer
                torch.nn.BatchNorm1d(hidden_dim),
                # Leaky ReLU activation function with negative slope of 0.2
                torch.nn.LeakyReLU(negative_slope=0.2),
            ])
            # End of ModuleList for this scale
            for scale in x_shape.keys()
        })

        # Iterate over each scale in the dictionary of x_shape
        for scale in x_shape.keys():
            # Iterate over each layer except the first one
            for i in range(1, nlayer):
                # Add a linear layer to inference_qyx at the current scale
                self.inference_qyx[scale].append(
                    torch.nn.Linear(hidden_dim, hidden_dim, bias=False))
                # Add batch normalization layer to inference_qyx at the current
                # scale
                self.inference_qyx[scale].append(
                    torch.nn.BatchNorm1d(hidden_dim))
                # Add a LeakyReLU activation layer to inference_qyx at the
                # current scale
                self.inference_qyx[scale].append(
                    torch.nn.LeakyReLU(negative_slope=0.2))
            # Create a sequential module from the layers added to inference_qyx
            # at the current scale
            self.inference_qyx[scale] = torch.nn.Sequential(
                *self.inference_qyx[scale])

        # Create a dictionary of GumbelSoftmax modules for each scale in
        # x_shape
        self.gumbel_softmax = torch.nn.ModuleDict({
            scale:
            GumbelSoftmax(hidden_dim, y_dim)
            for scale in x_shape.keys()
        })

        # q(z|y,x)
        self.inference_qzyx = torch.nn.ModuleDict({
            scale:
            torch.nn.ModuleList([
                # Add a linear layer to the current scale's ModuleList with
                # input dimensions of y_dim + x_shape[scale][1] and output
                # dimensions of hidden_dim
                torch.nn.Linear(x_shape[scale][1] + y_dim,
                                hidden_dim,
                                bias=False),
                # Add a batch normalization layer to the current scale's
                # ModuleList with input dimensions of x_shape[scale][0]
                torch.nn.BatchNorm1d(x_shape[scale][0]),
                # Add a LeakyReLU activation layer to the current scale's
                # ModuleList
                torch.nn.LeakyReLU(negative_slope=0.2),
                # Reshape the input tensor to a 2D matrix
                View((-1, x_shape[scale][0] * hidden_dim)),
                # Add another linear layer to the current scale's ModuleList
                # with input dimensions of x_shape[scale][0] * hidden_dim and
                # output dimensions of hidden_dim
                torch.nn.Linear(x_shape[scale][0] * hidden_dim,
                                hidden_dim,
                                bias=False),
                # Add another batch normalization layer to the current scale's
                # ModuleList
                torch.nn.BatchNorm1d(hidden_dim),
                # Add another LeakyReLU activation layer to the current scale's
                # ModuleList
                torch.nn.LeakyReLU(negative_slope=0.2)
            ])
            for scale in x_shape.keys()
        })

        # Add more layers to the q(z|y,x) ModuleList for each scale in the
        # dictionary of x_shape
        for scale in x_shape.keys():
            for i in range(1, nlayer):
                # Add a linear layer to the current scale's ModuleList with
                # input and output dimensions of hidden_dim
                self.inference_qzyx[scale].append(
                    torch.nn.Linear(hidden_dim, hidden_dim, bias=False))
                # Add a batch normalization layer to the current scale's
                # ModuleList with input dimensions of hidden_dim
                self.inference_qzyx[scale].append(
                    torch.nn.BatchNorm1d(hidden_dim))
                # Add a LeakyReLU activation layer to the current scale's
                # ModuleList
                self.inference_qzyx[scale].append(
                    torch.nn.LeakyReLU(negative_slope=0.2))
            # Create a sequential module from the layers added to the current
            # scale's ModuleList
            self.inference_qzyx[scale] = torch.nn.Sequential(
                *self.inference_qzyx[scale])

        # Create a dictionary of Gaussian modules for each scale in x_shape
        self.gaussian = torch.nn.ModuleDict(
            {scale: Gaussian(hidden_dim, z_dim)
             for scale in x_shape.keys()})

    def common_encoder(self, x: Dict[str,
                                     torch.Tensor]) -> Dict[str, torch.Tensor]:
        # Get the sizes of the input tensors
        split_sizes = [x[scale].shape[-1] for scale in x.keys()]

        # Concatenate the input tensors along the last dimension
        x_out = torch.cat([x[scale] for scale in x.keys()], dim=-1)

        # Pass through the common component multiscale data representation.
        x_out = self.common(x_out)

        # Split the concatenated tensor back into individual tensors and
        # reshape them
        x_out = torch.split(x_out, split_sizes, dim=-1)
        x_out = {scale: x_out[i] for i, scale in enumerate(x.keys())}

        # Return the mixed and transformed tensors
        return x_out

    def qyx(self, x: Dict[str, torch.Tensor], temperature: float,
            hard: bool) -> Dict[str, torch.Tensor]:
        """
        Computes the posterior q(y|x) for each scale in x using the Gumbel
        Softmax trick.

        Args:
            x (Dict[str, torch.Tensor]): dictionary of input tensors with each
                key representing a scale.
            temperature (float): temperature parameter used in the Gumbel
                Softmax trick.
            hard (bool): whether to use a hard sample from the Gumbel Softmax
                distribution or not

        Returns:
            ys (Dict[str, torch.Tensor]): dictionary of tensors representing
                the posterior q(y|x) for each scale in x
        """
        # Compute the output of the inference_qyx module for each scale in x
        ys = {scale: self.inference_qyx[scale](x[scale]) for scale in x.keys()}

        # Apply the Gumbel Softmax trick to each output tensor
        ys = {
            scale: self.gumbel_softmax[scale](ys[scale], temperature, hard)
            for scale in ys.keys()
        }

        return ys

    def qzxy(
        self, x: Dict[str, torch.Tensor], y: Dict[str, torch.Tensor]
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Compute the approximate posterior distribution q(z|x,y).

        Args:
            x (Dict[str, torch.Tensor]): Dictionary of tensors x at different
                scales.
            y (Dict[str, torch.Tensor]): Dictionary of tensors y at different
                scales.

        Returns:
            Dict[str, Tuple[torch.Tensor, torch.Tensor]]: Dictionary of tuples
                representing the mean and logvariance of the Gaussian
                distribution at each scale in x.
        """
        # Concatenate x and y tensors along the channel dimension for each
        # scale
        xy = {
            scale:
            torch.cat((x[scale], y[scale].unsqueeze(1).repeat(
                1, x[scale].shape[1], 1)),
                      dim=2)
            for scale in x.keys()
        }
        # Pass the concatenated tensors through the q(z|y,x) network
        xy = {
            scale: self.inference_qzyx[scale](xy[scale])
            for scale in xy.keys()
        }
        # Compute the mean and logvariance of the Gaussian distribution at each
        # scale
        xy = {scale: self.gaussian[scale](xy[scale]) for scale in xy.keys()}
        return xy

    def forward(self,
                x: Dict[str, torch.Tensor],
                temperature: float = 1.0,
                hard: int = 0) -> Dict[str, Dict[str, torch.Tensor]]:
        """Compute the forward pass of the model.

        Args:
            x (Dict[str, torch.Tensor]): Dictionary containing the input
            tensors at
                different scales.
            temperature (float): Temperature parameter for the Gumbel-Softmax
                distribution (default 1.0).
            hard (int): Flag indicating whether to use the "hard" version of
                the Gumbel-Softmax distribution (default 0).

        Returns:
            Dict[str, Dict[str, torch.Tensor]]: Dictionary containing the
                output tensors at different scales.
                - 'mean': Dictionary containing the mean tensors of the
                      Gaussian distributions at different scales.
                - 'var': Dictionary containing the variance tensors of the
                      Gaussian distributions at different scales.
                - 'gaussian': Dictionary containing the latent variable tensors
                      sampled from the Gaussian distributions at different
                      scales.
                - 'logits': Dictionary containing the logits tensors of the
                      categorical distributions at different scales.
                - 'prob_cat': Dictionary containing the probability tensors of
                      the categorical distributions at different scales.
                - 'categorical': Dictionary containing the sampled categorical
                      variable tensors at different scales.
        """
        # Common encoder.
        x = self.common_encoder(x)

        # q(y|x)
        qyx = self.qyx(x, temperature, hard)
        # Get logits, probabilities, and samples from categorical distribution
        # at each scale
        logits = {scale: qyx[scale][0] for scale in qyx.keys()}
        prob = {scale: qyx[scale][1] for scale in qyx.keys()}
        y = {scale: qyx[scale][2] for scale in qyx.keys()}

        # q(z|x,y)
        qzxy = self.qzxy(x, y)
        # Get mean, variance, and samples from the Gaussian distribution at
        # each scale
        mu = {scale: qzxy[scale][0] for scale in qzxy.keys()}
        var = {scale: qzxy[scale][1] for scale in qzxy.keys()}
        z = {scale: qzxy[scale][2] for scale in qzxy.keys()}

        # Store the results in a dictionary and return it
        output = {
            'mean': mu,
            'var': var,
            'gaussian': z,
            'logits': logits,
            'prob_cat': prob,
            'categorical': y
        }
        return output


class Decoder(torch.nn.Module):
    """
    A module to create a decoder network for the FactorialVAE.
    """

    def __init__(self, x_shape: Dict[str, Tuple[int, int, int]], z_dim: int,
                 y_dim: int, hidden_dim: int, nlayer: int) -> None:
        """
        Decoder network that generates an image given the latent variables z
        and label y.

        Args:
            x_shape (Dict[str, Tuple[int, int, int]]): A dictionary with keys
                representing different scales of the image and values
                representing the shape of the image at each scale.
            z_dim (int): Dimensionality of the latent variable z.
            y_dim (int): Dimensionality of the label variable y.
            hidden_dim (int): Dimensionality of the hidden layers.
            nlayer (int): Number of hidden layers.
        """
        super(Decoder, self).__init__()

        # p(z|y): GMMs for each scale.
        self.y_mu = torch.nn.ModuleDict(
            {scale: torch.nn.Linear(y_dim, z_dim)
             for scale in x_shape.keys()})
        self.y_var = torch.nn.ModuleDict(
            {scale: torch.nn.Linear(y_dim, z_dim)
             for scale in x_shape.keys()})

        # Define the p(x|z) generative network architecture.
        self.generative_pxz = torch.nn.ModuleDict({
            scale:
            torch.nn.ModuleList([
                # First layer: linear transformation from z_dim to hidden_dim.
                torch.nn.Linear(z_dim, hidden_dim, bias=False),
                # Batch normalization layer.
                torch.nn.BatchNorm1d(hidden_dim),
                # Activation function.
                torch.nn.LeakyReLU(negative_slope=0.2),
                # Second layer: linear transformation from hidden_dim to
                # x_shape[scale][0] * hidden_dim.
                torch.nn.Linear(hidden_dim,
                                x_shape[scale][0] * hidden_dim,
                                bias=False),
                # Batch normalization layer.
                torch.nn.BatchNorm1d(x_shape[scale][0] * hidden_dim),
                # Activation function.
                torch.nn.LeakyReLU(negative_slope=0.2),
            ])
            for scale in x_shape.keys()
        })

        # Add additional layers to the generative network architecture.
        for scale in x_shape.keys():
            for i in range(1, nlayer):
                self.generative_pxz[scale].append(
                    # Additional linear layer with x_shape[scale][0] *
                    # hidden_dim input and output dimensions.
                    torch.nn.Linear(x_shape[scale][0] * hidden_dim,
                                    x_shape[scale][0] * hidden_dim,
                                    bias=False))
                self.generative_pxz[scale].append(
                    # Additional batch normalization layer.
                    torch.nn.BatchNorm1d(x_shape[scale][0] * hidden_dim))
                self.generative_pxz[scale].append(
                    # Additional activation function.
                    torch.nn.LeakyReLU(negative_slope=0.2))
            self.generative_pxz[scale].append(
                # Reshape the output to match the desired output shape.
                View((-1, x_shape[scale][0], hidden_dim)))
            self.generative_pxz[scale].append(
                # Final linear layer with hidden_dim input dimension and
                # x_shape[scale][1] output dimension.
                torch.nn.Linear(hidden_dim, x_shape[scale][1]))
            self.generative_pxz[scale] = torch.nn.Sequential(
                # Create a sequential model by combining all layers of the
                # generative network architecture
                *self.generative_pxz[scale])

    def pzy(
        self, y: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Calculate the mean and variance of the distribution p(z|y) for each
        scale.

        Args:
            y (dict of torch.Tensor): A dictionary containing tensors of shape
            (batch_size, 1) for each scale.

        Returns:
            Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]: A tuple
                containing two dictionaries. The first dictionary contains the
                mean of the distribution p(z|y) for each scale. The second
                dictionary contains the variance of the distribution p(z|y) for
                each scale.
        """
        # Calculate the mean of p(z|y) for each scale
        y_mu = {scale: self.y_mu[scale](y[scale]) for scale in y.keys()}
        # Calculate the variance of p(z|y) for each scale
        y_var = {
            scale: F.softplus(self.y_var[scale](y[scale])) + 1.0
            for scale in y.keys()
        }
        return y_mu, y_var

    def pxz(self, z: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Computes the likelihood of data given the latent variable z.

        Args:
            z (Dict[str, torch.Tensor]): A dictionary of tensors with keys
                representing the different scales of the inputs.

        Returns:
            Dict[str, torch.Tensor]: A dictionary of tensors representing the
                probability distribution of the data given the latent variable
                z for each scale.
        """
        # Compute the probability distribution for each scale.
        pxz = {
            scale: self.generative_pxz[scale](z[scale])
            for scale in z.keys()
        }
        return pxz

    def forward(self, z: Dict[str, torch.Tensor],
                y: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Computes the output of the model given the input and condition.

        Args:
            z (Dict[str, torch.Tensor]): A dictionary of tensors with keys
                representing the different scales of the latent variable.
            y (torch.Tensor): A tensor representing the condition.

        Returns:
            Dict[str, torch.Tensor]: A dictionary of tensors representing the
            model output.
        """
        # Compute the probability distribution of the condition
        y_mu, y_var = self.pzy(y)

        # Compute the probability distribution of the data given the latent
        # variable
        x_rec = self.pxz(z)

        # Return the model output
        output = {'y_mean': y_mu, 'y_var': y_var, 'x_rec': x_rec}
        return output


class FactorialVAE(torch.nn.Module):
    """
    A module implementing a factorial variational autoencoder (FactorialVAE).

    Args:
        x_shape (tuple): The shape of the input data.
        z_dim (int): The dimensionality of the continuous latent variable.
        y_dim (int): The dimensionality of the categorical latent variable.
        init_temp (float): The initial value of the Gumbel-Softmax temperature
            parameter.
        hard_gumbel (bool): Whether to use the hard Gumbel-Softmax estimator.
        hidden_dim (int): The dimensionality of the hidden layers.
        nlayer (int): The number of hidden layers.

    Attributes:
        inference (Encoder): The encoder network.
        generative (Decoder): The decoder network.
        gumbel_temp (float): The Gumbel-Softmax temperature parameter.
        hard_gumbel (bool): Whether to use the hard Gumbel-Softmax estimator.
    """

    def __init__(self,
                 x_shape: tuple,
                 z_dim: int,
                 y_dim: int,
                 init_temp: float,
                 hard_gumbel: bool = False,
                 hidden_dim: int = 512,
                 nlayer: int = 3) -> None:
        super(FactorialVAE, self).__init__()

        # Instantiate the encoder and decoder networks.
        self.inference = Encoder(x_shape, z_dim, y_dim, hidden_dim, nlayer)
        self.generative = Decoder(x_shape, z_dim, y_dim, hidden_dim, nlayer)

        # Initialize the Gumbel-Softmax temperature parameter and hard Gumbel
        # flag.
        self.gumbel_temp = init_temp
        self.hard_gumbel = hard_gumbel

    def forward(self, x: torch.Tensor) -> dict:
        """
        Forward pass of the FactorialVAE.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            dict: A dictionary containing the output of the inference and
                generative networks.
        """
        # Pass the input data through the encoder network.
        out_inf = self.inference(x, self.gumbel_temp, self.hard_gumbel)

        # Extract the continuous and categorical latent variables.
        z, y = out_inf['gaussian'], out_inf['categorical']

        # Pass the latent variables through the decoder network.
        out_gen = self.generative(z, y)

        # Combine the output of the encoder and decoder networks.
        # out_inf.update(out_gen)?
        output = out_inf
        for key, value in out_gen.items():
            output[key] = value
        return output
