import torch


class Pooling(torch.nn.Module):

    def __init__(self, kernel_size):
        super(Pooling, self).__init__()
        self.pool = torch.nn.AvgPool1d(kernel_size)

    def forward(self, x):
        if x.dtype in [torch.complex64, torch.complex128]:

            # Separate real and imaginary parts.
            x_real = x.real
            x_imag = x.imag

            # Perform average pooling on real and imaginary parts separately.
            x_real = self.pool(x_real.view(x.shape[0], -1, x.shape[-1]))
            x_imag = self.pool(x_imag.view(x.shape[0], -1, x.shape[-1]))

            # Combine real and imaginary parts back into a complex tensor.
            y = torch.view_as_complex(torch.stack([x_real, x_imag], dim=-1))
        else:
            y = self.pool(x.view(x.shape[0], -1, x.shape[-1]))

        return y.view(x.shape[:-1] + (-1, ))
