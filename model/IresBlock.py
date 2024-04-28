import torch
from torch import nn
import numpy as np
from LogDet_Estimator import basic_logdet_estimator, neumann_logdet_estimator, mem_eff_wrapper


class iResBlock(nn.Module):
    """
    Implements an invertible residual block (iResBlock) with options for unbiased gradient estimation
    using the Neumann series approximation and gradient backpropagation during forward passes.

    Attributes:
        nnet (nn.Module): A neural network module that computes the transformation 'g' in the iResBlock.
        geom_p (float): The parameter of the geometric distribution used for stochastic depth.
        n_samples (int): Number of samples drawn from the geometric distribution to determine the depth.
        n_exact_terms (int): The number of terms that are always included from the Neumann series, regardless of sampling.
        neumann_grad (bool): Whether to use the Neumann series approximation for gradient computation.
        grad_in_forward (bool): Whether to compute gradients during the forward pass to reduce memory usage during training.
    """
    def __init__(
        self,
        net,
        geom_p,
        n_samples,
        n_exact_terms,
        neumann_grad,
        grad_in_forward,
    ):
        """
        Initializes the iResBlock with a specific neural network and parameters for managing depth and gradient computations.

        Parameters:
            net (nn.Module): Neural network module to use within the iResBlock.
            geom_p (float): Parameter for the geometric distribution to control the stochastic depth.
            n_samples (int): Number of samples to draw from the geometric distribution for determining the series depth.
            n_exact_terms (int): Number of exact terms from the series to always compute.
            neumann_grad (bool): If True, uses the Neumann series approximation for unbiased gradient estimation.
            grad_in_forward (bool): If True, computes gradients during the forward pass to optimize memory usage.
        """
        nn.Module.__init__(self)
        self.nnet = net
        self.geom_p = nn.Parameter(torch.tensor(np.log(geom_p) - np.log(1.0 - geom_p)))
        self.n_samples = n_samples
        self.n_exact_terms = n_exact_terms
        self.grad_in_forward = grad_in_forward
        self.neumann_grad = neumann_grad

        # store the samples of n.
        self.register_buffer("last_n_samples", torch.zeros(self.n_samples))
        self.register_buffer("last_firmom", torch.zeros(1))
        self.register_buffer("last_secmom", torch.zeros(1))

    def forward(self, x):
        """
        Processes input through the iResBlock, returning both the transformed output and the log determinant
        of the Jacobian of the transformation, which is crucial for models like normalizing flows.

        Parameters:
            x (torch.Tensor): Input tensor to the iResBlock.

        Returns:
            tuple: A tuple containing:
                - The transformed output (x + g(x)).
                - The negative log determinant of the Jacobian (-log |det J|).
        """
        g, logdetgrad = self._logdetgrad(x)
        return x + g, - logdetgrad.view(-1)#(bs, 1)→(bs)


    def _logdetgrad(self, x):
        """
        Computes the transformation 'g' and the logarithm of the determinant of the Jacobian matrix
        for the transformation 'x + g(x)', using sampled depth and exact terms from a geometric distribution.

        Parameters:
            x (torch.Tensor): Input tensor to compute the transformation and Jacobian.

        Returns:
            tuple: A tuple containing:
                - The transformation 'g(x)'.
                - The log determinant of the Jacobian matrix 'log |det J|'.
        """

        with torch.enable_grad():

            geom_p = torch.sigmoid(self.geom_p).item()
            sample_fn = lambda m: geometric_sample(geom_p, m)
            rcdf_fn = lambda k, offset: geometric_1mcdf(geom_p, k, offset)

            if self.training:
                # Unbiased estimation.
                n_samples = sample_fn(self.n_samples)
                n_power_series = max(n_samples) + self.n_exact_terms
                coeff_fn = (
                    lambda k: 1
                    / rcdf_fn(k, self.n_exact_terms)
                    * sum(n_samples >= k - self.n_exact_terms)
                    / len(n_samples)
                )

            else:
                # Unbiased estimation with more exact terms.
                n_samples = sample_fn(self.n_samples)
                n_power_series = max(n_samples) + 20
                coeff_fn = (
                    lambda k: 1
                    / rcdf_fn(k, 20)
                    * sum(n_samples >= k - 20)
                    / len(n_samples)
                )

            ####################################
            # Power series with trace estimator.
            ####################################
            vareps = torch.randn_like(x)

            # Choose the type of estimator.
            if self.training and self.neumann_grad:
                estimator_fn = neumann_logdet_estimator
            else:
                estimator_fn = basic_logdet_estimator

            # Do backprop-in-forward to save memory.
            if self.training and self.grad_in_forward:
                g, logdetgrad = mem_eff_wrapper(
                    estimator_fn,
                    self.nnet,
                    x,
                    n_power_series,
                    vareps,
                    coeff_fn,
                    self.training,
                )
            else:
                x = x.requires_grad_(True)
                g = self.nnet(x)
                logdetgrad = estimator_fn(
                    g, x, n_power_series, vareps, coeff_fn, self.training
                )

            if self.training:
                self.last_n_samples.copy_(
                    torch.tensor(n_samples).to(self.last_n_samples)
                )
                estimator = logdetgrad.detach()
                self.last_firmom.copy_(torch.mean(estimator).to(self.last_firmom))
                self.last_secmom.copy_(torch.mean(estimator**2).to(self.last_secmom))
            return g, logdetgrad.view(-1, 1)

    def extra_repr(self):
        return "n_samples={}, neumann_grad={}".format(
            self.n_samples,
            self.neumann_grad
        )


def geometric_sample(p, n_samples):
    return np.random.geometric(p, n_samples)


def geometric_1mcdf(p, k, offset):
    if k <= offset:
        return 1.0
    else:
        k = k - offset
    """P(n >= k)"""
    return (1 - p) ** max(k - 1, 0)



