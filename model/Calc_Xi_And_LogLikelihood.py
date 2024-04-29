import torch
import torch.nn as nn

class Calc_Xi_And_LogLikelihood(nn.Module):
    """
    A module that samples from a given distribution defined by mean (mu) and standard
    deviation (sigma), and computes the log likelihood of the samples. This operation is
    commonly used in variational inference methods to enable gradient backpropagation through
    stochastic nodes.

    Attributes:
        None

    Inputs:
        data_embed (dict): A dictionary containing:
            - mu (Tensor): Mean of the distribution, with shape (BS, num_clusters, TS, latent_dim).
            - sigma (Tensor): Standard deviation of the distribution, with shape (BS, num_clusters, TS, latent_dim).

    Outputs:
        dict: A dictionary containing:
            - xi (Tensor): Samples drawn from the distribution, same shape as mu and sigma.
            - LogLikelihood (Tensor): Log likelihood of each sample, summed over the time series (TS),
              with shape (BS, num_clusters).
    """
    def __init__(self):
        super(Calc_Xi_And_LogLikelihood, self).__init__()
    
    def reparameterize(self, mu, sigma):
        """
        Samples from the distribution defined by mu and sigma using the reparameterization trick,
        which enables gradient descent methods to be used in stochastic variational inference.

        Parameters:
            mu (Tensor): Mean of the Gaussian distribution.
            sigma (Tensor): Standard deviation of the Gaussian distribution.

        Returns:
            Tensor: Samples drawn from the Gaussian distribution.
        """
        eps = torch.randn_like(sigma)  # Standard normal random noise
        xi = mu + (sigma * eps)  # The reparameterized sample
        return xi
    
    def forward(self, data_embed):
        """
        Performs the forward pass to sample xi from the distribution and calculate its log likelihood.

        Parameters:
            data_embed (dict): A dictionary with tensors 'mu' and 'sigma'.

        Returns:
            dict: A dictionary with the sampled xi and their log likelihood values.
        """
        mu = data_embed["mu"]  # Mean tensor
        sigma = data_embed["sigma"] + 0.00001  # Standard deviation tensor, adjusted to avoid division by zero

        xi = self.reparameterize(mu, sigma)  # Sample xi using the reparameterized values

        # Create a diagonal covariance matrix for each point in the batch and cluster
        scale_tril = torch.diag_embed(sigma, dim1=-2, dim2=-1)

        # Multivariate normal distribution to calculate log probability of xi
        normal_dist = torch.distributions.MultivariateNormal(loc=mu, scale_tril=scale_tril)

        # Calculate log probability, sum over the time sequence dimension
        LogLikelihood_xi = normal_dist.log_prob(xi)  # Log likelihood of each sample
        LogLikelihood_xi = LogLikelihood_xi.sum(axis=2)  # Sum over time dimension

        return {"xi": xi, "LogLikelihood": LogLikelihood_xi}
