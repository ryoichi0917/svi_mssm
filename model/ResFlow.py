import numpy as np

import math
import torch

from torch import nn
import torch.nn.init as init
import torch.nn.functional as F
import collections.abc as container_abcs
from itertools import repeat
from Lipschitz import LipschitzCNN1D
from IresBlock import iResBlock
from Other_Flow import EXP_Flow


class NormalizingFlow(nn.Module):
    """
    Implements a Normalizing Flow model, which is a sequence of invertible transformations applied to an initial
    distribution (usually simple to sample from, like a Gaussian) to transform it into a more complex distribution
    that better approximates a target distribution. This is useful in tasks such as density estimation and
    variational inference.

    The class allows chaining multiple flow transformations, each of which must be an invertible module with a 
    computable Jacobian determinant.
    """

    def __init__(self, flows):
        """
        Initializes the Normalizing Flow model with a list of flow transformations.

        Parameters:
            flows (list of nn.Module): A list containing instantiated flow modules, each of which should
                                       be an invertible transformation.
        """
        super().__init__()
        self.flows = nn.ModuleList(flows)  # Store the flows in a ModuleList to properly register them

    def forward(self, z):
        """
        Applies the sequence of flow transformations to the input, computing the transformed output and the 
        cumulative log determinant of the Jacobian matrices of these transformations.

        Parameters:
            z (torch.Tensor): The input tensor to the flow model, typically samples from a simple initial 
                              distribution such as a standard Gaussian.

        Returns:
            tuple: A tuple containing:
                - The transformed tensor after all flow transformations have been applied. This tensor represents 
                  samples from the target distribution.
                - The cumulative log determinant of the Jacobian of the transformations. This value is necessary 
                  for correctly computing probabilities under the transformed distribution, important in scenarios 
                  such as variational inference where the log-likelihood is required.
        """
        log_det = torch.zeros(len(z), device=z.device)  # Initialize log determinant to zero
        for flow in self.flows:
            z, log_d = flow(z)  # Apply each flow in sequence and accumulate the log determinants
            log_det += log_d
        return z, log_det


class Resflow_Each_Cluster(nn.Module):
    """
    Implements a Residual Flow model for each cluster, which involves a sequence of iResBlocks
    each composed of a Lipschitz-constrained 1D convolutional neural network. This module can
    transform data using a specified number of flow layers, where each flow is defined to be 
    Lipschitz continuous. This is useful for tasks that require a stable transformation of data,
    such as variational autoencoders or other generative models.

    This model supports two modes: 'sl' which directly outputs transformed data and 'sir' which
    applies an exponential transformation to the output of the residual flows for further processing.
    """

    def __init__(self, base_param_dict, resflow_param_dict):
        """
        Initializes the Resflow_Each_Cluster model with a dictionary of parameters specifying the 
        configuration of the residual flows and any additional transformation layers.

        Parameters:
            resflow_param_dict (dict): Configuration dictionary with the following keys:
                - num_flow_module (int): Number of flow modules to stack.
                - kernel_size (int): Kernel size for the 1D convolutional layers.
                - dims (list of int): Dimension sizes for each layer in the CNNs within each flow.
                - bias (bool): Whether to add a bias term in the CNN layers.
                - coeff (float): Coefficient for spectral normalization.
                - n_iterations (int or None): Number of iterations for computing spectral normalization.
                - tolerance (float): Tolerance for the spectral normalization process.
                - reduce_memory (bool): Whether to use memory reduction techniques in flow computations.
                - mode (str): Operational mode of the model, either 'sl' for standard output or 'sir'
                  for applying an exponential transformation post residual flows.
        """
        super().__init__()
        num_flow_module = resflow_param_dict["num_flow_module"]
        dims = resflow_param_dict["dims"]
        kernel_size = resflow_param_dict["kernel_size"]
        self.mode = base_param_dict["mode"]

        # Build flow modules
        kernel_size_list = [kernel_size] * (len(dims) - 1)
        flows = []
        for i in range(num_flow_module):
            net = LipschitzCNN1D(
                channels=dims,
                kernel_size=kernel_size_list,
                lipschitz_const=resflow_param_dict["coeff"],
                max_lipschitz_iter=resflow_param_dict["n_iterations"],
                lipschitz_tolerance=resflow_param_dict["tolerance"]
            )
            flows.append(iResBlock(
                net=net,
                geom_p=0.5,
                n_samples=1,
                n_exact_terms=2,
                neumann_grad=resflow_param_dict["reduce_memory"],
                grad_in_forward=resflow_param_dict["reduce_memory"]
            ))
        self.cnn_res = NormalizingFlow(flows=flows)

        if self.mode == "sir":
            self.exp_flow = EXP_Flow()

    def forward(self, xi):
        """
        Transforms the input tensor through the sequence of residual flows and, depending on the mode,
        applies an exponential transformation to the output.

        Parameters:
            xi (torch.Tensor): Input tensor with shape (BS, TS, latent_dim).

        Returns:
            dict: A dictionary containing:
                - 'z': The transformed tensor.
                - 'log_det': The cumulative log determinant of the Jacobian matrices from the transformations.
                - 'z_norm' (optional): Normalized transformed tensor, provided only in 'sir' mode.
        """
        xi = xi.permute(0, 2, 1)  # (BS, latent_dim, TS)
        z, log_det = self.cnn_res(xi)  # Apply residual flows
        z = z.permute(0, 2, 1)  # Permute back to (BS, TS, latent_dim)

        if self.mode == "sl":
            return {"z": z, "log_det": log_det}

        elif self.mode == "sir":
            z, log_det_exp = self.exp_flow(z)  # Apply exponential transformation
            log_det += log_det_exp  # Update log determinant
            BS, TS, latent_dim = z.shape
            z_norm = z / (z.sum(dim=2, keepdim=True))  # Normalize z
            return {"z": z, "log_det": log_det, "z_norm": z_norm}


class Resflow_Multi_Cluster(nn.Module):
    """
    A neural network module that applies a set of Resflow_Each_Cluster models, one for each cluster, 
    to input data. This model is designed to process multi-cluster data where each cluster is treated 
    as a separate dimension in the input tensor. Each cluster-specific model is an instance of the 
    Resflow_Each_Cluster class, capable of transforming data using residual flows with optional
    spectral normalization and exponential transformations.

    Attributes:
        flow_models (nn.ModuleList): A list of Resflow_Each_Cluster models, one for each cluster.
    """

    def __init__(self, base_param_dict, resflow_param_dict):
        """
        Initializes the Resflow_Multi_Cluster model with specified parameters for each flow model
        per cluster.

        Parameters:
            base_param_dict (dict): Dictionary with base configuration parameters including:
                - input_dim (int): Dimension of input features.
                - latent_dim (int): Dimension of the latent space.
                - num_clusters (int): Number of clusters or separate models to instantiate.
                - TS (int): Number of time steps in each input sequence.
            
            resflow_param_dict (dict): Configuration for the residual flow modules including:
                - num_flow_module (int): Number of flow modules to stack.
                - kernel_size (int): Kernel size for the 1D CNNs in each flow module.
                - dims (list of int): Dimensions for each layer in the 1D CNNs.
                - bias (bool): Whether to add a bias term in the CNN layers.
                - coeff (float): Coefficient for spectral normalization.
                - n_iterations (int or None): Number of iterations for spectral normalization.
                - tolerance (float): Tolerance for spectral normalization.
                - reduce_memory (bool): Whether to use memory reduction techniques.
                - mode (str): Mode of operation, either 'sl' (simple latent) or 'sir' (scaled implicit reparam).
        """
        super().__init__()
        self.mode = base_param_dict["mode"]
        self.flow_models = nn.ModuleList()
        for i in range(base_param_dict["num_clusters"]):
            self.flow_models.append(Resflow_Each_Cluster(base_param_dict,
                                                         resflow_param_dict))

    def forward(self, x):
        """
        Processes the input tensor through each cluster-specific flow model, aggregating the outputs.

        Parameters:
            x (torch.Tensor): Input tensor with dimensions (BS, num_clusters, TS, latent_dim).

        Returns:
            dict: A dictionary containing:
                - 'z': Tensor with transformed outputs from each flow model stacked along the cluster dimension.
                - 'log_det': Tensor with log determinants of Jacobian matrices from each flow model stacked along the cluster dimension.
                - 'z_norm' (optional): Normalized output tensors provided only in 'sir' mode, stacked along the cluster dimension.
        """
        flow_out_list = []
        for i, flow_model in enumerate(self.flow_models):
            tmp_out = flow_model(x[:, i, :, :])
            flow_out_list.append(tmp_out)

        z = [flow_out["z"] for flow_out in flow_out_list]
        log_det = [flow_out["log_det"] for flow_out in flow_out_list]
        
        if self.mode == "sl":
            return {"z": torch.stack(z, dim=1), "log_det": torch.stack(log_det, dim=1)}
        elif self.mode == "sir":
            z_norm = [flow_out["z_norm"] for flow_out in flow_out_list]
            return {"z": torch.stack(z, dim=1), 
                    "log_det": torch.stack(log_det, dim=1), 
                    "z_norm": torch.stack(z_norm, dim=1)}
            
