import torch
import torch.nn as nn

class EXP_Flow(nn.Module):
    """
    A neural network module that applies an exponential transformation to its input.
    This transformation is often used in flow-based generative models where transformations
    need to be invertible and it's necessary to calculate the Jacobian determinant for the
    transformation efficiently.

    The exponential function is applied element-wise, and the log of the Jacobian determinant
    of this transformation is simply the sum of the input tensor, as the derivative of the
    exponential function is the exponential function itself.
    """

    def __init__(self):
        """
        Initializes the EXP_Flow module. This module does not have any parameters.
        """
        super().__init__()

    def forward(self, z):
        """
        Applies an exponential transformation to the input tensor and calculates the
        log determinant of the Jacobian matrix of this transformation.

        Parameters:
            z (torch.Tensor): The input tensor to the transformation.

        Returns:
            tuple: A tuple containing:
                - The transformed tensor after applying the exponential function.
                - The negative log determinant of the Jacobian matrix, which is needed
                  for change of variables in flow-based models.
        """
        z_exp = torch.exp(z)  # Element-wise exponential transformation
        log_det = z.sum(dim=[1, 2])  # Sum over all dimensions except the batch dimension
        return z_exp, -log_det