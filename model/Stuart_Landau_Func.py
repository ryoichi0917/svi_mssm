import torch
import torch.nn as nn

class System_Func(nn.Module):
    """
    Represents the transition function of a Stuart-Landau model. This module computes the expected next state of
    the latent variables based on the Stuart-Landau differential equations. It supports learning some of the 
    model parameters if they are not provided during initialization.

    Attributes:
        a1 (torch.Tensor or torch.nn.Parameter): Parameter controlling the linear part of the x-component.
        a2 (torch.Tensor or torch.nn.Parameter): Parameter controlling the nonlinear part of the x-component.
        omega1 (torch.Tensor): Positive parameter controlling the coupling between x and y components, 
                               enforced to be non-negative through exponential transformation.
        omega2 (torch.Tensor or torch.nn.Parameter): Parameter controlling the coupling and rotation dynamics.
    """
    def __init__(self, sl_model_param_dict):
        """
        Initializes the System_Func module with parameters for the Stuart-Landau model's transition function.

        Parameters:
            sl_model_param_dict (dict): Dictionary containing values or None for parameters a1, a2, omega1, omega2.
                                        Parameters set to None will be learned during training.
        """
        super().__init__()
        a1 = sl_model_param_dict["sl_a1"]
        a2 = sl_model_param_dict["sl_a2"]
        omega1 = sl_model_param_dict["sl_omega1"]
        omega2 = sl_model_param_dict["sl_omega2"]
        

        if a1 == None:
            self.a1 = nn.Parameter(torch.randn(1).normal_(0, 0.1))
            self.param_a1 = True
        else:
            self.a1=a1
            self.param_a1 = False
        

        if a2 == None:
            self.a2 = nn.Parameter(torch.randn(1).normal_(0, 0.1))
            self.param_a2 = True
        else:
            self.a2=a2
            self.param_a2 = False
            

        if omega1 == None:
            self.omega1_log = nn.Parameter(torch.randn(1).normal_(0, 0.1))
            self.omega1 = self.param_exp_trans(self.omega1_log)
            self.param_omega1 = True
        else:
            self.omega1=a1
            self.param_omega1 = False
            

        if omega2 == None:
            self.omega2 = nn.Parameter(torch.randn(1).normal_(0, 0.1))
            self.param_omega2 = True
        else:
            self.omega2=omega2
            self.param_omega2 = False
            

    def param_exp_trans(self, x_log):
        x_log = torch.exp(x_log)
        return x_log
        
    def calc_x(self, x, y):
        x = x+self.a1*x - self.omega1*y - (x**2+y**2)*(self.a2*x-self.omega2*y)
        return x
    
    def calc_y(self, x, y):
        y = y+self.a1*y + self.omega1*x - (x**2+y**2)*(self.a2*y+self.omega2*y)
        return y
    
    def calc_E(self, x, y):
        x, y = self.calc_x(x, y),  self.calc_y(x, y)
        return x, y
        
        

    def forward(self, xy):
        
        x = xy[:, :, 0]#shape:(BS, TS)
        y = xy[:, :, 1]#shape:(BS, TS)
        
        self.omega1 = self.param_exp_trans(self.omega1_log)
        
        x_t, y_t = self.calc_E(x=x, y=y)

        xy_t = torch.stack([x_t, y_t], dim=2)#shape:(BS, TS, latent_dim)
        return xy_t
    
class Observe_Func(nn.Module):
    """
    A module representing the observation function for a Stuart-Landau model.
    In this model, the observation function is an identity function that simply returns
    the second dimension of the input tensor, which corresponds to the observed variable 'y'.
    This function is used to model how the latent states are observed or measured, which in this
    case is a direct observation without any transformation.
    """

    def __init__(self):
        """
        Initializes the Observe_Func module.
        """
        super().__init__()

    def forward(self, xy):
        """
        Processes the input tensor to extract the observed component.

        Parameters:
            xy (torch.Tensor): The input tensor with dimensions (BS, TS, latent_dim), where
                               the last dimension is expected to be at least 2, representing at least
                               two variables (x, y) of which 'y' is observed.

        Returns:
            torch.Tensor: A tensor representing the observed 'y' component, extracted from the second
                          dimension of the input tensor.
        """
        return xy[:, :, 1]  # Return the second component of the input tensor, corresponding to 'y'