import torch
import torch.nn as nn

class System_Func(nn.Module):
    """
    Represents the transition function of an SIR model. This module computes the expected next state
    of the population's compartmental statuses (Susceptible, Infected, Recovered) based on the SIR
    model dynamics. It supports learning the transmission rate (beta) and recovery rate (gamma) if
    they are not provided during initialization.

    Attributes:
        beta (torch.Tensor or torch.nn.Parameter): Transmission rate of the disease. Learned if not provided.
        gamma (torch.Tensor or torch.nn.Parameter): Recovery rate of the disease. Learned if not provided.
    """
    def __init__(self, sir_param_dict):
        """
        Initializes the System_Func module with parameters controlling the dynamics of the SIR model.

        Parameters:
            sir_param_dict (dict): Dictionary containing optional parameters for the model. If any parameter
                                   is set to None, it becomes a learnable parameter of the model.
                                   - beta (float or None): Transmission rate of the disease.
                                   - gamma (float or None): Recovery rate of the disease.
        """
        super().__init__()
        beta = sir_param_dict["beta"]
        gamma = sir_param_dict["gamma"]
        
        if beta == None:
            self.beta_log = nn.Parameter(torch.randn(1).normal_(0, 0.1))
            self.beta = self.param_exp_trans(self.beta_log)
            self.param_beta= True
        else:
            self.beta=beta
            self.param_beta = False
        
        if gamma == None:
            self.gamma_log = nn.Parameter(torch.randn(1).normal_(0, 0.1))
            self.gamma = self.param_exp_trans(self.gamma_log)
            self.param_gamma= True
        else:
            self.gamma=gamma
            self.param_gamma = False    
        
            

    def param_exp_trans(self, x_log):
        """
        Transforms a log-transformed parameter back to its original scale using an exponential function.

        Parameters:
            x_log (torch.Tensor): The log-transformed parameter.

        Returns:
            torch.Tensor: The parameter in its original scale, guaranteed to be positive.
        """
        x = torch.exp(x_log)
        return x
    
    
    def calc_E(self, S, I, R):
        """
        Calculates the next state of the SIR model given the current state.

        Parameters:
            S (torch.Tensor): Current number of susceptible individuals.
            I (torch.Tensor): Current number of infected individuals.
            R (torch.Tensor): Current number of recovered individuals.

        Returns:
            tuple: Next number of susceptible, infected, and recovered individuals.
        """
        S_next = S - self.beta*S * I
        I_next = I + (self.beta*S*I-self.gamma*I)
        R_next = R + self.gamma*I
        return S_next, I_next, R_next
        
        

    def forward(self, SIR):
        """
        Propagates the input through the SIR model to compute the next state.

        Parameters:
            SIR (torch.Tensor): The current state of the system with dimensions (BS, TS, latent_dim=3),
                                where latent_dim includes compartments S, I, R.

        Returns:
            torch.Tensor: The next state of the system, maintaining the same dimensions as the input.
        """
        S = SIR[:, :, 0]#shape:(BS, TS)
        I = SIR[:, :, 1]#shape:(BS, TS)
        R = SIR[:, :, 2]#shape:(BS, TS)
        
        self.beta = self.param_exp_trans(self.beta_log)
        self.gamma = self.param_exp_trans(self.gamma_log)
        
        S_t, I_t, R_t = self.calc_E(S=S, I=I, R=R)

        SIR_t = torch.stack([S_t, I_t, R_t], dim=2)#shape:(BS, TS, latent_dim)
        return SIR_t
    
class Observe_Func(nn.Module):
    """
    A module representing the observation function for an SIR model. In this model,
    the observation function is an identity function that simply returns the second
    dimension of the input tensor. This dimension corresponds to the 'Infected' compartment
    (I), which is assumed to be the observed variable in the SIR model context.
    """

    def __init__(self):
        """
        Initializes the Observe_Func module.
        """
        super().__init__()

    def forward(self, SIR):
        """
        Extracts the observed component (Infected individuals) from the SIR state tensor.

        Parameters:
            SIR (torch.Tensor): The state tensor of the SIR model with dimensions (BS, TS, latent_dim=3),
                                where latent_dim is expected to include Susceptible (S), Infected (I), and Recovered (R)
                                compartments.

        Returns:
            torch.Tensor: A tensor representing the 'Infected' component, extracted directly from the second
                          dimension of the input tensor.
        """
        return SIR[:, :, 1]  # Return the 'Infected' compartment as the observed variable