import torch
import torch.nn as nn
import torch.nn.functional as F

from SIR_Func import System_Func, Observe_Func
from torch.distributions.dirichlet import Dirichlet
from torch.distributions.gamma import Gamma
from torch.distributions.beta import Beta


class Joint_Probability_Each_Cluster(nn.Module):
    """
    Computes the joint log probability for each cluster within an epidemiological model based on the SIR framework.
    The log probabilities for the latent states (x) given cluster (k) and the observed data (y) given x and k are calculated.
    This class does not handle the summation of log probabilities across clusters, which is done in the Joint_Probability_Multi_Cluster class.

    Attributes:
        system_func (System_Func): System function that models the transition between states in the SIR model.
        observe_func (Observe_Func): Observation function that models how the latent states are observed.
        init_SIR (torch.Tensor): Initial values or distribution parameters for the SIR model states.
    """
    def __init__(self, 
                 base_param_dict, 
                 joint_prob_param_dict, 
                 sir_model_param_dict,
                 GPU=False):
        """
        Initializes the module with SIR model parameters and the parameters for initial, transition, and observation distributions.

        Parameters:
            base_param_dict (dict): Dictionary with the base configuration such as the dimension of the latent space.
            joint_prob_param_dict (dict): Dictionary containing parameters for the model such as orders and initial states, 
                                          where parameters set to None are to be learned.
            sir_model_param_dict (dict): Dictionary containing SIR model parameters like beta and gamma.
            GPU (bool): If True, moves parameters and tensors to the GPU.
        """
        super().__init__()
        self.latent_dim = base_param_dict["latent_dim"]
        self.GPU = GPU
        
        k_order = joint_prob_param_dict["k_order"]
        lambda_order = joint_prob_param_dict["lambda_order"]
        alpha_order = joint_prob_param_dict["alpha_order"]
        init_SIR = joint_prob_param_dict["init_SIR"]
        
        
        normal_dist = torch.distributions.Normal(loc=1, scale=0.1)
        
        if k_order==None:
            self.k_order = nn.Parameter(normal_dist.sample())
            self.param_k_order= True
        else:
            self.k_order = torch.tensor(k_order)
            if self.GPU:
                self.k_order = self.k_order.to("cuda:0")
        
        
        if lambda_order==None:
            self.lambda_order = nn.Parameter(normal_dist.sample())
            self.param_lambda_order= True
        else:
            self.lambda_order = torch.tensor(lambda_order)
            if self.GPU:
                self.lambda_order = self.lambda_order.to("cuda:0")
                
                
        if alpha_order==None:
            self.alpha_order = nn.Parameter(normal_dist.sample())
            self.param_alpha_order= True
        else:
            self.alpha_order = torch.tensor(alpha_order)
            if self.GPU:
                self.alpha_order = self.alpha_order.to("cuda:0")
                
        if init_SIR==None:
            self.init_SIR_seed = nn.Parameter(torch.ones(self.latent_dim)/self.latent_dim)
            self.normalize_init_SIR()
            self.param_init_SIR= True
        else:
            self.init_SIR = torch.tensor(init_SIR)
            if self.GPU:
                self.init_SIR = self.init_SIR.to("cuda:0")
        
        self.system_func = System_Func(sir_model_param_dict)
        self.observe_func = Observe_Func()

    def _normalize(self, x):
        BS, TS, latent_dim = x.shape
        return x/(x.sum(axis=2).reshape(BS, TS, 1))
    
    def normalize_init_SIR(self):
        self.init_SIR = F.softmax(self.init_SIR_seed, dim=0)
        
        
    def forward(self, x, y):
        """
        Calculates and returns the joint log probability of latent states and observed data for each input sequence.

        Parameters:
            x (torch.Tensor): Tensor of latent states with shape (BS, TS, latent_dim).
            y (torch.Tensor): Tensor of observed data with shape (BS, TS, 1).

        Returns:
            torch.Tensor: Tensor of joint log probabilities for each sequence in the batch.
        """
        first_term_LLH = self.calc_first_term_LLH(x)#shape:(BS)
        other_term_LLH = self.calc_other_term_LLH(x)#shape:(BS)
        obs_term_LLH = self.calc_obs_term_LLH(x, y)#shape:(BS)
        return first_term_LLH + other_term_LLH + obs_term_LLH
        
        
    
    def calc_first_term_LLH(self, x):
        self.normalize_init_SIR()
        S_gamma = Gamma(10**self.alpha_order*self.init_SIR[0], 1)
        I_gamma = Gamma(10**self.alpha_order*self.init_SIR[1], 1)
        R_gamma = Gamma(10**self.alpha_order*self.init_SIR[2], 1)
        
        first_term_LLH_S = S_gamma.log_prob(x[:, 0, 0])
        first_term_LLH_I = I_gamma.log_prob(x[:, 0, 1])
        first_term_LLH_R = R_gamma.log_prob(x[:, 0, 2])
        
        return first_term_LLH_S + first_term_LLH_I + first_term_LLH_R
    
    def calc_other_term_LLH(self, x):
        x_norm = self._normalize(x)
        E_t1 = self.system_func(x_norm[:, :-1, :])
        E_t1 = torch.clamp(E_t1, min=0.0000001)
        
        S_gamma = Gamma(10**self.k_order*E_t1[:,:,0], 1)
        I_gamma = Gamma(10**self.k_order*E_t1[:,:,1], 1)
        R_gamma = Gamma(10**self.k_order*E_t1[:,:,2], 1)
        
        other_term_LLH_S = S_gamma.log_prob(x[:, 1:, 0])
        other_term_LLH_I = I_gamma.log_prob(x[:, 1:, 1])
        other_term_LLH_R = R_gamma.log_prob(x[:, 1:, 2])
        
        other_term_LLH = other_term_LLH_S + other_term_LLH_I + other_term_LLH_R
        return other_term_LLH.sum(axis=1)
        
    
    def calc_obs_term_LLH(self, x, y):      
        x_norm = self._normalize(x)
        mu_y = self.observe_func(x_norm)
        obs_beta = Beta(10**self.lambda_order*mu_y, 10**self.lambda_order*(1-mu_y))
        other_term_LLH = obs_beta.log_prob(y[:, :, 0])
        
        return other_term_LLH.sum(axis=1)
    
    
class Joint_Probability_Multi_Cluster(nn.Module):
    """
    A module that computes the joint probability across multiple clusters for the given latent states and observations.
    This module orchestrates multiple Joint_Probability_Each_Cluster modules, one for each cluster, aggregating their
    outputs to compute the overall joint probability of the data. If there are multiple clusters, this module also
    incorporates prior probabilities for each cluster.

    Attributes:
        num_clusters (int): The number of clusters in the model.
        latent_dim (int): The dimension of the latent space.
        ssm_omega (torch.nn.Parameter): Learnable parameters representing the log of the prior probabilities of each cluster.
        joint_probability_each_clusters (nn.ModuleList): A list containing a joint probability model for each cluster.
    """
    def __init__(self, 
                 base_param_dict, 
                 joint_prob_param_dict, 
                 sir_model_param_dict,
                 GPU=False):
        """
        Initializes the Joint_Probability_Multi_Cluster with necessary parameters and configurations for each cluster.

        Parameters:
            base_param_dict (dict): Base parameters including the number of clusters and the dimension of the latent space.
            joint_prob_param_dict (dict): Parameters specific to the joint probability calculations, shared across clusters.
            sir_model_param_dict (dict of dicts): SIR model parameters for each cluster, keyed by cluster index.
            GPU (bool): Indicates whether to use GPU acceleration.
        """
        super().__init__()
        self.num_clusters = base_param_dict["num_clusters"]
        self.latent_dim = base_param_dict["latent_dim"]
        self.GPU = GPU

        
        if self.num_clusters >= 2:
            self.ssm_omega = nn.Parameter(torch.ones(self.num_clusters))
        
        
        self.joint_probability_each_clusters = nn.ModuleList()
        for i in range(self.num_clusters):
            self.joint_probability_each_clusters.append(Joint_Probability_Each_Cluster(
                base_param_dict=base_param_dict, 
                joint_prob_param_dict=joint_prob_param_dict, 
                sir_model_param_dict=sir_model_param_dict[f"c{i}"],
                GPU=self.GPU))
            
    def forward(self, x, y):
        joint_probability_list = []
        for i, joint_probability_each_cluster in enumerate(self.joint_probability_each_clusters):
            tmp_res = joint_probability_each_cluster(x = x[:, i, :, :], y = y)
            joint_probability_list.append(tmp_res)

            
        #結果をスタック
        joint_probability_multi = torch.stack(joint_probability_list, dim=1)#shape:(BS, num_clusters)
        if self.num_clusters >= 2:
            prior_cluster_prob = torch.log(nn.functional.softmax(self.ssm_omega, dim=0))#shape:(num_clusters)
            joint_probability_multi = prior_cluster_prob + joint_probability_multi
        
        return joint_probability_multi
