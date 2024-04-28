import torch
import torch.nn as nn

from Stuart_Landau_Func import System_Func, Observe_Func

class Joint_Probability_Each_Cluster(nn.Module):
    """
    Attributes:
        C (torch.nn.Parameter): Lower triangular matrix of the Cholesky decomposition representing
                               the covariance of the latent variable's initial state.
        A (torch.nn.Parameter): Lower triangular matrix of the Cholesky decomposition for the
                               covariance matrix of the system function.
        B (torch.nn.Parameter or torch.Tensor): The observation noise parameter, learned if None 
                               is provided in initialization.
        mu (torch.nn.Parameter): Mean of the initial state of the latent variables.
        system_func (System_Func): The system function that models the dynamics of the latent state.
        observe_func (Observe_Func): The observation function that models how the latent state is observed.
    """

    def __init__(self, base_param_dict, joint_prob_param_dict, sl_model_param_dict, GPU=False):
        """
        Initializes the Joint_Probability_Each_Cluster module with specified parameters for 
        the distribution of initial states and dynamics, as well as observation parameters.

        Parameters:
            base_param_dict (dict): Contains basic configuration such as the dimension of the latent space.
                - latent_dim (int): Dimension of the latent space.

            joint_prob_param_dict (dict): Contains parameters for the observation error and possibly initial state.
                - B (float or None): Observation noise parameter, if None, B is learned.

            sl_model_param_dict (dict): Parameters for the system function model contained within.
            
            GPU (bool, optional): If True, tensors will be moved to GPU. Default is False.
        """
        super().__init__()
        self.latent_dim = base_param_dict["latent_dim"]
        self.GPU = GPU

        # Parameters for the initial state and dynamics
        halfnormal_dist = torch.distributions.HalfNormal(scale=0.1)
        C_seed = halfnormal_dist.sample() + 1
        A_seed = halfnormal_dist.sample() + 1
        
        self.C = nn.Parameter(torch.tril(torch.eye(self.latent_dim) * C_seed))
        self.A = nn.Parameter(torch.tril(torch.eye(self.latent_dim) * A_seed))
        self.mu = nn.Parameter(torch.randn(2).normal_(0, 0.1))

        # Define observation noise parameter B
        B = joint_prob_param_dict.get("B", None)
        if B is None:
            B_seed = halfnormal_dist.sample() + 1
            self.B = nn.Parameter(B_seed)
        else:
            self.B = torch.tensor(B, dtype=torch.float32)
            if self.GPU:
                self.B = self.B.to("cuda:0")

        # System and observation functions
        self.system_func = System_Func(sl_model_param_dict)
        self.observe_func = Observe_Func()

    def forward(self, x, y):
        """
        Calculates the joint log-probability for the inputs given the model parameters.

        Parameters:
            x (torch.Tensor): The latent variables tensor of shape (BS, TS, latent_dim).
            y (torch.Tensor): The observed variables tensor of shape (BS, TS, 1).

        Returns:
            torch.Tensor: The joint log-probability of the latent and observed variables.
        """
        first_term_LLH = self.calc_first_term_LLH(x)
        other_term_LLH = self.calc_other_term_LLH(x)
        obs_term_LLH = self.calc_obs_term_LLH(x, y)

        return first_term_LLH + other_term_LLH + obs_term_LLH

    def calc_first_term_LLH(self, x):
        """
        Calculates the log-probability of the first term of the latent state sequence.

        Parameters:
            x (torch.Tensor): The input tensor for the latent states.

        Returns:
            torch.Tensor: Log-probability of the first term.
        """
        normal_dist = torch.distributions.MultivariateNormal(loc=self.mu, scale_tril=self.C)
        return normal_dist.log_prob(x[:, 0, :])

    def calc_other_term_LLH(self, x):
        """
        Calculates the log-probability of the remaining terms of the latent state sequence based on the system dynamics.

        Parameters:
            x (torch.Tensor): The input tensor for the latent states.

        Returns:
            torch.Tensor: Sum of log-probabilities of all terms except the first.
        """
        mu_t1 = self.system_func(x[:, :-1, :])  # Predict next state
        normal_dist = torch.distributions.MultivariateNormal(loc=mu_t1, scale_tril=self.A)
        return normal_dist.log_prob(x[:, 1:, :]).sum(dim=1)

    def calc_obs_term_LLH(self, x, y):
        """
        Calculates the log-probability of the observed data given the latent states.

        Parameters:
            x (torch.Tensor): The latent states.
            y (torch.Tensor): The observed data.

        Returns:
            torch.Tensor: Log-probability of the observed data.
        """
        mu_y = self.observe_func(x)  # Predict observations from latent states
        cov_matrix = torch.eye(mu_y.shape[1]) * self.B**2 if not self.GPU else torch.eye(mu_y.shape[1], device="cuda:0") * self.B**2
        normal_dist = torch.distributions.MultivariateNormal(loc=mu_y, covariance_matrix=cov_matrix)
        return normal_dist.log_prob(y[:, :, 0])
    
    

class Joint_Probability_Multi_Cluster(nn.Module):
    """
    Attributes:
        num_clusters (int): The number of clusters.
        latent_dim (int): The dimension of the latent space.
        ssm_omega (torch.nn.Parameter, optional): A parameter representing the unnormalized log probabilities 
                                                  of each cluster, used only if num_clusters is 2 or more.
        joint_probability_each_clusters (nn.ModuleList): A list of Joint_Probability_Each_Cluster modules,
                                                         one for each cluster.
    """

    def __init__(self, 
                 base_param_dict, 
                 joint_prob_param_dict, 
                 sl_model_param_dict,
                 GPU=False):
        """
        Initializes the Joint_Probability_Multi_Cluster module with specified parameters and a list of
        joint probability models for each cluster.

        Parameters:
            base_param_dict (dict): Contains basic configuration such as the number of clusters and the dimension of the latent space.
                - num_clusters (int): Number of clusters.
                - latent_dim (int): Dimension of the latent space.
            
            joint_prob_param_dict (dict of dicts): A dictionary where each key corresponds to a cluster index prefixed by 'c' 
                                                  and each value is a dictionary of parameters for the joint probability model of that cluster.
            
            sl_model_param_dict (dict of dicts): A dictionary where each key corresponds to a cluster index prefixed by 'c' 
                                                and each value is a dictionary of parameters for the system function model of that cluster.

            GPU (bool): If True, moves parameters and tensors to the GPU.
        """
        super().__init__()
        self.num_clusters = base_param_dict["num_clusters"]
        self.latent_dim = base_param_dict["latent_dim"]
        self.GPU = GPU

        # Initialize prior probabilities if there are multiple clusters
        if self.num_clusters >= 2:
            self.ssm_omega = nn.Parameter(torch.ones(self.num_clusters))  # Unnormalized log probabilities

        # Create a module for each cluster to calculate its joint probability
        self.joint_probability_each_clusters = nn.ModuleList([
            Joint_Probability_Each_Cluster(
                base_param_dict=base_param_dict, 
                joint_prob_param_dict=joint_prob_param_dict[f"c{i}"], 
                sl_model_param_dict=sl_model_param_dict[f"c{i}"],
                GPU=GPU
            ) for i in range(self.num_clusters)
        ])

    def forward(self, x, y):
        """
        Calculates the joint log probability for all clusters given the input data.

        Parameters:
            x (torch.Tensor): The input tensor of latent variables with shape (BS, num_clusters, TS, latent_dim).
            y (torch.Tensor): The observed data tensor with shape (BS, TS, obs_dim).

        Returns:
            torch.Tensor: The tensor containing the joint log probability for each cluster, with shape (BS, num_clusters).
                          If there are multiple clusters, includes the contribution from the log prior probabilities.
        """
        joint_probability_list = [jp(x=x[:, i, :, :], y=y) for i, jp in enumerate(self.joint_probability_each_clusters)]
        joint_probability_multi = torch.stack(joint_probability_list, dim=1)  # Stack results along the cluster dimension

        # Add log prior probabilities if there are multiple clusters
        if self.num_clusters >= 2:
            prior_cluster_prob = torch.log_softmax(self.ssm_omega, dim=0)  # Convert log weights to log probabilities
            joint_probability_multi += prior_cluster_prob.unsqueeze(0)  # Broadcast and add to all batches

        return joint_probability_multi
