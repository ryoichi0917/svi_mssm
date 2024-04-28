from torch import nn
import torch

from Embedding_LSTM import LSTM_Embedder
from Calc_Xi_And_LogLikelihood import Calc_Xi_And_LogLikelihood
from ResFlow import Resflow_Multi_Cluster
from Joint_Prob_SL import Joint_Probability_Multi_Cluster as Joint_Probability_Multi_Cluster_SL
from Joint_Prob_SIR import Joint_Probability_Multi_Cluster as Joint_Probability_Multi_Cluster_SIR


class SGVB_SSM_TS_Clustering_Model_With_NF(nn.Module):
    """
    A variational Bayesian model for time series clustering with Normalizing Flows (NF) in a State Space Model (SSM) setting.
    This model integrates LSTM-based embedding, Residual Flows for distribution transformation, and joint probability computation
    across multiple clusters. It supports both 'sl' and 'sir' modes, potentially adapting the model for different types of
    data or assumptions about the underlying processes.

    Attributes:
        num_clusters (int): Number of clusters in the model.
        mode (str): Operating mode, which could be 'sl' or 'sir', determining which joint probability model to use.
        GPU (bool): Flag to utilize GPU acceleration.
    """
    def __init__(self, 
                 base_param_dict,
                 lstm_param_dict,
                 resflow_param_dict,
                 model_param_dict,
                 joint_prob_param_dict,
                 GPU=False):
        """
        Initializes the clustering model with all necessary components and configurations.

        Parameters:
            base_param_dict (dict): Basic configuration parameters like number of clusters and mode.
            lstm_param_dict (dict): Parameters for the LSTM embedder.
            resflow_param_dict (dict): Parameters for the residual flow models.
            model_param_dict (dict): Specific model parameters, differing based on the mode ('sl' or 'sir').
            joint_prob_param_dict (dict): Parameters for calculating the joint probability.
            GPU (bool): If True, utilize CUDA for operations.
        """
        super().__init__()
        self.num_clusters = base_param_dict["num_clusters"]
        self.mode = base_param_dict["mode"]
        self.GPU = GPU
    
        
        self.lstm_embedder = LSTM_Embedder(base_param_dict=base_param_dict, 
                                           lstm_param_dict=lstm_param_dict, 
                                           GPU=self.GPU)
        
        self.calc_xi_and_loglikelihood = Calc_Xi_And_LogLikelihood()
        
        self.resflow_multi_cluster = Resflow_Multi_Cluster(base_param_dict = base_param_dict,
                                                           resflow_param_dict = resflow_param_dict
                                                          )
        if self.mode == "sl":
            self.joint_probability_multi_cluster = Joint_Probability_Multi_Cluster_SL(base_param_dict=base_param_dict, 
                                                                                   joint_prob_param_dict=joint_prob_param_dict, 
                                                                                   sl_model_param_dict=model_param_dict,
                                                                                   GPU=self.GPU)
        elif self.mode == "sir":
            self.joint_probability_multi_cluster = Joint_Probability_Multi_Cluster_SIR(base_param_dict = base_param_dict, 
                                                                                   joint_prob_param_dict = joint_prob_param_dict, 
                                                                                   sir_model_param_dict = model_param_dict,
                                                                                   GPU=self.GPU)
            
    def forward(self,obs_data):
        """
        Processes the observed data through the model components to compute embeddings, transformations, and probabilities.

        Parameters:
            obs_data (torch.Tensor): The observed data tensor.

        Returns:
            dict: A dictionary containing tensors for embedded data, xi and log likelihood values, 
                    transformed data and log determinants, and joint probabilities.
        """
        data_embeded = self.lstm_embedder(obs_data)
        fw_res = self.forward_flow(data_embeded, obs_data)


        return {"data_embeded" : data_embeded,
                "xi_loglh" : fw_res["xi_loglh"],
                "z_logdet" : fw_res["z_logdet"],
                "joint_prob" : fw_res["joint_prob"]}

    def forward_flow(self,data_embeded, obs_data):
        xi_loglh = self.calc_xi_and_loglikelihood(data_embeded)
        z_logdet = self.resflow_multi_cluster(xi_loglh["xi"])
        joint_prob = self.joint_probability_multi_cluster(z_logdet["z"], obs_data)
        return {"xi_loglh":xi_loglh, "z_logdet":z_logdet, "joint_prob":joint_prob}


    def calc_loss(self,obs_data, alpha=None):
        """
        Calculates the loss for the model which is negative the Evidence Lower Bound (ELBO) with an optional entropy penalty for diversity.

        Parameters:
            obs_data (torch.Tensor): The observed data tensor.
            alpha (float, optional): Coefficient for the entropy penalty to encourage diversity among clusters.

        Returns:
            torch.Tensor: The calculated loss.
        """
        fw_res = self.forward(obs_data)#
        ELBO = -fw_res["xi_loglh"]["LogLikelihood"]- fw_res["z_logdet"]["log_det"] + fw_res["joint_prob"]#shape:(BS, num_clusters)
        
        if self.num_clusters >= 2:
            
            prob_cluster = fw_res["data_embeded"]["cluster"]
            log_prob_cluster = torch.log(prob_cluster)
            

                
            
            ELBO = -log_prob_cluster + ELBO 
            ELBO_sample_mean = (prob_cluster*ELBO).sum(axis=1)
            
            loss = -1*torch.mean(ELBO_sample_mean)
          
            if alpha is not None:
                penalty = (prob_cluster*(-log_prob_cluster)).sum(axis=1)
                penalty = penalty.mean()
                loss = loss - alpha*penalty

        elif self.num_clusters ==1:
            loss = -1*torch.mean(ELBO)
        
        return loss
