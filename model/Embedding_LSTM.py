import torch
import torch.nn as nn

class BiLSTM_Base(nn.Module):
    """
    A Bidirectional LSTM (BiLSTM) neural network module for embedding input data
    with shared weights across layers. This class initializes a BiLSTM that can
    operate with multiple layers and on GPU if specified.

    Attributes:
        input_dim (int): Dimensionality of the input feature space.
        hidden_dim (int): Dimensionality of the hidden state per LSTM direction.
                          The total hidden states for both directions is twice this number.
        num_layers (int): Number of LSTM layers stacked together.
        GPU (bool): Flag to determine whether to use GPU acceleration.
    """
    
    def __init__(self, input_dim, hidden_dim, num_layers, GPU=False):
        """
        Initializes the BiLSTM_Base model with specified parameters for the input dimensions,
        hidden dimensions, number of layers, and GPU usage.

        Parameters:
            input_dim (int): The dimension of the input feature vector.
            hidden_dim (int): The dimension of the LSTM's hidden layers. The LSTM will actually have 
                              `hidden_dim / 2` hidden units per direction due to bidirectionality.
            num_layers (int): The number of LSTM layers to be stacked.
            GPU (bool): Whether to utilize CUDA-capable GPUs for processing. Defaults to False.
        """
        super(BiLSTM_Base, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = int(hidden_dim / 2)  # Divide hidden_dim by 2 for bidirectional split
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=self.input_dim, 
                            hidden_size=self.hidden_dim, 
                            num_layers=self.num_layers, 
                            batch_first=True, 
                            bidirectional=True)
        self.GPU = GPU

    def forward(self, x):
        """
        Defines the forward pass of the BiLSTM.

        Parameters:
            x (Tensor): The input tensor containing features. Expected shape is
                        (batch_size, sequence_length, input_dim).

        Returns:
            Tensor: Output from the BiLSTM layer. The shape of the output tensor is
                    (batch_size, sequence_length, 2 * hidden_dim) due to bidirectionality.
        """
        batch_size = x.shape[0]
        h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_dim)  # Initial hidden state
        c0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_dim)  # Initial cell state

        if self.GPU:
            h0 = h0.to("cuda:0")  # Move to GPU if GPU is enabled
            c0 = c0.to("cuda:0")  # Move to GPU if GPU is enabled

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))

        return out

class BiLSTM_Infer_Cluster(nn.Module):
    """
    A Bidirectional LSTM (BiLSTM) neural network model that embeds input data and performs
    clustering on the embedded data. The model uses a BiLSTM layer followed by a fully connected
    layer and a softmax to estimate cluster assignments.

    Attributes:
        input_dim (int): Dimension of the input feature space.
        hidden_dim (int): Dimension of the hidden state per LSTM direction. The total hidden
                          states for both directions is twice this number.
        num_layers (int): Number of LSTM layers stacked together.
        num_clusters (int): Number of clusters to estimate.
        GPU (bool): Flag to determine whether to use GPU acceleration.
    """
    
    def __init__(self, input_dim, hidden_dim, num_layers, num_clusters, GPU=False):
        """
        Initializes the BiLSTM_Infer_Cluster model with specified parameters for the input dimensions,
        hidden dimensions, number of layers, number of clusters, and GPU usage.

        Parameters:
            input_dim (int): The dimension of the input feature vector.
            hidden_dim (int): The dimension of the LSTM's hidden layers. The LSTM will actually have
                              `hidden_dim / 2` hidden units per direction due to bidirectionality.
            num_layers (int): The number of LSTM layers to be stacked.
            num_clusters (int): The number of clusters for the clustering layer.
            GPU (bool): Whether to utilize CUDA-capable GPUs for processing. Defaults to False.
        """
        super(BiLSTM_Infer_Cluster, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = int(hidden_dim / 2)  # Divide hidden_dim by 2 for bidirectional split
        self.num_layers = num_layers
        self.num_clusters = num_clusters
        self.lstm = nn.LSTM(input_size=self.input_dim, 
                            hidden_size=self.hidden_dim, 
                            num_layers=self.num_layers, 
                            batch_first=True, 
                            bidirectional=True)
        self.fc = nn.Linear(self.hidden_dim * 2, self.num_clusters)  # Linear layer for clustering
        self.sm = nn.Softmax(dim=1)  # Softmax for probability distribution over clusters
        self.GPU = GPU

    def forward(self, x):
        """
        Defines the forward pass of the BiLSTM with clustering.

        Parameters:
            x (Tensor): The input tensor containing features. Expected shape is
                        (batch_size, sequence_length, input_dim).

        Returns:
            Tensor: The output probabilities of cluster assignments. The shape of the output tensor is
                    (batch_size, num_clusters).
        """
        batch_size = x.shape[0]  # Extract batch size from input
        h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_dim)  # Initial hidden state
        c0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_dim)  # Initial cell state

        if self.GPU:
            h0 = h0.to("cuda:0")  # Move to GPU if GPU is enabled
            c0 = c0.to("cuda:0")  # Move to GPU if GPU is enabled

        # Forward propagate LSTM
        _, (h_n, _) = self.lstm(x, (h0, c0))  # Only the last hidden state is used
        h_n = h_n.permute(1, 2, 0)#(2*self.num_layers, BS, self.hidden_dim)→(BS, self.hidden_dim, 2*self.num_layers)
        h_n = h_n[:, :, -2:].reshape(h_n.shape[0], -1)#shape(BS, 2*self.hidden_dim)
        h_n = self.fc(h_n)  # Pass through the linear layer
        h_n = self.sm(h_n)  # Apply softmax to get probabilities

        return h_n


class BiLSTM_Each_Cluster(nn.Module):
    """
    Neural network that embeds input data and estimates the mu and sigma of each cluster.
    
    Input dimension: (batch_size, time_steps, observation_dimension)
    Output dimension: (batch_size, time_steps, latent_dimension)
    
    Args:
        input_dim (int): Dimension of the input data.
        hidden_dim (int): Dimension of the hidden state in the LSTM layer.
        num_lstm_layers (int): Number of layers in the LSTM network.
        output_dim (int): Dimension of the output.
        tgt (str): Target parameter to estimate, either "mu" or "sigma".
                   If tgt="sigma", the output is transformed using exponential function.
        GPU (bool, optional): Whether to use GPU. Defaults to False.
    """
    def __init__(self, input_dim, hidden_dim, num_lstm_layers, output_dim, tgt, GPU=False):
        super().__init__()
        assert (tgt == "mu") or (tgt == "sigma"), "tgt must be 'mu' or 'sigma'."
        
        self.input_dim = input_dim
        self.hidden_dim = int(hidden_dim / 2)  # Dividing hidden_dim by 2 for bidirectional LSTM.
        self.num_lstm_layers = num_lstm_layers
        self.output_dim = output_dim
        self.tgt = tgt
        self.GPU = GPU

        self.lstm = nn.LSTM(input_size=self.input_dim, 
                            hidden_size=self.hidden_dim, 
                            num_layers=self.num_lstm_layers, 
                            batch_first=True, bidirectional=True)
        
        self.fc = nn.Linear(self.hidden_dim * 2, self.output_dim)

    def forward(self, x):
        BS = x.shape[0]
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_lstm_layers * 2, BS, self.hidden_dim)
        # Initialize cell state with zeros
        c0 = torch.zeros(self.num_lstm_layers * 2, BS, self.hidden_dim)
        if self.GPU:
            h0 = h0.to("cuda:0")
            c0 = c0.to("cuda:0")

        # Forward propagate LSTM
        x, (_, _) = self.lstm(x, (h0, c0))  # Utilize the last output of the LSTM hidden state for cluster estimation.
        
        # Convert to the dimension of latent variables in the state space model by passing through a linear layer.
        x = self.fc(x)  # x.shape: (BS, TS, self.hidden_dim) → (BS, TS, output_dim)

        if self.tgt == "sigma":
            x = torch.exp(x)  # Apply exponential function for sigma to ensure non-negativity.

        return x



class BiLSTM_Multi_Cluster(nn.Module):
    """
    A neural network module that embeds input data using multiple Bidirectional LSTM (BiLSTM) models,
    each corresponding to a different cluster. Each cluster model outputs estimates of mu and sigma
    for that cluster. The module handles multiple sequences and outputs a set of parameters for each cluster.

    Attributes:
        input_dim (int): Dimension of the input features.
        hidden_dim (int): Dimension of the hidden state in each LSTM layer.
        output_dim (int): Dimension of the output from each LSTM layer, typically the dimensionality
                          of the estimated parameters mu and sigma.
        num_lstm_layers (int): Number of LSTM layers in each cluster model.
        num_clusters (int): Number of distinct clusters/models to be estimated.
        tgt (bool): A flag or setting used in each cluster-specific LSTM model (purpose should be specified by user).
        GPU (bool): Flag to determine whether to use GPU acceleration.
    """
    
    def __init__(self, input_dim, hidden_dim, output_dim, num_lstm_layers, num_clusters, tgt, GPU=False):
        """
        Initializes the BiLSTM_Multi_Cluster model with the specified parameters. This includes
        setting up multiple BiLSTM models, one for each cluster.

        Parameters:
            input_dim (int): The dimension of the input feature vector.
            hidden_dim (int): The dimension of the LSTM's hidden layers.
            output_dim (int): The output dimension from the LSTM, typically the size needed for mu and sigma.
            num_lstm_layers (int): The number of LSTM layers in each cluster's model.
            num_clusters (int): The number of clusters to model.
            tgt (bool): Target-related flag or parameter passed to each LSTM model.
            GPU (bool): Whether to utilize CUDA-capable GPUs for processing. Defaults to False.
        """
        super(BiLSTM_Multi_Cluster, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_lstm_layers = num_lstm_layers
        self.num_clusters = num_clusters
        self.tgt = tgt
        self.GPU = GPU

        # Initialize a list of BiLSTM models, one for each cluster
        self.flow_input_models = nn.ModuleList([
            BiLSTM_Each_Cluster(input_dim=self.input_dim, 
                                hidden_dim=self.hidden_dim, 
                                num_lstm_layers=self.num_lstm_layers, 
                                output_dim=self.output_dim, 
                                tgt=self.tgt,
                                GPU=self.GPU)
            for _ in range(self.num_clusters)
        ])

    def forward(self, x):
        """
        Forward pass of the model which processes input through each cluster-specific BiLSTM model
        and aggregates their outputs.

        Parameters:
            x (Tensor): Input tensor of shape (batch_size, sequence_length, input_dim).

        Returns:
            Tensor: Aggregated output from all cluster-specific models, shaped as
                    (batch_size, num_clusters, sequence_length, output_dim).
        """
        flow_input_list = [model(x) for model in self.flow_input_models]
        
        # Stack outputs along a new dimension for each cluster
        return torch.stack(flow_input_list, dim=1)


class LSTM_Embedder(nn.Module):
    """
    A neural network that uses LSTM for embedding input data and simultaneously estimates
    the mean (mu) and standard deviation (sigma) for each cluster. The network comprises
    several components: a base embedding LSTM, cluster inference, and separate models
    for predicting mu and sigma for each cluster.

    The input to the model is expected to be a batch of sequences, and the output is structured
    to provide predictions of cluster assignments, mu, and sigma for each sequence in each cluster.

    Attributes:
        input_dim (int): Dimension of the input feature space.
        hidden_dim (int): Dimension of the hidden layers in LSTMs.
        num_lstm_layers_base (int): Number of LSTM layers in the base embedding model.
        num_lstm_layers_other (int): Number of LSTM layers in the cluster, mu, and sigma models.
        num_clusters (int): Number of clusters to be estimated.
        latent_dim (int): Dimension of the output from mu and sigma models.
        GPU (bool): Flag to determine whether to use GPU acceleration.
    """
    
    def __init__(self, base_param_dict, lstm_param_dict, GPU=False):
        """
        Initializes the LSTM_Embedder with specified parameters grouped in dictionaries
        for base and LSTM specific parameters, and an option for GPU usage.

        Parameters:
            base_param_dict (dict): Dictionary containing base parameters such as input_dim,
                                    num_clusters, and latent_dim.
            lstm_param_dict (dict): Dictionary containing LSTM specific parameters such as
                                    embed_hidden_dim, num_lstm_layers_base, and num_lstm_layers_other.
            GPU (bool): Whether to utilize CUDA-capable GPUs for processing. Defaults to False.
        """
        assert lstm_param_dict["embed_hidden_dim"] % 2 == 0, "hidden_dim must be even."
        
        super(LSTM_Embedder, self).__init__()
        self.input_dim = base_param_dict["input_dim"]
        self.hidden_dim = lstm_param_dict["embed_hidden_dim"]
        self.num_lstm_layers_base = lstm_param_dict["num_lstm_layers_base"]
        self.num_lstm_layers_other = lstm_param_dict["num_lstm_layers_other"]
        self.num_clusters = base_param_dict["num_clusters"]
        self.latent_dim = base_param_dict["latent_dim"]
        self.GPU = GPU
        
        # Base embedding LSTM
        self.base_model = BiLSTM_Base(
            input_dim=self.input_dim, 
            hidden_dim=self.hidden_dim, 
            num_layers=self.num_lstm_layers_base,
            GPU=self.GPU)
        
        # Cluster inference model
        self.infer_cluster_model = BiLSTM_Infer_Cluster(
            input_dim=self.hidden_dim, 
            hidden_dim=self.hidden_dim,
            num_layers=self.num_lstm_layers_other, 
            num_clusters=self.num_clusters,
            GPU=self.GPU)
        
        # Mu estimation model
        self.mu_model = BiLSTM_Multi_Cluster(
            input_dim=self.hidden_dim, 
            hidden_dim=self.hidden_dim, 
            output_dim=self.latent_dim, 
            num_lstm_layers=self.num_lstm_layers_other, 
            num_clusters=self.num_clusters, 
            tgt="mu",
            GPU=self.GPU)
        
        # Sigma estimation model
        self.sigma_model = BiLSTM_Multi_Cluster(
            input_dim=self.hidden_dim, 
            hidden_dim=self.hidden_dim, 
            output_dim=self.latent_dim, 
            num_lstm_layers=self.num_lstm_layers_other, 
            num_clusters=self.num_clusters, 
            tgt="sigma",
            GPU=self.GPU)

    def forward(self, x):
        """
        Defines the forward pass of the model. It first embeds the input using the base model,
        then uses the embedded output to predict cluster assignments, mu, and sigma.

        Parameters:
            x (Tensor): The input tensor containing features. Expected shape is
                        (batch_size, time_sequence, input_dim).

        Returns:
            dict: A dictionary containing the outputs 'cluster', 'mu', and 'sigma' from the respective
                  models, with each key corresponding to a Tensor representing the model's output.
        """
        x = self.base_model(x)
        cluster = self.infer_cluster_model(x)
        mu = self.mu_model(x)
        sigma = self.sigma_model(x)
        
        return {"cluster": cluster, "mu": mu, "sigma": sigma}



