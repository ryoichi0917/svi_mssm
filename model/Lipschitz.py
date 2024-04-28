
import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F






# Modified and used code from https://github.com/rtqichen/residual-flows


class LipschitzCNN1D(nn.Module):
    """
    A 1-dimensional convolutional neural network (CNN) that maintains a specified Lipschitz 
    constant.
    Attributes:
        n_layers (int): Number of convolutional layers in the network.
        channels (list of int): Specifies the number of channels for each convolutional layer.
        kernel_size (list of int): Specifies the kernel size for each convolutional layer.
        lipschitz_const (float): Maximum Lipschitz constant for each convolutional layer.
        max_lipschitz_iter (int): Maximum iterations for the spectral norm calculation to enforce
                                  the Lipschitz constraint.
        lipschitz_tolerance (float, optional): Tolerance level for the spectral norm calculation
                                               if `max_lipschitz_iter` is None.
    """
    
    def __init__(
        self,
        channels,
        kernel_size,
        lipschitz_const=0.97,
        max_lipschitz_iter=5,
        lipschitz_tolerance=None,
    ):
        """
        Initializes the LipschitzCNN1D network with the specified number of channels,
        kernel sizes, Lipschitz constant, and other spectral norm parameters.

        Parameters:
            channels (list of int): The number of channels for each layer in the network.
            kernel_size (list of int): The kernel size for each convolutional layer.
            lipschitz_const (float): Maximum allowed Lipschitz constant for each layer.
            max_lipschitz_iter (int): Maximum number of iterations used in the spectral
                                      normalization to ensure Lipschitz continuity. Defaults to 5.
            lipschitz_tolerance (float, optional): Tolerance used alongside or instead of
                                                   `max_lipschitz_iter` for the spectral normalization.
        """
        super(LipschitzCNN1D, self).__init__()
        self.n_layers = len(kernel_size)
        self.channels = channels
        self.kernel_size = kernel_size
        self.lipschitz_const = lipschitz_const
        self.max_lipschitz_iter = max_lipschitz_iter
        self.lipschitz_tolerance = lipschitz_tolerance

        # Build the network layers
        layers = []
        for i in range(self.n_layers):
            layers += [
                SpectralNormConv1d(
                    in_channels=channels[i],
                    out_channels=channels[i + 1],
                    kernel_size=kernel_size[i],
                    stride=1,
                    padding=kernel_size[i] - 1,  # Note: Custom padding implementation in SpectralNormConv1d
                    bias=True,
                    coeff=lipschitz_const,
                    n_iterations=max_lipschitz_iter,
                    atol=lipschitz_tolerance,
                    rtol=lipschitz_tolerance
                ),
                Swish(),
            ]
        layers = layers[:-1]  # Remove activation function from the final layer
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass of the LipschitzCNN1D network.

        Parameters:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, length).

        Returns:
            torch.Tensor: Output tensor after passing through the Lipschitz-constrained convolutional
                          layers and activation functions.
        """
        return self.net(x)



class SpectralNormConv1d(nn.Module):
    """
    A 1D convolutional layer with spectral normalization to ensure that the layer maintains
    Lipschitz continuity.

    Attributes:
        in_channels (int): Number of channels in the input.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the convolving kernel.
        stride (int): Stride of the convolution.
        padding (int): Padding added to both sides of the input.
        bias (bool): If True, adds a learnable bias to the output.
        coeff (float): Coefficient to scale the Lipschitz constant.
        n_iterations (int, optional): Number of iterations for power iteration method.
        atol (float, optional): Absolute tolerance for convergence in power iteration.
        rtol (float, optional): Relative tolerance for convergence in power iteration.
    """
    
    def __init__(
        self, in_channels, out_channels, kernel_size, stride, padding, bias=True, coeff=0.97, 
        n_iterations=None, atol=None, rtol=None
    ):
        """
        Initializes the SpectralNormConv1d module with the necessary parameters and 
        sets up the weight and optionally bias parameters with spectral normalization.

        Parameters:
            in_channels (int): Number of channels in the input image.
            out_channels (int): Number of channels produced by the convolution.
            kernel_size (int): Size of the convolving kernel.
            stride (int): Stride of the convolution.
            padding (int): Padding added to both sides of the input.
            bias (bool): If set to False, the layer will not learn an additive bias.
                         Default is True.
            coeff (float): Scaling factor for the Lipschitz constant, typically less than 1.
            n_iterations (int, optional): Number of power iterations to approximate the spectral norm.
            atol (float, optional): Absolute tolerance used for convergence in power iteration.
            rtol (float, optional): Relative tolerance used for convergence in power iteration.
        """
        super(SpectralNormConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = nn.Parameter(torch.Tensor(out_channels)) if bias else None
        self.coeff = coeff
        self.n_iterations = n_iterations
        self.atol = atol
        self.rtol = rtol
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size))
        self.reset_parameters()
        self.initialized = False
        self.register_buffer('spatial_dims', torch.tensor([1.]))
        self.register_buffer('scale', torch.tensor(0.))

    def reset_parameters(self):
        """
        Initialize the weight and bias parameters using He initialization.
        """
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def _initialize_u_v(self):
        """
        Initializes u and v for the power iteration used in spectral normalization.
        """
        c, w = self.in_channels, int(self.spatial_dims[0].item())
        with torch.no_grad():
            num_input_dim = c * w
            v = F.normalize(torch.randn(num_input_dim).to(self.weight), dim=0, eps=1e-12)
            u = F.conv1d(v.view(1, c, w), self.weight, stride=self.stride, padding=self.padding, bias=None)
            num_output_dim = u.shape[0] * u.shape[1] * u.shape[2]
            self.out_shape = u.shape
            u = F.normalize(torch.randn(num_output_dim).to(self.weight), dim=0, eps=1e-12)

            self.register_buffer('u', u)
            self.register_buffer('v', v)

    def compute_weight(self, update=True):
        """
        Computes the spectrally normalized weight using the power iteration method if update is True.
        Ensures the layer's Lipschitz constant is controlled by 'coeff'.
        
        Parameters:
            update (bool): Whether to update u and v vectors for spectral normalization.

        Returns:
            torch.Tensor: The normalized weight tensor.
        """
        if not self.initialized:
            self._initialize_u_v()
            self.initialized = True
        
        return self._compute_weight_1d(update)

    def _compute_weight_1d(self, update=True, n_iterations=None, atol=None, rtol=None):
        """
        Computes the spectrally normalized weight using power iteration to enforce a Lipschitz constraint,
        typically used to stabilize the training of neural networks by controlling the Lipschitz constant
        of convolution layers.

        Parameters:
            update (bool): If True, updates the estimates of u and v vectors used in power iteration.
            n_iterations (int, optional): The number of iterations to use in power iteration.
                                          Defaults to self.n_iterations if None.
            atol (float, optional): Absolute tolerance used for convergence in power iteration.
                                    Defaults to self.atol if None.
            rtol (float, optional): Relative tolerance used for convergence in power iteration.
                                    Defaults to self.rtol if None.

        Returns:
            torch.Tensor: The normalized weight tensor after applying spectral normalization.

        """
        n_iterations = self.n_iterations if n_iterations is None else n_iterations
        atol = self.atol if atol is None else atol
        rtol = self.rtol if rtol is None else atol

        if n_iterations is None and (atol is None or rtol is None):
            raise ValueError('Need one of n_iteration or (atol, rtol).')

        if n_iterations is None:
            n_iterations = 20000

        u = self.u
        v = self.v
        weight = self.weight
        c, w = self.in_channels, int(self.spatial_dims[0].item())
        if update:
            with torch.no_grad():
                itrs_used = 0
                for _ in range(n_iterations):
                    old_u = u.clone()
                    old_v = v.clone()
                    v_s = F.conv_transpose1d(
                        u.view(self.out_shape), weight, stride=self.stride, padding=self.padding, output_padding=0)
                    v = F.normalize(v_s.view(-1), dim=0, out=v)
                    u_s = F.conv1d(v.view(1, c, w), weight, stride=self.stride, padding=self.padding, bias=None)
                    u = F.normalize(u_s.view(-1), dim=0, out=u)
                    itrs_used = itrs_used + 1
                    if atol is not None and rtol is not None:
                        err_u = torch.norm(u - old_u) / (u.nelement()**0.5)
                        err_v = torch.norm(v - old_v) / (v.nelement()**0.5)
                        tol_u = atol + rtol * torch.max(u)
                        tol_v = atol + rtol * torch.max(v)
                        if err_u < tol_u and err_v < tol_v:
                            break

                if itrs_used > 0:
                    u = u.clone()
                    v = v.clone()

        weight_v = F.conv1d(v.view(1, c, w), weight, stride=self.stride, padding=self.padding, bias=None)
        weight_v = weight_v.view(-1)
        sigma = torch.dot(u.view(-1), weight_v)
        with torch.no_grad():
            self.scale.copy_(sigma)
        # soft normalization: only when sigma larger than coeff
        factor = torch.max(torch.ones(1).to(weight.device), sigma / self.coeff)
        weight = weight / factor
        return weight
    
    
    def same_value_padding_by_ts(self, input_data, padding_size):
        """
        Applies padding to the time steps dimension of the input tensor, using the first and last values
        of each time step for padding. 

        Parameters:
            input_data (torch.Tensor): The input tensor with dimensions (batch_size, channel_size, time_step_size).
            padding_size (int): The number of values to add at both the start and end of the time-step dimension.

        Returns:
            torch.Tensor: The output tensor with padding applied at the beginning and end of each time step,
                          maintaining the original batch size and channel size dimensions.


        """
        batch_size, channel_size, time_step_size = input_data.size()

        start_values = input_data[:, :, 0].reshape(batch_size, channel_size, -1) 
        end_values = input_data[:, :, -1].reshape(batch_size, channel_size, -1)  

        if padding_size==1:
            return torch.concat((start_values,input_data), dim=2)

        elif (padding_size % 2)==0:
            start_values = torch.concat([start_values]*int(padding_size/2), dim=2)
            end_values = torch.concat([end_values]*int(padding_size/2), dim=2)
            return torch.concat((start_values,input_data,end_values), dim=2)
    
        else:
            start_values = torch.concat([start_values]*(int(padding_size/2)+1), dim=2)
            end_values = torch.concat([end_values]*int(padding_size/2), dim=2)
            return torch.concat((start_values,input_data,end_values), dim=2)    
            

    def forward(self, input):
        """
        Applies the spectrally normalized convolutional operation to the input data after applying
        same-value edge padding.

        Parameters:
            input (torch.Tensor): Input data to the convolutional layer.

        Returns:
            torch.Tensor: The output of the convolutional operation.
        """
        if not self.initialized: 
            self.spatial_dims.copy_(torch.tensor(input.shape[2:4]).to(self.spatial_dims))
        weight = self.compute_weight(update=self.training)
        input = self.same_value_padding_by_ts(input, self.padding)
        return F.conv1d(input=input, weight=weight, bias=self.bias, stride=self.stride)

    def extra_repr(self):
        """
        Set the extra representation of the module, which will be displayed in print statements.
        """
        s = ('in_channels = {in_channels}, out_channels = {out_channels}, kernel_size={kernel_size}' 
             ', stride={stride}')
        if self.padding != (0,):
            s += ', padding={padding}'
        if self.bias is None:
            s += ', bias=False'
        s += ', coeff={}, n_iters={}, atol={}, rtol={}'.format(self.coeff, self.n_iterations, self.atol, self.rtol)
        return s.format(**self.__dict__)

    
    
# Code taken from https://github.com/rtqichen/residual-flows/blob/master/lib/layers/base/activations.py
class Swish(nn.Module):

    def __init__(self):
        super(Swish, self).__init__()
        self.beta = nn.Parameter(torch.tensor([0.5]))

    def forward(self, x):
        return (x * torch.sigmoid_(x * F.softplus(self.beta))).div_(1.1)