import torch  # Importing PyTorch for deep learning framework
import torch.nn as nn  # Importing neural network modules from PyTorch


# Define the LISTA model.
# The model uses convolutional layers to approximate sparse coding solutions.
# Args:
#   - m (int): Number of rows in the dictionary matrix.
#   - n (int): Number of columns in the dictionary matrix.
#   - Dict (torch.Tensor): Dictionary matrix.
#   - numIter (int): Number of iterations for the LISTA algorithm.
#   - alpha (float): Regularization parameter.
#   - device (torch.device): Device on which to run the model.

class LISTA(nn.Module):
    def __init__(self, m, n, Dict, numIter, alpha, device):
        super(LISTA, self).__init__()

        # Define convolutional layers for _W (Weights)
        self._W = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=128, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, padding=1, bias=False)
        )

        # Define convolutional layers for _S (Scaling)
        self._S = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=128, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, padding=1, bias=False)
        )

        # Initialize threshold parameter
        self.thr = nn.Parameter(torch.rand(numIter, 1), requires_grad=True)
        self.numIter = numIter
        self.A = Dict
        self.alpha = alpha
        self.device = device
        self.scale_S = (1 - 1 / alpha)
        self.scale_B = (1 / alpha)

    # Custom weights initialization called on the network
    def weights_init(self):
        # Initialize the threshold parameter
        thr = torch.ones(self.numIter, 1, 1, 1) * 0.1 / self.alpha

        # Initialize the weights of convolutional layers in _S
        for layer in self._S:
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')

        # Initialize the weights of convolutional layers in _W
        for layer in self._W:
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')

        # Assign the threshold parameter to the model
        self.thr.data = nn.Parameter(thr.to(self.device))

    # Forward pass through the LISTA model
    # Args:
    #   - y (torch.Tensor): Input tensor with shape [num_training_images, channels, height, width].
    # Returns:
    #   - List[torch.Tensor]: A list of tensors corresponding to the outputs at each iteration.

    def forward(self, y):
        from utils import soft_thr
        x = []
        d = torch.zeros_like(y, device=self.device)

        for iter in range(self.numIter):
            # Apply the LISTA update step with the learned weights and scaling
            d = soft_thr(self.scale_B * self._W(y) + self.scale_S * self._S(d) + d, self.thr[iter])
            x.append(d)
        return x


# Overparametrization model class with two learnable parameters lambda1 and lambda2, and both batch and layer normalizations are included.
# This model is designed with convolutional layers.
class Overparametrization(nn.Module):
    def __init__(self, m, n, Dict, numIter, alpha, device):
        super(Overparametrization, self).__init__()

        # Convolutional layers for _W with BatchNorm and LayerNorm
        self._W = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.LayerNorm([128, 70, 70]),  # LayerNorm applied after ReLU and BatchNorm
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.LayerNorm([64, 70, 70]),
            nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.LayerNorm([3, 70, 70])
        )

        # Convolutional layers for _S with BatchNorm and LayerNorm
        self._S = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.LayerNorm([128, 70, 70]),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.LayerNorm([64, 70, 70]),
            nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.LayerNorm([3, 70, 70])
        )

        self._thr = nn.Parameter(torch.rand(numIter, 1), requires_grad=True)
        self.numIter = numIter
        self.A = Dict
        self.alpha = alpha
        self.device = device
        self._lambda1 = nn.Parameter(torch.tensor(0.9), requires_grad=True)  # Learnable lambda1
        self._lambda2 = nn.Parameter(torch.tensor(0.1), requires_grad=True)  # Learnable lambda2

    # Function to initialize weights
    # This function initializes the weights for the convolutional layers in _W and _S
    def weights_init(self):
        A = self.A
        alpha = self.alpha
        _thr = torch.ones(self.numIter, 1, 1, 1) * 0.1 / alpha

        for layer in self._S:
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')

        for layer in self._W:
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')

        self._thr.data = nn.Parameter(_thr.to(self.device))
        self._lambda1.data = nn.Parameter(torch.tensor(0.9).to(self.device))
        self._lambda2.data = nn.Parameter(torch.tensor(0.1).to(self.device))

    # Forward pass of the LISTA model
    # The forward pass iteratively updates variables u and v, applying the learned thresholds and parameters.
    # Input:
    #   - y (torch.Tensor): The input tensor, which typically represents a noisy or corrupted image.
    # Output:
    #   - x (list of torch.Tensor): A list of tensors representing the reconstructed image at each iteration.
    def forward(self, y):
        x = []
        u = torch.ones_like(y)
        v = y.clone()

        lambda1_positive = torch.nn.functional.relu(self._lambda1)
        lambda2_positive = torch.nn.functional.relu(self._lambda2)

        for iter in range(self.numIter):
            u_new = u - v * (self._S(u * v) + self._W(y)) - lambda1_positive * u
            v_new = v - u * (self._S(u * v) + self._W(y)) - lambda2_positive * v

            u = u_new
            v = v_new

            x.append(u * v)
        return x
