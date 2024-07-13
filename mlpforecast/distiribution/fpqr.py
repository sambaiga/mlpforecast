import torch
import torch.nn as nn

def get_pinball(y, sample, alpha):
    """
    Compute the pinball loss between true values (y) and quantile estimates from a sample.

    Parameters:
    - y (torch.Tensor): True values.
    - sample (torch.Tensor): Sample of quantile estimates.
    - alpha (float): Quantile level.

    Returns:
    torch.Tensor: Pinball loss.
    """
    # Ensure alpha is on the same device as y
    alpha = torch.tensor(alpha).to(y.device)

    # Compute quantile estimates from the sample
    q_hat = torch.quantile(sample, alpha, dim=1).permute(1, 0, 2, 3)

    # Compute the error between true values and quantile estimates
    error = (y.unsqueeze(1).expand_as(q_hat) - q_hat)

    # Compute the pinball loss
    pinball_loss = torch.max(alpha[None, :, None, None] * error, (alpha[None, :, None, None] - 1) * error).mean()

    return pinball_loss

class ConfidenceQuantileProposal(nn.Module):
    """
    A neural network module for proposing confidence quantiles.

    This module generates quantile estimates based on the input features 
    using a linear layer followed by dropout and softmax normalization. 
    It computes cumulative probabilities and ensures that the resulting 
    quantiles are within specified bounds.

    Args:
        N (int): Number of quantiles to propose.
        M (int): Number of output features for each quantile.
        z_dim (int): Dimensionality of the input features.
        dropout (float): Dropout rate for regularization.
        alpha (float): Significance level for quantile estimation.

    Attributes:
        N (int): Number of quantiles.
        M (int): Number of output features.
        alpha (torch.Tensor): Significance level tensor.
        net (nn.Linear): Linear layer for transforming input features.
        dropout (nn.Dropout): Dropout layer for regularization.
        tau_0 (torch.Tensor): Initial tau value buffer.
    """

    def __init__(self, N=10, M=1, z_dim=64, dropout=0.1, alpha=0.05):
        super().__init__()
        self.N = N
        self.M = M
        self.alpha = torch.tensor([alpha])
        
        # Define layers
        self.net = nn.Linear(z_dim, N * M)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize tau_0 as a buffer
        self.tau_0 = torch.zeros(1, 1, 1)
        self.register_buffer('tau_0', self.tau_0)
        self.register_buffer('alpha', self.alpha)

        self._initialize_weights()

    def _initialize_weights(self):
        """Initializes the weights and biases of the linear layer."""
        nn.init.uniform_(self.net.weight, a=self.alpha.item() / 2.0, b=1 - (self.alpha.item() / 2.0))
        nn.init.constant_(self.net.bias, 0.0)

    def forward(self, phf):
        """
        Forward pass for generating quantile estimates.

        Args:
            phf (torch.Tensor): Input tensor of shape (B, z_dim) where B is the batch size.

        Returns:
            torch.Tensor: Tensor of shape (B, N, M) containing estimated quantiles.
        """
        B = phf.size(0)
        
        # Compute output from the linear layer with dropout
        z = self.dropout(self.net(phf)).reshape(-1, self.N, self.M)
        log_probs = torch.log_softmax(z, dim=1)
        probs = log_probs.exp()

        # Cumulative sum of probabilities
        tau_N = torch.cumsum(probs, dim=1)
        
        # Adjust the tau values to be within [alpha/2, 1 - alpha/2]
        tau_N = (self.alpha / 2) + (1 - self.alpha / 2.0) * tau_N

        # Repeat initial tau value for the batch size and output features
        tau_0 = torch.repeat_interleave(self.tau_0, B, dim=0)
        tau_0 = torch.repeat_interleave(tau_0, self.M, dim=-1)

        # Concatenate initial tau with computed tau_N
        taus = torch.cat((tau_0, tau_N), dim=1)
        
        # Calculate tau_hats as the average of consecutive taus
        tau_hats = (taus[:, :-1, :] + taus[:, 1:, :]) / 2.0
        assert taus.shape == (B, self.N + 1, self.M)

        # Clamp tau_hats to ensure they are within the desired bounds
        tau_hats = torch.clamp(tau_hats, min=self.alpha.item() / 2.0, max=1 - (self.alpha.item() / 2.0))

        # Calculate entropies of value distributions.
        entropies = torch.sum(-log_probs*probs, 1)
        assert entropies.shape == (B, self.M)

        return taus, tau_hats, entropies
