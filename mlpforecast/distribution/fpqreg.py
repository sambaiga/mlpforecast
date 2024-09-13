import torch
import torch.nn as nn
import torch.nn.functional as F
from mlpforecast.net.embending import Rotary
from mlpforecast.net.layers import FeedForward, MLPForecastNetwork
from mlpforecast.custom_loss.quantile import (quantile_huber_loss, 
                                              pin_ball_loss,
                                              quantile_penalty_loss,
                                              quantile_proposal_loss)


    
class QuantileProposal(nn.Module):
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
    def __init__(self, N=10, z_dim=64, dropout=0.1, alpha=0.05):
        super().__init__()

        self.net = nn.Linear(z_dim, N)
        alpha = torch.tensor([alpha])
        torch.nn.init.xavier_uniform_(self.net.weight, gain=0.01)
        
        self.dropout=nn.Dropout(dropout)
        self.N = N
        self.z_dim = z_dim
        tau_0 = torch.zeros(1, 1, 1)
        self.register_buffer('tau_0', tau_0)
        self.register_buffer('alpha', alpha)
        

    def forward(self, z):
        batch_size = z.shape[0]
        
        z_out = self.dropout(self.net(z)).reshape(batch_size, self.N, 1)
        log_probs = F.log_softmax(z_out, dim=1)
        probs = log_probs.exp()
        assert probs.shape == (batch_size, self.N, 1)
        
        tau_0 = torch.repeat_interleave(self.tau_0, batch_size, dim=0)
        taus_1_N = torch.cumsum(probs, dim=1)

        


        # Calculate \tau_i (i=0,...,N).
        taus = torch.cat((tau_0, taus_1_N), dim=1)
        
        assert taus.shape == (batch_size, self.N+1, 1)
        
        # Calculate \hat \tau_i (i=0,...,N-1).
        tau_hats = (taus[:, :-1, :] + taus[:, 1:, :]).detach() / 2.
        #tau_hats = torch.clamp(tau_hats, max=0.98)
        assert tau_hats.shape == (batch_size, self.N, 1)
        
        # Calculate entropies of value distributions.
        entropies = torch.sum(-log_probs*probs, 1)
        assert entropies.shape == (batch_size, 1)

        # Clamp tau_hats to ensure they are within the desired bounds
        tau_hats = torch.clamp(tau_hats, min=self.alpha.item() / 2.0, max=1 - (self.alpha.item() / 2.0))
        #tau_hats = tau_hats.sort(dim=1)[0]
        return taus, tau_hats, entropies


class FPQRNetwork(nn.Module):
    """
    FPQRNetwork is a neural network for forecasting quantiles using rotary embeddings and 
    a confidence quantile proposal mechanism. 

    Args:
        n_out (int): Number of output features. Default is 1.
        N (int): Dimension of the proposal mechanism. Default is 90.
        hidden_size (int): Size of the hidden layers. Default is 256.
        forecast_horizon (int): Number of time steps to forecast. \
              Default is 48.
        dropout_rate (float): Dropout rate for regularization. \
            Default is 0.25.
        alpha (float): Alpha parameter for the confidence quantile proposal. \
              Default is 0.05.
        kappa (float): Kappa parameter for the Quantile Huber loss. \
            Default is 0.25.
        eps (float): Epsilon parameter for numerical stability. \
            Default is 1e-6.
        activation_function (nn.Module): \
            Activation function to use in the feedforward layers.\
                  Default is nn.SiLU().
        out_activation_function (nn.Module): \
            Activation function to use in the output layer. \
                Default is nn.Identity().
    """
    
    def __init__(self, n_out=1, N=32, hidden_size=256, forecast_horizon=48,
                 dropout_rate=0.25, alpha=0.05, kappa=0.25, eps=1e-6,
                 activation_function=nn.SiLU(), out_activation_function=nn.Identity()):
        super().__init__()

        self.n_out = n_out
        self.forecast_horizon = forecast_horizon
        self.eps=eps
        self.kappa=kappa
        self.alpha=alpha

        self.tau_proposal= QuantileProposal(N=N, z_dim=hidden_size,
                                            dropout=dropout_rate, alpha=alpha)

        '''
        self.tau_proposal = ConfidenceQuantileProposal(
            N=N, M=forecast_horizon, z_dim=hidden_size, dropout=dropout_rate, alpha=alpha
        )
        '''
        self.tau_cosine = Rotary(dim=hidden_size)
        
        self.decoder = nn.Sequential(
            FeedForward(hidden_size, expansion_factor=1, dropout=dropout_rate, activation=activation_function, bn=False),
            nn.Linear(hidden_size, forecast_horizon * n_out),
            out_activation_function
        )

        
        
    def FPN(self, z):
        """
        Forward pass through the confidence quantile proposal network.

        Args:
            z (torch.Tensor): Input tensor.

        Returns:
            tuple: Taus, tau_hats, and entropies.
        """
        taus, tau_hats, entropies = self.tau_proposal(z)
        return taus, tau_hats, entropies
    
    def QVN(self, tau, z):
        """
        Quantile value network that combines rotary embeddings with the input tensor.

        Args:
            tau (torch.Tensor): Tau tensor from the proposal network.
            z (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor reshaped to forecast horizon and output size.
        """
        B, N, M = tau.size()
        q = self.tau_cosine(tau)
       
        output_p = torch.relu(torch.mul(z.unsqueeze(1), q))
        output_p = torch.add(output_p, z.unsqueeze(1).expand_as(output_p))
        
        out = self.decoder(output_p) 
        return out.reshape(B, N, self.forecast_horizon, self.n_out)
    
    def forecast(self, z):
        with torch.no_grad():
            taus, tau_hats, _, quantile_hats = self(z)
            loc = (quantile_hats*(taus[:, 1:, :, None] - taus[:, :-1,:, None])).sum(dim=1)
            
            
        return {"loc":loc, 
                'q_samples':quantile_hats,
                  "taus":taus, 
                  'taus_hats':tau_hats}
    
    
    def forward(self, z):
        """
        Forward pass through the entire network.

        Args:
            z (torch.Tensor): Input tensor.

        Returns:
            tuple: Taus, tau_hats, entropies, and q_hats.
        """
        taus, tau_hats, entropies = self.FPN(z)
        q_hats = self.QVN(tau_hats, z)
        return taus, tau_hats, entropies, q_hats
    
    def step(self, z, y, metric_fn, beta=0.5):
        """
        Perform a training step, calculating the loss and a given metric.

        Args:
            z (torch.Tensor): Input tensor.
            y (torch.Tensor): Target tensor.
            metric_fn (callable): Metric function to evaluate the predictions.

        Returns:
            tuple: Loss and metric value.
        """
        taus, tau_hats, entropies, quantile_hats = self(z)
        quantile = self.QVN(taus[:, 1:-1, :], z)
        q_m = (quantile_hats * (taus[:, 1:, :, None] - taus[:, :-1, :, None])).sum(dim=1)
        y_q = y.unsqueeze(1).expand_as(quantile_hats)

        if self.kappa<1e-2 or self.kappa==0.0:
            q_loss = pin_ball_loss(inputs=quantile_hats, 
                                     targets=y_q, 
                                     quantiles=tau_hats.unsqueeze(-1),
                                     kappa=self.kappa, 
                                     reduction='mean'
                                     )
        else:
            q_loss = quantile_huber_loss(inputs=quantile_hats, 
                                     targets=y_q, 
                                     quantiles=tau_hats.unsqueeze(-1),
                                     kappa=self.kappa, 
                                     reduction='mean')

        
        tau_loss = quantile_proposal_loss(quantile.detach(), 
                                          quantile_hats.detach(), 
                                          taus.unsqueeze(-1).detach(),
                                          reduction='mean')
        entropy_loss = -entropies.mean()
        penalty_loss = quantile_penalty_loss(quantile_hats, 
                                             self.kappa, 
                                             self.eps, 
                                             reduction='mean')
        
        median_loss = (
            beta * F.mse_loss(q_m, y, reduction="none").sum(dim=(1, 2)).mean()
            + (1 - beta) * F.l1_loss(q_m, y, reduction="none").sum(dim=(1, 2)).mean()
        )
        
        loss = q_loss+ tau_loss+ penalty_loss + entropy_loss+median_loss


        metric = metric_fn(quantile_hats, y_q)
        return loss, q_loss