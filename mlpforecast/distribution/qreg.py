import torch
import torch.nn as nn
import torch.nn.functional as F
from mlpforecast.net.embending import Rotary
from mlpforecast.net.layers import FeedForward, MLPForecastNetwork
from mlpforecast.custom_loss.quantile import pin_ball_loss, quantile_penalty_loss

class QRNetwork(nn.Module):
    """
    QRNetwork is a neural network for forecasting quantiles using a quantile value network.

    Args:
        quantiles (list): List of quantiles to forecast. Default is [0.05, 0.1, 0.5, 0.9, 0.95].
        n_out (int): Number of output features. Default is 1.
        hidden_size (int): Size of the hidden layers. Default is 256.
        forecast_horizon (int): Number of time steps to forecast. Default is 48.
        dropout_rate (float): Dropout rate for regularization. Default is 0.25.
        alpha (float): Alpha parameter for the confidence quantile proposal. Default is 0.05.
        kappa (float): Kappa parameter for the Quantile Huber loss. Default is 0.25.
        eps (float): Epsilon parameter for numerical stability. Default is 1e-6.
        activation_function (nn.Module): Activation function to use in the feedforward layers. Default is nn.SiLU().
        out_activation_function (nn.Module): Activation function to use in the output layer. Default is nn.Identity().
    """
    
    def __init__(self, quantiles=[0.05, 0.1, 0.5, 0.9, 0.95], n_out=1, hidden_size=256, 
                 forecast_horizon=48, dropout_rate=0.25, alpha=0.05, kappa=0.25, 
                 eps=1e-6, activation_function=nn.SiLU(), out_activation_function=nn.Identity()):
        super().__init__()

        self.n_out = n_out
        self.forecast_horizon = forecast_horizon
        self.kappa = kappa
        self.eps = eps

        taus = torch.tensor(quantiles)
        self.register_buffer('taus', taus)

        self.decoder = nn.Sequential(
            FeedForward(hidden_size, expansion_factor=1, dropout=dropout_rate, 
                        activation=activation_function, bn=False),
            nn.Linear(hidden_size, forecast_horizon * n_out * len(quantiles)),
            out_activation_function
        )

        
    def QVN(self, z):
        """
        Quantile value network that combines the input tensor with the decoder.

        Args:
            z (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor reshaped to forecast horizon and output size.
        """
        B = z.size(0)
        out = self.decoder(z)
        return out.reshape(B, -1, self.forecast_horizon, self.n_out)
    
    def forecast(self, z):
        """
        Forecast function to generate quantile predictions without gradient calculation.

        Args:
            z (torch.Tensor): Input tensor.

        Returns:
            dict: Dictionary containing location (median), quantile samples, and taus.
        """
        with torch.no_grad():
            quantile_hats = self(z)
            idx = len(self.taus) // 2
            loc = quantile_hats[:, idx, :, :]
            tau_hats = self.taus.to(loc.device)
            
        return {"loc": loc, 'q_samples': quantile_hats, "taus": tau_hats}

    def forward(self, z):
        """
        Forward pass through the entire network.

        Args:
            z (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Quantile predictions.
        """
        q_hats = self.QVN(z)
        return q_hats
    
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
        quantile_hats = self(z)
        y_q = y.unsqueeze(1).expand_as(quantile_hats)
        idx = len(self.taus) // 2
        q_m = quantile_hats[:, idx, :, :]

        # Expand quantiles to match the dimensions of inputs and targets
        quantiles = self.taus[None, :, None].to(y.device)
        quantiles = torch.repeat_interleave(quantiles, y.size(0), dim=0)
        quantiles = torch.repeat_interleave(quantiles, y.size(1), dim=2)
        quantiles = quantiles.unsqueeze(-1)

        q_loss = pin_ball_loss(quantile_hats, y_q, quantiles, 
                               kappa=self.kappa, reduction='mean')
        penalty_loss = quantile_penalty_loss(quantile_hats, self.kappa, self.eps, 
                                             reduction='mean')
        loss = q_loss + penalty_loss
        median_loss = (
            beta * F.mse_loss(q_m, y, reduction="none").sum(dim=(1, 2)).mean()
            + (1 - beta) * F.l1_loss(q_m, y, reduction="none").sum(dim=(1, 2)).mean()
        )
        
        loss = q_loss+  penalty_loss + median_loss

        metric = metric_fn(quantile_hats, y_q)
        return loss, q_loss