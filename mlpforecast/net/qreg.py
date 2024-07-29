from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from mlpforecast.net.layers import MLPForecastNetwork
from mlpforecast.distribution.qreg import QRNetwork

class MLPQRForecastNetwork(MLPForecastNetwork):
    """
    Multilayer Perceptron (MLP) Quantile Regression Forecast Network for time series forecasting.

    Args:
        n_target_series (int): Number of target series.
        n_unknown_features (int): Number of unknown time-varying features.
        n_known_calendar_features (int): Number of known categorical time-varying features.
        n_known_continuous_features (int): Number of known continuous time-varying features.
        embedding_size (int, optional): Dimensionality of the embedding space. Defaults to 28.
        embedding_type (str, optional): Type of embedding to use. Defaults to None. Options: 'PosEmb', 'RotaryEmb', 
            'CombinedEmb'.
        combination_type (str, optional): Type of combination to use. Defaults to 'attn-comb'. Options: 'attn-comb', 
            'weighted-comb', 'addition-comb'.
        expansion_factor (int, optional): Expansion factor for the encoder. Defaults to 2.
        residual (bool, optional): Whether to use residual connections in the encoder. Defaults to False.
        hidden_size (int, optional): Dimensionality of the hidden layers. Defaults to 256.
        num_layers (int, optional): Number of layers in the MLP. Defaults to 2.
        forecast_horizon (int, optional): Number of future time steps to forecast. Defaults to 48.
        input_window_size (int, optional): Size of the input window. Defaults to 96.
        activation_function (str, optional): Activation function. Defaults to 'SiLU'.
        out_activation_function (str, optional): Output activation function. Defaults to 'Identity'.
        dropout_rate (float, optional): Dropout probability. Defaults to 0.25.
        alpha (float, optional): Alpha parameter for the loss. Defaults to 0.1.
        num_attention_heads (int, optional): Number of heads in the multi-head attention. Defaults to 4.
        quantiles (list, optional): List of quantiles for quantile regression. Defaults to [0.05, 0.25, 0.75, 0.95].
        eps (float, optional): Epsilon for numerical stability. Defaults to 1e-6.
        kappa (float, optional): Kappa parameter for the Quantile Huber loss. Defaults to 0.25.
    """

    def __init__(self, n_target_series: int, n_unknown_features: int, n_known_calendar_features: int, 
                 n_known_continuous_features: int, embedding_size: int = 28, embedding_type: str = None, 
                 combination_type: str = "attn-comb", expansion_factor: int = 2, residual: bool = False, 
                 hidden_size: int = 256, num_layers: int = 2, forecast_horizon: int = 48, input_window_size: int = 96, 
                 activation_function: str = "SiLU", out_activation_function: str = "Identity", dropout_rate: float = 0.25, 
                 alpha: float = 0.1, num_attention_heads: int = 4, quantiles: list = [0.05, 0.25, 0.75, 0.95], 
                 eps: float = 1e-6, kappa: float = 0.25):

        super().__init__(n_target_series=n_target_series, n_unknown_features=n_unknown_features, 
                         n_known_calendar_features=n_known_calendar_features, 
                         n_known_continuous_features=n_known_continuous_features, embedding_size=embedding_size, 
                         embedding_type=embedding_type, combination_type=combination_type, expansion_factor=expansion_factor, 
                         residual=residual, hidden_size=hidden_size, num_layers=num_layers, forecast_horizon=forecast_horizon, 
                         input_window_size=input_window_size, activation_function=activation_function, 
                         out_activation_function=out_activation_function, dropout_rate=dropout_rate, alpha=alpha, 
                         num_attention_heads=num_attention_heads)

        self.qrnet = QRNetwork(quantiles=quantiles, n_out=self.n_out, hidden_size=hidden_size, 
                               forecast_horizon=forecast_horizon, dropout_rate=dropout_rate, alpha=alpha, eps=eps, 
                               kappa=kappa, activation_function=self.activation, out_activation_function=self.out_activation)

    def forward(self, x):
        """
        Forward pass through the MLPQRForecastNetwork.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor from the QRNetwork.
        """
        z = self.compute_combined_projection_feature(x)
        return self.qrnet(z)

    def step(self, batch: tuple, metric_fn: callable) -> tuple:
        """
        Training step for the MLPQRForecastNetwork.

        Args:
            batch (tuple): Tuple containing input and target tensors.
            metric_fn (callable): Metric function to evaluate.

        Returns:
            tuple: Tuple containing the loss and computed metric.
        """
        x, y = batch
        z = self.compute_combined_projection_feature(x)
        return self.qrnet.step(z, y, metric_fn)

    def forecast(self, x):
        """
        Generate forecasts using the MLPQRForecastNetwork.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            dict: Dictionary containing the forecast location, quantile samples, and taus.
        """
        z = self.compute_combined_projection_feature(x)
        return self.qrnet.forecast(z)


