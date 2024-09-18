from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from mlpforecast.net.layers import MLPGAMForecastNetwork

class MLPCRFForecastNetwork(MLPGAMForecastNetwork):
    """
    Multilayer Perceptron (MLP) Forecast Network for time series forecasting.

    Attributes:
        n_out (int): Number of target series.
        n_unknown (int): Number of unknown time-varying features.
        n_covariates (int): Number of known time-varying features.
        n_channels (int): Number of channels in the input.
        input_window_size (int): Size of the input window.
        forecast_horizon (int): Number of future time steps to forecast.
        out_activation (torch.nn.Module): Output activation function.
        activation (torch.nn.Module): Activation function.
        encoder (PastFutureEncoder): Encoder module.
        horizon (PastFutureEncoder): Horizon encoder module.
        combination_type (str): Type of combination to use.
        alpha (float): Alpha parameter for the loss.
        attention (nn.MultiheadAttention): Multi-head attention module.
        gate (nn.Linear): Linear layer for weighted combination.
        decoder (nn.Sequential): Decoder module.
        mu (nn.Linear): Linear layer for output.
    """

    def __init__(
        self,
        n_target_series: int,
        n_unknown_features: int,
        n_known_calendar_features: int,
        n_known_continuous_features: int,
        embedding_size: int = 28,
        embedding_type: str = None,
        combination_type: str = "attn-comb",
        expansion_factor: int = 2,
        residual: bool = False,
        hidden_size: int = 256,
        num_layers: int = 2,
        forecast_horizon: int = 48,
        input_window_size: int = 96,
        activation_function: str = "SiLU",
        out_activation_function: str = "Identity",
        dropout_rate: float = 0.25,
        alpha: float = 0.1,
        lambda_lasso:float=1e-3,
        normal_sigma:bool=False,
        sign_res:bool=False,
        num_attention_heads: int = 4,
    ):
        """
        Multilayer Perceptron (MLP) Forecast Network for time series forecasting.

        Args:
            n_target_series (int): Number of target series.
            n_unknown_features (int): Number of unknown time-varying features.
            n_known_calendar_features (int): Number of known categorical time-varying features.
            n_known_continuous_features (int): Number of known continuous time-varying features.
            embedding_size (int, optional): Dimensionality of the embedding space. Defaults to 28.
            embedding_type (str, optional): Type of embedding to use. Defaults to None. Options: 'PosEmb', 'RotaryEmb', 'CombinedEmb'.
            combination_type (str, optional): Type of combination to use.Defaults to 'attn-comb'. Options: 'attn-comb', 'weighted-comb', 'addition-comb'.
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
        """
        super().__init__(n_target_series=n_target_series,
        n_unknown_features=n_unknown_features,
        n_known_calendar_features=n_known_calendar_features,
        n_known_continuous_features=n_known_continuous_features,
        embedding_size=embedding_size,
        embedding_type=embedding_type,
        combination_type=combination_type,
        expansion_factor = expansion_factor,
        residual=residual,
        hidden_size=hidden_size,
        num_layers=num_layers,
        forecast_horizon=forecast_horizon,
        input_window_size=input_window_size,
        activation_function=activation_function,
        out_activation_function=out_activation_function,
        dropout_rate=dropout_rate,
        alpha=alpha,
        lambda_lasso=lambda_lasso,
        num_attention_heads=num_attention_heads)
        self.linear_to_sigma = nn.Linear(hidden_size, self.n_out * forecast_horizon)
        self.normal_sigma = normal_sigma
        self.sign_res = sign_res

    def initilise_sigma_layer(self):
        #initilise weights to small random values centered around zeros
        nn.init.normal_(self.linear_to_sigma.weight, mean=0.0, std=0.01)
        
        #initilise bias to zero
        nn.init.constant_(self.linear_to_sigma.bias, 0.0)


    def forecast(self, x: torch.Tensor) -> dict:
        """
        Generates forecasts for the input sequences.

        Args:
            x (torch.Tensor): Input tensor.

        Returns
        -------
            dict: Dictionary containing the forecast predictions.
        """
        with torch.no_grad():
            loc, scale = self(x)

        return {"loc": loc, "scale": scale}

 
    def forward_gam(self, x: torch.Tensor) -> torch.Tensor:
        # Process past features
        past_features = self.encoder(x[:, :self.input_window_size, :])
        past_features_transformed = self.past_feature_transform(past_features)

        if self.n_covariates > 0:
            # Process future features
            future_features = self.horizon(x[:, self.input_window_size:, self.n_unknown:])
            future_features_transformed = self.future_feature_transform(future_features)
        
            # Combine past and future feature outputs
            combined_output =  self.bias + past_features_transformed + future_features_transformed
            #sigma = self.linear_to_sigma(past_features+future_features)
        else:
            combined_output =  self.bias + past_features_transformed
        
            #sigma = self.linear_to_sigma(past_features)
        return combined_output
    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the MLPForecastNetwork.

        Args:
            x (torch.Tensor): Input tensor.

        Returns
        -------
            torch.Tensor: Output tensor after processing through the network.
        """
        # Process past features
        
        combined_output, sigma=self.forward_gam(x)
        loc = self.out_activation(combined_output)
        loc = loc.reshape(x.size(0), self.forecast_horizon, self.n_out)

        sigma = self.linear_to_sigma(combined_output)
        sigma = sigma.reshape(x.size(0), self.forecast_horizon, self.n_out)
        
        if  (self.normal_sigma):
            sigma = sigma.mul(0.5).exp()
        return loc, sigma
    
    def step_sigma(self, batch, metric_fn):
        x, y = batch
        loc, sigma = self(x)
        res  = (y-loc.data)

        if self.normal_sigma:
            pdsit = torch.distributions.Normal(torch.zeros_like(sigma), sigma)
            loss  = -pdsit.log_prob(res).mean() 
        else:
            if self.sign_res:
                loss = F.l1_loss(sigma, res)
            else:
                loss = F.l1_loss(sigma, res.abs())
        
        return loss, loss
    
    def step(self, batch: tuple, metric_fn: callable) -> tuple:
        """
        Training step for the MLPForecastNetwork.

        Args:
            batch (tuple): Tuple containing input and target tensors.
            metric_fn (callable): Metric function to evaluate.

        Returns
        -------
            tuple: Tuple containing the loss and computed metric.
        """
        x, y = batch

        
        loc, _= self(x)
        

        loss = (
            self.alpha * F.mse_loss(loc, y)
            + (1 - self.alpha) * F.l1_loss(loc, y)
        )
        metric = loss
        loss += self.lasso_penalty()
    
        return loss, metric

