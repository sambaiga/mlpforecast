from neuralforecast.models.timesnet import  TimesBlock
from neuralforecast.common._modules import TokenEmbedding, PositionalEmbedding, TimeFeatureEmbedding
from mlpforecast.net.neuralnet import BaseNeuralNet
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft

class TimesNet(BaseNeuralNet):
    """
    TimesNet: A neural network model designed for univariate time series forecasting, 
    addressing multiple intraperiod and interperiod temporal variations.

    Args:
        n_target_series (int): Number of target series to predict.
        n_unknown_features (int): Number of unknown exogenous features.
        n_known_calendar_features (int): Number of known calendar-based features.
        n_known_continuous_features (int): Number of known continuous features.
        hidden_size (int, optional): Size of hidden layers for embedding and encoders. Defaults to 256.
        num_layers (int, optional): Number of layers in the model. Defaults to 2.
        forecast_horizon (int, optional): Forecast horizon (number of future time steps to predict). Defaults to 48.
        input_window_size (int, optional): Size of the input window (number of historical time steps). Defaults to 96.
        dropout_rate (float, optional): Dropout rate for regularization, must be between 0 and 1. Defaults to 0.25.
        alpha (float, optional): Weighting parameter for certain loss or regularization functions. Defaults to 0.1.
        conv_hidden_size (int, optional): Number of channels in the convolutional hidden layers. Defaults to 64.
        out_activation_function (str, optional): Activation function for the output layer. Defaults to "Identity".
        top_k (int, optional): Number of periodic variations to capture. Defaults to 5.
        num_kernels (int, optional): Number of convolutional kernels in the model. Defaults to 6.
    """

    def __init__(
        self,
        n_target_series: int,
        n_unknown_features: int,
        n_known_calendar_features: int,
        n_known_continuous_features: int,
        hidden_size: int = 256,
        num_layers: int = 2,
        forecast_horizon: int = 48,
        input_window_size: int = 96,
        dropout_rate: float = 0.25,
        alpha: float = 0.1,
        conv_hidden_size: int = 64,
        out_activation_function: str = "Identity",
        top_k: int = 5,
        num_kernels: int = 6
    ):
        super().__init__(n_target_series=n_target_series,
        n_unknown_features=n_unknown_features,
        n_known_calendar_features=n_known_calendar_features,
        n_known_continuous_features=n_known_continuous_features,
        hidden_size =  hidden_size,
        num_layers = num_layers,
        forecast_horizon= forecast_horizon,
        input_window_size=input_window_size,
        dropout_rate=dropout_rate,
        alpha=alpha,
        out_activation_function=out_activation_function)
        self.model = nn.ModuleList(
            [
                TimesBlock(
                    input_size=self.input_window_size,
                    h=self.forecast_horizon,
                    k=top_k,
                    hidden_size=hidden_size,
                    conv_hidden_size=conv_hidden_size,
                    num_kernels=num_kernels,
                )
                for _ in range(num_layers)
            ]
        )
       
        self.value_embedding = TokenEmbedding(c_in=self.n_channels, hidden_size=hidden_size)
        self.position_embedding = PositionalEmbedding(hidden_size=hidden_size)
        self.temporal_embedding = TimeFeatureEmbedding(
                input_size=self.n_covariates, hidden_size=hidden_size
            )
        self.dropout = nn.Dropout(p=dropout_rate)
        self.out_activation = getattr(nn, out_activation_function)()
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.projection = nn.Linear(hidden_size, self.n_out, bias=True)



    def forward(self, x):

        past_feature=x[:, :self.input_window_size , :]
        # Convolution
        input_emb = self.value_embedding(past_feature)
        input_emb = input_emb + self.position_embedding(input_emb)
        
        if self.n_covariates > 0:
            futr_exog = x[:,  self.input_window_size:, self.n_unknown :]
            futr_exog=self.temporal_embedding(futr_exog)
            input_emb=torch.cat([input_emb, futr_exog], dim=1)

        enc_out = input_emb.permute(0, 2, 1).permute(0, 2, 1)  # align temporal dimension
        for model in self.model:
            enc_out = self.layer_norm(model(enc_out))

        dec_out = self.projection(enc_out)
        forecast = self.out_activation(dec_out[:, -self.forecast_horizon :])
        return forecast
    
    