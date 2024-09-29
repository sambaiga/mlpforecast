from neuralforecast.models.fedformer import  (SeriesDecomp, 
                                              FourierBlock, 
                                              FourierCrossAttention,
                                               Encoder, LayerNorm,
                                               AutoCorrelationLayer,
                                               Decoder, DecoderLayer,
                                               DataEmbedding, EncoderLayer)
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mlpforecast.net.neuralnet import BaseNeuralNet

class  FEDformer(BaseNeuralNet):
    """
    FEDformer: Fourier Enhanced Decomposition for Time Series Forecasting.

    This PyTorch module is designed for univariate or multivariate time series forecasting, 
    leveraging Fourier-based attention mechanisms, residual learning, and forecast decomposition. 
    The model integrates both covariate information and hierarchical interpolation for accurate 
    forecasting over multiple resolutions.

    Args:
        n_target_series (int): 
            Number of target time series to forecast.
        n_unknown_features (int): 
            Number of unknown covariates (dynamic exogenous variables).
        n_known_calendar_features (int): 
            Number of known calendar features (static exogenous variables).
        n_known_continuous_features (int): 
            Number of known continuous features (dynamic exogenous variables).
        forecast_horizon (int, optional): 
            Number of steps ahead to forecast. Defaults to 48.
        input_window_size (int, optional): 
            Size of the input window (number of past time steps to consider). Defaults to 96.
        dropout_rate (float, optional): 
            Dropout rate applied to MLP layers. Defaults to 0.25.
        alpha (float, optional): 
            Weighting parameter for combining MSE and L1 loss functions. Defaults to 0.1.
        version (str, optional): 
            Specifies the version of the Fourier block. Defaults to "Fourier".
        modes (int, optional): 
            Number of Fourier modes to be used in the Fourier block. Defaults to 64.
        mode_select (str, optional): 
            Method for selecting modes in the Fourier layer. Options include "random" and "top-k". Defaults to "random".
        hidden_size (int, optional): 
            Size of hidden layers. Defaults to 128.
        dropout (float, optional): 
            Dropout rate applied in the embedding layers. Defaults to 0.05.
        n_head (int, optional): 
            Number of attention heads used in multi-head attention layers. Defaults to 8.
        decoder_input_size_multiplier (float, optional): 
            Fraction of the input window size to use for the decoder's input. Defaults to 0.5.
        conv_hidden_size (int, optional): 
            Number of hidden units in the convolutional layers. Defaults to 32.
        activation (str or callable, optional): 
            Activation function used in the model. Defaults to "gelu".
        encoder_layers (int, optional): 
            Number of layers in the encoder. Defaults to 2.
        decoder_layers (int, optional): 
            Number of layers in the decoder. Defaults to 1.
        MovingAvg_window (int, optional): 
            Window size for moving average decomposition. Defaults to 25.
        out_activation_function (str, optional): 
            Output activation function applied to the final forecast. Defaults to "Identity".

    Attributes:
        n_out (int): 
            Number of target series to forecast.
        n_channels (int): 
            Total number of input channels, including covariates and target series.
        decomp (SeriesDecomp): 
            Series decomposition module based on moving average.
        out_activation (torch.nn.Module): 
            The activation function applied to the final forecast.
        enc_embedding (DataEmbedding): 
            Embedding module for encoder inputs.
        dec_embedding (DataEmbedding): 
            Embedding module for decoder inputs.
        encoder (Encoder): 
            Encoder consisting of multiple AutoCorrelation layers.
        decoder (Decoder): 
            Decoder consisting of multiple AutoCorrelation layers.

    Methods:
        forecast(x: torch.Tensor) -> dict:
            Generates forecasts based on the input sequences.

        forward(x: torch.Tensor) -> torch.Tensor:
            Executes the forward pass, producing the forecasted output.

        step(batch: tuple, metric_fn: callable) -> tuple:
            Performs a training step, computes loss, and evaluates metrics.

    Example:
        >>> model = FEDformer(
        >>>     n_target_series=1, 
        >>>     n_unknown_features=3, 
        >>>     n_known_calendar_features=2, 
        >>>     n_known_continuous_features=4
        >>> )
        >>> x = torch.randn(4, 96, 10)  # Batch size 4, window size 96, 10 features
        >>> forecast = model.forecast(x)
    """
    def __init__(
        self,
        n_target_series: int,
        n_unknown_features: int,
        n_known_calendar_features: int,
        n_known_continuous_features: int,
        forecast_horizon: int = 48,
        input_window_size: int = 96,
        dropout_rate: float = 0.0,
        alpha: float = 0.1,
        version: str = "Fourier",
        modes: int = 64,
        mode_select: str = "random",
        hidden_size: int = 128,
        dropout: float = 0.05,
        n_head: int = 8,
        decoder_input_size_multiplier:float=0.5,
        conv_hidden_size: int = 32,
        activation: str = "gelu",
        encoder_layers: int = 2,
        decoder_layers: int = 1,
        MovingAvg_window: int = 25,
        out_activation_function: str = "Identity"):
        super().__init__(n_target_series=n_target_series,
        n_unknown_features=n_unknown_features,
        n_known_calendar_features= n_known_calendar_features,
        n_known_continuous_features=n_known_continuous_features,
        forecast_horizon=forecast_horizon,
        input_window_size = input_window_size,
        dropout_rate = dropout_rate,
        hidden_size=hidden_size,
        alpha=alpha,
        out_activation_function=out_activation_function)
        

        self.stat_exog_size = 0
        self.decomp = SeriesDecomp(MovingAvg_window)
        self.output_attention = False
        self.enc_in = 1
        self.dec_in = 1
        self.label_len = int(np.ceil(input_window_size * decoder_input_size_multiplier))

        self.enc_embedding = DataEmbedding(
            c_in=self.enc_in,
            exog_input_size=self.n_covariates,
            hidden_size=hidden_size,
            pos_embedding=False,
            dropout=dropout,
        )
        self.dec_embedding = DataEmbedding(
            self.dec_in,
            exog_input_size=self.n_covariates,
            hidden_size=hidden_size,
            pos_embedding=False,
            dropout=dropout,
        )

        encoder_self_att = FourierBlock(
            in_channels=hidden_size,
            out_channels=hidden_size,
            seq_len=self.input_window_size,
            modes=modes,
            mode_select_method=mode_select,
        )
        decoder_self_att = FourierBlock(
            in_channels=hidden_size,
            out_channels=hidden_size,
            seq_len=self.input_window_size // 2 + self.forecast_horizon,
            modes=modes,
            mode_select_method=mode_select,
        )
        decoder_cross_att = FourierCrossAttention(
            in_channels=hidden_size,
            out_channels=hidden_size,
            seq_len_q=self.input_window_size // 2 + self.forecast_horizon,
            seq_len_kv=self.input_window_size,
            modes=modes,
            mode_select_method=mode_select,
        )

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(encoder_self_att, hidden_size, n_head),
                    hidden_size=hidden_size,
                    conv_hidden_size=conv_hidden_size,
                    MovingAvg=MovingAvg_window,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(encoder_layers)
            ],
            norm_layer=LayerNorm(hidden_size),
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AutoCorrelationLayer(decoder_self_att, hidden_size, n_head),
                    AutoCorrelationLayer(decoder_cross_att, hidden_size, n_head),
                    hidden_size=hidden_size,
                    c_out=self.n_out,
                    conv_hidden_size=conv_hidden_size,
                    MovingAvg=MovingAvg_window,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(decoder_layers)
            ],
            norm_layer=LayerNorm(hidden_size),
            projection=nn.Linear(hidden_size, self.n_out, bias=True),
        )
        
        
    def forward(self, x):
        insample_y = x[:, :self.input_window_size , :self.n_unknown]
        if self.n_covariates > 0:
            futr_exog  = x[:,  :, self.n_unknown :]
            x_mark_enc = futr_exog[:, : self.input_window_size, :]
            x_mark_dec = futr_exog[:, -(self.label_len + self.forecast_horizon) :, :]
        else:
            x_mark_enc = None
            x_mark_dec = None

        x_dec = torch.zeros(
            size=(len(insample_y), self.forecast_horizon, self.dec_in), device=insample_y.device
        )
        x_dec = torch.cat([insample_y[:, -self.label_len :, :], x_dec], dim=1)

        # decomp init
        mean = torch.mean(insample_y, dim=1).unsqueeze(1).repeat(1, self.forecast_horizon, 1)
        zeros = torch.zeros(
            [x_dec.shape[0], self.forecast_horizon, x_dec.shape[2]], device=insample_y.device
        )
        seasonal_init, trend_init = self.decomp(insample_y)
        # decoder input
        trend_init = torch.cat([trend_init[:, -self.label_len :, :], mean], dim=1)
        seasonal_init = torch.cat(
            [seasonal_init[:, -self.label_len :, :], zeros], dim=1
        )
        # enc
        enc_out = self.enc_embedding(insample_y, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        # dec
        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
        seasonal_part, trend_part = self.decoder(
            dec_out, enc_out, x_mask=None, cross_mask=None, trend=trend_init
        )
        # final
        dec_out = trend_part + seasonal_part

        forecast = self.out_activation(dec_out[:, -self.forecast_horizon :])
        return forecast
        
       
    
        
