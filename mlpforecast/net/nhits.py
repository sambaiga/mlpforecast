from neuralforecast.models.nhits import  NHITSBlock, _IdentityBasis
from mlpforecast.model.base_model import BaseForecastModel
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from mlpforecast.net.neuralnet import BaseNeuralNet

class  NHITS(BaseNeuralNet):
    """
    NHITS: Neural Hierarchical Interpolation for Time Series Forecasting.

    A PyTorch module designed for univariate or multivariate time series forecasting. 
    It employs residual learning, forecast decomposition, and MLP layers with 
    hierarchical pooling. NHITS consists of identity-based blocks that operate on 
    different resolutions within the time series, allowing for efficient hierarchical 
    forecasting.

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
        stack_types (list, optional): 
            Types of blocks used in the model. Currently, only 'identity' blocks are supported. Defaults to ["identity", "identity", "identity"].
        n_blocks (list, optional): 
            Number of blocks in each stack. Defaults to [1, 1, 1].
        mlp_units (list of lists, optional): 
            Number of units in each MLP layer for each stack. Defaults to 3*[[512, 512]].
        n_pool_kernel_size (list, optional): 
            Kernel sizes for downsampling in each stack. Defaults to [2, 2, 1].
        n_freq_downsample (list, optional): 
            Downsampling factors for frequency resolution in each block. Defaults to [4, 2, 1].
        pooling_mode (str, optional): 
            Type of pooling applied to downsample inputs. Defaults to "MaxPool1d".
        interpolation_mode (str, optional): 
            Mode for interpolation in the identity block. Defaults to "linear".
        activation (str or callable, optional): 
            Activation function used in MLP layers. Defaults to "ReLU".
        decompose_forecast (bool, optional): 
            Whether to return decomposed forecasts for each block. Defaults to False.
        out_activation_function (str, optional): 
            Output activation function applied to the final forecast. Defaults to "Identity".

    Attributes:
        blocks (torch.nn.ModuleList): 
            List of blocks forming the hierarchical stack for time series forecasting.
        n_out (int): 
            Number of target series (output time series).
        n_channels (int): 
            Total number of input channels, including covariates and target series.
        out_activation (torch.nn.Module): 
            The activation function applied to the final forecast.

    Methods:
        forecast(x: torch.Tensor) -> dict:
            Generates forecasts for the input sequences.

        forward(x: torch.Tensor) -> torch.Tensor:
            Forward pass of the model, returning the forecast or decomposed forecasts.

        step(batch: tuple, metric_fn: callable) -> tuple:
            Executes a training step, computes loss, and evaluates the metric function.

    Example:
        >>> model = NHITS(
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
        dropout_rate: float = 0.25,
        alpha: float = 0.1,
        stack_types: list = ["identity", "identity", "identity"],
        n_blocks: list = [1, 1, 1],
        mlp_units: list = 3 * [[512, 512]],
        n_pool_kernel_size: list = [2, 2, 1],
        n_freq_downsample: list = [4, 2, 1],
        pooling_mode: str = "MaxPool1d",
        interpolation_mode: str = "linear",
        activation="ReLU",
        decompose_forecast:bool=False,
        out_activation_function: str = "Identity"):
        super().__init__(n_target_series=n_target_series,
        n_unknown_features=n_unknown_features,
        n_known_calendar_features=n_known_calendar_features,
        n_known_continuous_features=n_known_continuous_features,
        forecast_horizon=forecast_horizon,
        input_window_size=input_window_size,
        dropout_rate=dropout_rate,
        alpha=alpha,
        out_activation_function=out_activation_function)
        
        self.stat_exog_size = 0
        self.decompose_forecast = decompose_forecast
        blocks = self.create_stack(
            h=self.forecast_horizon,
            input_size=self.input_window_size,
            stack_types=stack_types,
            futr_input_size=self.n_covariates,
            hist_input_size=n_unknown_features,
            stat_input_size=self.stat_exog_size,
            n_blocks=n_blocks,
            mlp_units=mlp_units,
            n_pool_kernel_size=n_pool_kernel_size,
            n_freq_downsample=n_freq_downsample,
            pooling_mode=pooling_mode,
            interpolation_mode=interpolation_mode,
            dropout_prob_theta=dropout_rate,
            activation=activation,
            n_out_size=self.n_out 
        )
        self.blocks = torch.nn.ModuleList(blocks)
        

    def create_stack(
        self,
        h,
        input_size,
        stack_types,
        n_blocks,
        mlp_units,
        n_pool_kernel_size,
        n_freq_downsample,
        pooling_mode,
        interpolation_mode,
        dropout_prob_theta,
        activation,
        futr_input_size,
        hist_input_size,
        stat_input_size,
        n_out_size,
    ):

        block_list = []
        for i in range(len(stack_types)):
            for block_id in range(n_blocks[i]):

                assert (
                    stack_types[i] == "identity"
                ), f"Block type {stack_types[i]} not found!"

                n_theta = input_size +n_out_size * max(
                    h // n_freq_downsample[i], 1
                )
                basis = _IdentityBasis(
                    backcast_size=input_size,
                    forecast_size=h,
                    out_features=n_out_size,
                    interpolation_mode=interpolation_mode,
                )

                nbeats_block = NHITSBlock(
                    h=h,
                    input_size=input_size,
                    futr_input_size=futr_input_size,
                    hist_input_size=hist_input_size,
                    stat_input_size=stat_input_size,
                    n_theta=n_theta,
                    mlp_units=mlp_units,
                    n_pool_kernel_size=n_pool_kernel_size[i],
                    pooling_mode=pooling_mode,
                    basis=basis,
                    dropout_prob=dropout_prob_theta,
                    activation=activation,
                )

                # Select type of evaluation and apply it to all layers of block
                block_list.append(nbeats_block)

        return block_list

    

    def forward(self, x):
        insample_y = x[:, :self.input_window_size , :self.n_out].squeeze(-1)
        futr_exog  = x[:,  :, self.n_unknown :]
        if self.n_unknown_features>0:
            hist_exog = x[:, :self.input_window_size , self.n_out:self.n_unknown_features].squeeze(-1)
        else:
            hist_exog = None
        stat_exog = None
        
        # insample
        residuals = insample_y.flip(dims=(-1,))  # backcast init
        

        forecast = insample_y[:, -1:, None]  # Level with Naive1
        
        block_forecasts = [forecast.repeat(1, self.forecast_horizon, 1)]
        
        for i, block in enumerate(self.blocks):
           
            backcast, block_forecast = block(
                insample_y=residuals,
                futr_exog=futr_exog,
                hist_exog=hist_exog,
                stat_exog=stat_exog,
            )
            residuals = (residuals - backcast) 
            forecast = forecast + block_forecast

            if self.decompose_forecast:
                block_forecasts.append(block_forecast)
        
        forecast = self.out_activation(forecast)
        if self.decompose_forecast:
            block_forecasts = torch.stack(block_forecasts)
            block_forecasts = block_forecasts.permute(1, 0, 2, 3)
            block_forecasts = block_forecasts.squeeze(-1)  # univariate output
            return block_forecasts
        else:
            return forecast
        
    