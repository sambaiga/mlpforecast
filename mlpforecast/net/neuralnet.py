from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseNeuralNet(nn.Module):
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
        out_activation_function: str = "Identity",
    ):
        super().__init__()
        
        self.n_out = n_target_series
        self.n_unknown = n_unknown_features + self.n_out
        self.n_covariates = n_known_calendar_features + n_known_continuous_features
        self.n_channels = self.n_unknown + self.n_covariates
        self.input_window_size = input_window_size
        self.forecast_horizon = forecast_horizon
        self.dropout = nn.Dropout(p=dropout_rate)
        self.out_activation = getattr(nn, out_activation_function)()
        self.n_unknown_features = n_unknown_features
        self.alpha = alpha
       

    def forecast(self, x: torch.Tensor) -> dict:
        """
        Generates forecasts for the input sequences.

        Args:
            x (torch.Tensor): Input tensor containing time series data.

        Returns:
            dict: A dictionary with the predicted forecast, where the key is 'loc' and 
                  the value is a tensor of predicted values.
        """
        with torch.no_grad():
            loc = self(x)

        return {"loc": loc}

    def forward(self, x):
        pass

        
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

        y_pred = self(x)

        loss = (
            self.alpha * F.mse_loss(y_pred, y, reduction="none").sum(dim=(1, 2)).mean()
            + (1 - self.alpha) * F.l1_loss(y_pred, y, reduction="none").sum(dim=(1, 2)).mean()
        )

        metric = metric_fn(y_pred, y)

        return loss, metric
    
    
        
        
       
        
