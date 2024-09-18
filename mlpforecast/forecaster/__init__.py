
"""
This module defines the mlp-based models based on the MLPF block used to produce potent and probabilistic forecasts.
"""
from mlpforecast.forecaster.mlp import MLPForecast
from mlpforecast.forecaster.quantile import MLPFQRForecast
from mlpforecast.forecaster.parametric import MLPLaplaceForecast
from mlpforecast.forecaster.conformal import MLPGAMCRForecast
__all__ = ["MLPForecast", 'MLPFQRForecast', 'MLPLaplaceForecast', "MLPGAMCRForecast"]