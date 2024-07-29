
"""
This module defines the mlp-based models based on the MLPF block used to produce potent and probabilistic forecasts.
"""
from mlpforecast.forecaster.mlp import MLPForecast
from mlpforecast.forecaster.mlpfqr import MLPFQRForecast
from mlpforecast.forecaster.mlplaplace import MLPLaplaceForecast
from mlpforecast.forecaster.mlpmultivariate import MLPMultivarGaussForecast
from mlpforecast.forecaster.mlpqr import MLPQRForecast
__all__ = ["MLPForecast", "MLPFQRForecast", "MLPQRForecast", "MLPMultivarGaussForecast", "MLPLaplaceForecast"]