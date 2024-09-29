
"""
This module defines the mlp-based models based on the MLPF block used to produce potent and probabilistic forecasts.
"""
from mlpforecast.forecaster.regressor import RegressorForecast
from mlpforecast.forecaster.nhits import NHITSForecast
from mlpforecast.forecaster.nbeats import NBEATSForecast
from mlpforecast.forecaster.timesnet import TimesNetForecast
from mlpforecast.forecaster.fedformer import FeDformerForecast
__all__ = ["RegressorForecast", 'NHITSForecast', 'NBEATSForecast', "TimesNetForecast", "FeDformerForecast"]