
"""
This module defines the mlp-based models based on the MLPF block used to produce potent and probabilistic forecasts.
"""
from mlpforecast.forecaster.regressor import CatBoostForecast, XGBoostForecast, LightGBMForecast, LinearRegForecast
from mlpforecast.forecaster.nhits import NHITSForecast
from mlpforecast.forecaster.nbeats import NBEATSForecast
from mlpforecast.forecaster.timesnet import TimesNetForecast
from mlpforecast.forecaster.fedformer import FeDformerForecast
from mlpforecast.forecaster.mlpf import MLPForecast
from mlpforecast.forecaster.mlpfgam import MLPGAMForecast
from mlpforecast.forecaster.mlpf_org import MLPForecastOG
__all__ = ["RegressorForecast", 'NHITSForecast', 'MLPForecastOG',
           'NBEATSForecast', "TimesNetForecast",
           "FeDformerForecast", "MLPForecast", "MLPGAMForecast",
           "XGBoostForecast", "LightGBMForecast", 
           "LinearRegForecast", "CatBoostForecast"]