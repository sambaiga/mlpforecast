import pandas as pd
import numpy as np
from IPython.display import clear_output
from aquarel import load_theme
from mlpforecast.data.transform import DatasetObjective
from sklearn.preprocessing import (
    MinMaxScaler,
    RobustScaler,
    PowerTransformer,
)

data = pd.read_parquet("albania_res.parquet")
data.index = pd.to_datetime(data.index, utc="UTC")

common_params = {
    "target_series": ["NetLoad"],
    "unknown_features": [],
    "calendar_variables": ["HOUR", "Session"],
    "known_calendar_features": ["HOUR-cosin", "Session-cosin"],
    "known_continuous_features": ["NetLoad_lag_48", "NetLoad_lag_336", "Temperature"],
    "input_window_size": 96,
    "forecast_horizon": 48,
}



data_params = {
    "input_scaler": PowerTransformer('yeo-johnson'),
    "target_scaler": PowerTransformer('box-cox'),
    "lags": [1, 7],
    "windows": [],
    "window_funcs": ["mean"],
    "period": "30min",
    "date_column": "timestamp",
}

data_params.update(common_params)
ds = DatasetObjective(**data_params)
clear_output()