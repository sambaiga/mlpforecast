import numpy as np
import pytimetk as tk
from sklearn.base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler

from mlpforecast.data.processing import (
    combine_past_future_exogenous,
    fourier_series_t,
    get_n_sample_per_day,
)


class DatasetObjective(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    """
    A class to handle dataset preprocessing for time series forecasting, including lag and rolling window feature
    augmentation, scaling, and feature extraction.

    Attributes:
        target_series (list): List of target series to forecast.
        unknown_features (list): List of unknown features.
        calender_variable (list): List of calendar variables.
        known_calender_features (list): List of known calendar features.
        known_continuous_features (list): List of known continuous features.
        input_scaler (object): Scaler for input features (default is MinMaxScaler).
        target_scaler (object): Scaler for target features (default is MinMaxScaler).
        lags (list): List of lag values for feature augmentation.
        window (list): List of window sizes for rolling window calculations.
        window_func (list): List of window functions for rolling window calculations.
        period (str): Period of the time series data.
        input_window_size (int): Size of the input window.
        forecast_horizon (int): Forecast horizon.
        date_column (str): Name of the date column.
    """

    def __init__(
        self,
        target_series=["NetLoad"],
        unknown_features=[],
        calender_variable=[],
        known_calender_features=[],
        known_continuous_features=[],
        input_scaler=MinMaxScaler(),
        target_scaler=MinMaxScaler(),
        lags=[1 * 48, 7 * 48],
        window=[1 * 48, 7 * 48],
        window_func=["mean"],
        period="30min",
        input_window_size=96,
        forecast_horizon=48,
        date_column="timestamp",
    ):
        """
        Initializes the DatasetObjective with the specified parameters.

        Args:
            target_series (list or str): List of target series to forecast. \
                If a single string, it will be converted to a list.
            unknown_features (list): List of unknown features.
            calender_variable (list): List of calendar variables.
            known_calender_features (list): List of known calendar features.
            known_continuous_features (list): List of known continuous features.
            input_scaler (object): Scaler for input features (default is MinMaxScaler).
            target_scaler (object): Scaler for target features (default is MinMaxScaler).
            lags (list): List of lag values for feature augmentation.
            window (list): List of window sizes for rolling window calculations.
            window_func (list): List of window functions for rolling window calculations.
            period (str): Period of the time series data (default is "30min").
            input_window_size (int): Size of the input window (default is 96).
            forecast_horizon (int): Forecast horizon (default is 48).
            date_column (str): Name of the date column (default is "timestamp").

        Raises:
            ValueError: If target_series is not a string or list of strings.
        """

        if isinstance(target_series, str):
            target_series = [target_series]
        elif not isinstance(target_series, list):
            raise ValueError(
                f"{target_series} should be a string or a list of strings."
            )

        self.numerical_features = unknown_features + known_continuous_features
        self.calender_variable = calender_variable
        self.date_column = date_column
        self.target_series = target_series
        self.unknown_features = unknown_features
        self.known_calender_features = known_calender_features
        self.known_continuous_features = known_continuous_features
        self.input_window_size = input_window_size
        self.forecast_horizon = forecast_horizon
        self.input_scaler = input_scaler
        self.target_scaler = target_scaler
        self.lags = lags
        self.window = window
        self.window_func = window_func
        self.period = period
        self.exog_periods = None
        self.n_samples = get_n_sample_per_day(self.period)
        self.max_data_drop=0

        steps = []

        if (self.lags is not None) and (len(self.lags) > 0):
            lags_scaled = [int(lag * self.n_samples) for lag in sorted(self.lags)]
            self.max_data_drop=max(self.max_data_drop, max(lags_scaled))
            transformer_lags = FunctionTransformer(
                tk.augment_lags,
                kw_args={
                    "date_column": date_column,
                    "value_column": target_series,
                    "lags": lags_scaled,
                },
            )
            steps += [("lags_step", transformer_lags)]

        if (self.window is not None) and (len(self.window) > 0):
            window_scaled = [
                int(wsize * self.n_samples) for wsize in sorted(self.window)
            ]
            self.max_data_drop=max(self.max_data_drop, max(window_scaled))
            transformer_rolling = FunctionTransformer(
                tk.augment_rolling,
                kw_args={
                    "date_column": date_column,
                    "window_func": window_func,
                    "value_column": target_series,
                    "window": window_scaled,
                },
            )
            steps += [("rolling_step", transformer_rolling)]

        steps += [("dropnan_step", FunctionTransformer(lambda x: x.dropna()))]
        feature_pipeline = Pipeline(steps=steps)

        transformers = []
        if len(self.numerical_features) > 0:
            numeric_transformer = Pipeline(steps=[("scaler", self.input_scaler)])
            transformers += [
                ("feat_scaler", numeric_transformer, self.numerical_features)
            ]

        target_transformer = Pipeline(steps=[("target_scaler", self.target_scaler)])
        transformers += [("target_scaler", target_transformer, target_series)]

        data_pipeline = ColumnTransformer(
            transformers=transformers,
            remainder="passthrough",
            verbose_feature_names_out=False,
        )
        # Set output transform to pandas format
        data_pipeline.set_output(transform="pandas")

        self.data_pipeline = Pipeline(
            steps=[("feature_extraction", feature_pipeline), ("scaling", data_pipeline)]
        )

    def fit(self, data, y=None):
        self.data_pipeline.fit(data)

        if len(self.calender_variable) > 0:
            exog = data[self.calender_variable].values
            self.exog_periods = [
                len(np.unique(exog[:, size])) for size in range(exog.shape[-1])
            ]

        return self

    def transform(self, data):
        data_transfomed = self.data_pipeline.transform(data.copy())
        data_transfomed = data_transfomed.sort_values(by=self.date_column)

        if len(self.calender_variable) > 0:
            exog = data_transfomed[self.calender_variable].astype(np.float32).values
            if self.exog_periods is None:
                self.exog_periods = [
                    len(np.unique(exog[:, size])) for size in range(exog.shape[-1])
                ]
            seasonalities = np.hstack([
                fourier_series_t(exog[:, i], self.exog_periods[i], 1)
                for i in range(len(self.exog_periods))
            ])
            for i, col in enumerate(self.calender_variable):
                data_transfomed[f"{col}-sin"] = seasonalities[:, i]
                data_transfomed[f"{col}-cosin"] = (
                    seasonalities[:, i] + seasonalities[:, i + 1]
                )
                data_transfomed[f"{col}-cos"] = seasonalities[:, i + 1]
                i += 2

        features = data_transfomed[
            self.target_series + self.unknown_features
        ].values.astype(np.float64)
        targets = data_transfomed[self.target_series].values.astype(np.float64)

        future_exogenous = data_transfomed[
            self.known_continuous_features + self.known_calender_features
        ].values.astype(np.float64)
        features = np.concatenate([features, future_exogenous], 1).astype(np.float64)

        assert (
            len(features) > 0
        ), "Ensure you have at least one historical features to train the model."

        if len(future_exogenous) > 0:
            future_exogenous = np.squeeze(
                np.lib.stride_tricks.sliding_window_view(
                    future_exogenous[self.input_window_size :],
                    (self.forecast_horizon, future_exogenous.shape[1]),
                ),
                axis=1,
            )
            future_exogenous = future_exogenous.reshape(
                future_exogenous.shape[0], self.forecast_horizon, -1
            )
        else:
            future_exogenous = None

        features = np.squeeze(
            np.lib.stride_tricks.sliding_window_view(
                features,
                window_shape=(self.input_window_size, features.shape[1]),
            ),
            axis=1,
        )
        features = features.reshape(features.shape[0], self.input_window_size, -1)[
            : -self.forecast_horizon
        ]

        targets = np.squeeze(
            np.lib.stride_tricks.sliding_window_view(
                targets[self.input_window_size :],
                (self.forecast_horizon, targets.shape[1]),
            ),
            axis=1,
        )
        targets = targets.reshape(targets.shape[0], self.forecast_horizon, -1)
        features = combine_past_future_exogenous(features, future_exogenous)

        return features, targets
    


