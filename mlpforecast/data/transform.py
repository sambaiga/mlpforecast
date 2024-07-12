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
from mlpforecast.data.utils import (
    _validate_target_series,
    extract_daily_sequences,
    extract_feature_sequences,
    extract_target_sequences,
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
        target_series: list[str] | str = ["NetLoad"],
        unknown_features: list[str] = [],
        calendar_variables: list[str] = [],
        known_calendar_features: list[str] = [],
        known_continuous_features: list[str] = [],
        input_scaler: object = MinMaxScaler(),
        target_scaler: object = MinMaxScaler(),
        lags: list[int] = [1, 7],
        windows: list[int] = [1, 7],
        window_funcs: list[str] = ["mean"],
        period: str = "30min",
        input_window_size: int = 96,
        forecast_horizon: int = 48,
        date_column: str = "timestamp",
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

        self.target_series = _validate_target_series(target_series)
        self.unknown_features = unknown_features
        self.calendar_variables = calendar_variables
        self.known_calendar_features = known_calendar_features
        self.known_continuous_features = known_continuous_features
        self.input_scaler = input_scaler
        self.target_scaler = target_scaler
        self.lags = lags
        self.windows = windows
        self.window_funcs = window_funcs
        self.period = period
        self.input_window_size = input_window_size
        self.forecast_horizon = forecast_horizon
        self.date_column = date_column
        self.n_samples = get_n_sample_per_day(self.period)
        self.max_data_drop = 0
        self.daily_features = False
        self.exog_periods = None
        self.data_pipeline = self._build_pipeline()

    def _build_pipeline(self) -> Pipeline:
        feature_pipeline = self._build_feature_pipeline()
        data_pipeline = self._build_data_pipeline()
        return Pipeline(
            steps=[("feature_extraction", feature_pipeline), ("scaling", data_pipeline)]
        )

    def _build_feature_pipeline(self) -> Pipeline:
        steps = []

        if self.lags:
            lags_scaled = [int(lag * self.n_samples) for lag in sorted(self.lags)]
            self.max_data_drop = max(self.max_data_drop, max(lags_scaled))
            transformer_lags = FunctionTransformer(
                tk.augment_lags,
                kw_args={
                    "date_column": self.date_column,
                    "value_column": self.target_series,
                    "lags": lags_scaled,
                },
            )
            steps.append(("lags_step", transformer_lags))

        if self.windows:
            windows_scaled = [int(w * self.n_samples) for w in sorted(self.windows)]
            self.max_data_drop = max(self.max_data_drop, max(windows_scaled))
            transformer_rolling = FunctionTransformer(
                tk.augment_rolling,
                kw_args={
                    "date_column": self.date_column,
                    "window_func": self.window_funcs,
                    "value_column": self.target_series,
                    "window": windows_scaled,
                },
            )
            steps.append(("rolling_step", transformer_rolling))

        steps.append(("dropnan_step", FunctionTransformer(lambda x: x.dropna())))
        return Pipeline(steps=steps)

    def _build_data_pipeline(self) -> ColumnTransformer:
        transformers = []
        if self.unknown_features or self.known_continuous_features:
            numerical_features = self.unknown_features + self.known_continuous_features
            numeric_transformer = Pipeline(steps=[("scaler", self.input_scaler)])
            transformers.append(
                (
                    "feat_scaler",
                    numeric_transformer,
                    numerical_features,
                )
            )

        target_transformer = Pipeline(steps=[("target_scaler", self.target_scaler)])
        transformers.append(("target_scaler", target_transformer, self.target_series))

        data_pipeline = ColumnTransformer(
            transformers=transformers,
            remainder="passthrough",
            verbose_feature_names_out=False,
        )
        # Set output transform to pandas format
        data_pipeline.set_output(transform="pandas")
        return data_pipeline

    def fit(self, data, y=None):
        """
        Fit the data pipeline to the given data.
        """
        self.data_pipeline.fit(data)
        if self.calendar_variables:
            exog = data[self.calendar_variables].values
            self.exog_periods = [
                len(np.unique(exog[:, size])) for size in range(exog.shape[-1])
            ]
        return self

    def transform(self, data):
        """
        Transform the data using the fitted pipeline.
        """
        data_transformed = self.data_pipeline.transform(data.copy())
        data_transformed = data_transformed.sort_values(by=self.date_column)
        self.data = data_transformed.copy()

        if self.calendar_variables:
            self._add_fourier_features(data_transformed)

        features, targets = self._extract_features_and_targets(data_transformed)
        future_exogenous = self._extract_future_exogenous(data_transformed)

        features = combine_past_future_exogenous(features, future_exogenous)
        self.data = data_transformed
        return features, targets

    def _add_fourier_features(self, data_transformed):
        exog = data_transformed[self.calendar_variables].astype(np.float32).values
        if self.exog_periods is None:
            self.exog_periods = [
                len(np.unique(exog[:, size])) for size in range(exog.shape[-1])
            ]
        seasonalities = np.hstack(
            [
                fourier_series_t(exog[:, i], self.exog_periods[i], 1)
                for i in range(len(self.exog_periods))
            ]
        )
        for i, col in enumerate(self.calendar_variables):
            data_transformed[f"{col}-sin"] = seasonalities[:, i]
            data_transformed[f"{col}-cosin"] = (
                seasonalities[:, i] + seasonalities[:, i + 1]
            )
            data_transformed[f"{col}-cos"] = seasonalities[:, i + 1]
            i += 2

    def _extract_features_and_targets(self, data_transformed):
        features = data_transformed[
            self.target_series + self.unknown_features
        ].values.astype(np.float64)
        targets = data_transformed[self.target_series].values.astype(np.float64)
        future_exogenous = data_transformed[
            self.known_continuous_features + self.known_calendar_features
        ].values.astype(np.float64)
        features = np.concatenate([features, future_exogenous], axis=1).astype(
            np.float64
        )
        assert (
            len(features) > 0
        ), "Ensure you have at least one historical feature to train the model."

        if not self.daily_features:
            features = extract_feature_sequences(
                features,
                input_window_size=self.input_window_size,
                forecast_horizon=self.forecast_horizon,
            )
            targets = extract_target_sequences(
                targets,
                input_window_size=self.input_window_size,
                forecast_horizon=self.forecast_horizon,
            )
        else:
            features = extract_daily_sequences(
                features,
                input_window_size=self.input_window_size,
                forecast_horizon=self.forecast_horizon,
                target_mode=False,
            )
            targets = extract_daily_sequences(
                targets,
                input_window_size=self.input_window_size,
                forecast_horizon=self.forecast_horizon,
                target_mode=True,
            )
        return features, targets

    def _extract_future_exogenous(self, data_transformed):
        future_exogenous = data_transformed[
            self.known_continuous_features + self.known_calendar_features
        ].values.astype(np.float64)

        if len(future_exogenous) == 0:
            future_exogenous = None

        if not self.daily_features:
            future_exogenous = extract_target_sequences(
                future_exogenous,
                input_window_size=self.input_window_size,
                forecast_horizon=self.forecast_horizon,
            )
        else:
            future_exogenous = extract_daily_sequences(
                future_exogenous,
                input_window_size=self.input_window_size,
                forecast_horizon=self.forecast_horizon,
                target_mode=True,
            )
        return future_exogenous
