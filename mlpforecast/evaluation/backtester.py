import logging
from copy import deepcopy
from functools import partial

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Forecaster")


class TimeSeriesSplitter:
    """Cross validation splitter for time series data"""

    def __init__(
        self,
        df,
        forecast_len=1,
        incremental_len=None,
        n_splits=None,
        min_train_len=None,
        window_type="expanding",
        date_col=None,
    ):
        """Initializes object with DataFrame and splits data

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame object containing time index, response, and other features
        forecast_len : int
            forecast length; default as 1
        incremental_len : int
            the number of observations between each successive backtest period; default as forecast_len
        n_splits : int; default None
            number of splits; when n_splits is specified, min_train_len will be ignored
        min_train_len : int
            the minimum number of observations required for the training period
        window_type : {'expanding', 'rolling }; default 'expanding'
            split scheme
        date_col : str
            optional for user to provide date columns; note that it stills uses discrete index
            as splitting scheme while `date_col` is used for better visualization only

        Attributes
        ----------
        _split_scheme : dict{split_meta}
            meta data of ways to split train and test set
        """

        self.df = df.copy()
        self.min_train_len = min_train_len
        self.incremental_len = incremental_len
        self.forecast_len = forecast_len
        self.n_splits = n_splits
        self.window_type = window_type
        self.date_col = None
        self.dt_array = None

        if date_col is not None:
            self.date_col = date_col
            # support cases for multiple observations
            self.dt_array = pd.to_datetime(np.sort(self.df[self.date_col].unique()))

        self._set_defaults()

        # validate
        self._validate_params()

        # init meta data of how to split
        self._split_scheme = {}

        # timeseries cross validation split
        self._set_split_scheme()

    def _set_defaults(self):
        if self.date_col is None:
            self._full_len = self.df.shape[0]
        else:
            self._full_len = len(self.dt_array)

        if self.incremental_len is None:
            self.incremental_len = self.forecast_len

        # if n_splits is specified, set min_train_len internally
        if self.n_splits:
            self.min_train_len = (
                self._full_len
                - self.forecast_len
                - (self.n_splits - 1) * self.incremental_len
            )

    def _validate_params(self):
        if self.min_train_len is None and self.n_splits is None:
            raise ValueError("min_train_len and n_splits cannot both be None...")

        if self.window_type not in ["expanding", "rolling"]:
            raise ValueError("unknown window type...")

        # forecast length invalid
        if self.forecast_len <= 0:
            raise ValueError("holdout period length must be positive...")

        # train + test length cannot be longer than df length
        if self.min_train_len + self.forecast_len > self._full_len:
            raise ValueError("required time span is more than the full data frame...")

        if self.n_splits is not None and self.n_splits < 1:
            raise ValueError("n_split must be a positive number")

        if self.date_col:
            if self.date_col not in self.df.columns:
                raise ValueError("date_col not found in df provided.")

    def _set_split_scheme(self):
        """set meta data of ways to split train and test set"""
        test_end_min = self.min_train_len - 1
        test_end_max = self._full_len - self.forecast_len
        test_seq = range(test_end_min, test_end_max, self.incremental_len)

        split_scheme = {}
        for i, train_end_idx in enumerate(test_seq):
            split_scheme[i] = {}
            train_start_idx = (
                train_end_idx - self.min_train_len + 1
                if self.window_type == "rolling"
                else 0
            )
            split_scheme[i]["train_idx"] = range(train_start_idx, train_end_idx + 1)
            split_scheme[i]["test_idx"] = range(
                train_end_idx + 1, train_end_idx + self.forecast_len + 1
            )

            if self.date_col is not None:
                split_scheme[i]["train_period"] = (
                    self.dt_array[train_start_idx],
                    self.dt_array[train_end_idx],
                )
                split_scheme[i]["test_period"] = (
                    self.dt_array[train_end_idx + 1],
                    self.dt_array[train_end_idx + self.forecast_len],
                )

        self._split_scheme = split_scheme
        self.n_splits = len(split_scheme)

    def get_scheme(self):
        return deepcopy(self._split_scheme)

    def split(self):
        """
        Returns
        -------
        iterables with (train_df, test_df, scheme, split_key) where
        train_df : pd.DataFrame
            data split for training
        test_df : pd.DataFrame
            data split for testing/validation
        scheme : dict
            derived from self._split_scheme
        split_key : int
             index of the iteration
        """
        if self.date_col is None:
            for split_key, scheme in self._split_scheme.items():
                train_df = self.df.iloc[scheme["train_idx"], :].reset_index(drop=True)
                test_df = self.df.iloc[scheme["test_idx"], :].reset_index(drop=True)
                yield train_df, test_df, scheme, split_key
        else:
            for split_key, scheme in self._split_scheme.items():
                train_df = self.df.loc[
                    (self.df[self.date_col] >= scheme["train_period"][0])
                    & (self.df[self.date_col] <= scheme["train_period"][1]),
                    :,
                ].reset_index(drop=True)
                test_df = self.df.loc[
                    (self.df[self.date_col] >= scheme["test_period"][0])
                    & (self.df[self.date_col] <= scheme["test_period"][1]),
                    :,
                ].reset_index(drop=True)
                yield train_df, test_df, scheme, split_key


class BacktestingForecast:
    def __init__(
        self,
        n_splits=10,
        forecast_len=3,
        incremental_len=1,
        min_train_len=6,
        window_type="expanding",
        n_sample_per_day: int = 48,
        date_column="timestamp",
    ):
        min_train_len = int(
            n_sample_per_day * 30 * min_train_len
        )  # minimal length of window length
        forecast_len = int(
            n_sample_per_day * 30 * forecast_len
        )  # length forecast window
        incremental_len = int(
            n_sample_per_day * 30 * incremental_len
        )  # step length for moving forward)
        self.date_column = date_column
        self.generator = partial(
            TimeSeriesSplitter,
            min_train_len=min_train_len,
            incremental_len=incremental_len,
            forecast_len=forecast_len,
            window_type=window_type,
            n_splits=n_splits,
            date_col=date_column,
        )

    def fit(
        self,
        data,
        model_instance,
        train_ratio=0.80,
        drop_last=False,
        num_worker=1,
        batch_size=64,
        pin_memory=True,
    ):
        self.generator=self.generator(df=data)
        backtest_metrics, backtest_df = pd.DataFrame(), pd.DataFrame()
        for train_df, test_df, scheme, key in self.generator.split():
            f"{key}_cross_validation"

            logger.info(
                f"---------------Fit  Backtesting expanding-{key + 1} \
                Cross validation Training --------------------------"
            )

            logger.info(
                f"Train_window: from {train_df[self.date_column].iloc[0]} to  {train_df[self.date_column].iloc[-1]} "
            )

            logger.info(
                f"Test_window: from {test_df[self.date_column].iloc[0]} to  {test_df[self.date_column].iloc[-1]}"
            )

            filename = f"{key + 1}_cross_validation"
            model = model_instance(file_name=filename)
            model_copy = deepcopy(model)
            model_copy.fit(
                train_df,
                train_ratio=train_ratio,
                drop_last=drop_last,
                num_worker=num_worker,
                batch_size=batch_size,
                pin_memory=pin_memory,
            )
            pred_df = model.predict(test_df)
            metrics = model.metrics
            metrics["Folds"] = key + 1
            pred_df["Folds"] = key + 1

            backtest_metrics = pd.concat([backtest_metrics, metrics], axis=0)
            backtest_df = pd.concat([backtest_df, pred_df], axis=0)

        return backtest_df, backtest_metrics
