from datetime import datetime

import numpy as np
import pandas as pd


def get_n_sample_per_day(period):
    period = int("".join(x for x in period if x.isdigit()))
    return int(24 * 60 / period)


def detect_missing_date(dataset, period=30):
    """
    Fill missing dates in a time series dataset with NaN values.

    Parameters:
    - dataset (pd.DataFrame): The input time series dataset with a datetime index.
    - period (int): The frequency, in minutes, for the new date range. Default is 30 minutes.

    Returns:
    - pd.DataFrame: The input dataset with missing dates filled with NaN values.
    """
    data = dataset.copy()
    index_name = data.index.name
    min_dt = min([data.index.min()])
    max_dt = max([data.index.max()]) + pd.Timedelta(minutes=period)
    dt_rng = pd.date_range(min_dt, max_dt, freq=f"{period}T")
    data = data.reindex(dt_rng)
    data.index.name = index_name
    return data


def get_periods_for_exog_variable(hparams, data):
    exog = data[hparams["time_varying_known_categorical_feature"]].values
    exog_periods = [len(np.unique(exog[:, size])) for size in range(exog.shape[-1])]
    return exog_periods


def add_time_features(results_pd, hemisphere="Northern"):
    results_pd = results_pd.reset_index()
    results_pd["quarter"] = results_pd.timestamp.dt.quarter.values
    results_pd["day"] = results_pd.timestamp.dt.day_name().values
    results_pd["hour"] = results_pd.timestamp.dt.hour.values
    results_pd["week"] = (
        results_pd.timestamp.dt.isocalendar().week
        if hasattr(results_pd.timestamp.dt, "isocalendar")
        else results_pd.timestamp.dt.week
    )
    results_pd["month"] = results_pd.timestamp.dt.month
    results_pd["year"] = results_pd.timestamp.dt.year
    results_pd["Session"] = results_pd["hour"].apply(day_night).values
    results_pd = results_pd.set_index("timestamp")

    season_list = []

    for month in results_pd["month"]:
        season = find_season(month, hemisphere)
        season_list.append(season)

    results_pd["Season"] = season_list
    return results_pd


def find_season(month, hemisphere):
    if hemisphere == "Southern":
        season_month_south = {
            12: "Summer",
            1: "Summer",
            2: "Summer",
            3: "Autumn",
            4: "Autumn",
            5: "Autumn",
            6: "Winter",
            7: "Winter",
            8: "Winter",
            9: "Spring",
            10: "Spring",
            11: "Spring",
        }
        return season_month_south.get(month)

    elif hemisphere == "Northern":
        season_month_north = {
            12: "Winter",
            1: "Winter",
            2: "Winter",
            3: "Spring",
            4: "Spring",
            5: "Spring",
            6: "Summer",
            7: "Summer",
            8: "Summer",
            9: "Autumn",
            10: "Autumn",
            11: "Autumn",
        }
        return season_month_north.get(month)
    else:
        print("Invalid selection. Please select a hemisphere and try again")


def loadData(
    a_path,
    a_cols,
    a_rename=None,
    a_idx_field="timestamp",
    a_period="1T",
    a_timezone=None,
):
    data = pd.DataFrame()
    for path in a_path:
        data_temp = pd.read_csv(path, usecols=a_cols)
        data = pd.concat([data, data_temp], ignore_index=True)

    data[a_idx_field] = pd.to_datetime(data[a_idx_field], utc=True)

    if a_rename is not None:
        data.rename(columns=a_rename, inplace=True)
    data.set_index(a_idx_field, inplace=True)
    # data = data.interpolate(method="time")
    if a_timezone is not None:
        data.index = data.index.tz_convert(a_timezone)

    # data.index = pd.to_datetime(data.index)
    data = data.resample(a_period).mean()
    # data = resample(data, rate=a_period, short_rate='T', max_gap="30T")

    return data


def day_night(x):
    if (x > 5) and (x <= 18):
        return 1
    else:
        return 0


def add_exogenous_variables(df, one_hot=False):
    """
    Augment the dataframe with exogenous features (date/time feature + holiday feature).
    The feature's values can be kept as they are or they can be one hot encoded
    :param df: the dataframe
    :param one_hot: if True, one hot encode all the features.
    :return: two matrix of exogenous features,
     the first for temperatures only the second one contains all the other variables.
    """
    data = df.copy().reset_index()
    data["DAYOFWEEK"] = data.timestamp.dt.dayofweek
    data["WEEK"] = (
        data.timestamp.dt.isocalendar().week
        if hasattr(data.timestamp.dt, "isocalendar")
        else data.timestamp.dt.week
    )
    data["DAYOFYEAR"] = data.timestamp.dt.dayofyear
    data["YEAR"] = data.timestamp.dt.year
    data["MONTH"] = data.timestamp.dt.month
    data["DAY"] = data.timestamp.dt.day
    data["HOUR"] = data.timestamp.dt.hour
    data["Session"] = data["HOUR"].apply(day_night).values

    data["WEEKDAY"] = 0
    data["WEEKDAY"] = ((data.timestamp.dt.dayofweek) // 5 == 0).astype(float)
    data["WEEKEND"] = 0
    data["WEEKEND"] = ((data.timestamp.dt.dayofweek) // 5 == 1).astype(float)

    data["SATURDAY"] = 0
    data["SATURDAY"] = (data.timestamp.dt.dayofweek == 5).astype(float)

    data["SUNDAY"] = 0
    data["SUNDAY"] = (data.timestamp.dt.dayofweek == 6).astype(float)

    if one_hot:
        ex_feat = pd.get_dummies(
            data,
            columns=[
                "MONTH",
                "DAY",
                "HOUR",
                "DAYOFWEEK",
                "DAYOFYEAR",
                "WEEKDAY",
                "WEEKEND",
                "SATURDAY",
                "SUNDAY",
            ],
        )
        return ex_feat
    else:
        return data


def fourier_series(dates, period, series_order):
    """Provides Fourier series components with the specified frequency and order.
    Note: Identical to OG Prophet.
    Args:
        dates (pd.Series): containing timestamps.
        period (float): Number of days of the period.
        series_order (int): Number of fourier components.
    Returns:
        Matrix with seasonality features.
    """
    # convert to days since epoch
    t = np.array((dates - datetime(1970, 1, 1)).dt.total_seconds().astype(float)) / (
        3600 * 24.0
    )
    return fourier_series_t(t, period, series_order)


def fourier_series_t(t, period, series_order):
    """Provides Fourier series components with the specified frequency and order.
    Note: Identical to OG Prophet.
    Args:
        t (pd.Series, float): containing time as floating point number of days.
        period (float): Number of days of the period.
        series_order (int): Number of fourier components.
    Returns:
        Matrix with seasonality features.
    """
    features = np.column_stack(
        [
            fun(2.0 * (i + 1) * np.pi * t / period)
            for i in range(series_order)
            for fun in (np.sin, np.cos)
        ]
    )
    return features


def get_index(test_df, hparams, test=True):
    index = test_df["timestamp"]

    list_range = (
        range(
            0,
            len(index) - hparams["window_size"] - hparams["horizon"] + 1,
            hparams["horizon"],
        )
        if test
        else range(0, len(index) - hparams["window_size"] - hparams["horizon"] + 1)
    )
    past_index, future_index = [], []

    for i in list_range:
        past_index.append(index[i : i + hparams["window_size"]])
        future_index.append(
            index[
                i
                + hparams["window_size"] : i
                + hparams["window_size"]
                + hparams["horizon"]
            ]
        )
    index = np.concatenate((np.array(past_index), np.array(future_index)), 1)

    return index


def compute_netload_ghi(load, ghi, SAMPLES_PER_DAY):
    print("Compute load ghi feature")
    Load_ghi = []
    for i in range(SAMPLES_PER_DAY, len(load) - SAMPLES_PER_DAY, SAMPLES_PER_DAY):
        load_i = load[i - SAMPLES_PER_DAY : i]
        ghi_i = ghi[i : SAMPLES_PER_DAY + i]
        load_i_norm = load_i / np.abs(load_i).max()
        ghi_i_norm = ghi_i / np.abs(ghi_i).max()
        lg = load_i_norm - ghi_i_norm
        Load_ghi.append(lg)

    Load_ghi = np.hstack(Load_ghi)

    return Load_ghi


def combine_past_future_exogenous(features: np.array, covariates: np.array = None):
    """
    Combines past and future exogenous features with optional covariates.

    Parameters:
    - features (np.array): A 2D or 3D numpy array of features. If 2D, it is reshaped to 3D.
                          Expected shape (N, C) for 2D or (B, N, C) for 3D, where:
                          - B is the batch size.
                          - N is the number of time steps.
                          - C is the number of features.
    - covariates (np.array, optional): A 2D or 3D numpy array of covariates. If 2D, it is reshaped to 3D.
                                       Expected shape (N, C) for 2D or (B, N, C) for 3D. Default is None.

    Returns:
    - np.array: A 3D numpy array with combined features and covariates along the second axis. The shape
                of the returned array is (B, 2N, C), where:
                - B is the batch size.
                - 2N is the combined number of time steps.
                - C is the number of features, padded with zeros if covariates had fewer features than features.

    Raises:
    - AssertionError: If `features` or `covariates` do not have 2 or 3 dimensions.
    - AssertionError: If `features` and `covariates` do not have the same batch size and time steps after reshaping.
    """

    assert features.ndim in (2, 3), "Features must be a 2D or 3D array."
    if features.ndim == 2:
        N, C = features.shape
        features = features.reshape((1, N, C))

    if covariates is not None:
        assert covariates.ndim in (
            2,
            3,
        ), "covariates must be a 2D or 3D array."
        if covariates.ndim == 2:
            N, C = covariates.shape
            covariates = covariates.reshape((1, N, C))

        diff = features.shape[2] - covariates.shape[2]
        if diff > 0:
            B, N, _ = covariates.shape
            zeros_diff = np.zeros((B, N, diff), dtype=covariates.dtype)
            covariates = np.concatenate([zeros_diff, covariates], axis=-1)
        features = np.concatenate([features, covariates], axis=1)

    return features
