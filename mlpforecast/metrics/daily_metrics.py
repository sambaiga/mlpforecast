import numpy as np
import pandas as pd
import scipy
from mlpforecast.metrics.deterministic import get_pointwise_metrics



def get_daily_pointwise_metrics(pred: np.ndarray, true: np.ndarray, target_range: float) -> pd.DataFrame:
    """
    Calculate daily pointwise metrics.

    This function computes various pointwise metrics for daily predictions.

    Parameters
    ----------
    pred : np.ndarray
        Predicted values. Must be 1-dimensional.
    true : np.ndarray
        True values. Must be 1-dimensional.
    target_range : float
        Target range for normalization.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the calculated metrics.

    Raises
    ------
    ValueError
        If `pred` or `true` are not 1-dimensional or if they do not have the same shape.

    Example
    -------
    >>> pred = np.array([1.0, 2.0, 3.0])
    >>> true = np.array([1.1, 2.1, 3.1])
    >>> target_range = 2.0
    >>> metrics = get_daily_pointwise_metrics(pred, true, target_range)
    >>> print(metrics)
    """
    if pred.ndim != 1:
        raise ValueError("pred must be 1-dimensional")
    if true.ndim != 1:
        raise ValueError("true must be 1-dimensional")
    if pred.shape != true.shape:
        raise ValueError("pred and true must have the same shape")

    # Get pointwise metrics
    metrics = get_pointwise_metrics(pred, true, target_range)
    metrics_df = pd.DataFrame.from_dict(metrics, orient="index").T
    return metrics_df


def get_daily_quantile_metrics(pred: np.ndarray, 
                               true: np.ndarray, 
                               quantile_hats: np.ndarray,
                               taus: np.ndarray,
                               upper: np.ndarray,
                               lower: np.ndarray,
                               alpha: float = 0.05,
                               target_range: float = None) -> pd.DataFrame:
    """
    Calculate daily quantile metrics.

    This function computes various pointwise, interval, and quantile metrics for daily predictions.

    Parameters
    ----------
    pred : np.ndarray
        Predicted values. Must be 1-dimensional.
    true : np.ndarray
        True values. Must be 1-dimensional.
    quantile_hats : np.ndarray
        Predicted quantiles.
    taus : np.ndarray
        Quantile levels.
    upper : np.ndarray
        Upper bound of the prediction intervals.
    lower : np.ndarray
        Lower bound of the prediction intervals.
    alpha : float, optional
        Significance level for the prediction intervals, by default 0.05.
    target_range : float, optional
        Target range for normalization, by default None.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the calculated metrics.

    Raises
    ------
    ValueError
        If `pred` or `true` are not 1-dimensional or if they do not have the same shape.

    Example
    -------
    >>> pred = np.array([1.0, 2.0, 3.0])
    >>> true = np.array([1.1, 2.1, 3.1])
    >>> quantile_hats = np.array([[0.9, 1.9, 2.9], [1.1, 2.1, 3.1]])
    >>> taus = np.array([0.1, 0.9])
    >>> upper = np.array([1.2, 2.2, 3.2])
    >>> lower = np.array([0.8, 1.8, 2.8])
    >>> metrics = get_daily_quantile_metrics(pred, true, quantile_hats, taus, upper, lower)
    >>> print(metrics)
    """
    if pred.ndim != 1:
        raise ValueError("pred must be 1-dimensional")
    if true.ndim != 1:
        raise ValueError("true must be 1-dimensional")
    if pred.shape != true.shape:
        raise ValueError("pred and true must have the same shape")

    # Get pointwise metrics
    metrics = get_pointwise_metrics(pred, true, target_range)
    
    metrics_df = pd.DataFrame.from_dict(metrics, orient="index").T
    return metrics_df

