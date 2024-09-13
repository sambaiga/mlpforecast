import numpy as np
import pandas as pd
import scipy
from mlpforecast.metrics.deterministic import get_pointwise_metrics
from mlpforecast.metrics.probabilistic import (continous_ranked_pscore_pwm, 
                                               laplace_nll, 
                                               winkler_score,
                                               average_coverage_error,
                                               prediction_intervals_coverage,
                                               median_prediction_width,
                                               get_standardized_gamma_nmpi,
                                               calculate_gamma_nmpi_score,
                                               calculate_gamma_coverage_score,
                                               pinball_score,
                                               cwe_score)


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
    
    # Get interval metrics
    interval_metrics = get_interval_metrics(loc=pred, true=true, lower=lower, upper=upper, alpha=alpha)
    metrics.update(interval_metrics)
    
    # Get quantile metrics
    quantile_metrics = get_quantile_metrics(loc=pred, true=true, taus=taus, quantile_hats=quantile_hats)
    metrics.update(quantile_metrics)

    metrics_df = pd.DataFrame.from_dict(metrics, orient="index").T
    return metrics_df



def get_parametric_metrics(loc: np.ndarray, 
                           true: np.ndarray, 
                           scale: np.ndarray,
                           samples: np.ndarray):
    """
    Calculate pointwise metrics.

    Args:
    -----
    loc: np.ndarray
        Predicted values.
    true: np.ndarray
        True values.
    scale: np.ndarray
        Scale parameter for the distribution.
    samples: np.ndarray
        Sample values.
    target_range: float, optional
        Target range for normalization.

    Returns:
    --------
    dict
        Dictionary containing the calculated metrics.
    """
    if loc.ndim != 1:
        raise ValueError("loc must be 1-dimensional")
    if true.ndim != 1:
        raise ValueError("true must be 1-dimensional")
    if loc.shape != true.shape:
        raise ValueError("loc and true must have the same shape")

    crsps=continous_ranked_pscore_pwm(true, samples)
    nll=laplace_nll(true, loc, scale)
    

    
    return {
        "CRSPS": crsps,
        "NLL": nll
    }


def get_quantile_metrics(loc: np.ndarray, 
                           true: np.ndarray, 
                           taus: np.ndarray,
                           quantile_hats: np.ndarray):
    """
    Calculate pointwise metrics.

    Args:
    -----
    loc: np.ndarray
        Predicted values.
    true: np.ndarray
        True values.
    scale: np.ndarray
        Scale parameter for the distribution.
    samples: np.ndarray
        Sample values.
    target_range: float, optional
        Target range for normalization.

    Returns:
    --------
    dict
        Dictionary containing the calculated metrics.
    """
    if loc.ndim != 1:
        raise ValueError("loc must be 1-dimensional")
    if true.ndim != 1:
        raise ValueError("true must be 1-dimensional")
    if loc.shape != true.shape:
        raise ValueError("loc and true must have the same shape")
    crsps=continous_ranked_pscore_pwm(true[ None, :], quantile_hats)
    pinball=pinball_score(true=true,q_values=quantile_hats, taus=taus)

    return {
        "CRSPS": crsps,
        "PINBALL":pinball
    }


def get_interval_metrics(loc: np.ndarray, 
                           true: np.ndarray, 
                           upper:np.ndarray,
                           lower:np.ndarray,
                           target_range: float = None,
                           alpha:float=0.05):
    """
    Calculate pointwise metrics.

    Args:
    -----
    loc: np.ndarray
        Predicted values.
    true: np.ndarray
        True values.
    scale: np.ndarray
        Scale parameter for the distribution.
    samples: np.ndarray
        Sample values.
    target_range: float, optional
        Target range for normalization.

    Returns:
    --------
    dict
        Dictionary containing the calculated metrics.
    """
    if loc.ndim != 1:
        raise ValueError("loc must be 1-dimensional")
    if true.ndim != 1:
        raise ValueError("true must be 1-dimensional")
    if loc.shape != true.shape:
        raise ValueError("loc and true must have the same shape")

    target_range = true.max() - true.min() if target_range is None else target_range
    winkler=winkler_score(loc, true=true, lower=lower, upper=upper, alpha=alpha)
    coverage=prediction_intervals_coverage(true=true, lower=lower, upper=upper)
    interval_width=median_prediction_width(lower=lower, upper=upper)
    coverage_error=average_coverage_error(coverage=coverage, alpha=alpha)

    kappa=scipy.stats.median_abs_deviation(true)
    gamma_mpi=get_standardized_gamma_nmpi(nmpi=interval_width, 
                                          true_nmpic=kappa)
    gamma_picp=calculate_gamma_coverage_score(picp=coverage, alpha=alpha)
    cwe=cwe_score(nmpi=interval_width, picp=coverage, error=None,
                  true_nmpic=kappa, alpha=alpha)

    
    return {
        "CWE": cwe,
        "Kappa": kappa,
        "Gamma-mpi": gamma_mpi,
        "Gamma-pcip": gamma_picp,
        "MPI": interval_width,
        "PICP(%)": coverage*100,
        "WSCORE": winkler,
        "ACE": coverage_error,
        "R":target_range
    }