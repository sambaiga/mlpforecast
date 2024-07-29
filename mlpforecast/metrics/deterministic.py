import numpy as np
import pandas as pd
from sklearn.metrics import (
    d2_absolute_error_score,
    max_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)


def get_nbias(y, y_hat, axis=0):
    """
    Calculates the normalized bias (NBias) between the true values and predicted values.

    NBias is defined as the sum of the difference between the true values and predicted values,
    normalized by the sum of the true and predicted values.

    Args:
        y (ndarray): The true values.
        y_hat (ndarray): The predicted values.
        axis (int, optional): The axis along which to compute the NBias. Default is 0.

    Returns
    -------
        float: The normalized bias value.
    """
    epsilon = np.finfo(np.float64).eps  # Small value to avoid division by zero
    scale = y + y_hat  # Sum of true and predicted values
    # Normalized bias calculation
    nbias = (y - y_hat) / np.maximum(scale, epsilon)
    output_errors = nbias
    return np.sum(output_errors, axis=axis)  # Sum of output errors along the specified axis


def get_smape(y, y_hat, axis=0):
    """
    Calculates the Symmetric Mean Absolute Percentage Error (SMAPE) between the true values and predicted values.

    SMAPE is defined as the average of the absolute percentage errors, normalized by the sum of the absolute values of
    the true and predicted values.

    Args:
        y (ndarray): The true values.
        y_hat (ndarray): The predicted values.
        axis (int, optional): \
              The axis along which to compute the SMAPE. Default is 0.

    Returns
    -------
        float: The symmetric mean absolute percentage error value.
    """
    epsilon = np.finfo(np.float64).eps
    # Small value to avoid division by zero
    scale = np.abs(y) + np.abs(y_hat)
    # Sum of absolute true and predicted values
    output_errors = 2 * (np.abs(y - y_hat) / np.maximum(scale, epsilon))  # SMAPE calculation
    return np.average(output_errors, axis=axis)


def get_pointwise_metrics(pred: np.array, true: np.array, target_range: float = None):
    """Calculate pointwise metrics
    Args:   pred: predicted values
            true: true values
            target_range: target range
    Returns:    rmse: root mean square error


    """
    assert pred.ndim == 1, "pred must be 1-dimensional"
    assert true.ndim == 1, "pred must be 1-dimensional"
    assert pred.shape == true.shape, "pred and true must have the same shape"
    target_range = true.max() - true.min() if target_range is None else target_range

    mse = mean_squared_error(true, pred)
    rmse = np.sqrt(mean_squared_error(true, pred))
    nrmse = min(rmse / target_range, 1)
    mae = mean_absolute_error(true, pred)
    mape = mean_absolute_percentage_error(true, pred)
    corr = np.corrcoef(true, pred)[0, 1]
    max_res = max_error(pred, true)
    d2_err = d2_absolute_error_score(pred, true)
    nbias = get_nbias(true, pred)
    corr = np.corrcoef(true, pred)[0, 1]
    r2 = r2_score(pred, true)
    smape = get_smape(true, pred)

    return {
        "MSE": mse,
        "RMSE": rmse,
        "NRMSE": nrmse,
        "MAE": mae,
        "MAPE(%)": mape * 100,
        "CORR": corr,
        "MAX-error": max_res,
        "D2-error": d2_err,
        "NBIAS": nbias,
        "R2-error": r2,
        "SMAPE(%)": smape * 100,
    }





