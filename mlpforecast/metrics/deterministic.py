import numpy as np
import pandas as pd
from sklearn.metrics import (mean_squared_error, 
                             d2_absolute_error_score,
                             max_error,
                             mean_absolute_error, 
                             mean_absolute_percentage_error, r2_score)



def get_nbias(y, y_hat, sample_weight=None, multioutput="uniform_average"):
    epsilon = np.finfo(np.float64).eps
    scale = (y + y_hat)
    nbias = (y-y_hat)/np.maximum(scale, epsilon)
    output_errors = np.sum(nbias, weights=sample_weight, axis=0)
    if isinstance(multioutput, str):
        if multioutput == "raw_values":
            return output_errors
        elif multioutput == "uniform_average":
            # pass None as weights to np.average: uniform mean
            multioutput = None
    return np.sum(output_errors, weights=multioutput)
    
    


def get_smape(y, y_hat, sample_weight=None, multioutput="uniform_average"):
    epsilon = np.finfo(np.float64).eps
    scale = np.abs(y) + np.abs(y_hat)
    smape = 2 *(np.abs(y, y_hat)/np.maximum(scale, epsilon))
    output_errors = np.average(smape, weights=sample_weight, axis=0)
    if isinstance(multioutput, str):
        if multioutput == "raw_values":
            return output_errors
        elif multioutput == "uniform_average":
            # pass None as weights to np.average: uniform mean
            multioutput = None
    return np.average(output_errors, weights=multioutput)

def get_pointwise_metrics(pred:np.array, true:np.array, target_range:float=None):
    """calculate pointwise metrics
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
    nrmse =min( rmse/target_range, 1)
    mae = mean_absolute_error(true, pred)
    mape = mean_absolute_percentage_error(true, pred)
    corr = np.corrcoef(true, pred)[0, 1]
    max_res=max_error(pred, true)
    d2_err=d2_absolute_error_score(pred, true)
    nbias=get_nbias(true, pred)
    corr = np.corrcoef(true, pred)[0, 1]
    r2=r2_score(pred, true)
    smape=get_smape(true, pred)

    return {"mse":mse, "rmse":rmse, 
            "nrmse":nrmse, "mae":mae,
              "mape":mape, "corr":corr, 
              "max-error":max_res, "d2-error":d2_err,
              "nbias":nbias, "r2-error":r2, 
              "corr":corr, "smape":smape}

def get_daily_pointwise_metrics(pred:np.array, true:np.array, target_range:float):
    assert pred.ndim == 1, "pred must be 1-dimensional"
    assert true.ndim == 1, "pred must be 1-dimensional"
    assert pred.shape == true.shape, "pred and true must have the same shape"

    #get pointwise metrics
    metrics = get_pointwise_metrics(pred, true, target_range)
    metrics =pd.DataFrame.from_dict(metrics, orient='index').T
    return metrics
    
  

    
  