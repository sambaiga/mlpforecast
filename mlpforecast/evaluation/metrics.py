from mlpforecast.metrics.daily_metrics import (get_daily_pointwise_metrics, 
                                               get_daily_quantile_metrics)
import pandas as pd




def evaluate_point_forecast(outputs):
    """
    Evaluates point forecasts by computing daily pointwise metrics.

    Args:
        outputs (dict): A dictionary containing the true values, predicted values, and associated metadata.
            Expected keys:
                'true' (ndarray): The true values.
                'loc' (ndarray): The predicted values.
                'index' (ndarray): The timestamps for each prediction.
                'targets' (list): The names of the target variables.
        show_fig (bool, optional): Whether to display a figure of the results. Default is False.

    Returns
    -------
        tuple: A tuple containing:
            - pd_metrics (dict): DataFrame of combined metrics for each target variable.
            - split_metrics (dict): Dictionary of metrics split by target variable.
            - logs (dict): Any additional logs generated during the evaluation.
    """
    pd_metrics = pd.DataFrame()
    for i in range(len(outputs["true"])):
        metrics = []

        for j in range(outputs["true"].shape[-1]):
            true = outputs["true"][i, :, j]
            pred = outputs["loc"][i, :, j]

            point_scores = get_daily_pointwise_metrics(pred, true, None)
            point_scores.insert(0, "target", outputs["targets"][j])
            metrics.append(point_scores)

        metrics_df = pd.concat(metrics)
        df = pd.DataFrame(outputs["index"][i], columns=["Date"])
        df["Date"] = pd.to_datetime(df["Date"], unit="ns")
        metrics_df.insert(0, "timestamp", df.Date.dt.round("D").unique()[-1])
        pd_metrics = pd.concat([pd_metrics, metrics_df], axis=0)
    pd_metrics.set_index("timestamp", inplace=True)

    return pd_metrics


def evaluate_quantile_forecast(outputs: dict, alpha: float = 0.05) -> pd.DataFrame:
    """
    Evaluates quantile forecasts by computing daily pointwise metrics.

    Parameters
    ----------
    outputs : dict
        A dictionary containing the true values, predicted values, and associated metadata.
        Expected keys:
            'true' (ndarray): The true values.
            'loc' (ndarray): The predicted values.
            'index' (ndarray): The timestamps for each prediction.
            'targets' (list): The names of the target variables.
            'q_samples' (ndarray): The quantile samples.
            'taus_hats' (ndarray): The quantile levels.
    alpha : float, optional
        The significance level for the prediction intervals. Default is 0.05.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the combined metrics for each target variable.

    Example
    -------
    >>> outputs = {
            'true': np.random.rand(2, 10, 3),
            'loc': np.random.rand(2, 10, 3),
            'index': [np.arange(10), np.arange(10)],
            'targets': ['target1', 'target2', 'target3'],
            'q_samples': np.random.rand(100, 2, 10, 3),
            'taus_hats': [np.linspace(0.1, 0.9, 100), np.linspace(0.1, 0.9, 100)]
        }
    >>> alpha = 0.05
    >>> metrics = evaluate_quantile_forecast(outputs, alpha)
    >>> print(metrics)
    """
    pd_metrics = pd.DataFrame()

    for i in range(len(outputs["true"])):
        metrics = []

        for j in range(outputs["true"].shape[-1]):
            true = outputs["true"][i, :, j]
            pred = outputs["loc"][i, :, j]
            q_hats = outputs['q_samples'][i, :, :, j]
            taus = outputs['taus_hats'][i].squeeze().numpy()
            

            _scores = get_daily_quantile_metrics(
                pred,
                true,
                quantile_hats=q_hats,
                taus=taus,
                upper=q_hats[-1],
                lower=q_hats[0],
                alpha=alpha
            )
            _scores.insert(0, "target", outputs["targets"][j])
            metrics.append(_scores)

        metrics_df = pd.concat(metrics)
        df = pd.DataFrame(outputs["index"][i], columns=["Date"])
        df["Date"] = pd.to_datetime(df["Date"], unit="ns")
        metrics_df.insert(0, "timestamp", df.Date.dt.round("D").unique()[-1])
        pd_metrics = pd.concat([pd_metrics, metrics_df], axis=0)

    pd_metrics.set_index("timestamp", inplace=True)
    return pd_metrics

