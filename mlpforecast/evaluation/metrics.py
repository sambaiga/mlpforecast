from mlpforecast.metrics.daily_metrics import get_daily_pointwise_metrics                                      
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

