import sys

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


def autoregressive_forecast(
    model,
    initial_hist_df,
    covariate_df,
    forecast_horizon,
    input_window_size,
    target_transformer,
    max_days=7,
):
    """
    Get autoregressive forecasts using a Multi-Layer Perceptron Forecasting (MLPF) model.

    Parameters:
    - model: The MLPF model for forecasting.
    - initial_hist_df: DataFrame containing initial historical data.
    - covariate_df: DataFrame containing covariate data.
    - forecast_horizon: Number of steps to forecast in each iteration.
    - input_window_size: Size of the input window for the model.
    - target_transformer: Transformer to invert scaling of predictions.
    - max_days: Maximum number of days for forecasting (default is 7).

    Returns:
    - DataFrame containing the autoregressive forecasts for each day.
    """
    # Prepare initial historical data for forecasting
    historical_slices = [
        initial_hist_df[i : forecast_horizon + i].values
        for i in range(0, input_window_size, forecast_horizon)
    ]
    day_1, day_2 = historical_slices[:2]

    predictions = []

    # Iterate over the specified number of forecast days
    with tqdm(total=max_days, file=sys.stdout) as pbar:
        for k in range(max_days):
            first_day_start = k * forecast_horizon
            first_day_end = first_day_start + forecast_horizon

            second_day_start = first_day_end
            second_day_end = second_day_start + forecast_horizon

            historical_data = covariate_df.iloc[first_day_start:second_day_end]
            future_data = covariate_df.iloc[
                second_day_end : second_day_end + forecast_horizon
            ]

            historical_target = np.concatenate([day_1, day_2], axis=0)
            unknown_features = np.concatenate(
                [historical_target, historical_data.values], axis=1
            ).astype(np.float64)
            known_features = future_data.values.astype(np.float64)

            features = torch.FloatTensor(unknown_features).unsqueeze(0)
            covariates = torch.FloatTensor(known_features).unsqueeze(0)

            # Pad covariate features with zero
            diff = features.shape[2] - covariates.shape[2]
            B, N, _ = covariates.shape
            covariates = torch.cat(
                [torch.zeros(B, N, diff, requires_grad=False), covariates], dim=-1
            )
            features = torch.cat([features, covariates], dim=1)

            model.to(features.device)
            out = model.forecast(features)

            day_1 = day_2
            day_2 = out["pred"].numpy().reshape(-1, 1)
            predictions.append(pd.DataFrame.from_dict(out, orient="index").T)

            pbar.set_description(f"Processed: {k}")
            pbar.update(1)

        predictions = pd.concat(predictions)

    return predictions
