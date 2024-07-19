import glob
import os

from mlpforecast.data.processing import (
    extract_daily_sequences,
    extract_target_sequences,
)


def format_target(targets, input_window_size, forecast_horizon, daily_feature=True):
    """
    Format the target data for training the model.

    Args:
        targets (pd.DataFrame): Target data.
        input_window_size (int): Input window size.
        forecast_horizon (int): Forecast horizon.
        daily_feature (bool, optional): Whether to use daily features. Defaults to True.

    Returns:
        np.ndarray: Formatted target data.
    """
    if daily_feature:
        return extract_daily_sequences(
            targets, input_window_size, forecast_horizon, target_mode=True
        )
    else:
        return extract_target_sequences(targets, input_window_size, forecast_horizon)


def get_latest_checkpoint(checkpoint_path):
    """
    Get the path of the latest checkpoint file.

    Args:
        checkpoint_path (str): Path to the directory containing the checkpoint files.

    Returns:
        str: Path of the latest checkpoint file.
    """
    checkpoint_path = str(checkpoint_path)
    list_of_files = glob.glob(checkpoint_path + "/*.ckpt")

    if list_of_files:
        latest_file = max(list_of_files, key=os.path.getctime)
    else:
        latest_file = None
    return latest_file
