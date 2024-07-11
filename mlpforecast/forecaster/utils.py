import glob
import os

import numpy as np


def format_target(targets, input_window_size, forecast_horizon):
    targets = np.squeeze(
        np.lib.stride_tricks.sliding_window_view(
            targets[input_window_size:],
            (forecast_horizon, targets.shape[1]),
        ),
        axis=1,
    )
    return targets.reshape(targets.shape[0], forecast_horizon, -1)


def get_latest_checkpoint(checkpoint_path):
    checkpoint_path = str(checkpoint_path)
    list_of_files = glob.glob(checkpoint_path + "/*.ckpt")

    if list_of_files:
        latest_file = max(list_of_files, key=os.path.getctime)
    else:
        latest_file = None
    return latest_file
