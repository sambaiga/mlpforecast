import logging
import torch
from optuna import Trial
from mlpforecast.model.base_model import BaseForecastModel
from mlpforecast.net.timesnet import TimesNet
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TimesNet")

class TimesNetForecastModel(BaseForecastModel):
    """
    MLP Forecast Model for time series point forecasting.

    Attributes:
        n_out (int): Number of output series.
        n_channels (int): Number of input channels.
        model (object): Model object.
        hparams (dict): Hyperparameters for the model.
    """

    def __init__(
        self,
        data_pipeline=None,
        target_series: list[str] | str = ["NetLoad"],
        unknown_features: list[str] = [],
        known_calendar_features: list[str] = [],
        known_continuous_features: list[str] = [],
        input_window_size: int = 96,
        forecast_horizon: int = 48,
        hidden_size: int = 64,
        out_activation_function: str = "Identity",
        dropout_rate: float = 0.25,
        alpha: float = 0.1,
        num_layers: int = 2,
        conv_hidden_size: int = 64,
        top_k: int = 5,
        num_kernels: int = 6,
        metric: str = "mae",
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-6,
        prob_decay_1: float = 0.75,
        prob_decay_2: float = 0.9,
        gamma: float = 0.01,
        max_epochs: int = 10,
        lambda_lasso:float=1e-3,
        gam_layer:bool=False,
    ):
        super().__init__(data_pipeline, metric)
        assert len(target_series) > 0, "target_series should not be empty."

        self.n_out = len(target_series)
        n_unknown = len(unknown_features) + self.n_out
        n_covariates = len(known_calendar_features) + len(known_continuous_features)
        self.n_channels = n_unknown + n_covariates
        self.model = TimesNet(
                n_target_series=self.n_out,
                n_unknown_features=len(unknown_features),
                n_known_calendar_features=len(known_calendar_features),
                n_known_continuous_features=len(known_continuous_features),
                hidden_size=hidden_size,
                forecast_horizon=forecast_horizon,
                input_window_size=input_window_size,
                num_layers=num_layers,
                out_activation_function=out_activation_function,
                dropout_rate=dropout_rate,
                alpha=alpha,
                conv_hidden_size = conv_hidden_size,
                top_k=top_k,
                num_kernels=num_kernels)
        
    def get_search_params(self, trial: Trial) -> dict:
        """
        Define the search space for hyperparameter optimization using Optuna.

        Args:
            trial: An Optuna trial object to suggest parameters.

        Returns
        -------
            dict: A dictionary containing suggested hyperparameters.
        """
        params = {}
        params["alpha"] = trial.suggest_float("alpha", 1e-3, 1.0, log=True)
        params["top_k"] = trial.suggest_int("top_k", 1, 5)
        params["num_kernels"] = trial.suggest_int("num_kernels", 2, 6)
        params["activation_function"]= trial.suggest_categorical("activation",
                ['ReLU', 'Softplus', 'Tanh', 'SELU', 'LeakyReLU', 'PReLU', 'Sigmoid']
            )
        params["hidden_size"] = trial.suggest_int("hidden_size", 8, 512, step=2)
        params["conv_hidden_size"] = trial.suggest_int("conv_hidden_size", 8, 512, step=2)
        params["dropout_rate"] = trial.suggest_float("dropout_rate", 0.0, 0.9, step=0.05)
        params["num_layers"] = trial.suggest_int("num_layers", 1, 5)
        return params