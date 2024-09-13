import logging
import torch
from mlpforecast.model.base_model import BaseForecastModel
from mlpforecast.net.parametric import MCDMultForecastNetwork, LaplaceForecastNetwork

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MLParametric")

class MLPLaplaceForecastModel(BaseForecastModel):
    def __init__(
        self,
        data_pipeline=None,
        target_series: list[str] | str = ["NetLoad"],
        unknown_features: list[str] = [],
        calendar_variables: list[str] = [],
        known_calendar_features: list[str] = [],
        known_continuous_features: list[str] = [],
        input_window_size: int = 96,
        forecast_horizon: int = 48,
        embedding_size: int = 28,
        embedding_type: str = None,
        combination_type: str = "addition-comb",
        hidden_size: int = 64,
        num_layers: int = 2,
        expansion_factor: int = 2,
        residual: bool = False,
        activation_function: str = "ReLU",
        out_activation_function: str = "Identity",
        dropout_rate: float = 0.25,
        alpha: float = 0.1,
        num_attention_heads: int = 4,
        metric: str = "mae",
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-6,
        prob_decay_1: float = 0.75,
        prob_decay_2: float = 0.9,
        gamma: float = 0.01,
        max_epochs: int = 10,
    ):
        """
        Initialize the Multilayer Perceptron (MLP) Forecast Model for time series forecasting.

        Args:
            data_pipeline (object, optional): Data pipeline object containing the series and features. Defaults to None.
            target_series (list[str] | str, optional): Target time series to forecast. Defaults to ["NetLoad"].
            unknown_features (list[str], optional): List of unknown features. Defaults to an empty list.
            calendar_variables (list[str], optional): Calendar variables to consider. Defaults to an empty list.
            known_calendar_features (list[str], optional): Known calendar features. Defaults to an empty list.
            known_continuous_features (list[str], optional): Known continuous features. Defaults to an empty list.
            input_window_size (int, optional): Size of the input window. Defaults to 96.
            forecast_horizon (int, optional): Forecast horizon. Defaults to 48.
            embedding_size (int, optional): Dimensionality of the embedding space. Defaults to 28.
            embedding_type (str, optional): Type of embedding to use. Options: 'PosEmb', 'RotaryEmb', 'CombinedEmb'. Defaults to None.
            combination_type (str, optional): Type of combination to use. Options: 'attn-comb', 'weighted-comb', 'addition-comb'. Defaults to 'addition-comb'.
            hidden_size (int, optional): Dimensionality of the hidden layers. Defaults to 64.
            num_layers (int, optional): Number of layers in the MLP. Defaults to 2.
            expansion_factor (int, optional): Factor to expand the size of layers. Defaults to 2.
            residual (bool, optional): Whether to use residual connections. Defaults to False.
            activation_function (str, optional): Activation function for hidden layers. Defaults to "ReLU".
            out_activation_function (str, optional): Activation function for the output layer. Defaults to "Identity".
            dropout_rate (float, optional): Dropout rate for regularization. Defaults to 0.25.
            alpha (float, optional): Alpha parameter for the loss function. Defaults to 0.1.
            num_attention_heads (int, optional): Number of attention heads. Defaults to 4.
            metric (str, optional): Metric for evaluation. Defaults to "mae".
            learning_rate (float, optional): Learning rate for the optimizer. Defaults to 1e-3.
            weight_decay (float, optional): Weight decay for the optimizer. Defaults to 1e-6.
            prob_decay_1 (float, optional): First probability decay rate. Defaults to 0.75.
            prob_decay_2 (float, optional): Second probability decay rate. Defaults to 0.9.
            gamma (float, optional): Gamma parameter. Defaults to 0.01.
            max_epochs (int, optional): Maximum number of epochs for training. Defaults to 10.

        Example:
            kwargs = {
                'data_pipeline': data_pipeline,
                'embedding_size': 20,
                'embedding_type': None,
                'combination_type': 'addition-comb',
                'hidden_size': 64,
                'num_layers': 2,
                'activation_function': 'ReLU',
                'out_activation_function': 'Identity',
                'dropout_rate': 0.25,
                'alpha': 0.25,
                'num_attention_heads': 4,
                'metric': 'smape',
                'learning_rate': 1e-4,
                'weight_decay': 1e-6,
                'prob_decay_1': 0.75,
                'prob_decay_2': 0.9,
                'gamma': 0.01,
                'max_epochs': 50
            }

            model = MLPFQRForecastModel(**kwargs)
        """
        super().__init__(data_pipeline, metric)

        assert len(target_series) > 0, "target_series should not be empty."

        self.n_out = len(target_series)
        n_unknown = len(unknown_features) + self.n_out
        n_covariates = len(known_calendar_features) + len(known_continuous_features)
        self.n_channels = n_unknown + n_covariates

        self.model = LaplaceForecastNetwork(
            n_target_series=self.n_out,
            n_unknown_features=len(unknown_features),
            n_known_calendar_features=len(known_calendar_features),
            n_known_continuous_features=len(known_continuous_features),
            embedding_size=embedding_size,
            embedding_type=embedding_type,
            combination_type=combination_type,
            hidden_size=hidden_size,
            num_layers=num_layers,
            forecast_horizon=forecast_horizon,
            input_window_size=input_window_size,
            activation_function=activation_function,
            out_activation_function=out_activation_function,
            dropout_rate=dropout_rate,
            alpha=alpha,
            num_attention_heads=num_attention_heads
        )
    
    def forecast(self, x):
        """
        Generate forecast for the given input.

        Args:
            x (tensor): Input data for forecasting.

        Returns:
            tensor: Forecasted values.
        """
        return self.model.forecast(x)

    def forward(self, x):
        """
        Forward pass through the model.

        Args:
            x (tensor): Input data.

        Returns:
            tensor: Output of the model.
        """
        return self.model(x)
    
    def sample(self, x, num_samples=500):
        """
        Generate samples using the network.

        Args:
            x (torch.Tensor): Input tensor.
            num_samples (int): Number of samples to generate.

        Returns:
            dict: Dictionary containing the forecast samples.
        """
        return self.model.sample(x, num_samples)
    

class MLPMultivarGaussModel(BaseForecastModel):
    def __init__(
        self,
        data_pipeline=None,
        target_series: list[str] | str = ["NetLoad"],
        unknown_features: list[str] = [],
        calendar_variables: list[str] = [],
        known_calendar_features: list[str] = [],
        known_continuous_features: list[str] = [],
        input_window_size: int = 96,
        forecast_horizon: int = 48,
        embedding_size: int = 28,
        embedding_type: str = None,
        combination_type: str = "addition-comb",
        hidden_size: int = 64,
        num_layers: int = 2,
        expansion_factor: int = 2,
        residual: bool = False,
        activation_function: str = "ReLU",
        out_activation_function: str = "Identity",
        dropout_rate: float = 0.25,
        alpha: float = 0.1,
        num_attention_heads: int = 4,
        metric: str = "mae",
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-6,
        prob_decay_1: float = 0.75,
        prob_decay_2: float = 0.9,
        gamma: float = 0.01,
        max_epochs: int = 10,
    ):
        """
        Initialize the Multilayer Perceptron (MLP) Forecast Model for time series forecasting.

        Args:
            data_pipeline (object, optional): Data pipeline object containing the series and features. Defaults to None.
            target_series (list[str] | str, optional): Target time series to forecast. Defaults to ["NetLoad"].
            unknown_features (list[str], optional): List of unknown features. Defaults to an empty list.
            calendar_variables (list[str], optional): Calendar variables to consider. Defaults to an empty list.
            known_calendar_features (list[str], optional): Known calendar features. Defaults to an empty list.
            known_continuous_features (list[str], optional): Known continuous features. Defaults to an empty list.
            input_window_size (int, optional): Size of the input window. Defaults to 96.
            forecast_horizon (int, optional): Forecast horizon. Defaults to 48.
            embedding_size (int, optional): Dimensionality of the embedding space. Defaults to 28.
            embedding_type (str, optional): Type of embedding to use. Options: 'PosEmb', 'RotaryEmb', 'CombinedEmb'. Defaults to None.
            combination_type (str, optional): Type of combination to use. Options: 'attn-comb', 'weighted-comb', 'addition-comb'. Defaults to 'addition-comb'.
            hidden_size (int, optional): Dimensionality of the hidden layers. Defaults to 64.
            num_layers (int, optional): Number of layers in the MLP. Defaults to 2.
            expansion_factor (int, optional): Factor to expand the size of layers. Defaults to 2.
            residual (bool, optional): Whether to use residual connections. Defaults to False.
            activation_function (str, optional): Activation function for hidden layers. Defaults to "ReLU".
            out_activation_function (str, optional): Activation function for the output layer. Defaults to "Identity".
            dropout_rate (float, optional): Dropout rate for regularization. Defaults to 0.25.
            alpha (float, optional): Alpha parameter for the loss function. Defaults to 0.1.
            num_attention_heads (int, optional): Number of attention heads. Defaults to 4.
            metric (str, optional): Metric for evaluation. Defaults to "mae".
            learning_rate (float, optional): Learning rate for the optimizer. Defaults to 1e-3.
            weight_decay (float, optional): Weight decay for the optimizer. Defaults to 1e-6.
            prob_decay_1 (float, optional): First probability decay rate. Defaults to 0.75.
            prob_decay_2 (float, optional): Second probability decay rate. Defaults to 0.9.
            gamma (float, optional): Gamma parameter. Defaults to 0.01.
            max_epochs (int, optional): Maximum number of epochs for training. Defaults to 10.

        Example:
            kwargs = {
                'data_pipeline': data_pipeline,
                'embedding_size': 20,
                'embedding_type': None,
                'combination_type': 'addition-comb',
                'hidden_size': 64,
                'num_layers': 2,
                'activation_function': 'ReLU',
                'out_activation_function': 'Identity',
                'dropout_rate': 0.25,
                'alpha': 0.25,
                'num_attention_heads': 4,
                'metric': 'smape',
                'learning_rate': 1e-4,
                'weight_decay': 1e-6,
                'prob_decay_1': 0.75,
                'prob_decay_2': 0.9,
                'gamma': 0.01,
                'max_epochs': 50
            }

            model = MLPFQRForecastModel(**kwargs)
        """
        super().__init__(data_pipeline, metric)

        assert len(target_series) > 0, "target_series should not be empty."

        self.n_out = len(target_series)
        n_unknown = len(unknown_features) + self.n_out
        n_covariates = len(known_calendar_features) + len(known_continuous_features)
        self.n_channels = n_unknown + n_covariates

        self.model = MCDMultForecastNetwork(
            n_target_series=self.n_out,
            n_unknown_features=len(unknown_features),
            n_known_calendar_features=len(known_calendar_features),
            n_known_continuous_features=len(known_continuous_features),
            embedding_size=embedding_size,
            embedding_type=embedding_type,
            combination_type=combination_type,
            hidden_size=hidden_size,
            num_layers=num_layers,
            forecast_horizon=forecast_horizon,
            input_window_size=input_window_size,
            activation_function=activation_function,
            out_activation_function=out_activation_function,
            dropout_rate=dropout_rate,
            alpha=alpha,
            num_attention_heads=num_attention_heads
        )
    
    def forecast(self, x):
        """
        Generate forecast for the given input.

        Args:
            x (tensor): Input data for forecasting.

        Returns:
            tensor: Forecasted values.
        """
        return self.model.forecast(x)

    def forward(self, x):
        """
        Forward pass through the model.

        Args:
            x (tensor): Input data.

        Returns:
            tensor: Output of the model.
        """
        return self.model(x)
    
    def sample(self, x, num_samples=500):
        """
        Generate samples using the network.

        Args:
            x (torch.Tensor): Input tensor.
            num_samples (int): Number of samples to generate.

        Returns:
            dict: Dictionary containing the forecast samples.
        """
        return self.model.sample(x, num_samples)
    



