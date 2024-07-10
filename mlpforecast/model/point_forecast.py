import torch
import pytorch_lightning as pl
import logging
import torchmetrics
from mlpforecast.net.layers import MLPForecastNetwork
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MLPF")


class MLPForecastModel(pl.LightningModule):

    def __init__(
        self,
        data_pipeline,
        target_series,
        unknown_features=[],
        known_calender_features=[],
        known_continuous_features=[],
        embedding_size: int = 28,
        embedding_type: str = None,
        combination_type: str = "attn-comb",
        hidden_size: int = 64,
        num_layers: int = 2,
        expansion_factor: int = 2,
        residual: bool = False,
        forecast_horizon: int = 48,
        input_window_size: int = 96,
        activation_function: str = "ReLU",
        out_activation_function: str = "Identity",
        dropout_rate: float = 0.25,
        alpha=0.1,
        num_attention_heads: int = 4,
        metric="mae",
        learning_rate=1e-3,
        weight_decay=1e-6,
        prob_decay_1=0.75,
        prob_decay_2=0.9,
        gamma=0.01,
        max_epochs=10,
    ):
        """
        Multilayer Perceptron (MLP) Forecast Model for time series forecasting.

        Parameters:
            targets (List): List of target variables.
            time_varying_unknown_feature (List, optional): \
                  List of unknown time-varying features. Defaults to [].
            time_varying_known_categorical_feature (List, optional): \
                  List of known categorical time-varying features. Defaults to [].
            time_varying_known_feature (List, optional): \
                List of known time-varying features. Defaults to [].
            emb_size (int, optional): Dimensionality of the embedding space. Defaults to 28.
            embed_type (String, optional): \
                Type of embedding to use. Defaults to None. Either -> 'PosEmb', 'RotaryEmb', 'CombinedEmb'
            comb_type (String, optional): \
                Type of combination to use. Defaults to 'attn-comb'. Either -> 'attn-comb', 'weighted-comb', 'addition-comb'
            latent_size: (int, optional): \
                Dimensionality of the latent space. Defaults to 64.
            depth (int, optional): Number of layers in the MLP. Defaults to 2.
            horizon (int, optional): Number of future time steps to forecast. Defaults to 48.
            window_size (int, optional): Size of the input window. Defaults to 96.
            activation (torch.nn.Module, optional): Activation function. Defaults to ReLU().
            dropout (float, optional): Dropout probability. Defaults to 0.25.
            num_head (int, optional): Number of heads in the multi-head attention. Defaults to 4.
            alpha (float, optional): Alpha parameter for the loss. Defaults to 0.01.
            learning_rate (float, optional): Learning rate for the optimizer. Defaults to 0.001.
            weight_decay (float, optional): Weight decay for the optimizer. Defaults to 1e-6.
            max_epochs (int, optional): Maximum number of epochs for training. Defaults to 50.

        Example:

            kwargs = {
                    'target_series': ['Load'],
                    'unknown_features': [],
                    'known_calender_features': ['hour'],
                    'known_continuous_features': ['Ghi'],
                    'embedding_size': 20,
                    'embedding_type': None,
                    'combination_type': 'Add',
                    'hidden_size': 64,
                    'num_layers': 2,
                    'forecast_horizon': 48,
                    'input_window_size': 48,
                    'activation_function': 'ReLU',
                    'out_activation_function': 'ReLU',
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

            m = MLPForecastModel(**kwargs)
        """

        super().__init__()
        # Ensure target_series is a list
        if isinstance(target_series, str):
            target_series = [target_series]
        elif not isinstance(target_series, list):
            raise ValueError("target_series should be a string or a list of strings.")

        # Assertion to ensure target_series is not empty
        assert len(target_series) > 0, "target_series should not be empty."
        self.n_out = len(target_series)
        n_unknown = len(unknown_features) + self.n_out
        n_covariates = len(known_calender_features) + len(known_continuous_features)
        self.n_channels = n_unknown + n_covariates
        self.data_pipeline = data_pipeline
        self.model = MLPForecastNetwork(
            n_target_series=len(target_series),
            n_unknown_features=len(unknown_features),
            n_known_calender_features=len(known_calender_features),
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
            num_attention_heads=num_attention_heads,
        )

        # get model size
        param_size = 0
        for param in self.model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in self.model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        self.size = (param_size + buffer_size) / 1024**2
        logger.info(f"Model size: {self.size:.3f}MB")

        # Initialize metric functions
        if metric == "mae":
            self.tra_metric_fcn = torchmetrics.MeanAbsoluteError()
            self.val_metric_fcn = torchmetrics.MeanAbsoluteError()

        elif metric == "mse":
            self.tra_metric_fcn = torchmetrics.MeanSquaredError()
            self.val_metric_fcn = torchmetrics.MeanSquaredError()

        elif metric == "smape":
            self.tra_metric_fcn = torchmetrics.SymmetricMeanAbsolutePercentageError()
            self.val_metric_fcn = torchmetrics.SymmetricMeanAbsolutePercentageError()
        else:
            raise ValueError("Invalid metric. Please select 'mae', 'smape', 'mse'.")
        self.save_hyperparameters()
    
    def forecast(self, x):
        return self.model.forecast(x)

    def training_step(self, batch, batch_idx):
        loss, metric = self.model.step(batch, self.tra_metric_fcn)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        self.log(f"train_{self.hparams['metric']}", metric, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, metric = self.model.step(batch, self.val_metric_fcn)
        self.log("val_loss", loss, prog_bar=True, logger=True)
        self.log(f"val_{self.hparams['metric']}", metric, prog_bar=True, logger=True)

    def configure_optimizers(self):
        p1 = int(self.hparams["prob_decay_1"] * self.hparams["max_epochs"])
        p2 = int(self.hparams["prob_decay_2"] * self.hparams["max_epochs"])

        list(self.parameters())
        optim = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams["learning_rate"],
            weight_decay=self.hparams["weight_decay"],
        )
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optim, milestones=[p1, p2], gamma=self.hparams["gamma"]
        )
        return [optim], [scheduler]
