import logging

import torch
import optuna

from mlpforecast.model.base_model import BaseForecastModel
from mlpforecast.model.mlpf_org.layers import MLPForecastNetwork
from optuna import Trial

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MLPF")


class MLPForecastModelOG(BaseForecastModel):
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
        hparams=None,
        metric: str = "mae",
        input_window_size:int= None,
        forecast_horizon:int=None
    ):
        r"""
        Multilayer Perceptron (MLP) Forecast Model for time series forecasting.

        Args:
            data_pipeline (object, optional): Data pipeline object containing the series and features. Defaults to None.
            embedding_size (int, optional): Dimensionality of the embedding space. Defaults to 28.
            embedding_type (str, optional): Type of embedding to use.\
                  Options: 'PosEmb', 'RotaryEmb', 'CombinedEmb'. Defaults to None.
            combination_type (str, optional): Type of combination to use. Options: \
                'attn-comb', 'weighted-comb', 'addition-comb'. Defaults to 'attn-comb'.
            hidden_size (int, optional): Dimensionality of the hidden layers. Defaults to 64.
            num_layers (int, optional): Number of layers in the MLP. Defaults to 2.
            expansion_factor (int, optional): Factor to expand the size of layers. Defaults to 2.
            residual (bool, optional): Whether to use residual connections. Defaults to False.
            activation_function (str, optional): \
                  Activation function to use in the hidden layers. Defaults to "ReLU".
            out_activation_function (str, optional): Activation function to use in the output layer. \
                Defaults to "Identity".
            dropout_rate (float, optional): Dropout rate for regularization. Defaults to 0.25.
            alpha (float, optional): Alpha parameter for the loss function. Defaults to 0.1.
            num_attention_heads (int, optional): Number of attention heads. Defaults to 4.
            metric (str, optional): Metric to evaluate the model. Defaults to "mae".
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
                'combination_type': 'Add',
                'hidden_size': 64,
                'num_layers': 2,
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

            model = MLPForecastModel(**kwargs)
        """
        super().__init__(data_pipeline, metric)

        #assert len(target_series) > 0, "target_series should not be empty."
        self.model = MLPForecastNetwork(hparams=hparams)
        self.save_hyperparameters()
        self.hparams.update(hparams)


    def forecast(self, x):
        """
        Generate forecast for the given input.

        Args:
            x (tensor): Input data for forecasting.

        Returns:
            (tensor): Forecasted values.
        """
        return self.model.forecast(x)


    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (tensor): Input data.
        """
        return self.model(x)


    def training_step(self, batch, batch_idx):
        """
        Perform a single training step.

        Args:
            batch (tuple): A batch of training data.
            batch_idx (int): Index of the batch.

        Returns:
            (tensor): The loss value for the batch.
        """
        loss, metric = self.model.step(batch, self.tra_metric_fcn)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        self.log(f"train_mae", metric, prog_bar=True, logger=True)
        return loss


    def validation_step(self, batch, batch_idx):
        """
        Perform a single validation step.

        Args:
            batch (tuple): A batch of validation data.
            batch_idx (int): Index of the batch.

        Returns:
            (tensor): The loss value for the batch.
        """
        loss, metric = self.model.step(batch, self.val_metric_fcn)
        self.log("val_loss", loss, prog_bar=True, logger=True)
        self.log(f"val_mae", metric, prog_bar=True, logger=True)

    
        

    def configure_optimizers(self):
        p1 = int(0.75 * self.hparams.max_epochs)
        p2 = int(0.9 * self.hparams.max_epochs)

        
        params  = list(self.parameters())
        optim = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate,  weight_decay=self.hparams.weight_decay)
           
        scheduler  = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[p1, p2], gamma=0.1)
        return [optim], [scheduler]
    

    def get_search_params(self, trial):
        # We optimize the number of layers, hidden units and dropout ratio in each layer.
        params = {}
        latent_size = {'latent_size': trial.suggest_categorical("latent_size", [16, 32, 64, 128, 256, 512] )}
        params.update(latent_size)

        depth = {'depth':trial.suggest_categorical("depth", [1, 2, 3, 4, 5])}
        params.update(depth)

        dropout = {'dropout':trial.suggest_float("dropout", 0.1, 0.9)}
        params.update(dropout)

        activation  = {'activation':trial.suggest_categorical("activation", [0, 1, 2, 3, 4])}
        params.update(activation)
    
        emb_type = {'emb_type':trial.suggest_categorical("emb_type",["None", 'PosEmb', 'RotaryEmb', 'CombinedEmb'])}
        params.update(emb_type)
        
        emb_size = {'emb_size':trial.suggest_categorical("emb_size",[8,  16, 32, 64])}
        params.update(emb_size)
        
        comb_type = {'comb_type':trial.suggest_categorical("comb_type",['attn-comb', 'weighted-comb', 'addition-comb'])}
        params.update(emb_size)
        if comb_type=='attn-comb':
            num_head = {'num_head':trial.suggest_categorical("num_head",[2, 4,  8, 16])}
            params.update(num_head)
        

        
        alpha = {'alpha':trial.suggest_float("alpha", 0.01, 0.9)}
        params.update(alpha)
       
        return params
        
            
        