import logging

import joblib
import lightning as pl
import torch
import torchmetrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MLPF")


class BaseForecastModel(pl.LightningModule):
    def __init__(self, data_pipeline=None, metric="mae"):
        super().__init__()

        self.model = None
        self.data_pipeline = data_pipeline
        # get model size

        if self.model is not None:
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
        self.checkpoint_path = "./"

    def on_save_checkpoint(self, checkpoint):
        # Save the pipeline to a file
        data_pipeline_path = f"{self.checkpoint_path}/data_pipeline.pkl"
        joblib.dump(self.data_pipeline, data_pipeline_path)
        # Add the pipeline file path to the checkpoint dictionary
        checkpoint["data_pipeline_path"] = data_pipeline_path

    def on_load_checkpoint(self, checkpoint):
        self.data_pipeline = joblib.load(checkpoint["data_pipeline_path"])

    def training_step(self, batch, batch_idx):
        """
        Perform a single training step.

        Args:
            batch (tuple): A batch of training data.
            batch_idx (int): Index of the batch.

        Returns
        -------
            tensor: The loss value for the batch.
        """
        loss, metric = self.model.step(batch, self.tra_metric_fcn)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        self.log(f"train_{self.hparams['metric']}", metric, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Perform a single validation step.

        Args:
            batch (tuple): A batch of validation data.
            batch_idx (int): Index of the batch.

        Returns
        -------
            tensor: The loss value for the batch.
        """
        loss, metric = self.model.step(batch, self.val_metric_fcn)
        self.log("val_loss", loss, prog_bar=True, logger=True)
        self.log(f"val_{self.hparams['metric']}", metric, prog_bar=True, logger=True)

    def configure_optimizers(self):
        """
        Configure optimizers and learning rate schedulers.

        Returns
        -------
            tuple: A tuple containing the optimizer and the scheduler.
        """
        p1 = int(self.hparams["prob_decay_1"] * self.hparams["max_epochs"])
        p2 = int(self.hparams["prob_decay_2"] * self.hparams["max_epochs"])

        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams["learning_rate"],
            weight_decay=self.hparams["weight_decay"],
        )
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[p1, p2], gamma=self.hparams["gamma"])
        return [optimizer], [scheduler]
