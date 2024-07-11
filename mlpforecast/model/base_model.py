import logging

import joblib
import pytorch_lightning as pl
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
