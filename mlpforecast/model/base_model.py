import logging

import joblib
import lightning as pl
import torchmetrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MLPF")


class BaseForecastModel(pl.LightningModule):
    """
    Base class for all forecasting models.

    Attributes:
        model (torch.nn.Module): PyTorch model.
        data_pipeline (sklearn.pipeline.Pipeline): Data pipeline.
        tra_metric_fcn (torchmetrics.Metric): Training metric function.
        val_metric_fcn (torchmetrics.Metric): Validation metric function.
        size (float): Model size in MB.
        checkpoint_path (str): Path to save checkpoints.
    """

    def __init__(self, data_pipeline=None, metric="mae"):
        """
        Initialize the model.

        Args:
            data_pipeline (sklearn.pipeline.Pipeline): Data pipeline.
            metric (str): Metric to use for evaluation. Options: 'mae', 'mse', 'smape'.
        """
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
        """
        Save the data pipeline to a file and add the file path to the checkpoint dictionary.

        Args:
            checkpoint (dict): Checkpoint dictionary.
        """
        # Save the pipeline to a file
        data_pipeline_path = f"{self.checkpoint_path}/data_pipeline.pkl"
        joblib.dump(self.data_pipeline, data_pipeline_path)
        # Add the pipeline file path to the checkpoint dictionary
        checkpoint["data_pipeline_path"] = data_pipeline_path

    def on_load_checkpoint(self, checkpoint):
        """
        Load the data pipeline from a file.

        Args:
            checkpoint (dict): Checkpoint dictionary.
        """
        self.data_pipeline = joblib.load(checkpoint["data_pipeline_path"])
