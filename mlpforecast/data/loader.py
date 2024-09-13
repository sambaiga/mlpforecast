import numpy as np
import pytorch_lightning as pl
import torch


class TimeSeriesDataset(torch.utils.data.Dataset):
    """
    A PyTorch Dataset for lazy loading of time series data.

    This dataset handles time series data by storing the inputs and targets as
    tensors. It provides the necessary methods to retrieve the length of the dataset
    and individual data points.

    Parameters
    ----------
    - inputs (np.array): A numpy array of input features. Expected shape is (num_samples, num_features).
    - targets (np.array): A numpy array of target values. Expected shape is (num_samples, target_dim).

    Methods
    -------
    - __len__(): Returns the number of samples in the dataset.
    - __getitem__(index): Returns the input features and target for a given index.

    Usage:
    >>> dataset = TimeSeriesLazyDataset(inputs, targets)
    >>> len(dataset)
    >>> features, target = dataset[index]

    Attributes
    ----------
    - inputs (torch.FloatTensor): The input features stored as a PyTorch FloatTensor.
    - targets (torch.FloatTensor): The target values stored as a PyTorch FloatTensor.
    """

    def __init__(
        self,
        inputs: np.array,
        targets: np.array,
    ):
        self.inputs = torch.FloatTensor(inputs.copy())
        self.targets = torch.FloatTensor(targets.copy())

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, index):
        features = self.inputs[index]
        target = self.targets[index]

        return features, target


class TimeseriesDataModule(pl.LightningDataModule):
    """
    A PyTorch Lightning DataModule for handling time series data.

    This DataModule prepares the data loaders for training and validation datasets.

    Parameters
    ----------
    - train_inputs (np.array): A numpy array of training input features.
    - train_targets (np.array): A numpy array of training target values.
    - val_inputs (np.array, optional): A numpy array of validation input features. Default is None.
    - val_targets (np.array, optional): A numpy array of validation target values. Default is None.
    - drop_last (bool, optional): Whether to drop the last incomplete batch in each epoch. Default is True.
    - num_worker (int, optional): Number of worker processes for data loading. Default is 1.
    - batch_size (int, optional): Number of samples per batch to load. Default is 64.
    - pin_memory (bool, optional): Whether to pin memory during data loading. Default is True.

    Methods
    -------
    - train_dataloader(): Returns the DataLoader for the training dataset.
    - val_dataloader(): Returns the DataLoader for the validation dataset if it exists, otherwise returns None.
    """

    def __init__(
        self,
        train_inputs: np.array,
        train_targets: np.array,
        val_inputs: np.array = None,
        val_targets: np.array = None,
        drop_last: bool = True,
        num_worker: int = 1,
        batch_size: int = 64,
        pin_memory: bool = True,
    ):
        super().__init__()
        self.train_dataset = TimeSeriesDataset(inputs=train_inputs, targets=train_targets)

        if val_inputs is not None:
            self.val_dataset = TimeSeriesDataset(inputs=val_inputs, targets=val_targets)
        else:
            self.val_dataset = None
        self.drop_last = drop_last
        self.num_worker = num_worker
        self.pin_memory = pin_memory
        self.batch_size = batch_size

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=self.drop_last,
            num_workers=self.num_worker,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        if self.val_dataset is not None:
            return torch.utils.data.DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                drop_last=self.drop_last,
                num_workers=self.num_worker,
                pin_memory=self.pin_memory,
            )
        else:
            return None
