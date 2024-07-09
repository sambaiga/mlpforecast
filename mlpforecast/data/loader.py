from torch.utils.data import DataLoader, TensorDataset, Dataset
import torch
import numpy as np
import pytorch_lightning as pl
from tqdm import tqdm

class TimeSeriesDataset(object):   
    def __init__(self, unknown_features, kown_features, targets, window_size=96, horizon=48, batch_size=64, shuffle=False, test=False, drop_last=True):
        self.inputs = unknown_features
        self.covariates = kown_features
        self.targets = targets
        self.window_size = window_size
        self.horizon = horizon
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.test = test
        self.drop_last= drop_last
        
    def frame_series(self):
        
        nb_obs, nb_features = self.inputs.shape
        features, targets, covariates = [], [], []
        

        list_range = range(0, nb_obs - self.window_size - self.horizon+1, self.horizon) if self.test else range(0, nb_obs - self.window_size - self.horizon+1)
        with tqdm(len(list_range)) as pbar:
            for i in list_range:
                features.append(torch.FloatTensor(self.inputs[i:i + self.window_size, :]).unsqueeze(0))
                targets.append(
                        torch.FloatTensor(self.targets[i + self.window_size:i + self.window_size + self.horizon]).unsqueeze(0))
                covariates.append(
                        torch.FloatTensor(self.covariates[i + self.window_size:i + self.window_size + self.horizon,:]).unsqueeze(0))

                pbar.set_description('processed: %d' % (1 + i))
                pbar.update(1)
            pbar.close() 

        features = torch.cat(features)
        targets, covariates = torch.cat(targets), torch.cat(covariates)
        
        
        
        #padd covariate features with zero
        diff = features.shape[2] - covariates.shape[2]
        B, N, _ = covariates.shape
        diff = torch.zeros(B, N, diff, requires_grad=False)
        covariates = torch.cat([diff, covariates], dim=-1)
        features = torch.cat([features, covariates], dim=1)
        
        del covariates
        del diff
        
       

        return TensorDataset(features,  targets)
        
    def get_loader(self):
        dataset = self.frame_series()
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=self.shuffle, drop_last=self.drop_last, num_workers=8, pin_memory=True)
        return loader


class TimeSeriesLazyDataset(Dataset):   
    def __init__(self, unknown_features, kown_features, targets, window_size=96, horizon=48):
        self.inputs = torch.FloatTensor(unknown_features)
        self.covariates = torch.FloatTensor(kown_features)
        self.targets = torch.FloatTensor(targets)
        self.window_size = window_size
        self.horizon = horizon
        
        
    def __len__(self):
        return self.inputs.shape[0]-self.window_size-self.horizon
    
    def __getitem__(self, index):
        features = self.inputs[index:index + self.window_size]
        target = self.targets[index+self.window_size:index+self.window_size+self.horizon]
        covariates = self.covariates[index+self.window_size:index+self.window_size+self.horizon]
        
        #padd covariate features with zero
        diff = features.shape[1] - covariates.shape[1]
        N, _ = covariates.shape
        diff = torch.zeros(N, diff, requires_grad=False)
        covariates = torch.cat([diff, covariates], dim=-1)
        features = torch.cat([features, covariates], dim=0)
        del covariates
        del diff
        return features, target
            

class TimeseriesDataModule(pl.LightningDataModule):
    def __init__(self, hparams, experiment, train_df, test_df, test=False):
        super().__init__()
       
        target, known_features, unkown_features = experiment.get_data(data=train_df)
        if test:
            self.train_dataset = TimeSeriesDataset(unkown_features, known_features, target, window_size=hparams['window_size'], horizon=hparams['horizon'],
                                                    batch_size=hparams['batch_size'], shuffle=True, test=False, drop_last=True)
        else:
            self.train_dataset = TimeSeriesLazyDataset(unkown_features, known_features, target, window_size=hparams['window_size'], horizon=hparams['horizon'])
      

        target, known_features, unkown_features = experiment.get_data(data=test_df)
        
        if test:
            self.test_dataset = TimeSeriesDataset(unkown_features, known_features, target, window_size=hparams['window_size'], horizon=hparams['horizon'],
                                                    batch_size=hparams['batch_size'], shuffle=False, test=False, drop_last=False)
        else:
            self.test_dataset = TimeSeriesLazyDataset(unkown_features, known_features, target, window_size=hparams['window_size'], horizon=hparams['horizon'])
        del target
        del known_features
        del unkown_features
        self.batch_size=hparams['batch_size']
        self.test = test

    def train_dataloader(self):
        if self.test:
            return self.train_dataset.get_loader()
        else:
            return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=8, pin_memory=True)

    def val_dataloader(self):
        if self.test:
            return self.test_dataset.get_loader()
        else:
            return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, drop_last=True, num_workers=8, pin_memory=True)





