
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pytorch_lightning as pl
import torchmetrics
from .layers import  MLPForecastNetwork
torch.set_float32_matmul_precision('high')





class MLPForecastModel(pl.LightningModule):
    
    def __init__(self,  hparams):
        super().__init__()
        self.model = MLPForecastNetwork(hparams=hparams)
        param_size = 0
        for param in self.model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in self.model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        self.size = (param_size + buffer_size) / 1024**2
        print('model size: {:.3f}MB'.format(self.size))
        
        self.tra_metric_fcn=torchmetrics.MeanAbsoluteError()
        self.val_metric_fcn=torchmetrics.MeanAbsoluteError()

        self.save_hyperparameters()
        self.hparams.update(hparams)
        
    def forecast(self, x):
        return self.model.forecast(x)
    
    def training_step(self, batch, batch_idx):
        
        loss, metric = self.model.step(batch, self.tra_metric_fcn)
        self.log("train_loss",loss, prog_bar=True, logger=True)
        self.log("train_mae",metric, prog_bar=True, logger=True)

        return loss
            
    
    def validation_step(self, batch, batch_idx):
        
        loss, metric = self.model.step(batch, self.val_metric_fcn) 
        self.log("val_loss",loss, prog_bar=True, logger=True)
        self.log("val_mae",metric, prog_bar=True, logger=True)

     

    def configure_optimizers(self):
        p1 = int(0.75 * self.hparams.max_epochs)
        p2 = int(0.9 * self.hparams.max_epochs)

        
        params  = list(self.parameters())
        optim = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate,  weight_decay=self.hparams.weight_decay)
           
        scheduler  = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[p1, p2], gamma=0.1)
        return [optim], [scheduler]
    

    def get_search_params(self, trial, params):
        # We optimize the number of layers, hidden units and dropout ratio in each layer.

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
        
            
        
       