from .cwe_score import get_cwi_score
from IPython.display import clear_output
from tqdm import tqdm
from .data_processing import add_time_features
from .conformal_numpy import non_conform_score_parametric, non_conform_score_quantile, non_conform_score_residual
import torch
from pyro.ops.stats import crps_empirical
from .metrics import  get_daily_metrics, get_pinball_score
import logging
import numpy as np
import pandas as pd
from scipy import stats
# Configure the logging module
logging.basicConfig(level=logging.INFO)
from pyro.contrib.forecast import  eval_crps

#from utils.visual_functions import set_matplolib_formart
#set_matplolib_formart()
hparams={}


def get_nll(true, loc, scale, dist_type='normal'):
    
    #nll=stats.multivariate_normal.logpdf(true, mean=loc, cov=scale, allow_singular=True)
        
    if dist_type in ['normal', 'multivariate']:
        nll=stats.norm.logpdf(true, loc=loc, scale=scale)
        
    if dist_type in ['laplace', 'lapace']:
        nll=stats.laplace.logpdf(true, loc=loc, scale=scale)
    
    if dist_type=='cauchy':
        nll=stats.cauchy.logpdf(true, loc=loc, scale=scale)
    return np.mean(nll)
   
    #return np.mean(np.log(np.sqrt(2*np.pi)*scale)+(true-loc)**2/2/scale**2)

def get_parametric_crps(sample, true):
    CRPS = crps_empirical(torch.from_numpy(sample), torch.from_numpy(true)).mean().item()
    return  CRPS

def crps_quantile(
    target: np.array,
    samples: np.array,
    quantiles: np.array = (np.arange(90) / 90.0)[1:]) -> np.float32:
    # Compute the CRPS using the quantile scores
    assert target.shape[0] == samples.shape[1]
    assert target.shape[1] == samples.shape[2]
    num_samples = samples.shape[0]

    sorted_samples = np.sort(samples, axis=0)

    result_sum = np.float32(0)

    for q in quantiles:
        # From 0 to num_samples - 1 so that the 100% quantile is the last sample
        q_idx = int(np.round(q * (num_samples - 1)))
        q_value = sorted_samples[q_idx, :, :]

        # The absolute value is just there in case of numerical inaccuracies
        quantile_score = np.abs(2 * ((target <= q_value) - q) * (q_value - target))

        result_sum += np.nanmean(quantile_score, axis=(0, 1))

    return result_sum / len(quantiles)

def get_quantile_realibility(true:np.array, q_pred:np.array, tau:np.array):
    #https://github.com/tony-psq/QRMGM_KDE/blob/master/QRMGM_KDE/evaluation/Evaluation.py
    assert true.ndim == 1, "pred must be 1-dimensional"
    assert q_pred.ndim == 2, "pred must be 1-dimensional"
    assert tau.ndim == 2, "pred must be 1-dimensional"
    assert tau.shape == q_pred.shape, "pred and true must have the same shape"
    assert len(true) == q_pred.shape[0], "pred and true must have the same shape"
    
    y_cdf = np.zeros((q_pred.shape[0], q_pred.shape[1] + 2))
    y_cdf[:, 1:-1] = q_pred
    y_cdf[:, 0] = 2.0 * q_pred[:, 1] - q_pred[:, 2]
    y_cdf[:, -1] = 2.0 * q_pred[:, -2] - q_pred[:, -3]
    
    
    qs = np.zeros((q_pred.shape[0], q_pred.shape[1] + 2))
    qs[:, 1:-1] = tau
    qs[:, 0] = 0.0
    qs[:, -1] = 1.0
    
    ind = np.zeros(y_cdf.shape)
    ind[y_cdf > true.reshape(-1, 1)] = 1.0
    CRPS = np.trapz((qs - ind) ** 2.0, y_cdf)
    CRPS = np.mean(CRPS)

    
    PIT = np.zeros(true.shape)
    for i in range(true.shape[0]):
        PIT[i] = np.interp(np.squeeze(true[i]), np.squeeze(y_cdf[i, :]), np.squeeze(qs[i, :]))
        
    return PIT, CRPS


def evaluate_prob_forecast(outputs, t_nmpic,  
                           conformalize=False, 
                           alpha=0.05, 
                           encoder_type='MLPFQR',
                           R=None, time_conformize=False, dist_type='normal', 
                           target_columns=['NetLoad']):
    
    quantiles = [alpha / 2, 1 - alpha / 2]
    q=np.arange(0.1, 1, 0.1).round(1).tolist()
    q.insert(0, quantiles[0])
    q.append( quantiles[-1])
    hparams.update({'quantiles':  q})
    pd_metrics, spilit_metrics = {}, {}
    logs = {}
    
    N, T, C = outputs['pred'].shape  if 'pred' in outputs.keys() else outputs['loc'].shape
    for j in range(C):
        metrics=[]
        q_hat=1
        for i in range(N):
            true = outputs['true'][i,:, j]
            pred = outputs['pred'][i,:, j] if 'pred' in outputs.keys() else outputs['loc'][i,:, j] 
            if conformalize:
                low, upp = outputs['lower-calib'][i,:, j], outputs['upper-calib'][i,:, j]
            else:
                if 'quantile_hats' in outputs.keys():
                    low, upp = outputs['quantile_hats'][i, 0, :, j], outputs['quantile_hats'][i, -1, :, j]
                else:
                    #low = outputs['loc'][i,  :, j]-outputs['scale'][i,  :, j]
                    #upp = outputs['loc'][i,  :, j]+outputs['scale'][i,  :, j]
                    low, upp = outputs['lower'][i,  :, j], outputs['upper'][i,  :, j] 
            
            q_p= outputs['q_sample'][:, i, :, j].T  if 'q_sample' in outputs.keys() else None
            tau = outputs['tau_hats'][i].T if 'quantile_hats' in outputs.keys() else None
            if 'quantile_hats' in outputs.keys():
                sample=outputs['quantile_hats'][i, :, :, j].T 
                
            elif 'samples' in  outputs.keys():
                sample=outputs['samples'][:, i, :, j].T
            else:
                sample=None
            if tau is not None:
                PIT, CRPS=get_quantile_realibility(true, sample, tau)
            elif sample is not None:
                CRPS = get_parametric_crps(sample.T, true)
            else:
                CRPS = 0.0
            
            
             
            #R=outputs['target-range'][j] if R is None else R
            R=true.max()
            df = pd.DataFrame(outputs['index'][i])
            df.columns=['Date']
            index=df.Date.dt.round("D").unique()[-1]
            if t_nmpic is None:
                t_nmpic = true.std()
            scores=get_daily_metrics(true, pred,None, low, upp,  alpha=1-alpha, q_hat=0, t_nmpic=t_nmpic, R=R)
            
            if 'dist_type' in list(outputs.keys()):
                scores['nll']=get_nll(outputs['true'][i,:, j], 
                                      outputs['loc'][i,:, j],
                                      outputs['scale'][i,:, j], dist_type=dist_type)
            
            if 'q_sample' in list(outputs.keys()):
                scores['PL']=get_pinball_score(true, outputs['q_sample'][:, i,:, j], hparams['quantiles'])
            else:
                scores['PL']  = 0.0
            
            if 'samples' in list(outputs.keys()):
                scores['crps'] = eval_crps(torch.tensor(outputs['samples'][:, i,:, j]), torch.tensor(true))
            else:
                scores['crps']  = 0.0
            scores['timestamp']=index
            scores['R']=R
            scores['CRPS']=CRPS
            scores['Bad-id']=outputs[f"{target_columns[j]}_bad"]
            scores['Good-id']=outputs[f"{target_columns[j]}_good"]
            metrics.append(scores)
           
        metrics = pd.concat(metrics)
        metrics['timestamp']=pd.to_datetime(metrics['timestamp'], utc=True)
        metrics=metrics.set_index('timestamp')
        metrics['target']=target_columns[j]
        outputs[f"{target_columns[j]}_metrics"]=metrics
    return outputs


## 
#
def get_metrics_per_dataset(dataset='austgrid_dataset', 
                                          window='expanding', 
                                          exp_name='TEST_NEW_PIPELINE',
                                         conformalize=False, 
                                         time_conformize=False,
                                          alpha=0.05, 
                                          conformal_loss=False,
                                          cqr_loss=False,
                                          fgm_loss=False,
                                          residual=False,
                                          sigma_loss=False,
                                          dist_type='laplace',
                                          conf_level=0.9,
                                          conf='SCP',
                                          n_folds=5,
                                          conformal_type='defaulty',
                                          target_columns=['NetLoad'],
                                          encoders=[ "MLPMCDForecast",  "MLPBNNForecast", "MLPGMM", "MLProbForecast",  "MLPQR","MLPFPQ"]):
    
    file_name=f'{exp_name}_{dataset}_{window}_conf_{conf_level}' 
   
    if conformal_loss:
        file_name=f'{file_name}_with_conformal_loss'
        
    if  cqr_loss:
        file_name=f'{file_name}_with_cqr_loss'
      
    if  sigma_loss:
        file_name=f'{file_name}_with_sigma_loss'
        

    if  residual:
        #file_name=f'{file_name}_and_residual'
        file_name=f'{file_name}_defaulty'
    
    if dist_type is not None:
        file_name=f'{file_name}_{dist_type}'
       
    logging.info(file_name)
    all_metrics=[]
    all_residual={}
    all_scale=[]
    all_data=[]
    for encoder_type in encoders:
        metrics_per_model=[]
        residual_per_model=[]
        scale_per_model=[]
        data_per_model=[]
        logging.info(encoder_type)
        
        for cross in tqdm(range(0, n_folds)):
            if conformalize:
                

                path=f"../results/{file_name}/{encoder_type}/{cross}_{conf}-conf-{conformal_type}_processed_results.npy"
                results=np.load(path, allow_pickle=True).item()
            else:
                print(cross)
                results=np.load(f"../results/{file_name}/{encoder_type}/{cross}_cross_validation_processed_results.npy", allow_pickle=True).item()
            true=results['true']
            pred=results['pred'] if 'pred' in results.keys() else results['loc']
            scale=results['scale'] if 'scale' in results.keys() else pred
            index=results['index'][:len(pred)]
                 
            scale = scale.reshape(-1, scale.shape[-1])
            pred = pred.reshape(-1, pred.shape[-1])
            true = true.reshape(-1, true.shape[-1])
            df_scale = pd.DataFrame(scale, columns=[f"{x}-scale" for x in target_columns], index=index.flatten())
            df_pred = pd.DataFrame(pred, columns=[f"{x}-loc" for x in target_columns], index=index.flatten())
            df_true = pd.DataFrame(true, columns=target_columns, index=index.flatten())
            df=pd.concat([df_true, df_pred, df_scale], axis=1)
            df['Model']=f'{encoder_type}'
            
            
            t = stats.norm.ppf((1 + alpha) / 2) 
            t_nmpic=None
           
            
           
            df.index=pd.to_datetime(df.index, utc=True)
           
            
            data_per_model.append(df)
            
            
            residual=true-pred
            residual_per_model.append(residual)
            results=evaluate_prob_forecast(results,  
                                           t_nmpic, 
                                           conformalize,  
                                           alpha, encoder_type, 
                                           R=None, 
                                           time_conformize=time_conformize, 
                                           target_columns=target_columns)
           
            metrics=[]
            for target_column in target_columns:
                metric=results[f'{target_column}_metrics']
                metrics.append(metric)
                
            metrics = pd.concat(metrics)
            metrics['Fold']=cross
            metrics['Model']=encoder_type
            #metrics['Train-time']=results['train-time'] if 'train-time' in results.keys() else results['train_time']
            #metrics['Test-time']=results['test-time'] if 'test-time' in results.keys() else results['test_time']
            
            metrics=add_time_features(metrics, hemisphere = 'Northern')
            metrics=metrics.replace("MLPMCDForecast", 'MLP-MCD').replace("MLPBNNForecast", 'MLP-BNN').replace("MLPGMM", 'MLP-MDN').replace("MLProbForecast", 'MLP-LD').replace("MLPFPQ", "MLP-FPQ").replace('MLPQR', 'MLP-QR')
            metrics_per_model.append(metrics)
            all_residual[encoder_type]=residual_per_model
        metrics_per_model=pd.concat(metrics_per_model)
        all_metrics.append(metrics_per_model)
        all_data.append(pd.concat(data_per_model))
        
    
    clear_output()
    metrics=pd.concat(all_metrics)
    all_data=pd.concat(all_data, axis=1)
   
    del all_metrics
    del df 
  
    del results
    del pred
    del true
    
    return metrics, all_residual, all_data


def get_results(window='expanding',
                exp_name="dist_analysis",
                alpha=0.1,
                encoders=["MLProbForecast"],
                datasets=[ 'pt_dataset', 'uk_dataset'],
                dists=[  'cauchy','lapace', 'multivariate'],
                conformalize=False,
                conf='SCP',
                conformal_type='sign-res',
                conformal_loss=False,
                cqr_loss=False,
                sigma_loss=False,
                n_folds=10,
                residual=False, target_columns=['NetLoad']):

    metrics_all=[]
    for dist_type in dists:
        for dataset in datasets:
            metrics, all_residual_pt, all_data_pt=get_metrics_per_dataset(dataset, 
                                                                          window, 
                                                                          exp_name, 
                                                                          encoders=encoders,
                                                                          conformalize=conformalize,
                                                                          conf=conf,
                                                                          conformal_type=conformal_type,
                                                                          alpha=alpha,
                                                                          conf_level=1-alpha,
                                                                          dist_type=dist_type,
                                                                          conformal_loss=conformal_loss,
                                                                          cqr_loss=cqr_loss,
                                                                          sigma_loss=sigma_loss,
                                                                          residual=residual,
                                                                          n_folds=n_folds[dataset],
                                                                          target_columns=target_columns[dataset]
                                                                )
            if dataset =='pt_dataset':
                data_name='MLVS-PT'
            elif dataset =='uk_dataset':
                data_name='SPS-UK'
            elif dataset =='pv_dataset':
                data_name='PV-UK'
            elif dataset =='dss_dataset':
                data_name='DSS-EU'
            elif dataset =='albania_dataset':
                 data_name='ALBANIA'
            elif dataset == 'pt_dataset_all_phases':
                data_name='MLVS-PT'
                
            metrics['Dataset']=data_name
            if dist_type is not None:
                metrics['dist-type']=dist_type.upper()
            metrics_all.append(metrics)

    metrics_all = pd.concat(metrics_all)
    return metrics_all