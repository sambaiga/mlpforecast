import numpy as np
import warnings
import math

class NonConformityScores(object):
    
    def __init__(self,  conformal_type='res', epislon=1e-10):
        self.conformal_type=conformal_type
        self.epislon=epislon
        
    def get_scores(self, predicts, y_truth):
        if self.conformal_type=='res':
            return self.calculate_residual_score(predicts, y_truth)
        
        elif self.conformal_type=='sign-res':
            return self.calculate_sign_residual_score(predicts, y_truth)
        
        elif self.conformal_type=='sigma-res':
            loc, sigma = predicts
            return self.calculate_sigma_res_score(loc, sigma, y_truth)
        
        elif self.conformal_type=='sigma-sign-res':
            loc, sigma = predicts
            return self.calculate_sigma_sign_res_score(loc, sigma, y_truth)
        
        elif self.conformal_type in ['acr', 'cqr']:
            lower, upper= predicts
            return self.calculate_quantile_score(lower, upper, y_truth)
        
       
    def calculate_residual_score(self, predicts, y_truth):
        if len(y_truth.shape) == 1:
            y_truth=np.expand_dims(y_truth, axis=1)
            y_truth = y_truth.unsqueeze(1)
        return np.abs( y_truth-predicts)
    
    def calculate_sign_residual_score(self, predicts, y_truth):
        if len(y_truth.shape) == 1:
            y_truth=np.expand_dims(y_truth, axis=1)
        return (y_truth-predicts)
    
    
    def calculate_sigma_res_score(self, loc,sigma,  y_truth):
        if len(y_truth.shape) == 1:
            y_truth=np.expand_dims(y_truth, axis=1)
        return np.abs(loc- y_truth)/sigma+self.epislon
    

    def calculate_sigma_sign_res_score(self, loc,sigma,  y_truth):
        if len(y_truth.shape) == 1:
            y_truth=np.expand_dims(y_truth, axis=1)
        return (loc- y_truth)/sigma+self.epislon
    
    def calculate_quantile_score(self, lower, upper, y_truth):
        if len(y_truth.shape) ==1:
            y_truth=np.expand_dims(y_truth, axis=1)
        return np.maximum(lower- y_truth, y_truth - upper)
    

class SplitConformal(NonConformityScores):
    """
    Distribution-Free Predictive Inference For Regression (Lei et al., 2017)
    paper: https://arxiv.org/abs/1604.04173
    
    :param model: a pytorch model for regression.
    """

    def __init__(self, 
                 conformal_type='res'):
        super().__init__(conformal_type=conformal_type)
        
    def calibrate(self, loc, ground_truth, confidence_level:float=0.1):
        self.calculate_threshold(loc, ground_truth, confidence_level)

    def calculate_threshold(self, loc, ground_truth, confidence_level:float=0.1):
        scores = self.get_scores(loc, ground_truth)
        self.q_hat = self.calculate_conformal_value(scores, confidence_level)

        
    def calculate_conformal_value(self, scores, alpha, dim=0):
        """
        Calculate the 1-alpha quantile of scores.
        
        :param scores: non-conformity scores.
        :param alpha: a significance level.
        
        :return: the threshold which is use to construct prediction sets.
        """
        if alpha >= 1 or alpha <= 0:
                raise ValueError("Significance level 'alpha' must be in (0,1).")
        if len(scores) == 0:
            warnings.warn(
                "The number of scores is 0, which is a invalid scores. To avoid program crash, the threshold is set as torch.inf.")
            return np.inf
        
        qunatile_value = math.ceil(scores.shape[0] + 1) * (1 - alpha) / scores.shape[0]

        if qunatile_value > 1:
            warnings.warn(
                "The value of quantile exceeds 1. It should be a value in (0,1). To avoid program crash, the threshold is set as torch.inf.")
            return np.inf
        
        return np.quantile(scores, qunatile_value, method='midpoint', axis=dim)


    
    def get_calibrated_pred(self, out):
        if self.conformal_type=='smape':
            out['upper-calib']=out['loc']*(1+self.q_hat)
            out['lower-calib']=out['loc']*(1-self.q_hat)
        else:
            out['upper-calib']=out['loc']+self.q_hat
            out['lower-calib']=out['loc']-self.q_hat
        return out
    

class ConformalResidualFitting(SplitConformal):
    """
    Conformalized Quantile Regression (Romano et al., 2019)
    paper: https://arxiv.org/abs/1905.03222

    :param model: a pytorch model that can output alpha/2 and 1-alpha/2 quantile regression.
    """

    def __init__(self, conformal_type='sigma-res'):
        super().__init__(conformal_type)
        
    def calibrate(self, loc, scale, ground_truth, confidence_level:float=0.1):
        predicts =(loc,  scale)
        self.calculate_threshold(predicts, ground_truth, confidence_level)

    def get_calibrated_pred(self, out):
        out['upper-calib']=out['loc']+out['scale']*self.q_hat
        out['lower-calib']=out['loc']-out['scale']*self.q_hat
        return out
    

class ConformalQuantileRegressor(SplitConformal):
    """
    Conformalized Quantile Regression (Romano et al., 2019)
    paper: https://arxiv.org/abs/1905.03222

    :param model: a pytorch model that can output alpha/2 and 1-alpha/2 quantile regression.
    """

    def __init__(self, conformal_type='cqr'):
        super().__init__(conformal_type)
        
        
    def calibrate(self, lower_q, upper_q, ground_truth, confidence_level:float=0.1):
        predicts =(lower_q, upper_q)
        self.calculate_threshold(predicts, ground_truth, confidence_level)

    
    def get_calibrated_pred(self, outs):
        outs['upper-calib']=outs['upper-q']+self.q_hat
        outs['lower-calib']=outs['lower-q']+self.q_hat
        return outs

      

      

    


    
    
    
    
    
    
    
    
    