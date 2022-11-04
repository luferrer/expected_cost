from psrcal.calibration import calibrate, AffineCalLogLoss
from sklearn.model_selection import StratifiedKFold
import torch
import numpy as np

def affine_calibration_with_crossval(logpost, targets, use_bias=True, priors=None):
    """ This is a utility method for performing calibration on the test scores
    using cross-validation. The calibration is done using an affine transformation
    of the form:
    
    logpostcal_i = logsoftmax ( scale * logpostraw_i + bias_i)

    The scale is the same for all classes, but the bias is class-dependent.
    If use_bias = False, this method does the usual temp-scaling.

    The method expects (potentially misscalibrated) log posteriors or log scaled
    likelihoods as input. In both cases, the output should be well-calibrated
    log posteriors. Note, though that when use_bias is False, it is 
    probably better to feed log posteriors since, without a bias term, the
    calibration cannot introduce the required priors to obtain a good log-posterior. 
    """

    logpostcal = np.zeros_like(logpost)
    skf = StratifiedKFold(n_splits=5, shuffle=True)
    for trni, tsti in skf.split(logpost, targets):
        trnf = torch.as_tensor(logpost[trni], dtype=torch.float32)
        tstf = torch.as_tensor(logpost[tsti], dtype=torch.float32)
        trnt = torch.as_tensor(targets[trni], dtype=torch.int64)
        tstf_cal, cal_params = calibrate(trnf, trnt, tstf, AffineCalLogLoss, bias=use_bias, priors=priors, quiet=True)
        logpostcal[tsti] = tstf_cal.detach().numpy()

    return logpostcal

