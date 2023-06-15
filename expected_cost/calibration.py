from psrcal.calibration import AffineCalLogLoss
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, GroupKFold, KFold
import torch
import numpy as np
from scipy.special import logsumexp 


def train_calibrator(logpost_trn, targets_trn, calparams={}, calmethod=AffineCalLogLoss):
    trnf = torch.as_tensor(logpost_trn, dtype=torch.float32)
    trnt = torch.as_tensor(targets_trn, dtype=torch.int64)
    calmodel = calmethod(trnf, trnt, **calparams)
    calmodel.train()
    return calmodel

def calibrate_scores(logpost_tst, calmodel):
    tstf = torch.as_tensor(logpost_tst, dtype=torch.float32)
    tstf_cal = calmodel.calibrate(tstf)
    logpostcal = tstf_cal.detach().numpy()

    # Normalize them to make them log posteriors
    logpostcal -= logsumexp(logpostcal, axis=1, keepdims=True)

    return logpostcal


def calibration_with_crossval(logpost, targets, calparams={}, calmethod=AffineCalLogLoss, seed=None, 
                              condition_ids=None, stratified=True, nfolds=5):
    
    """ 
    This is a utility method for performing calibration on the test scores using cross-validation.
    If calmethod is AffineCalLogLoss of AffineCalBrier, the calibration is done using an affine
    transformation of the form:
    
    logpostcal_i = logsoftmax ( scale * logpostraw_i + bias_i)

    The scale is the same for all classes, but the bias is class-dependent. If use_bias = False,
    this method does the usual temp-scaling.

    If calmethod is HistogramBinningCal, histogram binning is done instead.

    The method expects (potentially misscalibrated) log posteriors or log scaled likelihoods as
    input. In both cases, the output should be well-calibrated log posteriors. Note, though that
    when use_bias is False, it is probably better to feed log posteriors since, without a bias term,
    the calibration cannot introduce the required priors to obtain a good log-posterior. 

    The priors variable allows you to set external priors which overwrite the ones in the test data.
    This is useful when the test data has priors that do not reflect those we expect to see when we
    deploy the system.

    The condition_ids variable is used to determine the folds per condition. This is used when the
    data presents correlations due to some factor other than the class. In that case, the
    condition_ids variable should have indexes for the condition of each sample in the logpost
    array.

    Set stratified to True if you want to assume that the priors are always known with certainty.
    Else, if you want to consider the possible random variation in priors due to the sampling of the
    data, then set stratified to False.
    """

    logpostcal = np.zeros_like(logpost)
    
    if stratified:
        if condition_ids is not None:
            skf = StratifiedGroupKFold(n_splits=nfolds, shuffle=True, random_state=seed)
        else:
            # Use StratifiedKFold in this case for backward compatibility
            skf = StratifiedKFold(n_splits=nfolds, shuffle=True, random_state=seed)
    else:
        if condition_ids is not None:
            skf = GroupKFold(n_splits=nfolds)
        else:
            skf = KFold(n_splits=nfolds, shuffle=True, random_state=seed)

    for trni, tsti in skf.split(logpost, targets, condition_ids):

        calmodel = train_calibrator(logpost[trni], targets[trni], calparams=calparams, calmethod=calmethod)
        logpostcal[tsti] = calibrate_scores(logpost[tsti], calmodel)

    return logpostcal


def calibration_train_on_heldout(logpost_tst, logpost_trn, targets_trn, calmethod=AffineCalLogLoss, calparams={}, return_model=False):
    """ Same as calibration_with_crossval but doing cheating calibration.
    """
    calmodel = train_calibrator(logpost_trn, targets_trn, calparams=calparams, calmethod=calmethod)
    logpostcal = calibrate_scores(logpost_tst, calmodel)

    if return_model:
        return logpostcal, calmodel
    else:
        return logpostcal

def calibration_train_on_test(logpost, targets, calmethod=AffineCalLogLoss, calparams={} ,return_model=False):

    return calibration_train_on_heldout(logpost, logpost, targets, calmethod=calmethod, return_model=return_model, 
                                        calparams=calparams)

