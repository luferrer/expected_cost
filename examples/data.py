""" Utilities for loading scores or creating simulated data.
"""

import numpy as np
from scipy.stats import norm, multivariate_normal
from scipy.special import logsumexp 
from expected_cost import utils

try:
    import torch    
    from expected_cost.calibration import affine_calibration_with_crossval
    has_psr = True
    #print("Found psr calibration library = %s"%has_psr)
except:
    has_psr = False


def get_llks_for_multi_classif_task(dataset, priors=None, K=100000):
    """ Load or create multi-class log (potentially scaled) likelihoods (llks).
    The method outputs both the raw (potentially miscalibrated) scores and 
    calibrated scores.
         
    Two datasets are implemented here:
    * cifar10: pre-generated scores for cifar10 data
    * gaussian_sim: simulated data with Gaussian distributions. In this case,
      the priors and K need to be set to determine the class priors and the 
      total number of samples required.
      
    The cifar10 example can be used as template add a loader for your own set 
    of scores. If your scores are posteriors instead of llks, you can either: 
    * ignore this fact and have this method output the posteriors (in this case,  
      remember to call the expected cost methods with score_type="log-posteriors")
    * convert them to estimated llks by using logpost_to_log_scaled_lks where the 
      priors can be estimated as those used in training.

    """

    if 'cifar10' in dataset:

        # Load pre-generated scores for cifar10 data
        print("\n**** Loading in CIFAR10 data ****\n")
        preacts = np.load("data/resnet-50_cifar10/predictions.npy")
        targets = np.load("data/resnet-50_cifar10/targets.npy")

        # Compute log-softmax to get log posteriors. 
        raw_logpost = preacts - logsumexp(preacts, axis=-1, keepdims=True)

        if has_psr:
            # Calibrate the scores with cross-validation using an affine transform
            # trained with log-loss (cross-entropy)
            print("Calibrating data")
            cal_logpost = affine_calibration_with_crossval(raw_logpost, targets)
        else:
            # If calibration code is not available, load a pre-generated file
            cal_logpost = np.load("data/resnet-50_cifar10/predictions_cal_10classes.npy")

        # For this dataset, the posteriors coincide with the scaled likelihoods
        # because the priors are uniform. If that was not the case, we would use
        # utils.logpost_to_log_scaled_lks(logpost, priors), where the priors are 
        # those used in training.
        raw_llks = raw_logpost
        cal_llks = cal_logpost

    elif dataset == 'gaussian_sim':

        # Create scores using a Gaussian distribution for each class. These scores will not
        # necessarily be well calibrated. 
        print("\n**** Creating simulated data with Gaussian class distributions ****\n")
        np.random.seed(0)

        counts = np.array(np.array(priors)*K, dtype=int)
        C = len(counts)
        
        # We need C means, each of dimension C-1 (since the first 
        # dimension of the simulated scores is always 0, because the
        # scores are meant to be log scaled-likelihoods and the scale
        # is taken to be the likelihood for the first class)
        # We set these means to produce scores that tend to be larger
        # for the true class.
        alpha = 1.5
        m = -0.5 * np.ones(C-1)
        means = [m]
        for i in np.arange(1,C):
            m = np.zeros(C-1)
            m[i-1] = alpha
            means.append(m)

        # Use the same diagonal covariance matrix for all classes.
        std = 0.8
        stds   = np.ones([C,C-1]) * std

        # Draw scores from these distributions
        raw_llks, targets = draw_scores_for_gaussian_model(means, stds, counts)

        # Now get the LLRs given the model we chose for the distributions
        # These new scores are well-calibrated by definition
        cal_llks = get_llks_for_gaussian_model(raw_llks, means, stds)

    else:
        raise Exception("Unrecognized dataset name: %s"%dataset)


    return targets, raw_llks, cal_llks



def get_llrs_for_bin_classif_task(dataset, prior1=None, K=100000):
    """ Load or create log-likelihood ratios (llrs) for a binary classification 
    task. The method outputs both the raw (potentially miscalibrated) scores and 
    calibrated scores.
         
    Two datasets are implemented here:
    * cifar10_first2classes: pre-generated scores for cifar10 data. We simulate 
      a 2-class problem by selecting just the first two scores.
    * gaussian_sim: simulated data with Gaussian distributions. In this case,
      the prior for class 1 and K, the total number of required samples, need 
      to be set.
      
    The cifar10 example can be used as template add a loader for your own set 
    of scores. If your scores are posteriors of llrs, you can either: 
    * ignore this fact and have this method output the posteriors (in this case,  
      remember to call the expected cost methods with score_type="log-posteriors")
    * convert them to estimated llrs using logpost_to_llrs with priors estimated 
      based on the training data and procedure
    """

    if 'cifar10_first2classes' in dataset:

        # Load some scores from a resnet and targets for cifar10 data
        print("\n**** Loading in CIFAR10 data (first two classes) ****\n")
        preacts = np.load("data/resnet-50_cifar10/predictions.npy")
        targets = np.load("data/resnet-50_cifar10/targets.npy")
        
        # Keep only two of the scores and the samples from those two classes
        # to fake a 2-class problem.
        sel = targets <= 1
        targets = targets[sel]
        preacts = preacts[sel][:,:2]

        # Compute log-softmax to get log posteriors. 
        raw_logpost = preacts - logsumexp(preacts, axis=-1, keepdims=True)

        if has_psr:
            # Calibrate the scores with cross-validation using an affine transform
            # trained with log-loss (cross-entropy)
            cal_logpost = affine_calibration_with_crossval(raw_scores, targets)
        else:
            # If calibration code is not available, load a pre-generated file
            cal_logpost = np.load("data/resnet-50_cifar10/predictions_cal_first2classes.npy")

        # Turn the log-posteriors to LLRs. Priors are uniform on this
        # data so, to get the LLR, we just need to subtract the  log
        # posterior for class 1 from the log posterior from class 0. If
        # this was not the case, we would use
        # utils.logpost_to_llrs(logpost, priors), where the priors can 
        # be estimated as those used in training.
        raw_llrs = raw_logpost[:,1] - raw_logpost[:,0]
        cal_llrs = cal_logpost[:,1] - cal_logpost[:,0]


    elif dataset == 'gaussian_sim':

        # Create scores using a Gaussian distribution for each class. These scores will not
        # necessarily be well calibrated.
        print("\n**** Creating simulated data with Gaussian class distributions ****\n")
        np.random.seed(0)

        counts = [int(K * (1-prior1)), int(K * prior1)]
        means = [-1.5, 1.0]
        stds = [1.0, 1.0]

        # Generate random scores (raw log scaled-likelihoods) using the gaussian distribution for
        # each class. The raw_scaled_lks have a 0 as the value for the first class since the
        # method assumes the first lk is used as scale (ie, all lks are divided by that value). 
        # The value for the second class is then the llr.
        raw_scaled_lks, targets = draw_scores_for_gaussian_model(means, stds, counts)
        raw_llrs = raw_scaled_lks[:,1]

        # Now get the LLRs given the model we chose for the distributions
        # These new scores are well-calibrated by definition
        llks = get_llks_for_gaussian_model(raw_scaled_lks, means, stds)
        cal_llrs = llks[:,1] - llks[:,0]

    else:
        raise Exception("Unrecognized dataset name: %s"%dataset)


    return targets, raw_llrs, cal_llrs



def draw_scores_for_gaussian_model(means, stds, counts):
    """ 
    Draw scores for C classes. The scores are meant to be (potentially)
    misscalibrated log scaled likelihoods. We assume the scale is
    given by the lk for the first class. Hence, the first component 
    of the score vector is always 0 (log of 1). We assume that dimensions 
    2 through C are Gaussian with diagonal covariance, with different 
    means and stds for each class. 

    The means, stds and counts arguments are lists of size C
    containing the mean, std and number of samples for each class.
    Each mean and std is a vector of dimension C-1. """

    scores = []
    targets = []
    for i, (mean, std, count) in enumerate(zip(means, stds, counts)):
        scores.append(multivariate_normal.rvs(mean, std, count))
        targets.append(np.ones(count)*i)

    # Concatenate a column of 0s to the scores which corresponds to the
    # log scaled likelihood for the last class.
    scores = np.concatenate(scores)
    scores = np.c_[np.zeros(scores.shape[0]), scores]

    return scores, np.array(np.concatenate(targets), dtype=int)


def get_llks_for_gaussian_model(scores, means, stds):
    """ Assuming that the input scores were created with 
    draw_scores_for_gaussian_model, get the log likelihoods
    for each class for the input scores.
    """

    llks = []
    for mean, std in zip(means, stds):
        llks.append(np.atleast_2d(np.log(multivariate_normal(mean, std).pdf(scores[:,1:]))))

    return np.concatenate(llks).T

