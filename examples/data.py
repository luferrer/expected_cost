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


def get_scaled_llks_for_multi_classif_task(dataset, priors=None, K=100000):

    if 'cifar10' in dataset:

        # Load some scores from a resnet and targets for cifar10 data
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

        # Ccreate scores using a Gaussian distribution for each class. These scores will not
        # necessarily be well calibrated. 
        print("\n**** Creating simulated data with Gaussian class distributions and uniform priors ****\n")

        np.random.seed(0)

        counts = np.array(np.array(priors)*K, dtype=int)
        C = len(counts)
        
        # Draw the means from a gaussian distribution centered at 0 with 
        # diagonal covariance with std=10 for all dimensions
        # We need C means, each of dimension C-1 (since the first 
        # dimension of the simulated scores is always 0, because the
        # scores are meant to be log scaled-likelihoods and the scale
        # is taken to be the likelihood for the first class)
        # We set these means to produce scores that tend to be larger
        # for the true class.
        alpha = 1.5
        std = 0.8
        m = -0.5 * np.ones(C-1)
        means = [m]
        for i in np.arange(1,C):
            m = np.zeros(C-1)
            m[i-1] = alpha
            means.append(m)

        # All classes have identity covariance 
        stds   = np.ones([C,C-1]) * std

        raw_llks, targets = draw_scores_for_gaussian_model(means, stds, counts)

        # Now get the LLRs given the model we chose for the distributions
        # These new scores are well-calibrated by definition
        cal_llks = get_llks_for_gaussian_model(raw_llks, means, stds)

    else:
        raise Exception("Unrecognized dataset name: %s"%dataset)


    return targets, raw_llks, cal_llks



def get_llrs_for_bin_classif_task(dataset, prior1=None, K=100000):
    """ Load or simulate data for binary classification. For the simulations,
    K and P1 are needed, indicating the number if required samples, and the
    proportion of those samples that should be from class 1."""


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
        # utils.logpost_to_llrs(logpost, priors), where the priors are
        # those used in training.
        raw_llrs = raw_logpost[:,1] - raw_logpost[:,0]
        cal_llrs = cal_logpost[:,1] - cal_logpost[:,0]


    elif dataset == 'gaussian_sim':

        # Ccreate scores using a Gaussian distribution for each class. These scores will not
        # necessarily be well calibrated.
        print("\n**** Creating simulated data with Gaussian class distributions ****\n")

        np.random.seed(0)

        counts = [int(K * (1-prior1)), int(K * prior1)]
        means = [-1.5, 1.0]
        stds = [1.0, 1.0]

        # Generate random scores (raw log scaled-likelihoods) using the gaussian distribution for
        # each class. The raw_scaled_lks have a 1 as the value for the first class.
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
    
    """ Draw scores for each of C classes each with a Gaussian
    distribution. The scores are meant to be (potentially)
    misscalibrated log scaled likelihoods. We assume the scale is
    given by  the lk for the first class (any scale that is not class
    dependent will work). Hence, the log scaled lks live in  a C-1
    dimensional space. We assume that those C-1 dimensional vectors 
    are Gaussian with diagonal covariance, with different means and 
    stds for each class. 

    The means, stds and counts arguments are lists of size C
    containing the mean, std and number of samples for each class.
    Each mean and std is a vector of dimension C-1. """

    scores = []
    targets = []
    for i, (mean, std, count) in enumerate(zip(means, stds, counts)):
        scores.append(multivariate_normal.rvs(mean, std, count))
        targets.append(np.ones(count)*i)

    # Concatenate a column of 1s to the scores which corresponds to the
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

