""" Methods for computing the average cost on hard decisions and 
on optimal Bayes decisions based on system scores.
Written by Luciana Ferrer.
"""

import numpy as np
from expected_cost import utils
from scipy.sparse import coo_matrix
from sklearn.metrics._classification import  _check_targets, check_consistent_length



def average_cost(targets, decisions, costs=None, priors=None, sample_weight=None, adjusted=False):
    """Compute the average cost.

    The average cost is a generalization of both  the error rate (1-accuracy)
    and balanced error rate (1-balanced accuracy) for cases in which the
    different  types of error have different costs.  It is given by:

    1/N sum_n cost(n) 

    where the sum goes over all the N test samples and cost(n) is the cost
    incurred for sample n. This expression can be  converted into the
    following double sum, by observing that the cost is the same for all
    samples for which the class and the decision is  the same:

    sum_j sum_i c_{ij} P_i R_{ij}

    where c_{ij} is the cost assigned to decision j when the true class is i,
    P_i is the fraction of samples that belong to class i, ie, the empirical
    prior of class i in the test data, and R_{ij}  is the fraction of samples
    of class i for which decision j was made.

    Note that class priors can be set to arbitrary values, which is  useful
    when the priors in the evaluation data do not coincide with the ones
    expected at deployment of the system. In this case, the priors can  be set
    to the expected ones and the priors in the data will be ignored.

    Finally, the average cost allows the set of the decisions to be different
    from the set of targets. This can be used, for example, in cases in which
    what needs to be evaluated are actions taken based on the system's output.
    The decisions could, for example, include an ``abstain'' option to make no
    decision about the class when the system is not certain enough to  select
    any class.

    When using adjusted=True, the average cost is divided by the performance
    of a naïve system that always chooses the same lowest-cost decision. Hence,
    for the adjusted cost, any value above 1.0 implies that the system is worse
    than the naïve system.

    For a comparison of the average cost to many other classification metrics,
    including the F-score, standard and balanced accuracy, Mathew's
    correlation coefficient and others, see [1].

    Parameters 
    ----------

    targets : 1d array-like of size N
        Ground truth (correct) target values for a set of N samples. 
        Should take values between 0 and C-1, where C is the number
        of possible class targets.

    decisions : 1d array-like of size N
        Decisions made for each of the N samples above. Should take
        values between 0 and D-1, where D is the number of possible 
        decisions. In the standard case, the set of decisions and the 
        set of targets are the same (ie, C=D), but this function is 
        more general and allows for the case in which the set of 
        decisions is not the same as the set of targets.

    sample_weight : array-like of size N, default=None
        Sample weights used to compute the confusion matrix from 
        which the cost is then computed.

    costs : an object of class cost_matrix specifying the cost
        for each combination of i and j, where i and j
        are the true class and the decision indices for a sample. If 
        set to None, the standard zero-one cost matrix is used, which
        results in the average_cost coinciding with the error rate.

    priors : the class priors required for evaluation. If set to None,
        the priors are taken from the data. 

    adjusted : bool, default=False
        When true, the result is adjusted for chance, so that a naïve 
        system would score exactly 1.0. 

    Returns
    -------

    average_cost : float
        The average cost over the data.
    
    See Also
    --------

    cost_matrix : Class to define the cost matrix including methods
        to conver from utility to cost matrix and to standardize it.

    average_cost_for_bayes_decisions : Average cost taking continuous
        scores as input instead of decisions, assuming decisions are
        made using Bayes decision theory.

    bayes_decisions : method to make optimal decisions using Bayes
        decision theory.

    References
    ----------
    .. [1] Ferrer, L. "Analysis and comparison of classification metrics"
           arXiv:2209.05355

    Examples
    --------
    >>> targets   = [0, 1, 0, 0, 1, 0]
    >>> decisions = [0, 1, 0, 0, 2, 2]
    >>> costs = cost_matrix([[0, 1, 0.5], [1, 0, 0.5]])
    >>> average_cost(targets, decisions, costs)
    0.16666
    """

    if priors is None:
        priors = np.bincount(targets)/len(targets)
    priors = priors[:,np.newaxis]

    if costs is None:
        costs = CostMatrix.zero_one_costs(len(priors))
    cmatrix = costs.get_matrix()

    # The confusion matrix, when normalized by the true class (the target)
    # contains the R_ij we need for computing the cost.
    R = generalized_confusion_matrix(targets, decisions, sample_weight=sample_weight, normalize="true",
        num_targets = cmatrix.shape[0], num_decisions = cmatrix.shape[1])

    # Return average cost  
    return average_cost_from_confusion_matrix(R, priors, costs, adjusted)

def average_cost_from_confusion_matrix(R, priors, costs, adjusted=False):
    """Compute the average cost as in the average_cost method but taking
    a confusion matrix as input.

    Parameters 
    ----------

    R : confusion matrix, where Rij contains the fraction of samples of
        class i for which decision j was made (can be obtained with
        generalized_confusion_matrix with normalize="true")

    costs : an object of class cost_matrix specifying the cost
        for each combination of i and j, where i and j
        are the true class and the decision indices for a sample. If 
        set to None, the standard zero-one cost matrix is used, which
        results in the average_cost coinciding with the error rate.

    priors : the class priors required for evaluation. If set to None,
        the priors are taken from the data. 

    adjusted : bool, default=False
        When true, the result is adjusted for chance, so that a naïve 
        system would score exactly 1.0. 

    Returns
    -------

    average_cost : float
        The average cost over the data.
    """    

    cmatrix = costs.get_matrix()

    if priors.ndim==1:
        priors = priors[:,np.newaxis]

    # Compute the average cost as the sum of all c_ij P_i R_ij
    ave_cost = np.sum(priors * cmatrix * R)

    if adjusted:
        # When adjusted is true, normalize the average cost
        # with the cost of a naive system that always makes
        # the min cost decision.
        norm_value = np.min(np.dot(priors.T, cmatrix))
    else:
        norm_value = 1.0

    return ave_cost / norm_value

def get_posteriors_from_scores(scores, priors=None, score_type='log_posteriors'):
    """ Convert scores into posteriors depending on their type. See method
    bayes_decisions for more details."""

    if score_type in ['posteriors', 'log_posteriors']:
        # In this case, priors are ignored
        posteriors = np.exp(scores) if score_type == "log_posteriors" else scores

    else:
        # If the inputs are not posteriors, turn them into posteriors
        # using the provided priors.
        if priors is None:
            raise ValueError(
                f"Prior needs to be provided when using score_type {score_type}"
            )

        priors = np.array(priors)/np.sum(priors)

        if score_type == "log_likelihoods":
            posteriors = np.exp(utils.llks_to_logpost(scores, priors))

        elif score_type == "log_likelihood_ratios":
            posteriors = np.exp(utils.llrs_to_logpost(scores, priors))

        else:
            raise ValueError(f"Score type {score_type} not implemented")

    # Make sure the posteriors sum to 1
    posteriors /= np.sum(posteriors, axis=1, keepdims=True)

    return posteriors

def average_cost_for_bayes_decisions(targets, scores, costs=None, priors=None, sample_weight=None, 
    adjusted=False, score_type='log_posteriors', silent=False):
    """ Average cost for Bayes decisions given the provided scores. 
    The decisions are optimized for the provided costs and priors,
    assuming that the scores can be used to obtain well-calibrated
    posteriors. 

    Note that if the scores are posteriors or log-posteriors and the
    priors in the data, or those provided externally, are not well
    matched to those used while training the system, the cost will 
    not be optimal. 
    
    Parameters 
    ----------

    targets : 1d array-like of size N
        Ground truth (correct) target values for a set of N samples. 
        Should take values between 0 and C-1, where C is the number
        of possible class targets.

    scores : array-like of size NxC
        Scores can be posteriors, log-posteriors, log-likelihoods
        or log-likelihood ratios (see method bayes_decisions for 
        a detailed explanation)

    sample_weight : array-like of size N, default=None
        Sample weights used to compute the confusion matrix from 
        which the cost is then computed.

    costs : an object of class cost_matrix specifying the cost
        for each combination of i and j, where i and j
        are the true class and the decision indices for a sample. If 
        set to None, the standard zero-one cost matrix is used, which
        results in the average_cost coinciding with the error rate.

    priors : the class priors required for evaluation. If set to None,
        the priors are taken from the data. 

    adjusted : bool, default=False
        When true, the result is adjusted for chance, so that a naïve 
        system would score exactly 1.0. 

    score_type: string, default='log_posteriors'
        The type of scores provided. Can be posteriors, log-posteriors,
        log_likelihoods or log_likelihood_ratios.

    silent : If true, a warning is output when posteriors and priors are 
        provided.

    Returns
    -------

    average_cost : float
        The average cost over the data for Bayes decisions.

    decisions : array of size N
        The Bayes decisions that correspond to the computed cost

    """

    decisions, posteriors = bayes_decisions(scores, costs, priors, score_type, silent=silent)
    cost = average_cost(targets, decisions, costs, priors, sample_weight, adjusted)

    return cost, decisions

def average_cost_for_optimal_decisions(targets, scores, costs=None, priors=None, sample_weight=None, 
    adjusted=False, score_type='log_posteriors'):
    """ Average cost for optimal decisions given the provided scores. 
    Only applicable to binary classification when the cost matrix has 
    the following form:
    
                             0  c01
                            c10  0

    The optimal decisions are made by choosing the decision threshold on the
    posterior for class 1 to the one that optimizes the cost function defined by
    the costs and priors. This is the minimum cost that can be obtained on this
    data if one could estimate the threshold perfectly, i.e., it is optimistic
    as it cheats in the selection of the threshold, but it is useful as a reference
    of what is the best cost one can get on this data.
    
    """

    if np.max(targets)>1:
        raise ValueError("This method can only be used for binary classification.")

    cmatrix = costs.get_matrix()

    if np.any(np.array(cmatrix.shape) != 2) or cmatrix[0,0] != 0 or cmatrix[1,1] != 0:
        raise ValueError("This method is only valid for cost matrices of the form: [[0, c01], [c10, 0]]")

    if sample_weight is not None:
        raise ValueError("sample_weight option not implemented for this method")


    posteriors = get_posteriors_from_scores(scores, priors, score_type)
            
    N = len(targets)
    N1 = np.sum(targets==1)
    N0 = np.sum(targets==0)

    # Create an array with posteriors for class 1 and targets
    post1_with_target = np.c_[posteriors[:,1], targets]

    # Sort by the posterior for class 1
    post1_with_target = post1_with_target[post1_with_target[:,0].argsort(),]

    # Below, we create vectors R01 and R10 which will contain
    # the two error rates for all possible threshold values.
    # Rij is the fraction of samples from class i labelled as
    # class j

    sum1 = np.cumsum(post1_with_target[:,1])
    sum0 = N0 - (np.arange(1,N+1)-sum1)

    R10     = np.zeros(N+1,np.float32) # 1 more for the boundaries
    R10[0]  = 0.0
    R10[1:] = sum1 / N1
    
    R01     = np.zeros_like(R10)
    R01[0]  = 1.0
    R01[1:] = sum0 / N0

    # Now, using those two vectors we can compute the corresponding 
    # vector of average costs

    if priors is None:
        priors = np.bincount(targets)/len(targets)

    ave_cost = cmatrix[0,1] * priors[0] * R01 + cmatrix[1,0] * priors[1] * R10

    if adjusted:
        # When adjusted is true, normalize the average cost
        # with the cost of a naive system that always makes
        # the min cost decision.
        norm_value = np.min(np.dot(priors.T, cmatrix))
    else:
        norm_value = 1.0

    return np.min(ave_cost)/norm_value
class CostMatrix:
    """ 
    Utility class to define and work with cost matrices. The cost matrix has one
    row per true class and  one column per decision. Entry (i,j) in the matrix
    corresponds  to the cost we want the model to incur when it decides j for a
    sample with true class i.
    """

    def __init__(self,costs):
        self.costs = np.array(costs)
        if np.any(costs)<0:
            print("Cost matrix contains negative elements. Consider running self.normalize "+
                "to make sure all components are positive. This transformation does not change "+ 
                "the optimal decisions or the ranking of systems evaluated with this cost and "+
                "it ensures that the minimum value of the average_cost is 0 making it easier "+
                "to interpret.")


    def normalize(self):
        """ Subtract the minimum from each row (ie, from the costs for
        all decisions given the same class). This transformation does not change 
        the optimal decisions or the ranking of systems evaluated with this cost and 
        it ensures that the minimum value of the average_cost is 0.
        """
        self.cost -= self.cost.min(axis=0)

    def get_matrix(self):
        return self.costs

    @staticmethod
    def from_utilities(utilities):
        """ Obtain a cost matrix from a utility matrix where better
        decisions are given higher values. """
        return CostMatrix(-utilities).normalize()

    @staticmethod
    def zero_one_costs(C, abstention_cost=None):
        """ Create a cost_matrix object with costs of 0 in the 
        diagonal and 1 elsewhere. This is the cost matrix that
        leads to the average_cost coinciding with the usual error 
        rate. The parameter C indicates the size of the matrix 
        (ie, the number of possible targets and decisions). 
        If abstention_cost is not None, an additional decision
        is included as the last column with cost given by the
        value of this argument. 
        """
        c = 1-np.eye(C)
        if abstention_cost is not None:
            c = np.c_[c, abstention_cost*np.ones(c.shape[0])]
        
        return CostMatrix(c)

def generalized_confusion_matrix(targets, decisions, sample_weight=None, normalize=None, num_targets=None, num_decisions=None):
    """ Get the confusion matrix between targets and decisions. This is a
    generalization of the usual confusion matrix where the set of decisions are
    restricted to be the same as the set of targets. The element ij of the
    confusion matrix contains the (weighted) number of samples of class i for
    which decision j was made.
 
    Parameters 
    ----------

    targets : 1d array-like of size N
        Ground truth (correct) target values for a set of N samples. 
        Should take values between 0 and C-1, where C is the number
        of possible targets.

    decisions : 1d array-like of size N
        Decisions made for each of the N samples above. Should take
        values between 0 and D-1, where D is the number of possible 
        decisions.

    sample_weight : array-like of size N, default=None
        Sample weights.

    normalize :  normalize the confusion matrix by row (should be 
        "true"), by column ("pred"), or by the total number of
        samples ("all").

    num_targets :  total number of target classes. If None, the max
        index in targets + 1 is assumed to be the total number of 
        classes.

    num_decisions :  total number of possible decisions. If None, 
        the max index in decisions + 1 is assumed to be the total 
        number of classes.

    Returns
    -------
    confusion_matrix : matrix with the, optionally normalized, counts
        for each type of error

    """

    lab_type, targets, decisions = _check_targets(targets, decisions)
    if lab_type not in ("binary", "multiclass"):
        raise ValueError(f"{lab_type} is not supported")

    if sample_weight is None:
        sample_weight = np.ones(targets.shape[0], dtype=np.int64)
    else:
        sample_weight = np.asarray(sample_weight)

    dtype = np.int64 if sample_weight.dtype.kind in {"i", "u", "b"} else np.float64
    check_consistent_length(targets, decisions, sample_weight)

    if num_targets is None:
        num_targets = np.max(targets)+1
    if num_decisions is None:
        num_decisions = np.max(decisions)+1

    cm = coo_matrix((sample_weight, (targets, decisions)), shape=(num_targets, num_decisions), dtype=dtype).toarray()

    with np.errstate(all="ignore"):
        if normalize == "true":
            cm = cm / cm.sum(axis=1, keepdims=True)
        elif normalize == "pred":
            cm = cm / cm.sum(axis=0, keepdims=True)
        elif normalize == "all":
            cm = cm / cm.sum()
        cm = np.nan_to_num(cm)

    return cm

def bayes_decisions(scores, costs, priors=None, score_type='log_posteriors', silent=False):
    """ Make Bayes decisions for the given costs and scores. Bayes decision 
    are those that optimize the given cost function assuming the system produces
    well-calibrated posteriors. They are given by:

    argmin_j sum_i c_{ij} p_i

    where p_i is the posterior for class i for the sample.
    
    For flexibility, three options are considered for the input scores:

        * Scores are posteriors. In this case, the prior argument should be
          None. Note that these posteriors will only be good for optimizing
          a cost function that assumes the same priors that are implicit in the
          posteriors (usually, the same priors as in the training data, unless
          balancing or resampling approaches were used for training).

        * Scores are log_posteriors. Same comments as above apply.

        * Scores are log potentially-scaled likelihoods. In this case, the
          priors are used to convert these scores into posteriors using Bayes
          rule:

          p_i = lk_i prior_i / sum_j  lk_j prior_j
          
          where lk is exp(log_likelihood). 

          Note that, given that a normalization is needed to obtained the
          posteriors the likelihoods provided can be scaled by a factor that
          does not depend on the classes and it will not affect results. 

        * Scores are log-likelihoood-ratio. This is only a valid input score when
          the task is binary classification. In this case, the posterior for class
          0 is obtained as:

          p_0 = sigmoid(log-likelihoood-ratio + np.log(prior_0/prior_1))
          
          and p_1 = 1 - p_0

    Parameters 
    ----------
    
    scores :  array of dimension N x C, where N is the number of samples and C is
        the number of target classes, except in the case of log-likelihood ratios
        where the dimension should be N x 1.

    costs :  the cost_matrix object that defines the cost function along with the priors

    priors : array if dimension C. The priors are used to convert scores that are 
        not posteriors into posteriors. The decisions will be optimized for a
        cost function with  these same priors. Should be None if scores are
        posteriors. In that case it is expected that the posteriors are
        consistent with the priors in the evaluation data. If this is not the
        case, decisions will not be  optimal for the cost function that
        computes the priors from the data. The priors are normalized to sum to
        1 by the method.
    
    score_type : string describing the type of input score. Default=log_posteriors.
        It can be: posteriors, log_posteriors, log_likelihoods or log_likelihood_ratios.

    silent : If False, a warning is output when posteriors and priors are provided.

    """

    cmatrix = costs.get_matrix()

    if score_type == "log_likelihood_ratio" and cmatrix.shape[0] != 2:
        raise ValueError("Score type log_likelihood_ratio can only be used for binary "+
            "classification tasks, but your cost matrix has more than two targets.") 
    
    if 'posterior' in score_type and priors is not None and not silent:
        print("You provided posteriors and priors as input to bayes_decisions. "+
            "When posteriors are provided as score_type, priors are ignored. "+
            "The decisions will only be optimal for cost functions that assume the same "+
            "priors that are implicit in the posteriors provided.")

    posteriors = get_posteriors_from_scores(scores, priors, score_type)

    return (posteriors @ cmatrix).argmin(axis=-1), posteriors