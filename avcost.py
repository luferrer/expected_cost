""" Methods for compute the average cost on hard decisions and 
on optimal Bayes decisions based on system scores.
Written by Luciana Ferrer.
"""

from sklearn.metrics._classification import  _check_targets, check_consistent_length
import numpy as np
from scipy.sparse import coo_matrix
from scipy.special import logsumexp 


def average_cost(targets, decisions, costs=None, priors=None, sample_weight=None, adjusted=False):
    """Compute the average cost.

    The average cost is a generalization of both  the error rate (1-accuracy)
    and balanced error rate (1-balanced accuracy) for cases in which the
    different  types of error have different costs.  It is given by:

    :math:`\sum_j \sum_i c_{ij} P_i R_{ij}`

    where :math:`c_{ij}` is the cost assigned to decision j when the true
    class is i, :math:`P_i` is the prior probability of class i, and
    :math:`R_{ij}`  is the fraction of samples of class i for which decision j
    was made.

    Note that class priors can be set to arbitrary values, which is 
    useful when the priors in the evaluation data do not coincide with the
    ones expected at deployment of the system. In this case, the priors can 
    be set to the expected ones and the priors in the data will be ignored. 

    Finally, the average cost allows the set of the decisions to be different
    from the set of targets. This can be used, for example, in cases in which
    what needs to be evaluated are actions taken based on the system's output.
    The decisions could,  for example, include a ``reject'' option that makes
    no decision about the class when the system is not certain enough to 
    select any class.

    When using adjusted=True, the average cost is divided by the performance
    of a naïve system that always chooses the same lower-cost decision. Hence,
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
        Balanced accuracy score.
    
    See Also
    --------

    cost_matrix : Class to define the cost matrix including method
        to conver from utility to cost matrix, to check degeneracy
        and to standardize it.

    average_cost_for_bayes_decisions : Average cost taking continuous
        scores as input instead of decisions, assuming decisions are
        made optimally using Bayes decision theory for the specified 
        cost function (given by the cost matrix and priors).

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
        costs = cost_matrix.zero_one_costs(len(priors))
    cmatrix = costs.get_matrix()

    # The confusion matrix, when normalized by the true class (the target)
    # contains the R_ij we need for computing the cost.
    R = generalized_confusion_matrix(targets, decisions, sample_weight=sample_weight, normalize="true",
        num_targets = cmatrix.shape[0], num_decisions = cmatrix.shape[1])

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
        "true"), by column (should be "pred"), or by the total number 
        of samples ("all".

    num_targets :  total number of target classes. If None, the max
        index in targets + 1 is assumed to be the total number of 
        classes.

    num_decisions :  total number of possible decisions. If None, 
        the max index in targets + 1 is assumed to be the total number 
        of classes.

    Returns
    -------
    confusion_matrix : matrix with the, optionally normalized, counts
        for each type of error

    """

    lab_type, targets, decisions = _check_targets(targets, decisions)
    if lab_type not in ("binary", "multiclass"):
        raise ValueError("%s is not supported" % lab_type)

    if sample_weight is None:
        sample_weight = np.ones(targets.shape[0], dtype=np.int64)
    else:
        sample_weight = np.asarray(sample_weight)

    if sample_weight.dtype.kind in {"i", "u", "b"}:
        dtype = np.int64
    else:
        dtype = np.float64

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




def bayes_decisions(scores, costs, priors=None, score_type='log_posteriors'):
    """ Make Bayes decisions for the given costs and scores. Bayes decision 
    are those that optimize the given cost function assuming the system produces
    well-calibrated posteriors. They are given by:

    :math:`\argmin_j \sum_i c_{ij} p_i`

    where :math:`p_i` is the posterior for class i for the sample.
    
    For flexibility, three options are considered for the input scores:

        * Scores are posteriors. In this case, the prior argument should be
          None. Note that these posteriors will only be good for optimizing
          a cost function that assumes the same priors that are implicit in the
          posteriors (usually, the same priors as in the training data, unless
          balancing or resampling approaches were used for training).

        * Scores are log_posteriors. Same comments as above apply.

        * Scores are log_likelihoods. In this case, the priors are used to 
          convert these scores into posteriors using Bayes rule:

          :math:`p_i = lk_i prior_i / \sum_j  lk_j prior_j`
          
          where lk is exp(log_likelihood). 

          Note that, given that a normalization is needed to obtained the
          posteriors the likelihoods provided can be scaled by a factor that
          does not depend on the classes and it will not affect results. For
          example, the scores could simply be

          score_i = log(posterior_from_model_for_class_i/prior_class_i)
          
          where the prior should be the one obtained by averaging the posterior
          over the evaluation data. These scores can then be used as input to 
          this method setting score_type="log_likelihoods". For more on this, 
          see the average_cost_for_bayes_decisions method.
          Note that the log-scale is important here to avoid numerical issues 
          during the normalization.

        * Scores are log-likelihoood-ratio. This is only a valid input score when
          the task is binary classification. In this case, the posterior for class
          0 is obtained as:

          :math:`p_0 = sigmoid(log-likelihoood-ratio + np.log(prior_0/prior_1))`
          
          and :math:`p_1 = 1 - p_0`

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
    
    score_type : string describing the type of input score. See description above.

    """

    cmatrix = costs.get_matrix()

    if score_type == "log_likelihood_ratio" and cmatrix.shape[0] != 2:
        raise ValueError("Score type log_likelihood_ratio can only be used for binary "+
            "classification tasks, but your cost matrix has more than two targets.") 

    if 'posterior' in score_type and priors is not None:
        print("You provided posteriors and priors as input to bayes_decisions. "+
            "When posteriors are provided as score_type, priors are ignored. "+
            "The decisions will only be optimal for cost functions that assume the same "+
            "priors that are implicit in the posteriors provided.")

    posteriors = get_posteriors_from_scores(scores, priors, score_type)

    return (posteriors @ cmatrix).argmin(axis=-1), posteriors


def get_posteriors_from_scores(scores, priors=None, score_type='log_posteriors'):

    if 'posterior' in score_type:

        # In this case, priors are ignored
        
        posteriors = np.exp(scores) if score_type == "log_posteriors" else scores

    else:

        # If the inputs are not posteriors, turn them into posteriors
        # using the provided priors.

        if priors is None:
            raise ValueError("Prior needs to be provided when using score_type %s"%score_type)

        priors = np.array(priors)
        priors /= np.sum(priors)

        if score_type == "log_likelihoods":
            log_posteriors_unnormed = scores + np.log(priors)
            posteriors = np.exp(log_posteriors_unnormed - logsumexp(log_posteriors_unnormed))
        
        elif score_type == "log_likelihood_ratio":

            log_odds = scores + np.log(priors[0]/priors[1])
            posterior0 = 1/(1+np.exp(-log_odds))
            posteriors = np.c_[posterior0, 1-posterior0]
        
        else:
            raise ValueError("Score type %s not implemented"%score_type)

    return posteriors


class cost_matrix:
    """ Utility class to define and work with cost matrices.
    The cost matrix has one row per true class and 
    one column per decision. Entry (i,j) in the matrix corresponds 
    to the cost we want the model to incur when it decides j
    for a sample with true class i.
    """
    def __init__(self,costs):
        self.costs = np.array(costs)
        if np.any(costs)<0:
            print("Cost matrix contains negative elements. Consider running self.normalize "+
                "to make sure all components are positive. This transformation does not change "+ 
                "the optimal decisions or the ranking of systems evaluated with this cost and "+
                "it ensures that the minimum value of the average_cost is 0.")


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
    def from_utilities(utilites):
        """ Obtain a cost matrix from a utility matrix where better
        decisions are given higher values """
        return cost_matrix(-utilities).normalize()

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
        
        return cost_matrix(c)


def average_cost_for_bayes_decisions(targets, scores, costs=None, priors=None, sample_weight=None, 
    adjusted=False, score_type='log_posteriors'):
    """ Average cost for Bayes decisions given the provided scores.
    The decisions are optimized for the provided costs and priors, assuming
    that the scores can be used to obtain well-calibrated posteriors.
    Note that the decisions will not be optimal if the prior implicit in the
    posteriors on the evaluation data does not coincide with the priors
    used in the cost function. The method does a sanity check of this and
    reports a warning when this is not the case.
    See the average_cost method for a description of the inputs. The only
    difference between the two methods is that average_cost takes decisions
    as input, while this method takes scores. The types of scores allowed as
    input are described in the bayes_decision method.
    """

    decisions, posteriors = bayes_decisions(scores, costs, priors, score_type)
    cost = average_cost(targets, decisions, costs, priors, sample_weight, adjusted)

    # When posteriors are provided, we also return the cost
    # could have obtained if we converted the posteriors to scaled-likelihoods
    # and then back to posteriors using perfectly matched priors.
    # When other score types are used, the posteriors are computed for the
    # right priors so there is nothing to check in that case.
    if 'posterior' in score_type:

        priors_from_posterior = np.mean(posteriors, axis=0, keepdims=True)
        if score_type == 'posteriors':
            scores_without_prior = scores / priors_from_posterior
        elif score_type == 'log_posteriors':
            scores_without_prior = scores - np.log(priors_from_posterior)
        else:
            raise ValueError('Score type %s not implemented'%score_type)

        if priors is None:
            priors = np.bincount(targets)/len(targets)

        decisions_for_matched_priors, _ = bayes_decisions(scores_without_prior, costs, priors, score_type='log_likelihoods')
        cost_for_matched_priors = average_cost(targets, decisions_for_matched_priors, costs, priors, sample_weight, adjusted)

        return cost, decisions, cost_for_matched_priors

    else:

        return cost, decisions



def average_cost_for_optimal_decisions(targets, scores, costs=None, priors=None, sample_weight=None, 
    adjusted=False, score_type='log_posteriors'):
    """ Average cost for optimal decisions given the provided scores. Only 
    applicable to binary classification. The optimal decisions are made
    by choosing the decision threshold on the posterior for class 0 to
    that optimizes the cost function defined by the costs and priors.
    This is the minimum cost that can be obtained on this data if one
    could estimate the threshold perfectly.
    """

    posteriors = get_posteriors_from_scores(scores, priors, score_type)
    
    # Now

    cost = average_cost(targets, decisions, costs, priors, sample_weight, adjusted)

    return cost, decisions



