import numpy as np
from scipy.special import logsumexp 
from expected_cost import ec, utils
from data import get_scaled_llks_for_multi_classif_task
import sys

adjusted_cost = True

# Template for analysis of multi-class classification scores. To change the input
# data, you can add your own loading process in the get_scaled_llks_for_multi_classif_task.
# Note that the method should return log (potentially scaled) likelihoods. If your method
# generates posteriors instead of likelihoods, you can estimate likelihoods by dividing
# the posteriors by the priors used for training the model or, alternatively, by the
# the average posterior over the data. 
C = 10
p0 = 0.8
data_priors =  np.ones(C)/C
targets, raw_llks, cal_llks = get_scaled_llks_for_multi_classif_task('gaussian_sim', data_priors, 100000)

# For this exercise, we want to compare metrics computed with scaled lks as scores vs
# those computed with posteriors. Hence, we generate posteriors using the priors
# found in the data. These are well-calibrated posteriors for this data, but if the
# priors change, these posteriors are no longer well-calibrated.
raw_logpost = utils.llks_to_logpost(raw_llks, data_priors)
cal_logpost = utils.llks_to_logpost(cal_llks, data_priors)

num_targets = np.max(targets)+1

# Make maximum a posteriori decisions. 
map_decisions = np.argmax(raw_logpost, axis=-1)
cost_01 = ec.cost_matrix.zero_one_costs(num_targets)
ac_01 = ec.average_cost(targets, map_decisions, cost_01, adjusted=adjusted_cost)

# Now make Bayes decisions based on the posteriors
ac_01_bayes,           _ = ec.average_cost_for_bayes_decisions(targets, raw_logpost, cost_01, adjusted=adjusted_cost)
ac_01_bayes_after_cal, _ = ec.average_cost_for_bayes_decisions(targets, cal_logpost, cost_01, adjusted=adjusted_cost)

print("*** Average cost for 0-1 cost matrix")
print("    Using MAP decisions:                        %.4f"%ac_01)
print("    Using Bayes decisions:                      %.4f"%ac_01_bayes)
print("    Using Bayes decisions on calibrated scores: %.4f"%ac_01_bayes_after_cal)
print("\nThese costs are the same because Bayes decisions for 0-1 cost coincide with MAP decisions and because the scores are well-calibrated for this specific cost function.\n")

# Try a cost with an abstention decision and evaluate the MAP decisions.
ab_cost = 0.4
cost_ab = ec.cost_matrix.zero_one_costs(num_targets, abstention_cost=ab_cost)
ac_ab = ec.average_cost(targets, map_decisions, cost_ab, adjusted=adjusted_cost)

# Now make Bayes decisions for that cost function
ac_ab_bayes, bayes_decisions_ab = ec.average_cost_for_bayes_decisions(targets, raw_logpost, cost_ab, adjusted=adjusted_cost)
ac_ab_bayes_after_cal,       _  = ec.average_cost_for_bayes_decisions(targets, cal_logpost, cost_ab, adjusted=adjusted_cost)

print("*** Average cost for cost matrix with abstention cost of %.1f"%ab_cost)
print("    Using MAP decisions:                        %.4f"%ac_ab)
print("    Using Bayes decisions:                      %.4f"%ac_ab_bayes)
print("    Using Bayes decisions on calibrated scores: %.4f"%ac_ab_bayes_after_cal)
print("\nThe first two costs are not the same because optimal (Bayes) decisions include %d abstentions on the hardest samples. "%len(np.where(bayes_decisions_ab!= map_decisions)[0]))
print("The fact that the last two costs are not the same means that the original scores were, in fact, not that well-calibrated since they were suboptimal for this cost function.\n")

# Non let's assume we expect that, on the actual data the system will be used for the priors
# are not uniform but rather one of the classes is much more frequent than the others:
p0 = 0.8
nu_priors = np.array([(1-p0)/(C-1)]*(C-1) + [p0] )

# Let's find the average cost for the 0-1 cost matrix when making Bayes
# decisions with the posteriors given by the system. The Bayes decisions will
# not change since the posteriors do not change, but the cost will. 
ac_01_nup_bayes, _ = ec.average_cost_for_bayes_decisions(targets, cal_logpost, cost_01, nu_priors, adjusted=adjusted_cost, silent=True)

# Now, instead of using the posteriors to make decisions, use the scale llks.
# In this case, the method computes the posteriors using the provided priors.
# Hence, the cost in this case is smaller because there is no mismatch in the priors.
ac_01_nup_bayes_with_matched_priors, _ = ec.average_cost_for_bayes_decisions(targets, cal_llks, cost_01, nu_priors, adjusted=adjusted_cost, score_type='log_likelihoods')

print("*** Average cost for 0-1 cost matrix and non-uniform priors %s"%" ".join(['{:.2f}'.format(i) for i in nu_priors]))
print("    Using Bayes decisions on calibrated scores, computing the posteriors with data priors:    %.4f"%ac_01_nup_bayes)
print("    Using Bayes decisions on calibrated scores, computing the posteriors with cost priors:    %.4f"%ac_01_nup_bayes_with_matched_priors)
print("\nThe second average cost is much lower because the posteriors computed with the data priors are quite bad for making decisions for the priors in the cost function\n")



